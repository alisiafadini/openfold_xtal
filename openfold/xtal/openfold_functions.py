import torch
from openfold.np import residue_constants
from openfold.utils.feats import atom14_to_atom37
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from torch.optim.lr_scheduler import ExponentialLR
import torch.linalg as LA


from openfold.alisia.structurefactor_functions import (
    IndexedDataset,
    run_ML_binloop,
    set_new_positions,
    update_sfcalculator,
)
from xtalLLG import dsutils, structurefactors, coordinates, sigmaa


def extract_allatoms(outputs, feats):
    atom_types = residue_constants.atom_types
    pdb_lines = []
    atom_mask = outputs["final_atom_mask"]
    aatype = feats["aatype"]
    atom_positions = outputs["final_atom_positions"]
    residue_index = feats["residue_index"].to(torch.int32)

    n = aatype.shape[0]
    # Add all atom sites.
    for i in range(n):
        for atom_name, pos, mask in zip(atom_types, atom_positions[i], atom_mask[i]):
            if mask < 0.5:
                continue

            pdb_lines.append(pos)

    return torch.stack(pdb_lines)


def run_structure_module(evo_output, feats, AF_model):
    results = AF_model.structure_module(
        evo_output,
        feats["aatype"],
        mask=feats["seq_mask"],
        inplace_safe=False,
        _offload_inference=False,
    )
    results.update(AF_model.aux_heads(results))
    results["final_atom_mask"] = feats["atom37_atom_exists"]
    results["final_atom_positions"] = atom14_to_atom37(results["positions"][-1], feats)
    current_pos_atoms = extract_allatoms(results, feats)

    return current_pos_atoms, results["final_atom_positions"], results["plddt"]


def convert_feat_tensors_to_numpy(dictionary):
    numpy_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            numpy_dict[key] = value.detach().cpu().numpy()
        else:
            numpy_dict[key] = value
    return numpy_dict


def kabsch_align_matrices(tensor1, tensor2):
    # Center the atoms by subtracting their centroids
    centroid1 = torch.mean(tensor1, dim=0, keepdim=True)
    tensor1_centered = tensor1 - centroid1
    centroid2 = torch.mean(tensor2, dim=0, keepdim=True)
    tensor2_centered = tensor2 - centroid2

    # Calculate the covariance matrix
    covariance_matrix = torch.matmul(tensor2_centered.t(), tensor1_centered)

    # Perform Singular Value Decomposition (SVD) on the covariance matrix
    U, _, Vt = LA.svd(covariance_matrix)

    # Calculate the rotation matrix
    rotation_matrix = torch.matmul(U, Vt)

    return centroid1, centroid2, rotation_matrix


def align_tensors(tensor1, centroid1, centroid2, rotation_matrix):
    tensor1_centered = tensor1 - centroid1

    # Apply the rotation and translation to align the first tensor to the second one
    aligned_tensor1 = torch.matmul(tensor1_centered, rotation_matrix.t()) + centroid2

    return aligned_tensor1


def align_positions(current_pos, target_pos):
    # Perform alignment of positions
    with torch.no_grad():
        centroid1, centroid2, rotation_matrix = kabsch_align_matrices(
            current_pos, target_pos
        )
    aligned_pos = align_tensors(current_pos, centroid1, centroid2, rotation_matrix)
    return aligned_pos


def transfer_positions(aligned_pos, sfcalculator_model):
    # Transfer positions to sfcalculator
    frac_pos = coordinates.fractionalize_torch(
        aligned_pos,
        sfcalculator_model.unit_cell,
        sfcalculator_model.space_group,
    )
    sfcalculator_model = set_new_positions(aligned_pos, frac_pos, sfcalculator_model)
    sfcalculator_model = update_sfcalculator(sfcalculator_model)
    return sfcalculator_model


def normalize_modelFs(sfcalculator_model, data_dict):
    # Normalize for Emodel and Edata
    fmodel_amps = structurefactors.ftotal_amplitudes(sfcalculator_model, "Ftotal_HKL")
    sigmaP = structurefactors.calculate_Sigma_atoms(
        fmodel_amps, data_dict["EPS"], data_dict["BIN_LABELS"]
    )
    Emodel = structurefactors.normalize_Fs(
        fmodel_amps, data_dict["EPS"], sigmaP, data_dict["BIN_LABELS"]
    )
    data_dict["SIGMAP"] = sigmaP
    return Emodel, data_dict


def run_optimization_step(
    model,
    data_dict,
    feats,
    target_pos,
    epoch,
    optimizer,
    update,
    batches=1,
    verbose=False,
):
    batch_size = int(len(data_dict["EDATA"]) / batches)
    new_Edata = IndexedDataset(data_dict["EDATA"])
    data_loader = DataLoader(new_Edata, batch_size=batch_size, shuffle=False)

    for batch_data, batch_indices in data_loader:
        optimizer.zero_grad()

        current_pos, af2_pos, plddt = run_structure_module(
            model, feats, model["AF_model"]
        )

        aligned_pos = align_positions(current_pos, target_pos)
        sfcalculator_model = transfer_positions(
            aligned_pos, model["sfcalculator_model"]
        )
        Emodel, data_dict = normalize_modelFs(sfcalculator_model, data_dict)

        if epoch % update == 0:
            # Code for updating sigmaA (example)
            sigmaA = sigmaa.sigmaA_from_model(
                data_dict["ETRUE"],
                data_dict["PHITRUE"],
                sfcalculator_model,
                data_dict["EPS"],
                data_dict["SIGMAP"],
                data_dict["BIN_LABELS"],
            )
            sigmaA = torch.clamp(
                torch.tensor(sigmaA).to(dsutils.try_gpu()), 0.001, 0.999
            )
            data_dict["SIGMAA"] = sigmaA

        else:
            sigmaA = data_dict["SIGMAA"]

        llg = run_ML_binloop(
            data_dict["UNIQUE_LABELS"],
            batch_data,
            Emodel[batch_indices],
            data_dict["BIN_LABELS"][batch_indices],
            sigmaA,
            data_dict["CENTRIC"][batch_indices],
            data_dict["DOBS"][batch_indices],
        )

        loss = -llg
        loss.backward()
        optimizer.step()

        # Print the loss during training (example)
        if verbose:
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
                print("SigmaA tensor: ", sigmaA)

    return af2_pos, loss


def XYZ_refine(
    AF_model,
    sfcalculator_model,
    data_dict,
    evo_output,
    feats,
    target_pos,
    noise_term,
    lr_p,
    lr_s,
    num_epochs,
    gamma,
    batches=1,
    verbose=False,
    update=50,
    evo_out=False,
):
    # copy evo_output to prevent carrying on gradients across iterations
    new_evo = copy.deepcopy(evo_output)
    single_noise = torch.randn_like(evo_output["single"]).to(torch.device("cuda"))
    new_evo["single"] = new_evo["single"] + single_noise * (
        noise_term  # * new_evo["single"].data.abs().sqrt()
    )
    single = new_evo["single"]
    single.requires_grad_(True)

    # single = new_evo["single"].requires_grad_(True)
    pair = new_evo["pair"].requires_grad_(True)

    model = {
        "single": single,
        "pair": pair,
        "AF_model": AF_model,
        "sfcalculator_model": sfcalculator_model,
    }

    optimizer = torch.optim.Adam(
        [{"params": single, "lr": lr_s}, {"params": pair, "lr": lr_p}]
    )

    scheduler = ExponentialLR(optimizer, gamma=gamma, verbose=True)

    for epoch in tqdm(range(num_epochs)):
        result = run_optimization_step(
            model,
            data_dict,
            feats,
            target_pos,
            epoch,
            optimizer,
            update,
            batches,
            verbose,
        )

        scheduler.step()

    if evo_out:
        return result, new_evo
    else:
        return result
