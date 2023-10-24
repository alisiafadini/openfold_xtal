import torch
from openfold.np import residue_constants
from openfold.utils.feats import atom14_to_atom37
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
import torch.linalg as LA
import openfold.np.protein as protein
from openfold.utils.tensor_utils import tensor_tree_map

from openfold.xtal.structurefactor_functions import (
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


def extract_bfactors(prot):
    atom_mask = prot.atom_mask
    aatype = prot.aatype
    b_factors = prot.b_factors

    b_factor_lines = []

    n = aatype.shape[0]
    # Add all atom sites.
    for i in range(n):
        for mask, b_factor in zip(atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue

            b_factor_lines.append(b_factor)

    return np.array(b_factor_lines)


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


def runner_prediction(af2runner, feats):
    results = af2runner.af2(feats)
    current_pos_atoms = extract_allatoms(results, feats)

    new_features_np = tensor_tree_map(
        lambda t: t[..., -1].cpu().detach().numpy(), feats
    )
    results_np = tensor_tree_map(lambda t: t.cpu().detach().numpy(), results)

    plddt_factors = results_np["plddt"][:, None] * results_np["final_atom_mask"]
    prot = protein.from_prediction(new_features_np, results_np, b_factors=plddt_factors)
    plddt_factors_out = extract_bfactors(prot)
    pseudo_bfactors = update_bfactors(plddt_factors_out)
    # prot.b_factors = update_bfactors(prot.b_factors)
    prot = protein.from_prediction(
        new_features_np, results_np, b_factors=update_bfactors(prot.b_factors)
    )

    return current_pos_atoms, results["plddt"], prot, pseudo_bfactors


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


def select_confident_atoms(current_pos, target_pos, bfacts):
    # Boolean mask for confident atoms
    mask = bfacts < 11.5
    reshaped_mask = mask.unsqueeze(1).expand_as(current_pos)

    # Select confident atoms using the mask
    current_pos_conf = torch.flatten(current_pos)[torch.flatten(reshaped_mask)]
    target_pos_conf = torch.flatten(target_pos)[torch.flatten(reshaped_mask)]

    N = current_pos_conf.numel() // 3

    return current_pos_conf.view(N, 3), target_pos_conf.view(N, 3)


def align_positions(current_pos, target_pos, bfacts=None):
    # Perform alignment of positions
    # TO DO : add statement for when bfacts are not provided
    with torch.no_grad():
        current_pos_conf, target_pos_conf = select_confident_atoms(
            current_pos, target_pos, bfacts
        )

        centroid1, centroid2, rotation_matrix = kabsch_align_matrices(
            current_pos_conf, target_pos_conf
        )

    aligned_pos = align_tensors(current_pos, centroid1, centroid2, rotation_matrix)
    return aligned_pos


def transfer_positions(aligned_pos, sfcalculator_model, b_factors):
    # Transfer positions to sfcalculator
    frac_pos = coordinates.fractionalize_torch(
        aligned_pos,
        sfcalculator_model.unit_cell,
        sfcalculator_model.space_group,
    )
    sfcalculator_model = set_new_positions(aligned_pos, frac_pos, sfcalculator_model)

    # add bfactor calculation based on plddt
    sfcalculator_model.atom_b_iso = b_factors
    sfcalculator_model = update_sfcalculator(sfcalculator_model)
    return sfcalculator_model


def update_bfactors(plddts):
    # Use Tom Terwilliger's formula to convert plddt to Bfactor and update sfcalculator instance
    deltas = 1.5 * np.exp(4 * (0.7 - 0.01 * plddts))
    b_factors = (8 * np.pi**2 * deltas**2) / 3

    return b_factors


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


def LLG_computation(
    epoch,
    runner,
    feats,
    data_dict,
    sfcalculator,
    target_pos,
    update=50,
    verbose=False,
    device=dsutils.try_gpu(),
):
    # predict positions
    current_pos, plddt, pdb, b_factors = runner_prediction(runner, feats)
    aligned_pos = align_positions(
        current_pos,
        target_pos,
        bfacts=torch.from_numpy(b_factors).to(device),
    )

    # convert to XYZ model

    sfcalculator_model = transfer_positions(
        aligned_pos, sfcalculator, torch.tensor(b_factors).to(dsutils.try_gpu())
    )

    Emodel, data_dict = normalize_modelFs(sfcalculator_model, data_dict)

    # update sigmaA if needed

    if epoch % update == 0:
        # update sigmaA
        sigmaA = sigmaa.sigmaA_from_model(
            data_dict["ETRUE"],
            data_dict["PHITRUE"],
            sfcalculator_model,
            data_dict["EPS"],
            data_dict["SIGMAP"],
            data_dict["BIN_LABELS"],
        )
        sigmaA = torch.clamp(torch.tensor(sigmaA).to(dsutils.try_gpu()), 0.001, 0.999)
        data_dict["SIGMAA"] = sigmaA

    else:
        sigmaA = data_dict["SIGMAA"]

    # calculate LLG

    llg = run_ML_binloop(
        data_dict["UNIQUE_LABELS"],
        data_dict["EDATA"],
        Emodel,
        data_dict["BIN_LABELS"],
        sigmaA,
        data_dict["CENTRIC"],
        data_dict["DOBS"],
    )

    if verbose:
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: -{llg.item()}")
            print("SigmaA tensor: ", sigmaA)

    return llg, plddt, pdb, aligned_pos


"""
def af2runner_optimization_step(
    pdb,
    si,
    # evo_output_dict,
    seq,
    af2_runner,
    sf_model,
    data_dict,
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
        af2_runner.precompute(pdb)
        outputs = af2_runner.compute(si, seq)
        # outputs, new_features = af2_runner.compute_alisia(si, evo_output_dict)
        current_pos = outputs["current_XYZ"].to(dsutils.try_gpu())
        af2_pos = outputs["prot"].atom_positions
        plddt = outputs["plddt"]
        target_pos = target_pos.to(dsutils.try_gpu())
        aligned_pos = align_positions(current_pos, target_pos, plddt).to(
            dsutils.try_gpu()
        )
        sfcalculator_model = transfer_positions(aligned_pos, sf_model)
        Emodel, data_dict = normalize_modelFs(sfcalculator_model, data_dict)
        if epoch % update == 0:
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

        loss_function = torch.nn.MSELoss()
        loss = loss_function(aligned_pos, target_pos)

        # loss = -llg
        loss.backward()
        optimizer.step()
        # print("param grad", optimizer.param_groups[0]["params"])
        # print("param grad", optimizer.param_groups[0]["params"][0].grad)

        # Print the loss during training (example)
        if verbose:
            if epoch % 1 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
                # print("SigmaA tensor: ", sigmaA)
                print("LLG", llg)

    return {
        "af2_positions": af2_pos,
        "prot": outputs["prot"],
        "loss": loss,
    }

"""
