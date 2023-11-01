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
    run_ML_binloop,
    set_new_positions,
    update_sfcalculator,
)
from xtalLLG import dsutils, structurefactors, coordinates, sigmaa


def move_tensors_to_device_inplace(processed_features, device=dsutils.try_gpu()):
    """
    Moves PyTorch tensors in a dictionary to the specified device in-place.

    Args:
        processed_features (dict): Dictionary containing tensors.
        device (str): Device to move tensors to (e.g., "cuda:0", "cpu").
    """
    # Iterate through the keys and values in the input dictionary
    for key, value in processed_features.items():
        # Check if the value is a PyTorch tensor
        if isinstance(value, torch.Tensor):
            # Move the tensor to the specified device in-place
            processed_features[key] = value.to(device)


def move_tensors_to_device(processed_features, device=dsutils.try_gpu()):
    """
    Moves PyTorch tensors in a dictionary to the specified device.

    Args:
        processed_features (dict): Dictionary containing tensors.
        device (str): Device to move tensors to (e.g., "cuda:0", "cpu").

    Returns:
        dict: Dictionary with tensors moved to the specified device.
    """
    # Create a new dictionary to store processed features with tensors moved to the device
    processed_features_on_device = {}

    # Iterate through the keys and values in the input dictionary
    for key, value in processed_features.items():
        # Check if the value is a PyTorch tensor
        if isinstance(value, torch.Tensor):
            # Move the tensor to the specified device
            value = value.to(device)
        # Add the key-value pair to the new dictionary
        processed_features_on_device[key] = value

    # Return the new dictionary with tensors moved to the device
    return processed_features_on_device


def extract_allatoms(outputs, feats):
    atom_types = residue_constants.atom_types
    pdb_lines = []
    atom_mask = outputs["final_atom_mask"]
    aatype = feats["aatype"]
    atom_positions = outputs["final_atom_positions"]
    # residue_index = feats["residue_index"].to(torch.int32)

    n = aatype.shape[0]
    # Add all atom sites.
    for i in range(n):
        for atom_name, pos, mask in zip(atom_types, atom_positions[i], atom_mask[i]):
            if mask < 0.5:
                continue
            pdb_lines.append(pos)

    return torch.stack(pdb_lines)


def extract_atoms_and_backbone(outputs, feats):
    atom_types = residue_constants.atom_types
    atom_mask = outputs["final_atom_mask"]
    pdb_lines = []
    aatype = feats["aatype"]
    atom_positions = outputs["final_atom_positions"]
    selected_atoms_mask = []

    n = aatype.shape[0]
    for i in range(n):
        for atom_name, pos, mask in zip(atom_types, atom_positions[i], atom_mask[i]):
            if mask < 0.5:
                continue
            pdb_lines.append(pos)

            if atom_name in ["C", "CA", "O", "N"]:
                selected_atoms_mask.append(torch.tensor(1, dtype=torch.bool))
            else:
                selected_atoms_mask.append(torch.tensor(0, dtype=torch.bool))

    return torch.stack(pdb_lines), torch.stack(selected_atoms_mask)


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
    # current_pos_atoms = extract_allatoms(results, feats)
    current_pos_atoms, backbone_mask = extract_atoms_and_backbone(results, feats)

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

    return current_pos_atoms, results["plddt"], prot, pseudo_bfactors, backbone_mask


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


def select_confident_atoms(current_pos, target_pos, bfacts=None):
    if bfacts is None:
        # If bfacts is None, set mask to all True
        reshaped_mask = torch.ones_like(current_pos, dtype=torch.bool)
    else:
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


def transfer_positions(
    aligned_pos, sfcalculator_model, b_factors, device=dsutils.try_gpu()
):
    # Transfer positions to sfcalculator
    frac_pos = coordinates.fractionalize_torch(
        aligned_pos,
        sfcalculator_model.unit_cell,
        sfcalculator_model.space_group,
        device=device,
    )
    sfcalculator_model = set_new_positions(
        aligned_pos, frac_pos, sfcalculator_model, device=device
    )

    # add bfactor calculation based on plddt
    sfcalculator_model.atom_b_iso = b_factors
    sfcalculator_model = update_sfcalculator(sfcalculator_model)
    return sfcalculator_model


def update_bfactors(plddts):
    # Use Tom Terwilliger's formula to convert plddt to Bfactor and update sfcalculator instance
    deltas = 1.5 * np.exp(4 * (0.7 - 0.01 * plddts))
    b_factors = (8 * np.pi**2 * deltas**2) / 3

    return b_factors


def normalize_modelFs(sfcalculator_model, data_dict, batch_indices=None):
    # Normalize for Emodel and Edata
    fmodel_amps = structurefactors.ftotal_amplitudes(sfcalculator_model, "Ftotal_HKL")
    if batch_indices is not None:
        fmodel_amps = fmodel_amps[batch_indices]
    sigmaP = structurefactors.calculate_Sigma_atoms(
        fmodel_amps,
        data_dict["EPS"],
        data_dict["BIN_LABELS"],
    )

    Emodel = structurefactors.normalize_Fs(
        fmodel_amps,
        data_dict["EPS"],
        sigmaP,
        data_dict["BIN_LABELS"],
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
    batch_indices=None,
    verbose=False,
    device=dsutils.try_gpu(),
):
    # predict positions
    current_pos, plddt, pdb, b_factors, backbone_mask = runner_prediction(runner, feats)
    aligned_pos = align_positions(
        current_pos, target_pos, bfacts=torch.from_numpy(b_factors).to(device)
    )

    # convert to XYZ model

    sfcalculator_model = transfer_positions(
        aligned_pos, sfcalculator, torch.tensor(b_factors).to(device), device=device
    )

    Emodel, data_dict = normalize_modelFs(sfcalculator_model, data_dict, batch_indices)

    # update sigmaA if needed

    if epoch % update == 0:
        # update sigmaA
        sigmaA = sigmaa.sigmaA_from_model(
            data_dict["ETRUE"],
            dsutils.assert_numpy(data_dict["PHITRUE"]),
            sfcalculator_model,
            data_dict["EPS"],
            data_dict["SIGMAP"],
            data_dict["BIN_LABELS"],
        )

        sigmaA = torch.clamp(torch.tensor(sigmaA).to(device), 0.001, 0.999)
        data_dict["SIGMAA"] = sigmaA

    else:
        sigmaA = data_dict["SIGMAA"]

    # calculate LLG

    llg = run_ML_binloop(
        torch.unique(data_dict["BIN_LABELS"]),
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
