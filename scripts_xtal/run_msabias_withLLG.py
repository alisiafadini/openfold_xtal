from openfold.xtal.af2runner import AF2Runner

import copy
import torch
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from openfold.xtal.openfold_functions import LLG_computation, move_tensors_to_device
from openfold.xtal.structurefactor_functions import (
    load_tng_data,
    initialize_model_frac_pos,
)
from openfold.np import protein
from xtalLLG import dsutils


# General settings
preset = "model_1"
device = "cuda:0"
template_mask_sequence = False
template_mask_sidechain = False

# Load external files
tng_file = "3hak/3hak-tng.mtz"
tng_dict = load_tng_data(tng_file, device=device)

input_pdb = "3hak/phaserpred-aligned.pdb"

Etrue = np.load("3hak/3hak-Etrue.npy")
phitrue = np.load("3hak/3hak-phitrue.npy")
tng_dict["ETRUE"] = Etrue
tng_dict["PHITRUE"] = phitrue

with open("3hak/3hak_processed_feats.pickle", "rb") as file:
    # Load the data from the pickle file
    processed_features = pickle.load(file)

device_processed_features = move_tensors_to_device(processed_features, device=device)
del processed_features

# Initiate runer and sfcalculator instances
af2_runner = AF2Runner(preset, device, template_mask_sequence, template_mask_sidechain)
sfcalculator_model, target_pos = initialize_model_frac_pos(
    input_pdb, tng_file, device=device
)

# Initiate bias tensor
msa_params = torch.zeros((512, 103, 23, 21), requires_grad=True, device=device)
device_processed_features["msa_feat_bias"] = msa_params

# Optimizer loop
# Define optimizer
lr_s = 1e-4  # OG: 0.0001
optimizer = torch.optim.Adam(
    [{"params": device_processed_features["msa_feat_bias"], "lr": lr_s}]
)

num_epochs = 2000
pdbs = []
learning_rates = []
plddt_losses = []
total_losses = []


for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()

    feats_copy = copy.deepcopy(device_processed_features)
    feats_copy["msa_feat_bias"] = device_processed_features["msa_feat_bias"].clone()
    llg, plddt, pdb, aligned_pos = LLG_computation(
        epoch,
        af2_runner,
        feats_copy,
        tng_dict,
        sfcalculator_model,
        target_pos,
        update=50,
        verbose=False,
        device=device,
    )
    loss = -llg

    mean_plddt = torch.round(torch.mean(plddt), decimals=3)
    plddt_losses.append(mean_plddt.item())
    total_losses.append(loss.item())
    pdbs.append(pdb)

    current_lr = optimizer.param_groups[0]["lr"]
    learning_rates.append(current_lr)

    loss.backward()
    optimizer.step()

# Write out final model
with open(
    "3hak/output_pdbs/LLG_{it}_{lr}_norec_nodropout.pdb".format(it=epoch, lr=lr_s), "w"
) as file:
    file.write(protein.to_pdb(pdb))

# with open("3hak/output_pdbs/LLG_1000_0.001_norec.pdb", "w") as file:
#    file.write(protein.to_pdb(pdbs[999]))

# Plot results
iterations = np.arange(1, num_epochs + 1)  # Example: Iterations from 1 to 100

n = 0
# Create a figure and two subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(7, 6))

# Plot pLDDT and LLG components in the first subplot
ax1.plot(iterations[n:], plddt_losses[n:], "b-", label="pLDDT", linewidth=2)
ax1.set_ylabel("pLDDT")
ax1.legend()

# Plot total loss and weight values in the second subplot
ax2.plot(iterations[n:], total_losses[n:], "m-", label="LLG Loss", linewidth=2)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Total Loss")
ax2.legend()

# Plot total loss and weight values in the second subplot
ax3.plot(iterations[n:], learning_rates[n:], "y-", linewidth=2)
ax3.set_xlabel("Iteration")
ax3.set_ylabel("Learning rate")

plt.suptitle("MSA Biasing with MSE Backprop")

# Adjust spacing between subplots
plt.tight_layout()
fig.savefig("3hak/output_pdbs/LLG_{it}_{lr}_norec.png".format(it=epoch, lr=lr_s))
