from openfold.xtal.af2runner import AF2Runner
import copy
import torch
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from openfold.xtal.openfold_functions import (
    runner_prediction,
    align_positions,
    move_tensors_to_device,
)
from openfold.xtal.structurefactor_functions import (
    load_tng_data,
    initialize_model_frac_pos,
)
from openfold.np import protein

# General settings

preset = "model_1"
device = "cuda:1"
template_mask_sequence = False
template_mask_sidechain = False


# Load external files and initiate runner instance
tng_file = "../../run_openfold/3hak/3hak/3hak-tng.mtz"
tng_dict = load_tng_data(tng_file, device=device)

input_pdb = "../../run_openfold/3hak/3hak/3hak_noalts.pdb"

Etrue = np.load("../../run_openfold/3hak/3hak/3hak-Etrue.npy")
phitrue = np.load("../../run_openfold/3hak/3hak/3hak-phitrue.npy")
tng_dict["ETRUE"] = Etrue
tng_dict["PHITRUE"] = phitrue

af2_runner = AF2Runner(preset, device, template_mask_sequence, template_mask_sidechain)

with open("../../run_openfold/3hak/3hak/3hak_processed_feats.pickle", "rb") as file:
    # Load the data from the pickle file
    processed_features = pickle.load(file)

device_processed_features = move_tensors_to_device(processed_features, device=device)
del processed_features

# Initiate sfcalculator
sfcalculator_model, target_pos = initialize_model_frac_pos(
    input_pdb, tng_file, device=device
)

msa_params = torch.zeros((512, 103, 23, 21), requires_grad=True, device=device)
device_processed_features["msa_feat_bias"] = msa_params

# Define optimizer
lr_s = 1e-3  # OG: 0.0001
optimizer = torch.optim.Adam(
    [{"params": device_processed_features["msa_feat_bias"], "lr": lr_s}]
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=20, threshold=1e-3, verbose=True
)

loss_function = torch.nn.MSELoss(reduction="sum")

num_epochs = 6000
pdbs = []
learning_rates = []
plddt_losses = []
total_losses = []


for epoch in tqdm(range(num_epochs)):
    feats_copy = copy.deepcopy(device_processed_features)
    feats_copy["msa_feat_bias"] = device_processed_features["msa_feat_bias"].clone()
    current_pos, plddt, pdb, pseudo_bs = runner_prediction(af2_runner, feats_copy)
    mean_plddt = torch.round(torch.mean(plddt), decimals=3)
    aligned_pos = align_positions(current_pos, target_pos).to(device)
    loss = loss_function(aligned_pos, target_pos)

    # print("Total loss : ", loss.item())
    # pdbs.append(pdb)
    plddt_losses.append(mean_plddt.item())
    total_losses.append(loss.item())

    current_lr = optimizer.param_groups[0]["lr"]
    learning_rates.append(current_lr)

    loss.backward()
    optimizer.step()

    if epoch == 5999:
        print("Final loss is ", loss)
    #    optimizer.param_groups[0]["lr"] = 1e-4
    # scheduler.step(loss)


# Write out final model
with open("3hak/output_pdbs/MSE_{it}_{lr}.pdb".format(it=epoch, lr=lr_s), "w") as file:
    file.write(protein.to_pdb(pdb))

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
fig.savefig("3hak/output_pdbs/MSE_{it}_{lr}.png".format(it=epoch, lr=lr_s))
