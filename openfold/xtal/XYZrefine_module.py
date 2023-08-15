import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
from openfold_functions import run_structure_module, IndexedDataset


# Define the XYZRefineModel class
class XYZRefineModel(nn.Module):
    def __init__(self, AF_model, sfcalculator_model):
        super(XYZRefineModel, self).__init__()

        # Copy the models and make tensors require gradients
        self.AF_model = AF_model
        self.sfcalculator_model = sfcalculator_model
        self.single = nn.Parameter(
            copy.deepcopy(sfcalculator_model["single"]), requires_grad=True
        )
        self.pair = nn.Parameter(
            copy.deepcopy(sfcalculator_model["pair"]), requires_grad=True
        )

    def forward(self, feats):
        # Run one forward pass through structure module
        current_pos, af2_pos, plddt = run_structure_module(
            {"single": self.single, "pair": self.pair},
            feats,
            self.AF_model,
        )

        # realign to initial positions
        with torch.no_grad():
            centroid1, centroid2, rotation_matrix = kabsch_align_matrices(
                current_pos, target_pos
            )
            aligned_pos = align(
                current_pos, target_pos, centroid1, centroid2, rotation_matrix
            )

            # Transfer positions to sfcalculator
            frac_pos = coordinates.fractionalize_torch(
                aligned_pos,
                sfcalculator_model.unit_cell,
                sfcalculator_model.space_group,
            )

            sfcalculator_model = set_new_positions(
                aligned_pos, frac_pos, sfcalculator_model
            )
            sfcalculator_model = update_sfcalculator(sfcalculator_model)

            # Normalize for Emodel and Edata
            fmodel_amps = structurefactors.ftotal_amplitudes(
                sfcalculator_model, "Ftotal_HKL"
            )
            sigmaP = structurefactors.calculate_Sigma_atoms(
                fmodel_amps, data_dict["EPS"], data_dict["BIN_LABELS"]
            )
            Emodel = structurefactors.normalize_Fs(
                fmodel_amps, data_dict["EPS"], sigmaP, data_dict["BIN_LABELS"]
            )

        return current_pos, af2_pos, plddt


# Define the training function
def train_model(
    model,
    data_dict,
    feats,
    target_pos,
    lr_p,
    lr_s,
    num_epochs,
    batches,
    verbose=False,
    update=50,
):
    batch_size = int(len(data_dict["EDATA"]) / batches)

    # Create a DataLoader object to iterate over the training data in batches
    new_Edata = IndexedDataset(data_dict["EDATA"])
    data_loader = DataLoader(new_Edata, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(
        [{"params": model.single, "lr": lr_s}, {"params": model.pair, "lr": lr_p}]
    )

    for epoch in tqdm(range(num_epochs)):
        for batch_data, batch_indices in data_loader:
            optimizer.zero_grad()

            # Run the model forward
            current_pos, af2_pos, plddt = model(feats)

            # Backward pass and optimization
            loss = -llg
            loss.backward()
            optimizer.step()

        # Print the loss during training
        if verbose:
            if (epoch) % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
                print("SigmaA tensor: ", sigmaA)

    return model, current_pos, af2_pos, plddt
