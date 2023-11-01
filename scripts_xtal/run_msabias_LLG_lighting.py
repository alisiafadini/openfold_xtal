import torch
from torch import nn
import numpy as np
from torch.optim import Adam
import pickle
import pytorch_lightning as pl
from pytorch_lightning.profiler import PyTorchProfiler
from torch.utils.data import DataLoader
from openfold.xtal.af2runner import AF2Runner
from openfold.xtal.openfold_functions import LLG_computation, move_tensors_to_device
from openfold.xtal.structurefactor_functions import (
    load_tng_data,
    initialize_model_frac_pos,
    IndexedDataset,
)


class MsaBiasModule(pl.LightningModule):
    def __init__(
        self,
        lr,
        device,
        num_epochs,
        af2_runner,
        feats_copy,
        sfcalculator_model,
        target_pos,
        update=50,
        batch_num=1,
    ):
        super().__init__()
        self.lr = lr
        self.cuda_device = device
        self.num_epochs = num_epochs
        self.af2_runner = af2_runner
        self.feats_copy = feats_copy
        self.sfcalculator_model = sfcalculator_model
        self.target_pos = target_pos
        self.update = update
        self.batches = batch_num
        self.profiler = PyTorchProfiler(use_cuda=True)

        # Define msa_params tensor as a parameter
        self.msa_params = nn.Parameter(
            torch.zeros((512, 103, 23, 21), requires_grad=True, device=self.cuda_device)
        )  # TO DO: first two dimensions not hard-coded

    def forward(self, batch, batch_indices):
        # LLG computation
        llg, plddt, pdb, aligned_pos = LLG_computation(
            self.current_epoch,
            self.af2_runner,
            self.feats_copy,
            batch,
            self.sfcalculator_model,
            self.target_pos,
            self.update,
            batch_indices,
            verbose=False,
            device=self.cuda_device,
        )
        loss = -llg
        return loss, plddt, pdb

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch):  # Depending on batch use
        loss, plddt, pdb = self.forward(batch[0], batch[1].to(device=device))
        self.log("plddt", torch.mean(plddt))
        return loss

    def training_epoch_end(self, outputs):
        # Log mean loss at the end of each epoch
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log("avg_loss", avg_loss, prog_bar=True)


# General parameters

lr_s = 1e-3
num_epochs = 2
device = "cuda:0"
preset = "model_1"
template_mask_sequence = False
template_mask_sidechain = False
batches = 2
update = 1000


# Load data

tng_file = "3hak/3hak-tng.mtz"
tng_dict = load_tng_data(tng_file, device=device)

input_pdb = "3hak/phaserpred-aligned.pdb"

Etrue = np.load("3hak/3hak-Etrue.npy")
phitrue = np.load("3hak/3hak-phitrue.npy")
tng_dict["ETRUE"] = Etrue
tng_dict["PHITRUE"] = phitrue

new_Edata = IndexedDataset(tng_dict)
batch_sizes = int(len(tng_dict["EDATA"]) / batches)
data_loader = DataLoader(new_Edata, batch_size=batch_sizes, shuffle=False)

# for batch_data, batch_indices in data_loader:
#    print(batch_data["EDATA"].shape)
#    print(batch_indices)


with open("3hak/3hak_processed_feats.pickle", "rb") as file:
    # Load the data from the pickle file
    processed_features = pickle.load(file)

device_processed_features = move_tensors_to_device(processed_features, device=device)
del processed_features


# Initialize af2_runner and sfcalculator_model

af2_runner = AF2Runner(preset, device, template_mask_sequence, template_mask_sidechain)
sfcalculator_model, target_pos = initialize_model_frac_pos(
    input_pdb, tng_file, device=device
)

# Initiate bias tensor
msa_params = torch.zeros((512, 103, 23, 21), requires_grad=True, device=device)
device_processed_features["msa_feat_bias"] = msa_params


# Initialize lightning module

lightning_module = MsaBiasModule(
    lr_s,
    device,
    num_epochs,
    af2_runner,
    device_processed_features,
    sfcalculator_model,
    target_pos,
    update,
    batches,
)


# Initialize a Lightning Trainer
trainer = pl.Trainer(
    max_epochs=num_epochs,
    gpus=1 if device == "cuda:0" else 0,
    profiler=lightning_module.profiler,
)

trainer.fit(lightning_module, data_loader)

print("DONE")
exit()

# print(trainer.profiler_results)
