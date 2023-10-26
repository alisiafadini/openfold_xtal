import torch
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
from pytorch_lightning.profiler import PyTorchProfiler
from openfold.xtal.openfold_functions import LLG_computation, move_tensors_to_device


class MyLightningModule(pl.LightningModule):
    def __init__(
        self,
        device,
        lr,
        num_epochs,
        af2_runner,
        feats_copy,
        tng_dict,
        sfcalculator_model,
        target_pos,
    ):
        super().__init()
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.af2_runner = af2_runner
        self.feats_copy = feats_copy
        self.tng_dict = tng_dict
        self.sfcalculator_model = sfcalculator_model
        self.target_pos = target_pos

        # Define msa_params tensor as a parameter
        self.msa_params = nn.Parameter(
            torch.zeros((512, 103, 23, 21), requires_grad=True, device=device)
        )  # TO DO: first two dimensions not hard-coded

    def forward(self):
        # LLG computation
        llg, plddt, pdb, aligned_pos = LLG_computation(
            self.current_epoch,
            self.af2_runner,
            self.feats_copy,
            self.tng_dict,
            self.sfcalculator_model,
            self.target_pos,
            update=50,
            verbose=False,
            device=self.device,
        )
        loss = -llg
        return loss, plddt, pdb

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        loss, plddt, pdb = self.forward()
        self.log("plddt", torch.mean(plddt))
        return loss

    def training_epoch_end(self, outputs):
        # Log mean loss at the end of each epoch
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log("avg_loss", avg_loss, prog_bar=True)


# Initialize your lightning module
lr_s = 1e-3
num_epochs = 2000
lightning_module = MyLightningModule(
    device,
    lr_s,
    num_epochs,
    af2_runner,
    feats_copy,
    tng_dict,
    sfcalculator_model,
    target_pos,
)

# Initialize a Lightning Trainer
trainer = pl.Trainer(
    max_epochs=num_epochs,
    gpus=1 if device == "cuda:0" else 0,  # Specify the number of GPUs
    progress_bar_refresh_rate=10,  # Adjust this value as needed
)

# Train the model
trainer.fit(lightning_module)
