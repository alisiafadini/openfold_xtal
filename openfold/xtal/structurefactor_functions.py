import torch
from torch.utils.data import Dataset
from xtalLLG import dsutils, targets, structurefactors
from SFC_Torch import SFcalculator


class IndexedDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.data = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch = self.data[idx]
        batch_indices = idx

        return batch, batch_indices


def initialize_model_frac_pos(model_file, tng_file):
    sfcalculator_model = SFcalculator(
        model_file,
        tng_file,
        expcolumns=["FP", "SIGFP"],
        set_experiment=True,
        testset_value=0,
    )
    target_pos = sfcalculator_model.atom_pos_orth
    sfcalculator_model.atom_pos_frac = sfcalculator_model.atom_pos_frac * 0.00
    sfcalculator_model.atom_pos_orth = sfcalculator_model.atom_pos_orth * 0.00
    sfcalculator_model.atom_pos_frac.requires_grad = True

    return sfcalculator_model, target_pos


def set_new_positions(orth_pos, frac_pos, sfmodel):
    sfmodel.atom_pos_orth = torch.squeeze(orth_pos, dim=1).to(dsutils.try_gpu())
    sfmodel.atom_pos_frac = torch.squeeze(frac_pos, dim=1).to(dsutils.try_gpu())
    return sfmodel


def update_sfcalculator(sfmodel):
    sfmodel.inspect_data(verbose=False)
    sfmodel.calc_fprotein()
    sfmodel.calc_fsolvent()
    sfmodel.init_scales(requires_grad=True)
    sfmodel.calc_ftotal()
    return sfmodel


def load_tng_data(tng_file):
    tng = dsutils.load_mtz(tng_file).dropna()

    # Generate PhaserTNG tensors
    eps = torch.tensor(tng["EPS"].values, device=dsutils.try_gpu())
    centric = torch.tensor(tng["CENT"].values, device=dsutils.try_gpu()).bool()
    dobs = torch.tensor(tng["DOBS"].values, device=dsutils.try_gpu())
    feff = torch.tensor(tng["FEFF"].values, device=dsutils.try_gpu())
    bin_labels = torch.tensor(tng["BIN"].values, device=dsutils.try_gpu())
    unique_labels = torch.unique(bin_labels)

    sigmaN = structurefactors.calculate_Sigma_atoms(feff, eps, bin_labels)
    Edata = structurefactors.normalize_Fs(feff, eps, sigmaN, bin_labels)

    data_dict = {
        "EDATA": Edata,
        "EPS": eps,
        "CENTRIC": centric,
        "DOBS": dobs,
        "FEFF": feff,
        "BIN_LABELS": bin_labels,
        "UNIQUE_LABELS": unique_labels,
    }

    return data_dict


def run_ML_binloop(unique_labels, Edata, Emodel, bin_labels, sigmaA, centric, dobs):
    losses = [[] for _ in range(len(unique_labels))]

    for i, label in enumerate(unique_labels):
        bin_indices = bin_labels == label
        bin_Edata = Edata[bin_indices]
        bin_Emodel = Emodel[bin_indices]

        # Compute LLG score
        llg = targets.llgItot_calculate(
            sigmaA[i],
            dobs[bin_indices],
            bin_Edata,
            bin_Emodel,
            centric[bin_indices],
        )
        losses[i].append(llg)

    loss_sum = torch.zeros_like(losses[0][0])
    for tensor in losses:
        loss_sum += tensor[0]

    return loss_sum
