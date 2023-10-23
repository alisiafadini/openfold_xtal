import numpy as np

import torch
from torch.optim import Adam

from openfold.config import model_config
from openfold.model.model import AlphaFold

import openfold.data.parsers as parsers
import openfold.data.data_pipeline as data_pipeline
import openfold.data.feature_pipeline as feature_pipeline
import openfold.data.data_transforms as data_transforms

import openfold.np.protein as protein
import openfold.np.residue_constants as residue_constants

from openfold.utils.script_utils import get_model_basename
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.xtal.openfold_functions import extract_allatoms

# from utils import compare


class AF2Runner:
    """
    Runner for AlphaFold2, which uses OpenFold implementation for efficent
    gradient backpropagation and memory usage.
    """

    def __init__(
        self,
        preset,
        device,
        template_mask_sequence=False,
        template_mask_sidechain=False,
    ):
        """
        Initialization.

        Params:
                preset: model name, for example, model_1_ptm
                device: device name
                template_mask_sequence: whether to mask template sequence to alanines
                template_mask_sidechain: whether to mask sidechain atoms
        """
        self.preset = preset
        self.device = device
        self.template_mask_sequence = template_mask_sequence
        self.template_mask_sidechain = template_mask_sidechain

        # load model configuration
        # Note: recycling is disabled
        self.config = model_config(preset, train=True)
        self.config.data.common.max_recycling_iters = 0  # 50

        print(
            "Number of recycling iterations is",
            self.config.data.common.max_recycling_iters,
        )

        # laod feature pipeline
        self.pipeline = feature_pipeline.FeaturePipeline(self.config.data)

        # load model
        self.af2 = self.load_model()

    def load_model(self):
        """
        Load AlphaFold2 model.
        """
        model = AlphaFold(self.config)
        path = f"/net/cci-gpu-00/raid1/scratch1/alisia/programs/openfold/openfold_xtal/openfold/resources/params/params_{self.preset}.npz"
        # path = f"/net/cci-gpu-00/raid1/scratch1/alisia/programs/openfold/openfold_xtal/openfold/resources/openfold_params/finetuning_3.pt"
        model_basename = get_model_basename(path)
        model_version = "_".join(model_basename.split("_")[1:])
        import_jax_weights_(model, path, version=model_version)
        model = model.to(self.device)
        return model

    def precompute(self, input_pdb):
        """
        Set up for computing plddt given single representation. This
        creates a precompute feature dictionary, containing sequence
        information, mask information and so on. Since sequence is
        fixed here, later computation only requires to replace template
        information without the need to modify the rest. Note that this
        function MUST be called before calling compute.
        TODO: add strict constraint on function dependency

        Params:
                input_pdb: pdb filepath for initialization
        """
        self.features, seq = self.compute_features(input_pdb)

    def compute(self, act, seq):
        """
        Evaluate single representations, which are decoded into a template
        and subsequently used for AlphaFold2 prediction.

        Params:
                act: single representation
        """

        # decode single representation into a protein
        outputs = self.decode(
            {
                "aatype": self.features["aatype"][:, -1].squeeze(-1),
                "single": act,
                "residx_atom37_to_atom14": self.features["residx_atom37_to_atom14"][
                    :, :, -1
                ].squeeze(-1),
                "atom37_atom_exists": self.features["atom37_atom_exists"][
                    :, :, -1
                ].squeeze(-1),
            }
        )

        # perform masking
        template_aatype = self.features["template_aatype"][:1].squeeze(-1)
        template_all_atom_positions = outputs["positions"].unsqueeze(0)
        template_all_atom_mask = self.features["template_all_atom_mask"][:, :, :, -1][
            :1
        ].squeeze(-1)

        if self.template_mask_sequence:
            #
            is_gly = torch.eq(template_aatype, residue_constants.restype_order["G"])
            replace_beta_mask = torch.cat(
                [
                    torch.zeros_like(template_all_atom_mask[:, :, :3]),
                    is_gly.unsqueeze(-1),
                    torch.zeros_like(template_all_atom_mask[:, :, 4:]),
                ],
                dim=-1,
            ).bool()

            #
            replace_beta = torch.cat(
                [
                    torch.zeros_like(template_all_atom_positions[:, :, :3, :]),
                    template_all_atom_positions[:, :, 1, :].unsqueeze(-2),
                    torch.zeros_like(template_all_atom_positions[:, :, 4:, :]),
                ],
                dim=-2,
            )

            #
            template_all_atom_positions = (
                ~replace_beta_mask.unsqueeze(-1)
            ) * template_all_atom_positions + replace_beta_mask.unsqueeze(
                -1
            ) * replace_beta

            #
            template_aatype = torch.zeros_like(template_aatype)

            #
            template_mask = torch.cat(
                [
                    torch.ones_like(template_all_atom_mask[:, :, :5]),
                    torch.zeros_like(template_all_atom_mask[:, :, 5:]),
                ],
                dim=-1,
            )

            #
            template_all_atom_positions = (
                template_all_atom_positions * template_mask.unsqueeze(-1)
            )

        elif self.template_mask_sidechain:
            #
            template_all_atom_mask = template_all_atom_mask * torch.cat(
                [
                    template_all_atom_mask[:, :, :5],
                    torch.zeros_like(template_all_atom_mask[:, :, 5:]),
                ],
                dim=-1,
            )

            #
            template_all_atom_positions = (
                template_all_atom_positions * template_all_atom_mask.unsqueeze(-1)
            )

        # create template from the decoded protein
        template = {
            "template_aatype": template_aatype[:, :, -1],
            "template_all_atom_positions": template_all_atom_positions,
            "template_all_atom_mask": template_all_atom_mask,
        }
        template = data_transforms.make_pseudo_beta("template_")(template)
        template = data_transforms.atom37_to_torsion_angles("template_")(template)

        # create features dictionary
        temp_features = dict([(key, self.features[key]) for key in self.features])
        new_features = dict([(key, self.features[key]) for key in self.features])

        for key in template:
            temp_features[key] = torch.cat(
                [
                    template[key],
                    torch.zeros(
                        (3, *template[key].shape[1:]),
                        dtype=temp_features[key].dtype,
                        device=temp_features[key].device,
                    ),
                ]
            ).unsqueeze(-1)

            num_copies = self.config.data.common.max_recycling_iters + 1

            repeats = [
                1,
            ] * len(temp_features[key].shape) + [
                num_copies,
            ]
            expanded_tensor = temp_features[key].repeat(*repeats[1:])
            new_features[key] = expanded_tensor

            # print("after")

        # run alphafold
        results = self.af2(new_features)

        # create copy of numpy arrays
        new_features_np = tensor_tree_map(
            lambda t: t[..., -1].cpu().detach().numpy(), new_features
        )
        results_np = tensor_tree_map(lambda t: t.cpu().detach().numpy(), results)

        # create output protein
        b_factors = results_np["plddt"][:, None] * results_np["final_atom_mask"]
        prot = protein.from_prediction(new_features_np, results_np, b_factors=b_factors)

        # XYZs compatible with SFCalculator
        current_XYZs = extract_allatoms(results, new_features)

        # create output dictionary
        outputs = {
            "plddt": results["plddt"],
            "prot": prot,
            "current_XYZ": current_XYZs,
        }

        return outputs, results

    def predict(self, template_pdb):
        """
        Standard use of AlphaFold2 with templates. Here, the target sequence
        matches the sequence in the template.

        Params:
                template_pdb: pdb filepath for template protein
        """
        features = self.compute_features(template_pdb)
        outputs = self.af2(features)
        print(outputs["plddt"].mean())

        return None

    def encode(self, template_pdb):
        """
        Generate single representation given input template pdb. Similarly, the
        target sequence here matches the sequence in the template. In addition,
        the single representation in use here is the single representation before
        the first layer of backbone update (aka. after the first IPA layer and
        layer normalization). Note that no recyling is used in this model.

        Params:
                template_pdb: pdb filepath for template protein
        """
        features, seq = self.compute_features(template_pdb)
        with torch.no_grad():
            outputs = self.af2(features)
        return outputs["sm"]["states"][0], seq

    def decode(self, batch):
        """
        Decode single representation into its corresponding structure.

        Params:
                batch: a dictionary containing
                        - aatype: one-hot amino acid representation
                        - single: single representation
                        - residx_atom37_to_atom14: residue representation transformation mask
                        - atom37_atom_exists: atom mask
        """
        return self.af2.structure_module.decode(batch)

    ############################
    ###   Helper Functions   ###
    ############################

    def compute_features(self, template_pdb):
        """
        Compute features given template pdb.
        """
        prot = protein.from_pdb_string(self.pdb_to_string(template_pdb))
        seq = data_pipeline._aatype_to_str_sequence(prot.aatype)
        template = self.create_template(seq, prot)
        features = self.make_processed_feature_dict(seq, template)
        return features, seq

    def pdb_to_string(self, pdb):
        """
        Read in a PDB file from a path.
        """
        lines = []
        for line in open(pdb, "r"):
            if line[:6] == "HETATM" and line[17:20] == "MSE":
                line = "ATOM  " + line[6:17] + "MET" + line[20:]
            if line[:4] == "ATOM":
                lines.append(line)
        return "".join(lines)

    def create_template(self, seq, prot):
        """
        Create template from protein.
        """
        return {
            "template_aatype": residue_constants.sequence_to_onehot(
                seq, residue_constants.HHBLITS_AA_TO_ID
            ).astype(np.int64)[None],
            "template_all_atom_mask": np.expand_dims(prot.atom_mask, axis=0).astype(
                np.float32
            ),
            "template_all_atom_positions": np.expand_dims(
                prot.atom_positions, axis=0
            ).astype(np.float32),
            "template_domain_names": np.asarray(["None"]),
        }

    def make_processed_feature_dict(self, seq, template):
        """
        Create a feature dictionary for input to AlphaFold.
        """

        # create features
        feature_dict = {}
        feature_dict.update(
            data_pipeline.make_sequence_features(seq, "input", len(seq))
        )
        msa, deletion_matrix = parsers.parse_a3m(">1\n%s" % seq)
        feature_dict.update(
            data_pipeline.make_msa_features([(msa)], [(deletion_matrix)])
        )
        feature_dict.update(template)

        # process features
        processed_feature_dict = self.pipeline.process_features(
            feature_dict, mode="predict"
        )

        # update device
        processed_feature_dict = tensor_tree_map(
            lambda t: t.to(self.device), processed_feature_dict
        )

        return processed_feature_dict
