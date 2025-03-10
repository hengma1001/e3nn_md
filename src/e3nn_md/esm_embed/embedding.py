import torch

from e3nn_md.esm_embed.utils import get_seq


class ESM_embedder(object):
    def __init__(self) -> None:
        self.model, self.alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")

    def cal_embedding(self, pdb_file):
        sequence = get_seq(pdb_file)

        batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()  # disables dropout for deterministic results

        # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
        data = [
            ("protein1", sequence),
        ]
        _, _, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        return token_representations[0, 1 : batch_lens[0] - 1]
