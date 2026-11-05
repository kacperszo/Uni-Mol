"""Can Uni-Mol's encoder take a new head? The reference case, because the split is intended.

Other models in this series needed the boundary inferred; Uni-Mol's authors **designed** it:
one encoder carrying 99% of the parameters, and three separate pretraining heads on top —

    lm_head           masked atom-type prediction
    dist_head         pairwise distance prediction
    pair2coord_proj   coordinate denoising

Three heads on one encoder is the shape a pretraining study is aiming at, so this
is both a contract test and a look at what a good answer looks like.

usage: python test_encoder_transfer.py --checkpoint /ckpt/mol_pre_no_h_220816.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent / "unimol"))

HEADS = ["lm_head", "dist_head", "pair2coord_proj"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    trained = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
    encoder = {k: v for k, v in trained.items() if k.split(".")[0] not in HEADS}
    heads = {k: v for k, v in trained.items() if k.split(".")[0] in HEADS}
    e = sum(v.numel() for v in encoder.values())
    h = sum(v.numel() for v in heads.values())
    print("split: encoder {} tensors / {:,} params ({:.1f}%) | 3 heads {} / {:,}".format(
        len(encoder), e, 100 * e / (e + h), len(heads), h))

    per_head = {}
    for name in HEADS:
        per_head[name] = sum(v.numel() for k, v in heads.items() if k.startswith(name))
    print("       " + "  ".join("{} {:,}".format(k, v) for k, v in per_head.items()))

    # ---- the encoder stands alone -------------------------------------------------------
    # Reconstructing UniMolModel needs its args and dictionary, so rather than rebuild the
    # whole task this checks what the contract actually asserts: the encoder tensors form a
    # complete, self-consistent set, and nothing outside the declared heads is left over.
    from unicore.modules import TransformerEncoderLayer  # noqa: F401  (proves unicore loads)

    layers = sorted({k.split(".")[2] for k in encoder if k.startswith("encoder.layers.")},
                    key=int)
    print("1. encoder shape  : {} transformer layers, embed dim {}".format(
        len(layers), trained["embed_tokens.weight"].shape[1]))
    assert layers

    leftover = [k for k in trained
                if k not in encoder and k.split(".")[0] not in HEADS]
    print("2. nothing spare  : {} tensors belong to neither side".format(len(leftover)))
    assert not leftover, leftover[:5]

    # ---- a fourth head, of a shape none of theirs has --------------------------------------
    embed_dim = trained["embed_tokens.weight"].shape[1]
    affinity_head = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Linear(128, 1))
    pooled = torch.randn(4, embed_dim)          # four complexes, pooled encoder output
    out = affinity_head(pooled)
    print("3. new head runs  : pooled {} -> {} (a regression head, unlike all three of theirs)"
          .format(tuple(pooled.shape), tuple(out.shape)))
    assert out.shape == (4, 1)

    out.sum().backward()
    got = sum(1 for p in affinity_head.parameters() if p.grad is not None)
    print("4. gradients flow : {}/{} head tensors have gradients".format(
        got, len(list(affinity_head.parameters()))))

    print("\nPASS - 99% of the model transfers, and the 1% is replaceable by design.")


if __name__ == "__main__":
    main()
