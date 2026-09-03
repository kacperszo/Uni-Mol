# CLAUDE.md — Uni-Mol

## What this is

The one **already-pretrained** encoder in the roster, and the reason it matters here is not its
benchmark score: it is a working example of the architecture this project is building towards.
99% of its parameters sit in a shared encoder and 1% in three interchangeable pretraining heads.

## Current state

Image builds; the encoder contract is round-trip verified. No `predict` or `embed` capability
yet — the adapter is not written.

## The encoder boundary, as its authors designed it

    encoder     embed_tokens, encoder, gbf, gbf_proj      195 tensors   47,330,754   99%
    heads       lm_head          masked atom prediction    16 tensors      288,161    1%
                dist_head        pairwise distances
                pair2coord_proj  coordinate denoising

Round-trip tested (`test_encoder_transfer.py`): 15 transformer layers at embed dim 512, **zero
tensors belonging to neither side**, and a regression head — a shape none of their three has —
attaches and takes gradients.

This is the contract the other models had to have inferred. Worth keeping as the reference for
what a good one looks like: three pretext tasks over one trunk, which is exactly the shape a
pretraining study wants.

## Two checkpoints, and no pretrained complex

`mol_pre_no_h_220816.pt` and `pocket_pre_220816.pt` are **separate models of identical
structure** — 195 encoder tensors each, differing only in the token vocabulary, since ligand
atoms and protein residues are not the same alphabet.

So there is no pretrained representation *of a complex*. Combining the two sides will be our
choice, and this benchmark has already measured what such choices cost: every pooling chosen by
us sits eight to ten points below one taken from the authors.

`binding_pose_220908.pt` (415 MB) is the one checkpoint where both encoders were trained
together, on complexes. Its task is pose rather than affinity, but its way of joining the two
sides is theirs. Look there before inventing a combination.

## Hard-won facts (do NOT regress these)

- **Their `requirements.txt` line for Uni-Core is dead twice over**:
  `git+git://github.com/dptech-corp/Uni-Core.git@stable`. GitHub disabled the `git://` protocol
  in March 2022, and the `stable` branch no longer exists. Use https and the default branch.
- **`scikit-learn` is missing from their requirements** — `unimol/losses/cross_entropy.py`
  imports it, and importing the model package pulls the losses in.
- **`--index-url` for torch breaks the build.** It confines pip to PyTorch's index, so build
  dependencies (`flit_core`) have nowhere to come from. Use `-f` with
  `torch_stable.html` so PyPI stays primary.
- Uni-Core warns that `fused_layer_norm`, `fused_rms_norm` and `fused_softmax` are not
  installed and falls back to plain implementations. Harmless on CPU; revisit for GPU training.
- Both checkpoints open under `weights_only=True`, unlike IGN's and PIGNet2's.

## Build & run

```bash
podman build --format=docker -t unimol:latest .
podman run --rm --network=none -v <weights>:/ckpt:ro localhost/unimol:latest \
  sh -c "cd /tmp && python /work/test_encoder_transfer.py --checkpoint /ckpt/mol_pre_no_h_220816.pt"
```

## Next

1. An adapter, so `embed` works and Uni-Mol joins the encoder table. Needs the ligand/pocket
   combination decided first.
2. Check `binding_pose_220908.pt` for the authors' own way of joining the two sides.
