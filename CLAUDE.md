# CLAUDE.md — Uni-Mol

## What this is

The one **already-pretrained** encoder in the roster, and the reason it matters here is not its
benchmark score: it is a working example of the architecture this project is building towards.
99% of its parameters sit in a shared encoder and 1% in three interchangeable pretraining heads.

## Current state

Image builds; the encoder contract is round-trip verified. No `predict` or `embed` capability
yet — the adapter is not written.

**Not a reproduction target here**, and the reason is specific rather than sweeping: running their
pose checkpoint needs a pocket annotation keyed to their own copy of CASF-2016, and applying it to
the canonical structures would take a per-complex audit of the residue mapping. See below for what
that was measured to be. This model is in the roster as a pretrained encoder to transfer from,
which is what made it interesting in the first place — and its pocket *pretraining* recipe is
fully reproducible from this repo.

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

`binding_pose_220908.pt` (416 MB) is the one checkpoint where both encoders were trained
together, on complexes. Its task is pose rather than affinity, but its way of joining the two
sides is theirs, and that way is in the repo: `unimol/unimol/models/docking_pose.py` concatenates
the two token sequences, builds a block attention bias — ligand-ligand from the ligand pair
representation, pocket-pocket from the pocket's, the cross blocks left at zero — and runs a
four-layer `TransformerEncoderWithPair` over the result. Nothing about the joining has to be
invented.

## Why there is no `predict` or `embed` yet

Running `binding_pose_220908.pt` needs a pocket, and the pocket is not computed — it is read from
`casf2016.pocket.json`, shipped in `CASF-2016.tar.gz` on their v0.1 release, as a per-complex list
of `chain_id + residue_number`. Their loader says it cannot derive one locally.

That annotation is keyed to their copy of the structures. Against the canonical CASF core set:
**12 of 285** pockets match zero residues, **35** start at a different residue number, and **19**
have the same ATOM count. So it needs a per-complex audit of the mapping before it can be used,
and their structures are not an alternative — every other model here is scored on one canonical
copy of CASF, and we cannot redistribute theirs.

**So Uni-Mol is a training target, not a reproduction one**, and the pocket selection will be ours.
Choose it by measurement: `CroppingPocketDockingPoseDataset` caps a pocket at 256 atoms and
**randomly samples above that**, so an overshoot makes inference stochastic. Whole residues within
a radius of any ligand atom, heavy atoms only, over the core set:

| selection | median atoms | over the cap |
|---|---|---|
| CASF's `_pocket.pdb` (10 A) | 390 | 274 of 285 |
| within 6 A | 229 | 97 |
| within 5 A | 166 | 16 |
| within 4 A | 137 | 3 |

No radius clears it for all 285, so this is a recorded trade rather than a default.

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

## What is left to build

1. **An entry script and an adapter**, for `embed` then `train`. The featurisation is the work:
   atom tokens plus coordinates per side, the dictionaries from `example_data` with `[MASK]`
   appended as `tasks/docking_pose.py:93` does — giving 31 and 10, which matches the checkpoint's
   embedding tables — and the crystal ligand pose rather than the thousand conformers the docking
   demo generates and clusters.
2. **Our pocket selection**, declared as ours, from the table above.
3. **Train the joint decoder**, initialised from `mol_pre_no_h_220816.pt` and
   `pocket_pre_220816.pt`.

Nothing here needs inventing the combination of the two sides: `models/docking_pose.py` does it —
concatenate the token sequences, build a block attention bias (ligand-ligand from the ligand pair
representation, pocket-pocket from the pocket's, cross blocks zero), run a four-layer
`TransformerEncoderWithPair`. `binding_pose_220908.pt` holds trained weights for exactly that:
450 tensors, 108,780,492 parameters, 512 embedding dimensions and 64 attention heads read off the
shapes, and it opens under `weights_only=True`.

## The pocket pretraining recipe, which is reproducible

`unimol/README.md:164` carries the command that produced `pocket_pre_220816.pt`, with every
hyperparameter:

    --task unimol_pocket --loss unimol --arch unimol_base --dict-name dict_coarse.txt
    --masked-token-loss 1 --masked-coord-loss 1 --masked-dist-loss 1
    --x-norm-loss 0.01 --delta-pair-repr-norm-loss 0.01
    --mask-prob 0.15 --noise-type uniform --noise 1.0
    adam (0.9, 0.99) eps 1e-6, clip-norm 1.0, wd 1e-4, lr 1e-4
    polynomial_decay, 10000 warmup, 1e6 total updates, batch 16, fp16

Masked atom type, masked coordinate and masked distance, with two representation-norm
regularisers. `tasks/unimol_pocket.py` reads an LMDB of four fields — `pdbid`, `atoms`,
`coordinates`, `residue` — all derivable from any structure we hold, with nothing keyed to a
preprocessed copy of anything. **This half is reproducible from the repo**; the only choice left is
which pockets to pretrain on, which for this programme was always going to be our own data.

So: the pose/finetuning path is dropped, the pretraining path is complete. The second is why this
model is in the roster.
