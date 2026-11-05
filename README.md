# Uni-Mol — a pretrained transformer, here as an encoder to transfer from

> A maintained fork. The authors' own README is kept as
> [README.upstream.md](README.upstream.md) for attribution and for their description of
> the method — **its build and run instructions are not current for this fork.**

## Upstream

Uni-Mol by DP Technology: <https://github.com/deepmodeling/Uni-Mol>, MIT licence, unchanged.
The paper's citation is in [README.upstream.md](README.upstream.md). Everything outside the
following list is upstream, unmodified: the fork adds `Containerfile`, `Containerfile.gpu`,
`.containerignore`, `test_encoder_transfer.py`, `README.upstream.md` and this README.

## What it is

A 15-layer transformer at embedding dimension 512 over atoms and their pairwise
distances, pretrained self-supervised on masked atom types, masked coordinates and masked distances.
Two separate checkpoints of identical structure, one for ligands and one for protein pockets, plus a
four-layer joint decoder that combines them and was trained on complexes.

**Its authors designed an explicit encoder/head split**: 99% of the parameters sit in a
shared encoder and 1% in three interchangeable pretraining heads. That makes it a clean reference for
where an encoder ends and a task head begins.

## State

| what | state |
|---|---|
| encoder boundary | 195 encoder tensors / 47,330,754 params (**99%**) against 16 head tensors |
| contract | round-trip verified — a regression head of a shape none of theirs has attaches and trains |
| capabilities | **none yet**, no adapter |

**Not a reproduction target.** Running their pose checkpoint needs a pocket annotation keyed to their
own copy of CASF-2016; applied to canonical structures it leaves 12 of 285 pockets empty and would
need a per-complex audit. The pocket *pretraining* recipe, by contrast, is complete in this repo.

## Build

```bash
podman build --format=docker -t unimol:latest .
podman build --format=docker -f Containerfile.gpu -t unimol-gpu:latest .
```

## Run it

The one thing that runs is the encoder-contract test, which needs a checkpoint from the authors'
v0.1 release (listed in `unimol/README.md`; they are not tracked here and `weights/` is
gitignored):

- `mol_pre_no_h_220816.pt`, 182 MB, ligand encoder:
  <https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt>
- `pocket_pre_220816.pt`, 182 MB, pocket encoder:
  <https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/pocket_pre_220816.pt>
- `binding_pose_220908.pt`, 416 MB, both encoders plus the joint decoder, trained on complexes:
  <https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/binding_pose_220908.pt>

```bash
podman run --rm --network=none -v "$PWD/weights:/ckpt:ro" localhost/unimol:latest \
  sh -c "cd /tmp && python /work/test_encoder_transfer.py --checkpoint /ckpt/mol_pre_no_h_220816.pt"
```

Nothing else runs yet: there is no `predict` or `embed` entry script, for the reasons below.

## What is missing

1. **An entry script for `embed`, then `train`.** The featurisation is the work: atom tokens plus
   coordinates per side, the dictionaries from `unimol/example_data` with `[MASK]` appended as
   `tasks/docking_pose.py:93` does (giving 31 and 10 entries, which match the checkpoints'
   embedding tables), and the crystal ligand pose rather than the thousand conformers the docking
   demo generates and clusters.
2. **A pocket selection of our own**, chosen by measurement (table below).
3. **Training the joint decoder**, initialised from the two pretrained encoders.

Nothing here needs inventing the combination of the two sides: `unimol/unimol/models/docking_pose.py`
concatenates the two token sequences, builds a block attention bias (ligand-ligand from the ligand
pair representation, pocket-pocket from the pocket's, cross blocks zero) and runs a four-layer
`TransformerEncoderWithPair`. `binding_pose_220908.pt` holds trained weights for exactly that: 450
tensors, 108,780,492 parameters, 512 embedding dimensions and 64 attention heads read off the
shapes. It opens under `weights_only=True`, as do both pretrained checkpoints.

## Porting notes

**Two checkpoints, and no pretrained complex.** `mol_pre_no_h_220816.pt` and `pocket_pre_220816.pt`
are separate models of identical structure, 195 encoder tensors each, differing only in the token
vocabulary, since ligand atoms and protein residues are not the same alphabet. There is no
pretrained representation of a complex; only `binding_pose_220908.pt` trained both sides together,
and it solves pose, not affinity.

**The encoder boundary.**

    encoder     embed_tokens, encoder, gbf, gbf_proj      195 tensors   47,330,754   99%
    heads       lm_head          masked atom prediction    16 tensors      288,161    1%
                dist_head        pairwise distances
                pair2coord_proj  coordinate denoising

`test_encoder_transfer.py` checks 15 transformer layers at embedding dimension 512, zero tensors
belonging to neither side, and that a regression head of a shape none of the three has attaches
and takes gradients.

**Why the pose checkpoint cannot be run on arbitrary structures.** It needs a pocket, and the
pocket is not computed: it is read from `casf2016.pocket.json`, shipped in the authors'
`CASF-2016.tar.gz`, as a per-complex list of `chain_id + residue_number`. Their loader says it
cannot derive one locally. That annotation is keyed to their copy of the structures. Against the
canonical CASF-2016 core set, 12 of 285 pockets match zero residues, 35 start at a different
residue number and 19 have the same ATOM count, so it needs a per-complex audit of the mapping.

**Pocket selection.** `CroppingPocketDockingPoseDataset` caps a pocket at 256 atoms and randomly
samples above that, so an overshoot makes inference stochastic. Whole residues within a radius of
any ligand atom, heavy atoms only, over the 285 core-set complexes:

| selection | median atoms | over the cap |
|---|---|---|
| CASF's `_pocket.pdb` (10 A) | 390 | 274 of 285 |
| within 6 A | 229 | 97 |
| within 5 A | 166 | 16 |
| within 4 A | 137 | 3 |

No radius clears the cap for all 285, so this is a recorded trade rather than a default.

**The pocket pretraining recipe is reproducible from this repository.** `unimol/README.md:164`
carries the command that produced `pocket_pre_220816.pt`, with every hyperparameter:

    --task unimol_pocket --loss unimol --arch unimol_base --dict-name dict_coarse.txt
    --masked-token-loss 1 --masked-coord-loss 1 --masked-dist-loss 1
    --x-norm-loss 0.01 --delta-pair-repr-norm-loss 0.01
    --mask-prob 0.15 --noise-type uniform --noise 1.0
    adam (0.9, 0.99) eps 1e-6, clip-norm 1.0, wd 1e-4, lr 1e-4
    polynomial_decay, 10000 warmup, 1e6 total updates, batch 16, fp16

That is masked atom type, masked coordinate and masked distance, with two representation-norm
regularisers. `tasks/unimol_pocket.py` reads an LMDB of four fields (`pdbid`, `atoms`,
`coordinates`, `residue`), all derivable from any structure, with nothing keyed to a preprocessed
copy of anything. The only choice left open is which pockets to pretrain on.

**Environment traps** (each is why the Containerfile differs from `requirements.txt`):

- The authors' Uni-Core line, `git+git://github.com/dptech-corp/Uni-Core.git@stable`, is dead
  twice over: GitHub disabled the `git://` protocol in March 2022, and the `stable` branch no
  longer exists. The Containerfile uses https and the default branch, which is not pinned to a
  commit.
- `scikit-learn` is missing from their requirements, but `unimol/losses/cross_entropy.py` imports
  it and importing the model package pulls the losses in.
- `rdkit-pypi` pinned at 2022.9.3 has no wheel for current Python; the image installs `rdkit`
  unpinned.
- An `--index-url` for torch breaks the build: it confines pip to PyTorch's index, so build
  dependencies (`flit_core`) have nowhere to come from. Use `-f` with `torch_stable.html`.
- Uni-Core warns that `fused_layer_norm`, `fused_rms_norm` and `fused_softmax` are not installed
  and falls back to plain implementations. Harmless on CPU; worth revisiting for GPU training.
