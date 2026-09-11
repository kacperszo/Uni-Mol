# CLAUDE.md — Uni-Mol

## What this is

The one **already-pretrained** encoder in the roster, and the reason it matters here is not its
benchmark score: it is a working example of the architecture this project is building towards.
99% of its parameters sit in a shared encoder and 1% in three interchangeable pretraining heads.

## Current state

Image builds; the encoder contract is round-trip verified. No `predict` or `embed` capability
yet — the adapter is not written.

**Not a reproduction target.** Their published pose numbers cannot be reproduced from what is
published; see below. This model is here as a pretrained encoder to transfer from, which is what
made it interesting in the first place.

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

## Their published pose numbers cannot be reproduced from what is published

Settled 2026-09-12, and it is the reason this model is a **training** target here rather than a
reproduction one.

Running `binding_pose_220908.pt` needs a pocket, and the pocket is not computed — it is read from
`casf2016.pocket.json`, a per-complex list of `chain_id + residue_number` strings shipped in
`CASF-2016.tar.gz` on their v0.1 release. Their own loader says as much: *"Currently not support
parsing pdb and pocket info from local files."*

That json is keyed to **their** copy of CASF-2016, not to the canonical one, and the two are not
the same files:

| 4jia_protein.pdb | theirs | ours (official CASF) |
|---|---|---|
| first residue | `SER A 1` | `SER A 833` |
| numbering | 1-300 | 833-1132 |
| ATOM records | 4921 | 4878 |
| first coordinate | 15.912, -56.379, -45.462 | 15.894, -56.369, -45.445 |

So the residues were renumbered from 1, 43 atoms were added or kept that the canonical file does
not have, and even the coordinates differ in the third decimal. Applied to the canonical
structures, their annotation leaves **12 of 285 pockets with zero matching residues** — and the
other 273 are worse than that, because a shifted numbering means `A47` in their file and `A47` in
ours are simply different residues. Those 273 "successes" produce plausibly sized pockets made of
the wrong atoms, which nothing downstream would report.

**The data is published; the preprocessing that produced it is not.** Recovering it would mean
reverse-engineering a residue mapping per complex — matchable by coordinates, since they agree to
about 2e-2 A — and that is work on their bookkeeping rather than on the science. It is also a
redistribution of CASF-2016, which PDBbind's terms do not provide for.

So: **no reproduction of their docking numbers, and none claimed.** What stands is the part that
does not depend on their preprocessing — the architecture, the two pretrained encoders, the
dictionaries, and the encoder boundary their authors designed. We train the head ourselves, which
was always the plan for this model, and our pocket selection is declared as ours.

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

Both open questions from the previous round are now closed, and not in the direction expected.

1. ~~Needs the ligand/pocket combination decided first.~~ **It does not.** The combination is in
   the repo — `models/docking_pose.py` concatenates the two token sequences and runs a four-layer
   `TransformerEncoderWithPair` over them with a block attention bias. Nothing to invent.
2. ~~Check `binding_pose_220908.pt` for the authors' own way of joining the two sides.~~ Checked.
   It holds 450 tensors and 108,780,492 parameters: `mol_model` 193, `pocket_model` 193,
   `concat_decoder` 54, plus 10 in the two pose heads. The architecture config comes out of the
   shapes — 512 embedding dimensions, 64 attention heads — and the checkpoint opens under
   `weights_only=True`. What it cannot do is run, because the pocket annotation it needs is keyed
   to a preprocessed copy of CASF-2016 rather than to the canonical one.

So what remains:

1. **Our own pocket selection, declared as ours.** Theirs is not reproducible from what is
   published, so this is a choice we have to make and should make by measurement rather than by
   default — `CroppingPocketDockingPoseDataset` caps a pocket at 256 atoms and **randomly samples**
   above that, so a selection that overshoots makes inference stochastic. Measured on the
   canonical CASF core set, whole residues within a radius of any ligand atom, heavy atoms only:

   | selection | median atoms | over the 256 cap |
   |---|---|---|
   | CASF's own `_pocket.pdb` (10 A) | 390 | 274 of 285 |
   | residues within 6 A | 229 | 97 of 285 |
   | residues within 5 A | 166 | 16 of 285 |
   | residues within 4 A | 137 | 3 of 285 |

   No radius clears the cap for all 285, so the choice is a trade and has to be recorded as one.
   For reference, their own annotation has a median of 28 residues — roughly half of CASF's
   pocket — which is why the cap makes sense in their pipeline and not in ours.

2. **An entry script and an adapter**, for `embed` and then `train`. The featurisation is the work:
   atom tokens plus coordinates per side, the dictionaries from `example_data` with `[MASK]`
   appended as `tasks/docking_pose.py:93` does, and the crystal ligand pose rather than the
   thousand generated conformers the docking demo clusters — the same decision already taken and
   documented for EquiBind's embedding path.

3. **Train the joint decoder ourselves**, initialised from `mol_pre_no_h_220816.pt` and
   `pocket_pre_220816.pt`. That was always the plan for this model; it is now also the only way it
   runs.
