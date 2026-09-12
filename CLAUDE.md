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

## Running their pose checkpoint needs an annotation that does not transfer to our data

Settled 2026-09-12. This is the reason Uni-Mol is a **training** target here rather than a
reproduction one. The claim is deliberately narrower than the first version of this section, which
over-reached from two samples and was corrected by measuring all 285 — see the end.

Running `binding_pose_220908.pt` needs a pocket, and the pocket is not computed: it is read from
`casf2016.pocket.json`, a per-complex list of `chain_id + residue_number` strings shipped in
`CASF-2016.tar.gz` on their v0.1 release. Their own loader is explicit about the limitation —
*"Currently not support parsing pdb and pocket info from local files."*

**That annotation is keyed to their copy of the structures, and their copy is not ours.** Measured
across all 285:

| | |
|---|---|
| files with the same number of ATOM records as the canonical CASF file | **19 of 285** |
| files whose first residue number differs from ours | **35 of 285** |
| pockets that match **zero** residues when applied to canonical structures | **12 of 285** |

So the annotation cannot be applied blindly. For the 12 it fails visibly; for the 35 whose
numbering starts elsewhere it is suspect and would need checking one complex at a time; for the
remaining 250 the numbering starts at the same residue and probably lines up, though a shared
starting number does not by itself guarantee the same numbering throughout.

Using it properly therefore means auditing the mapping per complex. That is a tractable job — and
it is bookkeeping about their preprocessing rather than progress on the question this project is
asking, so it was not done. We also cannot redistribute CASF-2016 ourselves, so their structures
are not an option for this benchmark even where they would work: every other model here is scored
on one canonical copy, and a model scored on a different one does not belong in the same table.

**What stands**, because none of it depends on their preprocessing: the architecture, the way the
two sides are joined (`models/docking_pose.py`, in this repo), both pretrained encoders, the
dictionaries, and the encoder boundary their authors designed. We train the head ourselves, which
was always the plan, and our pocket selection is declared as ours.

### Two claims withdrawn, and how

The first version of this section said the residues were "renumbered from 1" and that coordinates
"differ in the third decimal", and concluded that the 273 non-zero-overlap complexes were *worse*
than the 12 because their pockets would be built from the wrong atoms. All three came from
inspecting `4jia` and `2v7a`, which turned out to be in the 35-complex tail rather than typical.

Measuring all 285: their first residue is 1 in only **82** of them, and matches ours in **250**.
The coordinate claim came from comparing the first ATOM line of each file — one atom — and a
proper comparison is confounded anyway, because the atom ordering differs between the two copies,
which makes an element-wise difference meaningless (it reads 9 A at the median for the 19 files
with matching atom counts).

The conclusion survives; the reasoning for it did not. Worth keeping as a reminder that two
samples out of 285 are a hypothesis.

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

## The pocket *pretraining* recipe is in the repo, unlike the pose data

And this is the part worth having, because it is what this project is actually building towards.
`unimol/README.md:164` carries the command that produced `pocket_pre_220816.pt`, with every
hyperparameter:

    --task unimol_pocket --loss unimol --arch unimol_base --dict-name dict_coarse.txt
    --masked-token-loss 1 --masked-coord-loss 1 --masked-dist-loss 1
    --x-norm-loss 0.01 --delta-pair-repr-norm-loss 0.01
    --mask-prob 0.15 --noise-type uniform --noise 1.0
    adam (0.9, 0.99) eps 1e-6, clip-norm 1.0, wd 1e-4, lr 1e-4
    polynomial_decay, 10000 warmup, 1e6 total updates, batch 16, fp16

A self-supervised objective over protein pockets — masked atom type, masked coordinate and masked
distance, with two representation-norm regularisers — and the dictionary is `dict_coarse.txt`,
which is in the repo and which we already confirmed matches the checkpoint's 10-token embedding
table.

**Its input is four fields, not an annotation.** `tasks/unimol_pocket.py` reads an LMDB of
`pdbid`, `atoms`, `coordinates`, `residue`. Nothing keyed to a preprocessed copy of anything: a
list of heavy atoms, their coordinates and their residue labels, which is derivable from any
structure we hold. So unlike the pose task, **this one is reproducible from the repo**, and the only
choice left to us is which pockets to pretrain on — which for this programme was always going to be
our own data anyway.

That makes Uni-Mol's standing here precise: its *finetuning* story is unreproducible and dropped,
while its *pretraining* story is a published, complete, self-supervised recipe for a 99%-of-
parameters encoder. The second is why the model is in the roster.
