# Uni-Mol — a pretrained transformer, here as an encoder to transfer from

> A fork maintained for [gnn-benchmark](../../README.md). The authors' own README is
> kept as [README.upstream.md](README.upstream.md) for attribution and for their
> description of the method — **its build and run instructions are not current for
> this fork.**

## What it is

A 15-layer transformer at embedding dimension 512 over atoms and their pairwise
distances, pretrained self-supervised on masked atom types, masked coordinates and masked distances.
Two separate checkpoints of identical structure, one for ligands and one for protein pockets, plus a
four-layer joint decoder that combines them and was trained on complexes.

**Its authors designed the encoder/head split this project needs**: 99% of the parameters sit in a
shared encoder and 1% in three interchangeable pretraining heads. That makes it the reference for a
contract the other models had to have inferred.

## State

| | |
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

## Run it, without the harness

Nothing to run yet: this model declares no capability, because it has no adapter.
`CLAUDE.md` says what is missing and why.

## Maintainer notes

`CLAUDE.md` in this directory holds what breaks if it is changed back.
