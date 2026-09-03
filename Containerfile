# Uni-Mol, CPU reference tier.
#
# The one already-pretrained encoder in the roster, and the reference for what an encoder /
# head boundary looks like when the authors designed it: 99% of parameters in the encoder,
# three pretraining heads (masked atoms, distances, coordinate denoising) sitting on top.
#
# Two things their requirements.txt gets wrong on a current machine:
#   * it installs Uni-Core over `git+git://@stable`, and *both halves* are dead: GitHub
#     disabled the git:// protocol in March 2022, and the `stable` branch no longer exists.
#     https and the default branch work.
#   * rdkit-pypi is pinned to 2022.9.3, which has no wheel for current Python.
FROM docker.io/library/python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

# -f, not --index-url: the latter confines pip to PyTorch's index, and build dependencies
# (flit_core here) then have nowhere to come from. This keeps PyPI as the primary index.
RUN pip install --no-cache-dir torch==2.1.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Uni-Core is a fairseq-style training framework; the model class inherits from it, so it is
# needed even to instantiate a model, not only to train one.
RUN pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/dptech-corp/Uni-Core.git

RUN pip install --no-cache-dir \
    rdkit \
    "numpy<2" \
    scipy \
    pandas \
    tqdm \
    ml-collections \
    lmdb \
    # not in their requirements.txt, but unimol/losses/cross_entropy.py imports it and the
    # model package pulls the losses in on import
    scikit-learn

RUN python -c "import torch, unicore; from unicore.models import BaseUnicoreModel; \
print('env ok', torch.__version__)"

WORKDIR /work
COPY . /work
RUN cd /work/unimol && python -c "import sys; sys.path.insert(0,'.'); \
from unimol.models.unimol import UniMolModel; print('unimol imports ok')"
