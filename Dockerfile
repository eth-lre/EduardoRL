# Eduardo on veRL — GH200 image, based on the NGC vLLM container for aarch64.
# verl is installed with `--no-deps` so pip does not replace the NGC-provided
# torch/vllm wheels; remaining deps are pinned in requirements-gh200.txt.
FROM nvcr.io/nvidia/vllm:26.03.post1-py3

ARG DEBIAN_FRONTEND=noninteractive
ARG VERL_VERSION=v0.7.1

# ── GH200 environment defaults ──────────────────────────────────────────────
# verl requires exactly one of {CUDA,ROCR}_VISIBLE_DEVICES to be set.
ENV ROCR_VISIBLE_DEVICES=""
ENV NCCL_NET=Socket
ENV VLLM_USE_V1=1
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# ── System deps ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        build-essential \
        libsndfile1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (everything not pre-installed by NGC) ───────────────────────
COPY requirements-gh200.txt /app/requirements-gh200.txt
RUN pip install --no-cache-dir -r /app/requirements-gh200.txt

# ── veRL at a pinned tag (package only; deps already satisfied above) ──────
RUN git clone --branch ${VERL_VERSION} --depth 1 \
        https://github.com/volcengine/verl.git /opt/verl \
    && pip install --no-cache-dir -e /opt/verl --no-deps \
    && python -c "import verl; print('verl', verl.__version__)"

CMD ["/bin/bash"]
