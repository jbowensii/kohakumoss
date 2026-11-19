# KohakuMossTrainer for Tesla V100 (sm_70). CLI-only: container idles, you `docker exec` in.
# -devel base: TileLang JIT-compiles sm70 kernels at runtime and needs nvcc.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=7.0 \
    KOHAKUMOSS_KERNEL_CACHE=/workspace/runs/.cache/kohakumoss_kernels \
    TILELANG_CACHE_DIR=/workspace/runs/.cache/tilelang \
    TRITON_CACHE_DIR=/workspace/runs/.cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/workspace/runs/.cache/inductor \
    HF_HOME=/workspace/runs/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev build-essential git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && pip install --upgrade pip

# V100 = sm_70. cu126 wheel: newer CUDA wheels (cu128+) drop Volta or lack torch>=2.13.
RUN pip install "torch==2.14.0" torchvision --index-url https://download.pytorch.org/whl/cu126

# [cuda] = tilelang + lightning + bitsandbytes. Constraint stops a different torch build being pulled in.
RUN echo "torch==2.14.0+cu126" > /tmp/c.txt \
    && pip install "kohakumoss[cuda]==0.1.0" -c /tmp/c.txt \
       --extra-index-url https://download.pytorch.org/whl/cu126 \
    && rm /tmp/c.txt

# Fail the build if this torch wheel lacks Volta kernels (checkable without a GPU).
RUN python -c "import torch; a=torch.cuda.get_arch_list(); print(a); assert 'sm_70' in a, a"

# README commands use `scripts/...` relative paths; the pip package ships them in kohakumoss/bundle.
WORKDIR /workspace
RUN B=$(python -c "import kohakumoss,os;print(os.path.join(os.path.dirname(kohakumoss.__file__),'bundle','scripts'))") \
    && test -d "$B" && ln -s "$B" /workspace/scripts \
    && mkdir -p /workspace/configs/user /workspace/runs /workspace/datasets

# V100 notes: compute dtype is fp16 (no bf16/fp8); the fp16 residual fix is automatic.
# Use the vram_16g layer when creating configs (kohakumoss-config wizard -> GPU: 16 GB).
CMD ["sleep", "infinity"]
