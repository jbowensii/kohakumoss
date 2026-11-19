# kohakumoss (Docker, Tesla V100)

Containerized [KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer) for training
LoRAs for **OpenMOSS MOSS-Image** on an NVIDIA **Tesla V100** (Volta, sm_70, FP16 — no bf16/fp8).

Built in GitHub Actions and published to `ghcr.io/jbowensii/kohakumoss:latest`. This is **not** the
upstream project — just a build wrapper for homelab use (Tower06 / Unraid).

## What it is

- Base: `nvidia/cuda:12.6.3-devel-ubuntu24.04` (the `-devel` base is required — TileLang JIT-compiles
  sm_70 kernels at runtime and needs `nvcc`).
- PyTorch `2.14.0+cu126` (the cu126 wheel still ships Volta/sm_70 kernels; cu128+ drop them).
- `pip install kohakumoss[cuda]==0.1.0`.
- **CLI only** — there is no web UI. The container idles (`sleep infinity`); you `docker exec` in to train.

## Using it (once the Unraid container is running)

```bash
docker exec -it KohakuMossTrainer bash
kohakumoss-doctor                        # expect sm70 listed, tilelang/triton OK
ls /models/diffusion_models              # MOSS-Image base weights (mounted read-only)
kohakumoss-config wizard                 # choose 16 GB GPU (vram_16g), LoRA, dataset, weights=/models
kogine run scripts/data/cache_text.py    --config configs/user/<name>_data.py
kogine run scripts/data/cache_latents.py --config configs/user/<name>_data.py
kogine run scripts/train/moss.py --config configs/user/<name>.py --set STEPS=2000
```

Outputs land in `/workspace/runs/moss/<name>/` (mounted to `ai-lab/training/output`).

## Volta / 16 GB notes

- Use the `vram_16g` memory layer. FP16 compute with the automatic residual-overflow fix (no bf16).
- Expect roughly RTX-2080-Ti-class speed. Host RAM (20+ GiB) is the likely constraint.

## Mounts (Unraid template)

| Container | Host |
|---|---|
| `/workspace/datasets` | `/mnt/user/ai-lab/training/datasets` |
| `/workspace/runs` | `/mnt/user/ai-lab/training/output` |
| `/workspace/configs/user` | `/mnt/user/ai-lab/training/configs` |
| `/models` (ro) | `/mnt/cache_ssd/ai-models/comfyui/basedir/models` |
