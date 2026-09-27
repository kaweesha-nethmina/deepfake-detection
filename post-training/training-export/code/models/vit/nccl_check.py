"""GPU to GPU communication preflight, run under torchrun before any training.

All reduces a buffer the size of ViT-B/16's fp32 gradients, checks the result and
times it. A broken PCIe peer to peer path shows up here within seconds (the notebook
then retries with NCCL_P2P_DISABLE=1) instead of as a hang in the middle of training.
Without CUDA it runs the same check over gloo on the CPU, which is only for testing.
"""
import time

import torch
import torch.distributed as dist

from models.vit import distributed as dist_utils

VIT_B16_PARAMETERS = 86_000_000


def main():
    cuda = torch.cuda.is_available()
    ctx = dist_utils.setup("cuda" if cuda else "cpu", timeout_minutes=3)
    try:
        if ctx.world < 2:
            print("Single process: no inter GPU communication to check.", flush=True)
            return
        elements = VIT_B16_PARAMETERS if cuda else 1_000_000
        buffer = torch.full((elements,), float(ctx.rank + 1), device=ctx.device)
        dist.all_reduce(buffer)
        if cuda:
            torch.cuda.synchronize(ctx.device)
        expected = ctx.world * (ctx.world + 1) / 2
        if not (torch.all(buffer[:1024] == expected) and torch.all(buffer[-1024:] == expected)):
            raise RuntimeError("All reduce returned wrong values.")
        repeats = 5
        start = time.perf_counter()
        for _ in range(repeats):
            dist.all_reduce(buffer)
        if cuda:
            torch.cuda.synchronize(ctx.device)
        milliseconds = 1000 * (time.perf_counter() - start) / repeats
        names = dist_utils.all_gather_object(ctx, torch.cuda.get_device_name(ctx.device) if cuda else "cpu")
        if ctx.main:
            backend = "NCCL" if cuda else "gloo (CPU test mode)"
            print(f"{backend} OK across {ctx.world} devices {names}. All reduce of {elements * 4 / 2**20:.0f} MiB "
                  f"(a full ViT-B/16 fp32 gradient on GPU): {milliseconds:.0f} ms; DDP overlaps it with backward.",
                  flush=True)
    finally:
        dist_utils.cleanup(ctx)


if __name__ == "__main__":
    main()
