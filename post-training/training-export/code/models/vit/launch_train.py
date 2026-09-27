"""torchrun entry point, one process per GPU:

    python -m torch.distributed.run --standalone --nproc_per_node=2 -m models.vit.launch_train CONFIG.json
"""
import json
import sys

from models.vit.wish_standalone import train_or_resume


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: -m models.vit.launch_train CONFIG.json")
    with open(sys.argv[1]) as handle:
        train_or_resume(json.load(handle))


if __name__ == "__main__":
    main()
