"""Audit Wish images using Member A's explicit assignments; never resplit."""
import argparse
import json

from models.vit.manifest import audit_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output", default="data/manifests/wish_v1")
    parser.add_argument("--near-distance", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(audit_manifest(args.manifest, args.data_root, args.output,
                                   args.dataset_version, args.near_distance), indent=2))


if __name__ == "__main__":
    main()
