"""Single cropped-face demo using exactly the evaluation preprocessing."""
import argparse
import json

from PIL import Image

from models.vit.runtime import device_for, load_checkpoint, load_config, predict_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, help="RGB face crop, not an uncropped group photograph")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = device_for(args.device)
    model, metadata = load_checkpoint(load_config(args.config), args.checkpoint, device)
    with Image.open(args.image) as image:
        probability = predict_image(model, image, metadata["preprocessing"], device)
    print(json.dumps({"fake_probability": probability,
                      "prediction": "fake" if probability >= metadata["threshold"] else "real",
                      "threshold": metadata["threshold"], "run_name": metadata["run_name"],
                      "note": "Experimental score, not forensic proof."}, indent=2))


if __name__ == "__main__":
    main()
