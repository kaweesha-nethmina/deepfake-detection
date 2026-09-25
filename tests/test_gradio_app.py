import hashlib
from pathlib import Path

from PIL import Image
import pytest
import torch

from gradio_app import (CROP_MODE, PHOTO_MODE, DEFAULT_CHECKPOINT, Detector,
                        prepare_face, read_config)
from models.vit.runtime import predict_image


def test_missing_pointer_and_invalid_checkpoints(tmp_path):
    with pytest.raises(FileNotFoundError, match="git lfs"):
        read_config(tmp_path / "missing.pt")
    path = tmp_path / "pointer.pt"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n")
    with pytest.raises(ValueError, match="pointer"):
        read_config(path)
    torch.save({"metadata": {"module": "arbitrary.module"}}, path)
    with pytest.raises(ValueError, match="ViT checkpoint"):
        read_config(path)


def test_input_handling(monkeypatch):
    import cv2
    with pytest.raises(ValueError, match="Upload"):
        prepare_face(None, CROP_MODE)
    with pytest.raises(ValueError, match="small"):
        prepare_face(Image.new("RGB", (5, 5)), CROP_MODE)
    with pytest.raises(ValueError, match="No frontal"):
        prepare_face(Image.new("RGB", (128, 128), "white"), PHOTO_MODE)
    assert prepare_face(Image.new("RGBA", (64, 64)), CROP_MODE).mode == "RGB"
    class DetectorFixture:
        def empty(self):
            return False
        def detectMultiScale(self, *args, **kwargs):
            return [(10, 10, 40, 40), (60, 10, 40, 40)]
    monkeypatch.setattr(cv2, "CascadeClassifier", lambda *args: DetectorFixture())
    with pytest.raises(ValueError, match="Multiple"):
        prepare_face(Image.new("RGB", (128, 128)), PHOTO_MODE)


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.is_file() or DEFAULT_CHECKPOINT.stat().st_size < 1000,
                    reason="Download the Git LFS checkpoint for the real-weight test")
def test_provided_weights_and_inference_agreement():
    torch.set_num_threads(2)
    assert hashlib.file_digest(DEFAULT_CHECKPOINT.open("rb"), "sha256").hexdigest() == "be78ec4969aaf73dc82918f6c9aa794e6d196287920d1957c511106f458a452c"
    detector = Detector(device="cpu")
    # Synthetic input checks wiring, not detector accuracy.
    image = Image.new("RGB", (299, 299), (100, 140, 170))
    before = detector.model.classifier.weight.detach().clone()
    label, scores, crop, check = detector.predict(image)
    direct = predict_image(detector.model, image, detector.metadata["preprocessing"], detector.device)
    assert scores["Fake"] == pytest.approx(direct, abs=1e-7)
    assert sum(scores.values()) == pytest.approx(1)
    assert label == ("Fake" if direct >= .5 else "Real")
    assert "unknown" in check
    assert detector.predict(image, reference=label)[3].startswith("Correct")
    other = "Fake" if label == "Real" else "Real"
    assert detector.predict(image, reference=other)[3].startswith("Incorrect")
    assert torch.equal(before, detector.model.classifier.weight)
    assert not detector.model.training
