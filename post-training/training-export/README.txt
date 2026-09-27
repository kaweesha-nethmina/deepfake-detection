Wish ViT-B/16 real vs fake face detector, run vit_wish_s42_ddp2_v1
best_model.pt sha256: be78ec4969aaf73dc82918f6c9aa794e6d196287920d1957c511106f458a452c

Scope: trained on aligned face crops, FFHQ and CelebA (real) against StyleGAN (fake).
Images from other generators (diffusion models, Gemini, Midjourney) are outside its evidence.

Install:  pip install torch torchvision timm==1.0.30 "gradio>=5,<7" pandas pillow scikit-learn matplotlib
Web app:  python gradio_app/wish_gradio_app.py --run-dir runs/vit_wish_s42_ddp2_v1 --code-dir code   (add --share for a public link)
Python:
    import sys; sys.path[:0] = ["code", "gradio_app"]
    from wish_gradio_app import Detector
    detector = Detector("runs/vit_wish_s42_ddp2_v1")
    print(detector.predict_path("face.jpg"))
