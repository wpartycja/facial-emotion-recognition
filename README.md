# facial-emotion-recogniton

A Bachelor's thesis project for recognizing facial emotions from images, covering dataset preparation, model training, and a demo web app for real-time inference.

## Contents

- `dataset_prep/` – scripts for preparing and preprocessing facial emotion datasets (e.g. AffectNet, ExpW), including face cropping and train/test splitting.
- `models/` – emotion recognition model implementations used in the project (DAN, DMUE, POSTER_V2).
- `ran_models/` – training scripts run for specific model/dataset combinations.
- `app_feraina/` – demo application (FastAPI/Flask + ONNX) with a simple web interface for real-time emotion prediction, including a camera-based demo.

## Getting started

Install the app dependencies:

```bash
pip install -r app_feraina/requirements.txt
```

A pretrained ONNX model is linked in the app for inference; see `app_feraina/server/server.py` for how it's loaded and served.
[Link](https://1drv.ms/u/c/e651ca9104cac9d9/EUm603VCMypEhHPWT-gNuq4BdO6dscj-FmE2uRr01BVTwg?e=ZYnT9X) to `onnx` model
