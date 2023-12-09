import argparse
import logging
import pathlib

import torch
from PIL import Image
from torchvision import transforms

from resnet import networks

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    args = parser.parse_args()

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)
    logger = logging.getLogger(__name__)
    logger.info("Prediction mode.")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    img = Image.open(PROJECT_DIR / args.img_path)
    img = img.resize((224, 224)).convert(mode="RGB")
    img = transform(img).unsqueeze(dim=0)  # type: ignore

    model = networks.SEResNet()
    model.load_state_dict(torch.load(PROJECT_DIR / args.model_path))
    model.eval()

    with torch.inference_mode():
        y_logit = model(img).squeeze()

    y_proba = torch.sigmoid(y_logit).item()
    y_pred = int(y_proba > 0.5)
    tyre_quality = {1: "Good Quality Condition", 0: "Defective Quality Condition"}
    confidence = y_proba if y_pred else 1 - y_proba

    print()
    print(f"Prediction: {tyre_quality.get(y_pred)}")
    print(f"Confidence: {confidence:.3f}")
    print()

    logger.info("All done!")


if __name__ == "__main__":
    main()
