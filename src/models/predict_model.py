import argparse

import torch
from PIL import Image
from torchvision import transforms

from resnet import networks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--input_size", type=int, required=True)
    args = parser.parse_args()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    img = Image.open(args.img_path)
    img = img.resize((args.input_size, args.input_size)).convert(mode="RGB")
    img = transform(img).unsqueeze(dim=0)  # type: ignore

    model = networks.SEResNet(args.input_size)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    with torch.inference_mode():
        y_logit = model(img).squeeze()

    y_proba = torch.sigmoid(y_logit)
    y_pred = torch.round(y_proba)

    print(f"Prediction: {y_pred.item():.2f} - Confidence: {y_proba.item():.2f}")


if __name__ == "__main__":
    main()
