import argparse
import json
import logging
import math
import pathlib
import time

import torch
import torch.nn as nn
import torchinfo
from torcheval import metrics
from torchvision import datasets, transforms

from resnet import loaders, networks, trainers

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]


def get_args():
    parser = argparse.ArgumentParser()

    train_path = PROJECT_DIR / "data/processed/train/"
    valid_path = PROJECT_DIR / "data/processed/valid/"
    parser.add_argument("--train_path", type=str, required=False, default=train_path)
    parser.add_argument("--valid_path", type=str, required=False, default=valid_path)

    parser.add_argument("--model_info", type=bool, required=False, default=False)

    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--epochs", type=int, required=False, default=50)
    parser.add_argument("--max_lr", type=float, required=False, default=1e-4)

    return parser.parse_args()


def main():
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)
    logger = logging.getLogger(__name__)
    logger.info("Training model. This may take a while.")

    run_time = time.strftime("%Y_%m_%d_%H_%M_%S")
    args = get_args()

    logger.info("Training settings:")
    print()
    args_str = "\n".join(f"{k}:".ljust(15) + f"{v}" for k, v in vars(get_args()).items())
    print(args_str)
    print()

    train_dataset = datasets.ImageFolder(args.train_path)
    valid_dataset = datasets.ImageFolder(args.valid_path)

    train_transform = transforms.Compose(
        [
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=72),
            transforms.ColorJitter(contrast=0.3, saturation=0.3, brightness=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    train_loader = loaders.MutableDataLoader(
        train_dataset,
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    valid_loader = loaders.MutableDataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        eval_transform=eval_transform,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    resnet = networks.SEResNet().to(DEVICE)

    total_scheduler_steps = args.epochs * int(math.ceil(len(train_dataset) / args.batch_size))
    loss_fn = nn.BCEWithLogitsLoss()
    binary_acc_metric = metrics.BinaryAccuracy()
    optimizer = torch.optim.AdamW(resnet.parameters(), amsgrad=True, fused=True)
    onecycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        div_factor=10,
        total_steps=total_scheduler_steps,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
    )

    history = trainers.train(
        model=resnet,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_fn=loss_fn,
        metric=binary_acc_metric,
        optimizer=optimizer,
        scheduler=onecycle_scheduler,
        epochs=args.epochs,
        device=DEVICE,
    )

    run_save_path = PROJECT_DIR / "models" / run_time
    run_save_path.mkdir(parents=True, exist_ok=False)

    torch.save(resnet.state_dict(), run_save_path / "state_dict.pt")

    with open(run_save_path / "history.json", "w") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    if args.model_info:
        with open(run_save_path / "architecture.txt", "w") as f:
            architecture = torchinfo.summary(
                resnet, input_size=(args.batch_size, 3, 224, 224), verbose=0
            )
            f.write(str(architecture))

    print()
    logger.info("All finished!")


if __name__ == "__main__":
    main()
