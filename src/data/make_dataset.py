# -*- coding: utf-8 -*-
import glob
import logging
import pathlib
import shutil
import subprocess
from pathlib import Path

import click
import numpy as np
from dotenv import find_dotenv, load_dotenv
from PIL import Image
from sklearn.model_selection import train_test_split


def download_from_kaggle(expr, /, data_dir=None):
    """Download all files from the Kaggle competition/dataset.

    Args:
        expr: Match expression to be used by kaggle API, e.g.
            "kaggle competitions download -c competition" or
            "kaggle datasets download -d user/dataset".
        data_dir: Optional. Directory path where to save files. Default to `None`,
        which means that files will be downloaded to `data` directory.

    Notes:
        If the associated files already exists, then it does nothing.
    """

    if data_dir is None:
        data_dir = pathlib.Path("../data/raw/")
    else:
        data_dir = pathlib.Path(data_dir)

    match expr.split():
        case ["kaggle", _, "download", *args] if args:
            data_dir.mkdir(parents=True, exist_ok=True)
            filename = args[-1].split("/")[-1] + ".zip"
            if not (data_dir / filename).is_file():
                subprocess.run(expr)
                shutil.unpack_archive(filename, data_dir)
                shutil.move(filename, data_dir)
        case _:
            raise SyntaxError("Invalid expression!")


def get_train_valid_subsets(data_path, /, valid_ratio=0.2, seed=None):
    defective_paths = glob.glob(str(data_path / "defective/*"))
    good_paths = glob.glob(str(data_path / "good/*"))
    paths = np.concatenate((defective_paths, good_paths))

    defective_labels = [0] * len(defective_paths)
    good_labels = [1] * len(good_paths)
    labels = np.concatenate((defective_labels, good_labels))

    return train_test_split(
        paths, labels, test_size=valid_ratio, random_state=seed, stratify=labels
    )


def preprocess_and_save(target_dir, paths, labels, subset, size=(224, 224)):
    defective_path = pathlib.Path(target_dir) / subset / "defective"
    good_path = pathlib.Path(target_dir) / subset / "good"
    defective_path.mkdir(parents=True, exist_ok=True)
    good_path.mkdir(parents=True, exist_ok=True)

    for k, (path, label) in enumerate(zip(paths, labels)):
        target_path = defective_path if label == 0 else good_path
        with Image.open(path) as image:
            img = image.resize(size)
            img.convert(mode="RGB").save(target_path / f"{k}.jpg")


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    dataset = "tyre-quality-classification"
    user = "warcoder"
    expr = f"kaggle datasets download -d {user}/{dataset}"
    download_from_kaggle(expr, data_dir=input_filepath)

    dataset_name = "Digital images of defective and good condition tyres"
    dataset_path = pathlib.Path(input_filepath) / dataset_name
    train_paths, valid_paths, train_labels, valid_labels = get_train_valid_subsets(
        dataset_path, valid_ratio=0.125, seed=42
    )

    preprocess_and_save(output_filepath, train_paths, train_labels, subset="train")
    preprocess_and_save(output_filepath, valid_paths, valid_labels, subset="valid")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
