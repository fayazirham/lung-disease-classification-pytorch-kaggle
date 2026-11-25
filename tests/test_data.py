import pytest
import torch
from PIL import Image

from lung_cnn.config import CLASS_NAMES
from lung_cnn.data import (
    LungXRayDataset,
    build_image_dataframe,
    find_dataset_root,
    split_dataframe,
)


@pytest.fixture
def image_dataset(tmp_path):
    class_root = tmp_path / "input" / "lung-disease"
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = class_root / class_name
        class_dir.mkdir(parents=True)
        for image_index in range(20):
            image = Image.new("RGB", (12, 12), color=(label * 40, image_index, 0))
            image.save(class_dir / f"{image_index:02d}.png")
    return class_root


def test_find_build_and_split_dataset(image_dataset):
    root = find_dataset_root(image_dataset.parent)
    dataframe = build_image_dataframe(root)

    assert len(dataframe) == 60
    assert dataframe["label_name"].unique().tolist() == sorted(CLASS_NAMES)

    first_split = split_dataframe(dataframe, seed=7)
    second_split = split_dataframe(dataframe, seed=7)
    assert [len(frame) for frame in first_split] == [42, 9, 9]
    assert [frame["filepath"].tolist() for frame in first_split] == [
        frame["filepath"].tolist() for frame in second_split
    ]


def test_dataset_returns_normalized_chw_tensor(image_dataset):
    dataframe = build_image_dataframe(image_dataset)
    dataset = LungXRayDataset(dataframe.iloc[:1], img_size=8)

    image, label = dataset[0]

    assert tuple(image.shape) == (3, 8, 8)
    assert image.dtype == torch.float32
    assert label.item() == 0


def test_find_dataset_root_reports_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        find_dataset_root(tmp_path / "missing")
