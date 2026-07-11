import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from anaprior.features.region_pooling import RegionBox, pool_region_features


def test_pool_region_features_uses_bbox_covered_cells() -> None:
    features = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    boxes = [RegionBox(name="right half", x1=112, y1=0, x2=224, y2=224)]

    result = pool_region_features(features, boxes, image_size=(224, 224))

    assert result.region_names == ["right half"]
    assert result.features.shape == (1, 1, 1)
    assert torch.allclose(result.features[0, 0], torch.tensor([8.5]))
    assert result.cell_counts.tolist() == [8]
    assert result.valid_mask.tolist() == [True]


def test_pool_region_features_handles_single_image_feature_map() -> None:
    features = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ]
    )
    boxes = [RegionBox(name="top row", x1=0, y1=0, x2=224, y2=112)]

    result = pool_region_features(features, boxes, image_size=(224, 224))

    assert result.features.shape == (1, 2)
    assert torch.allclose(result.features[0], torch.tensor([1.5, 15.0]))
    assert result.cell_counts.tolist() == [2]


def test_pool_region_features_marks_empty_clipped_boxes_invalid() -> None:
    features = torch.ones(1, 3, 4, 4)
    boxes = [RegionBox(name="outside", x1=300, y1=300, x2=320, y2=320)]

    result = pool_region_features(features, boxes, image_size=(224, 224))

    assert torch.allclose(result.features, torch.zeros(1, 1, 3))
    assert result.cell_counts.tolist() == [0]
    assert result.valid_mask.tolist() == [False]
