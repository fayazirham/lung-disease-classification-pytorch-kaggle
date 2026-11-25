"""Shared defaults for the lung disease classification experiments."""

DEFAULT_SEED = 999
DEFAULT_IMAGE_SIZE = 224
CLASS_NAMES = ("Lung_Opacity", "Normal", "Viral Pneumonia")
VALID_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp"})
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
