import os
import threading
import urllib.request

import cv2
import numpy as np
import folder_paths

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - handled at runtime when dependency is missing
    mp = None


MODEL_FOLDER_NAME = "mediapipe"
MODEL_FILE_NAME = "selfie_multiclass_256x256.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)

CLASS_LABELS = {
    0: "background",
    1: "hair",
    2: "body-skin",
    3: "face-skin",
    4: "clothes",
    5: "others",
}

_SEGMENTER = None
_SEGMENTER_LOCK = threading.Lock()


def _ensure_mediapipe_available():
    if mp is None:
        raise ImportError(
            "mediapipe is not installed. Please install the dependency listed in requirements.txt."
        )


def _get_model_path():
    model_dir = os.path.join(folder_paths.models_dir, MODEL_FOLDER_NAME)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, MODEL_FILE_NAME)
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def _get_segmenter():
    global _SEGMENTER
    _ensure_mediapipe_available()
    if _SEGMENTER is None:
        with _SEGMENTER_LOCK:
            if _SEGMENTER is None:
                BaseOptions = mp.tasks.BaseOptions
                ImageSegmenter = mp.tasks.vision.ImageSegmenter
                ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
                VisionRunningMode = mp.tasks.vision.RunningMode
                options = ImageSegmenterOptions(
                    base_options=BaseOptions(model_asset_path=_get_model_path()),
                    running_mode=VisionRunningMode.IMAGE,
                    output_confidence_masks=True,
                    output_category_mask=False,
                )
                _SEGMENTER = ImageSegmenter.create_from_options(options)
    return _SEGMENTER


def _to_uint8_image(image):
    if image.dtype == np.uint8:
        return image
    image = np.clip(image, 0.0, 1.0) if image.max() <= 1.5 else np.clip(image, 0, 255)
    if image.max() <= 1.5:
        image = image * 255.0
    return image.astype(np.uint8)


def _to_mediapipe_image(image):
    image_uint8 = _to_uint8_image(image)
    if image_uint8.ndim != 3:
        raise ValueError("MediaPipe segmentation expects an HWC image.")
    if image_uint8.shape[2] == 4:
        image_format = mp.ImageFormat.SRGBA
    else:
        image_format = mp.ImageFormat.SRGB
    return mp.Image(image_format=image_format, data=image_uint8)


def _apply_mask_dilation(mask, mask_dilation):
    if mask_dilation == 0:
        return mask
    kernel_size = 2 * abs(mask_dilation) + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
        (abs(mask_dilation), abs(mask_dilation)),
    )
    mask_uint8 = mask.astype(np.uint8)
    if mask_dilation > 0:
        return cv2.dilate(mask_uint8, kernel)
    return cv2.erode(mask_uint8, kernel)


def _ensure_2d_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3 and mask.shape[-1] == 1:
        return mask[..., 0]
    if mask.ndim == 3:
        return np.any(mask > 0, axis=-1).astype(mask.dtype)
    raise ValueError(f"Unsupported mask shape for bbox extraction: {mask.shape}")


def _mask_to_bbox(mask):
    mask = _ensure_2d_mask(mask)
    y_coords, x_coords = np.nonzero(mask)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    x1 = int(x_coords.min())
    y1 = int(y_coords.min())
    x2 = int(x_coords.max()) + 1
    y2 = int(y_coords.max()) + 1
    return (x1, y1, x2, y2)


def generate_mediapipe_bboxes(images_np, enabled_class_ids, confidence_threshold=0.25, mask_dilation=0):
    if not enabled_class_ids:
        return []

    segmenter = _get_segmenter()
    ordered_class_ids = sorted({class_id for class_id in enabled_class_ids if class_id in CLASS_LABELS})
    if not ordered_class_ids:
        return []

    # Match the existing bbox output semantics used by the video workflow:
    # produce a single combined prompt box from the first frame only.
    first_image = images_np[0]
    segmented_masks = segmenter.segment(_to_mediapipe_image(first_image))
    merged_mask = None

    for class_id in ordered_class_ids:
        confidence_mask = segmented_masks.confidence_masks[class_id].numpy_view()
        binary_mask = (confidence_mask > confidence_threshold).astype(np.uint8)
        binary_mask = _ensure_2d_mask(binary_mask)
        binary_mask = _apply_mask_dilation(binary_mask, mask_dilation)
        if merged_mask is None:
            merged_mask = binary_mask.astype(np.uint8)
        else:
            merged_mask = np.maximum(merged_mask, binary_mask.astype(np.uint8))

    if merged_mask is None:
        return []

    bbox = _mask_to_bbox(merged_mask)
    if bbox is None:
        return []

    return [bbox]
