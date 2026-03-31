import os
import torch
from tqdm import tqdm
import numpy as np
import folder_paths
import cv2
import json
import logging
script_directory = os.path.dirname(os.path.abspath(__file__))

from comfy import model_management as mm
from comfy.utils import ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))

from .models.onnx_models import ViTPose, Yolo
from .pose_utils.pose2d_utils import load_pose_metas_from_kp2ds_seq, crop, bbox_from_detector
from .utils import get_face_bboxes, padding_resize, resize_by_area, resize_to_bounds
from .pose_utils.human_visualization import AAPoseMeta, draw_aapose_by_meta_new
from .retarget_pose import get_retarget_pose
from .mediapipe_bbox import generate_mediapipe_bboxes

class OnnxDetectionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vitpose_model": (folder_paths.get_filename_list("detection"), {"tooltip": "这些模型从 'ComfyUI/models/detection' 文件夹加载。",}),
                "yolo_model": (folder_paths.get_filename_list("detection"), {"tooltip": "这些模型从 'ComfyUI/models/detection' 文件夹加载。",}),
                "onnx_device": (["CUDAExecutionProvider", "CPUExecutionProvider"], {"default": "CUDAExecutionProvider", "tooltip": "用于运行 ONNX 模型的设备。"}),
            },
        }

    RETURN_TYPES = ("POSEMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Loads ONNX models for pose and face detection. ViTPose for pose estimation and YOLO for object detection."

    def loadmodel(self, vitpose_model, yolo_model, onnx_device):

        vitpose_model_path = folder_paths.get_full_path_or_raise("detection", vitpose_model)
        yolo_model_path = folder_paths.get_full_path_or_raise("detection", yolo_model)

        vitpose = ViTPose(vitpose_model_path, onnx_device)
        yolo = Yolo(yolo_model_path, onnx_device)

        model = {
            "vitpose": vitpose,
            "yolo": yolo,
        }

        return (model, )

class PoseAndFaceDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1, "tooltip": "生成结果的宽度。"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1, "tooltip": "生成结果的高度。"}),
            },
            "optional": {
                "retarget_image": ("IMAGE", {"default": None, "tooltip": "用于姿态重定向的可选参考图像。"}),
                "face_padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1, "tooltip": "大于 0 时，对检测到的人脸区域扩边后再缩放到 512x512。"}),
                "use_mediapipe_bbox": ("BOOLEAN", {"default": False, "tooltip": "启用 MediaPipe 多类别分割，用于生成 SAM2 的 bbox 提示。"}),
                "mediapipe_0_background": ("BOOLEAN", {"default": False, "tooltip": "MediaPipe 类别 0：背景。"}),
                "mediapipe_1_hair": ("BOOLEAN", {"default": False, "tooltip": "MediaPipe 类别 1：头发。"}),
                "mediapipe_2_body_skin": ("BOOLEAN", {"default": False, "tooltip": "MediaPipe 类别 2：身体皮肤。"}),
                "mediapipe_3_face_skin": ("BOOLEAN", {"default": False, "tooltip": "MediaPipe 类别 3：面部皮肤。"}),
                "mediapipe_4_clothes": ("BOOLEAN", {"default": False, "tooltip": "MediaPipe 类别 4：衣物。"}),
                "mediapipe_5_others": ("BOOLEAN", {"default": False, "tooltip": "MediaPipe 类别 5：其他（配饰等）。"}),
                "mediapipe_confidence_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "将 MediaPipe 置信度图转为二值区域时使用的阈值。"}),
                "mediapipe_mask_dilation": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": "提取 bbox 前对 mask 做形态学处理：大于 0 为膨胀，小于 0 为腐蚀。"}),
            },
        }

    RETURN_TYPES = ("POSEDATA", "IMAGE", "STRING", "BBOX", "BBOX", "BBOX", "INT")
    RETURN_NAMES = ("pose_data", "face_images", "key_frame_body_points", "bboxes", "face_bboxes", "mediapipe_bbox", "mediapipe_bbox_frame_idx")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Detects human poses and face images from input images. Optionally retargets poses based on a reference image."

    def process(
        self,
        model,
        images,
        width,
        height,
        retarget_image=None,
        face_padding=0,
        use_mediapipe_bbox=False,
        mediapipe_0_background=False,
        mediapipe_1_hair=False,
        mediapipe_2_body_skin=False,
        mediapipe_3_face_skin=False,
        mediapipe_4_clothes=False,
        mediapipe_5_others=False,
        mediapipe_confidence_threshold=0.25,
        mediapipe_mask_dilation=0,
    ):
        detector = model["yolo"]
        pose_model = model["vitpose"]
        B, H, W, C = images.shape

        shape = np.array([H, W])[None]
        images_np = images.numpy()

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution=(256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()
        if retarget_image is not None:
            refer_img = resize_by_area(retarget_image[0].numpy() * 255, width * height, divisor=16) / 255.0
            ref_bbox = (detector(
                cv2.resize(refer_img.astype(np.float32), (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])

            if ref_bbox is None or ref_bbox[-1] <= 0 or (ref_bbox[2] - ref_bbox[0]) < 10 or (ref_bbox[3] - ref_bbox[1]) < 10:
                ref_bbox = np.array([0, 0, refer_img.shape[1], refer_img.shape[0]])

            center, scale = bbox_from_detector(ref_bbox, input_resolution, rescale=rescale)
            refer_img = crop(refer_img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (refer_img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            ref_keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            refer_pose_meta = load_pose_metas_from_kp2ds_seq(ref_keypoints, width=retarget_image.shape[2], height=retarget_image.shape[1])[0]

        comfy_pbar = ProgressBar(B*2)
        progress = 0
        bboxes = []
        for img in tqdm(images_np, total=len(images_np), desc="Detecting bboxes"):
            bboxes.append(detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        kp2ds = []
        for img, bbox in tqdm(zip(images_np, bboxes), total=len(images_np), desc="Extracting keypoints"):
            if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                bbox = np.array([0, 0, img.shape[1], img.shape[0]])

            bbox_xywh = bbox
            center, scale = bbox_from_detector(bbox_xywh, input_resolution, rescale=rescale)
            img = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            kp2ds.append(keypoints)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()

        kp2ds = np.concatenate(kp2ds, 0)
        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)

        def has_valid_face(meta, min_confidence=0.35, min_visible_points=5):
            keypoints_face = np.asarray(meta.get("keypoints_face", []))
            if keypoints_face.ndim != 2 or keypoints_face.shape[0] == 0:
                return False
            if keypoints_face.shape[1] < 3:
                return False
            conf = keypoints_face[:, 2]
            x = keypoints_face[:, 0]
            y = keypoints_face[:, 1]
            visible = (conf >= min_confidence) & (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)
            return int(np.count_nonzero(visible)) >= min_visible_points

        first_valid_face_idx = None
        for idx, meta in enumerate(pose_metas):
            if has_valid_face(meta):
                first_valid_face_idx = idx
                break

        if first_valid_face_idx is None:
            logging.warning("No valid face found in sequence. Falling back to frame 0 for face outputs.")
            first_valid_face_idx = 0

        face_images = []
        face_bboxes = []
        for idx in range(first_valid_face_idx, len(pose_metas)):
            meta = pose_metas[idx]
            face_bbox_for_image = get_face_bboxes(meta['keypoints_face'][:, :2], scale=1.3, image_shape=(H, W))
            x1, x2, y1, y2 = face_bbox_for_image
            if face_padding > 0:
                x1 = max(0, x1 - face_padding)
                y1 = max(0, y1 - face_padding)
                x2 = min(W, x2 + face_padding)
                y2 = min(H, y2 + face_padding)
            face_bboxes.append((x1, y1, x2, y2))
            face_image = images_np[idx][y1:y2, x1:x2]
            # Check if face_image is valid before resizing
            if face_image.size == 0 or face_image.shape[0] == 0 or face_image.shape[1] == 0:
                logging.warning(f"Empty face crop on frame {idx}, creating fallback image.")
                # Create a fallback image (black or use center crop)
                fallback_size = int(min(H, W) * 0.3)
                fallback_x1 = (W - fallback_size) // 2
                fallback_x2 = fallback_x1 + fallback_size
                fallback_y1 = int(H * 0.1)
                fallback_y2 = fallback_y1 + fallback_size
                face_image = images_np[idx][fallback_y1:fallback_y2, fallback_x1:fallback_x2]

                # If still empty, create a black image
                if face_image.size == 0:
                    face_image = np.zeros((fallback_size, fallback_size, C), dtype=images_np.dtype)
            face_image = cv2.resize(face_image, (512, 512))
            face_images.append(face_image)

        face_images_np = np.stack(face_images, 0)
        face_images_tensor = torch.from_numpy(face_images_np)

        if retarget_image is not None and refer_pose_meta is not None:
            retarget_pose_metas = get_retarget_pose(pose_metas[0], refer_pose_meta, pose_metas, None, None)
        else:
            retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in pose_metas]

        bbox = np.array(bboxes[0]).flatten()
        if bbox.shape[0] >= 4:
            bbox_ints = tuple(int(v) for v in bbox[:4])
        else:
            bbox_ints = (0, 0, 0, 0)

        key_frame_num = 4 if B >= 4 else 1
        key_frame_step = len(pose_metas) // key_frame_num
        key_frame_index_list = list(range(0, len(pose_metas), key_frame_step))

        key_points_index = [0, 1, 2, 5, 8, 11, 10, 13]

        for key_frame_index in key_frame_index_list:
            keypoints_body_list = []
            body_key_points = pose_metas[key_frame_index]['keypoints_body']
            for each_index in key_points_index:
                each_keypoint = body_key_points[each_index]
                if None is each_keypoint:
                    continue
                keypoints_body_list.append(each_keypoint)

            keypoints_body = np.array(keypoints_body_list)[:, :2]
            wh = np.array([[pose_metas[0]['width'], pose_metas[0]['height']]])
            points = (keypoints_body * wh).astype(np.int32)
            points_dict_list = []
            for point in points:
                points_dict_list.append({"x": int(point[0]), "y": int(point[1])})

        pose_data = {
            "retarget_image": refer_img if retarget_image is not None else None,
            "pose_metas": retarget_pose_metas,
            "refer_pose_meta": refer_pose_meta if retarget_image is not None else None,
            "pose_metas_original": pose_metas,
        }

        mediapipe_bbox = []
        mediapipe_bbox_frame_idx = -1
        if use_mediapipe_bbox:
            enabled_class_ids = []
            if mediapipe_0_background:
                enabled_class_ids.append(0)
            if mediapipe_1_hair:
                enabled_class_ids.append(1)
            if mediapipe_2_body_skin:
                enabled_class_ids.append(2)
            if mediapipe_3_face_skin:
                enabled_class_ids.append(3)
            if mediapipe_4_clothes:
                enabled_class_ids.append(4)
            if mediapipe_5_others:
                enabled_class_ids.append(5)

            images_for_mediapipe = images_np[first_valid_face_idx:] if len(images_np) > first_valid_face_idx else images_np
            mediapipe_bbox, local_frame_idx = generate_mediapipe_bboxes(
                images_for_mediapipe,
                enabled_class_ids=enabled_class_ids,
                confidence_threshold=mediapipe_confidence_threshold,
                mask_dilation=mediapipe_mask_dilation,
                return_frame_index=True,
            )
            if local_frame_idx >= 0:
                mediapipe_bbox_frame_idx = first_valid_face_idx + local_frame_idx
            if not mediapipe_bbox:
                mediapipe_bbox = [bbox_ints]
                mediapipe_bbox_frame_idx = int(first_valid_face_idx)

        return (pose_data, face_images_tensor, json.dumps(points_dict_list), [bbox_ints], face_bboxes, mediapipe_bbox, int(mediapipe_bbox_frame_idx))

class DrawViTPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1, "tooltip": "生成结果的宽度。"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1, "tooltip": "生成结果的高度。"}),
                "retarget_padding": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1, "tooltip": "大于 0 时，对重定向后的姿态图进行补边并缩放到目标尺寸。"}),
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "身体骨架线宽。设为 0 可关闭身体绘制，-1 为自动。"}),
                "hand_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "手部骨架线宽。设为 0 可关闭手部绘制，-1 为自动。"}),
                "draw_head": ("BOOLEAN", {"default": "True", "tooltip": "是否绘制头部关键点。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("pose_images", )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Draws pose images from pose data."

    def process(self, pose_data, width, height, body_stick_width, hand_stick_width, draw_head, retarget_padding=64):

        retarget_image = pose_data.get("retarget_image", None)
        pose_metas = pose_data["pose_metas"]

        draw_hand = hand_stick_width != 0
        use_retarget_resize = retarget_padding > 0 and retarget_image is not None

        comfy_pbar = ProgressBar(len(pose_metas))
        progress = 0
        crop_target_image = None
        pose_images = []

        for idx, meta in enumerate(tqdm(pose_metas, desc="Drawing pose images")):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            pose_image = draw_aapose_by_meta_new(canvas, meta, draw_hand=draw_hand, draw_head=draw_head, body_stick_width=body_stick_width, hand_stick_width=hand_stick_width)

            if crop_target_image is None:
                crop_target_image = pose_image

            if use_retarget_resize:
                pose_image = resize_to_bounds(pose_image, height, width, crop_target_image=crop_target_image, extra_padding=retarget_padding)
            else:
                pose_image = padding_resize(pose_image, height, width)

            pose_images.append(pose_image)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_images_np = np.stack(pose_images, 0)
        pose_images_tensor = torch.from_numpy(pose_images_np).float() / 255.0

        return (pose_images_tensor, )

class PoseRetargetPromptHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", )
    RETURN_NAMES = ("prompt", "retarget_prompt", )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Generates text prompts for pose retargeting based on visibility of arms and legs in the template pose. Originally used for Flux Kontext"

    def process(self, pose_data):
        refer_pose_meta = pose_data.get("refer_pose_meta", None)
        if refer_pose_meta is None:
            return ("Change the person to face forward.", "Change the person to face forward.", )
        tpl_pose_metas = pose_data["pose_metas_original"]
        arm_visible = False
        leg_visible = False

        for tpl_pose_meta in tpl_pose_metas:
            tpl_keypoints = tpl_pose_meta['keypoints_body']
            tpl_keypoints = np.array(tpl_keypoints)
            if np.any(tpl_keypoints[3]) != 0 or np.any(tpl_keypoints[4]) != 0 or np.any(tpl_keypoints[6]) != 0 or np.any(tpl_keypoints[7]) != 0:
                if (tpl_keypoints[3][0] <= 1 and tpl_keypoints[3][1] <= 1 and tpl_keypoints[3][2] >= 0.75) or (tpl_keypoints[4][0] <= 1 and tpl_keypoints[4][1] <= 1 and tpl_keypoints[4][2] >= 0.75) or \
                    (tpl_keypoints[6][0] <= 1 and tpl_keypoints[6][1] <= 1 and tpl_keypoints[6][2] >= 0.75) or (tpl_keypoints[7][0] <= 1 and tpl_keypoints[7][1] <= 1 and tpl_keypoints[7][2] >= 0.75):
                    arm_visible = True
            if np.any(tpl_keypoints[9]) != 0 or np.any(tpl_keypoints[12]) != 0 or np.any(tpl_keypoints[10]) != 0 or np.any(tpl_keypoints[13]) != 0:
                if (tpl_keypoints[9][0] <= 1 and tpl_keypoints[9][1] <= 1 and tpl_keypoints[9][2] >= 0.75) or (tpl_keypoints[12][0] <= 1 and tpl_keypoints[12][1] <= 1 and tpl_keypoints[12][2] >= 0.75) or \
                    (tpl_keypoints[10][0] <= 1 and tpl_keypoints[10][1] <= 1 and tpl_keypoints[10][2] >= 0.75) or (tpl_keypoints[13][0] <= 1 and tpl_keypoints[13][1] <= 1 and tpl_keypoints[13][2] >= 0.75):
                    leg_visible = True
            if arm_visible and leg_visible:
                break

        if leg_visible:
            if tpl_pose_meta['width'] > tpl_pose_meta['height']:
                tpl_prompt = "Change the person to a standard T-pose (facing forward with arms extended). The person is standing. Feet and Hands are visible in the image."
            else:
                tpl_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. The person is standing. Feet and Hands are visible in the image."

            if refer_pose_meta['width'] > refer_pose_meta['height']:
                refer_prompt = "Change the person to a standard T-pose (facing forward with arms extended). The person is standing. Feet and Hands are visible in the image."
            else:
                refer_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. The person is standing. Feet and Hands are visible in the image."
        elif arm_visible:
            if tpl_pose_meta['width'] > tpl_pose_meta['height']:
                tpl_prompt = "Change the person to a standard T-pose (facing forward with arms extended). Hands are visible in the image."
            else:
                tpl_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. Hands are visible in the image."

            if refer_pose_meta['width'] > refer_pose_meta['height']:
                refer_prompt = "Change the person to a standard T-pose (facing forward with arms extended). Hands are visible in the image."
            else:
                refer_prompt = "Change the person to a standard pose with the face oriented forward and arms extending straight down by the sides. Hands are visible in the image."
        else:
            tpl_prompt = "Change the person to face forward."
            refer_prompt = "Change the person to face forward."

        return (tpl_prompt, refer_prompt, )

class PoseDetectionOneToAllAnimation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 2, "tooltip": "生成结果的宽度。"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 2, "tooltip": "生成结果的高度。"}),
                "align_to": (["ref", "pose", "none"], {"default": "ref", "tooltip": "姿态对齐模式。"}),
                "draw_face_points": (["full", "weak", "none"], {"default": "full", "tooltip": "是否在姿态图中绘制面部关键点。"}),
                "draw_head": (["full", "weak", "none"], {"default": "full", "tooltip": "是否在姿态图中绘制头部关键点。"}),
            },
            "optional": {
                "ref_image": ("IMAGE", {"default": None, "tooltip": "用于姿态重定向的可选参考图像。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK",)
    RETURN_NAMES = ("pose_images", "ref_pose_image", "ref_image", "ref_mask")
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Specialized pose detection and alignment for OneToAllAnimation model https://github.com/ssj9596/One-to-All-Animation. Detects poses from input images and aligns them based on a reference image if provided."

    def process(self, model, images, width, height, align_to, draw_face_points, draw_head, ref_image=None):
        from .onetoall.infer_function import aaposemeta_to_dwpose, align_to_reference, align_to_pose
        from .onetoall.utils import draw_pose_aligned, warp_ref_to_pose
        detector = model["yolo"]
        pose_model = model["vitpose"]
        B, H, W, C = images.shape

        shape = np.array([H, W])[None]
        images_np = images.numpy()

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution=(256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()

        if ref_image is not None:
            refer_img_np = ref_image[0].numpy() * 255
            refer_img = resize_by_area(refer_img_np, width * height, divisor=16) / 255.0
            ref_bbox = (detector(
                cv2.resize(refer_img.astype(np.float32), (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])

            if ref_bbox is None or ref_bbox[-1] <= 0 or (ref_bbox[2] - ref_bbox[0]) < 10 or (ref_bbox[3] - ref_bbox[1]) < 10:
                ref_bbox = np.array([0, 0, refer_img.shape[1], refer_img.shape[0]])

            center, scale = bbox_from_detector(ref_bbox, input_resolution, rescale=rescale)
            refer_img = crop(refer_img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (refer_img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            ref_keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            refer_pose_meta = load_pose_metas_from_kp2ds_seq(ref_keypoints, width=ref_image.shape[2], height=ref_image.shape[1])[0]

            ref_dwpose = aaposemeta_to_dwpose(refer_pose_meta)

        comfy_pbar = ProgressBar(B*2)
        progress = 0
        bboxes = []
        for img in tqdm(images_np, total=len(images_np), desc="Detecting bboxes"):
            bboxes.append(detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        kp2ds = []
        for img, bbox in tqdm(zip(images_np, bboxes), total=len(images_np), desc="Extracting keypoints"):
            if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                bbox = np.array([0, 0, img.shape[1], img.shape[0]])

            bbox_xywh = bbox
            center, scale = bbox_from_detector(bbox_xywh, input_resolution, rescale=rescale)
            img = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            kp2ds.append(keypoints)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()

        kp2ds = np.concatenate(kp2ds, 0)
        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)
        tpl_dwposes = [aaposemeta_to_dwpose(meta) for meta in pose_metas]

        ref_pose_image_tensor = None
        if ref_image is not None:
            if align_to == "ref":
                ref_pose_image =  draw_pose_aligned(ref_dwpose, height, width, without_face=True)
                ref_pose_image_np = np.stack(ref_pose_image, 0)
                ref_pose_image_tensor = torch.from_numpy(ref_pose_image_np).unsqueeze(0).float() / 255.0
                tpl_dwposes = align_to_reference(refer_pose_meta, pose_metas, tpl_dwposes, anchor_idx=0)
                image_input_tensor = ref_image
                image_mask_tensor = torch.zeros(1, ref_image.shape[1], ref_image.shape[2], dtype=torch.float32, device="cpu")
            elif align_to == "pose":
                image_input, ref_pose_image_np, image_mask = warp_ref_to_pose(refer_img_np, tpl_dwposes[0], ref_dwpose)
                ref_pose_image_np = np.stack(ref_pose_image_np, 0)
                ref_pose_image_tensor = torch.from_numpy(ref_pose_image_np).unsqueeze(0).float() / 255.0
                tpl_dwposes = align_to_pose(ref_dwpose, tpl_dwposes, anchor_idx=0)
                image_input_tensor = torch.from_numpy(image_input).unsqueeze(0).float() / 255.0
                image_mask_tensor = torch.from_numpy(image_mask).unsqueeze(0).float() / 255.0
            elif align_to == "none":
                ref_pose_image =  draw_pose_aligned(ref_dwpose, height, width, without_face=True)
                ref_pose_image_np = np.stack(ref_pose_image, 0)
                ref_pose_image_tensor = torch.from_numpy(ref_pose_image_np).unsqueeze(0).float() / 255.0
                image_input_tensor = ref_image
                image_mask_tensor = torch.zeros(1, ref_image.shape[1], ref_image.shape[2], dtype=torch.float32, device="cpu")
        else:
            ref_pose_image_tensor = torch.zeros((1, height, width, 3), dtype=torch.float32, device="cpu")
            image_input_tensor = torch.zeros((1, height, width, 3), dtype=torch.float32, device="cpu")
            image_mask_tensor = torch.zeros(1, height, width, dtype=torch.float32, device="cpu")

        pose_imgs = []
        for pose_np in tpl_dwposes:
            pose_img = draw_pose_aligned(pose_np, height, width, without_face=(draw_face_points=="none"), face_change=(draw_face_points=="weak"), head_strength=draw_head)
            pose_img = torch.from_numpy(np.array(pose_img))
            pose_imgs.append(pose_img)

        pose_tensor = torch.stack(pose_imgs).cpu().float() / 255.0

        return (pose_tensor, ref_pose_image_tensor, image_input_tensor, image_mask_tensor)

NODE_CLASS_MAPPINGS = {
    "OnnxDetectionModelLoader": OnnxDetectionModelLoader,
    "PoseAndFaceDetection": PoseAndFaceDetection,
    "DrawViTPose": DrawViTPose,
    "PoseRetargetPromptHelper": PoseRetargetPromptHelper,
    "PoseDetectionOneToAllAnimation": PoseDetectionOneToAllAnimation,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OnnxDetectionModelLoader": "ONNX Detection Model Loader",
    "PoseAndFaceDetection": "Pose and Face Detection",
    "DrawViTPose": "Draw ViT Pose",
    "PoseRetargetPromptHelper": "Pose Retarget Prompt Helper",
    "PoseDetectionOneToAllAnimation": "Pose Detection OneToAll Animation",
}
