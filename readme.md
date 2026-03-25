## ComfyUI helper nodes for [Wan video 2.2 Animate preprocessing](https://github.com/Wan-Video/Wan2.2/tree/main/wan/modules/animate/preprocess)


Nodes to run the ViTPose model, get face crops and keypoint list for SAM2 segmentation.

`PoseAndFaceDetection` now also includes an optional MediaPipe multiclass segmentation path that can output `mediapipe_bbox` prompts for SAM2. The MediaPipe classes are exposed as node parameters so you can enable any combination of:

- `0 - background`
- `1 - hair`
- `2 - body-skin`
- `3 - face-skin`
- `4 - clothes`
- `5 - others (accessories)`

When enabled, the node downloads the official `selfie_multiclass_256x256.tflite` model automatically and returns a combined `XYXY` prompt box in `mediapipe_bbox` by merging the selected classes from the first frame. This keeps the output compatible with SAM2 video workflows.

Models:

to `ComfyUI/models/detection` (subject to change in the future)

YOLO:

https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/blob/main/process_checkpoint/det/yolov10m.onnx

ViTPose ONNX:

Use either the Large model from here:

https://huggingface.co/JunkyByte/easy_ViTPose/tree/main/onnx/wholebody

Or the Huge model like in the original code, it's split into two files due to ONNX file size limit:

Both files need to be in same directory, and the onnx file selected in the model loader:

`vitpose_h_wholebody_data.bin` and `vitpose_h_wholebody_model.onnx`

https://huggingface.co/Kijai/vitpose_comfy/tree/main/onnx


![example](example.png)
