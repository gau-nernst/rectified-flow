from pathlib import Path

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from .iresnet import load_adaface_ir101


# https://github.com/hpc203/yolov8-face-landmarks-opencv-dnn
class YOLOv8Face:
    input_size = 640
    reg_max = 16  # for distribution focal loss

    def __init__(self):
        url = "https://github.com/hpc203/yolov8-face-landmarks-opencv-dnn/raw/1f91851f/weights/yolov8n-face.onnx"
        local_path = Path(__file__).parent / url.split("/")[-1]
        if not local_path.exists():
            resp = requests.get(url)
            resp.raise_for_status()
            open(local_path, "wb").write(resp.content)

        import onnxruntime

        self.sess = onnxruntime.InferenceSession(local_path)
        self.output_names = [x.name for x in self.sess.get_outputs()]
        self.anchors = dict()

        for stride in (8, 16, 32):
            x = torch.arange(self.input_size // stride)
            xs, ys = torch.meshgrid(x, x, indexing="xy")
            self.anchors[stride] = torch.stack([xs, ys], dim=-1)

    def _resize_img(self, img: Image.Image):
        # resize long edge to 640, then pad the short edge to 640
        scale = self.input_size / max(img.size)
        h = round(img.height * scale)
        w = round(img.width * scale)

        if img.mode != "RGB":
            img = img.convert("RGB")
        img_np = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        img_np[:h, :w] = img.resize((w, h), Image.Resampling.BILINEAR)
        return img_np, 1 / scale

    def __call__(self, img: Image.Image, score_th: float = 0.6, nms_th: float = 0.8):
        """Returns bboxes (xyxy), scores, keypoints"""
        img_np, scale = self._resize_img(img)
        img_np = img_np.astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW

        # 3 outputs, correspond to 3 strides
        outputs = self.sess.run(self.output_names, dict(images=img_np[None]))

        all_bboxes = []
        all_scores = []
        all_kpts = []
        for out in outputs:
            stride = self.input_size // out.shape[2]
            points = self.anchors[stride]

            # NOTE: these can be included in the ONNX graph
            out = torch.from_numpy(out).permute(0, 2, 3, 1)  # NCHW -> NHWC
            bboxes, logits, kpts = out.split_with_sizes([self.reg_max * 4, 1, 15], dim=-1)
            scores = logits.sigmoid().view(-1)

            bboxes = bboxes.unflatten(-1, (4, self.reg_max)).softmax(-1)  # N, H, W, 4, reg_max
            bboxes = bboxes @ torch.arange(self.reg_max, dtype=torch.float)  # N, H, W, 4
            x1y1 = points + 0.5 - bboxes[..., :2]
            x2y2 = points + 0.5 + bboxes[..., 2:]
            bboxes = torch.cat([x1y1, x2y2], dim=-1).view(-1, 4) * (stride * scale)

            # ignore keypoints score
            kpts_x = kpts[..., 0::3] * 2.0 + points[..., :1]
            kpts_y = kpts[..., 1::3] * 2.0 + points[..., 1:]
            kpts = torch.stack([kpts_x, kpts_y], dim=-1).view(-1, 5, 2) * (stride * scale)

            mask = scores >= score_th
            all_bboxes.append(bboxes[mask])
            all_scores.append(scores[mask])
            all_kpts.append(kpts[mask])

        all_bboxes = torch.cat(all_bboxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_kpts = torch.cat(all_kpts, dim=0)

        indices = torchvision.ops.nms(all_bboxes, all_scores, nms_th)
        return all_bboxes[indices], all_scores[indices], all_kpts[indices]


# https://github.com/deepinsight/insightface/blob/2e29b41a226d5344aab36d3a470a55a33bd50af0/recognition/arcface_mxnet/common/face_align.py
ARCFACE_IMG_SIZE = 112
ARCFACE_KEYPOINTS = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def arcface_crop(img: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    # For AffinePartial2D, M has the following form
    # [s*cos(theta), -s*sin(theta),  tx]
    # [s*sin(theta),  s*cos(theta),  ty]
    # where s is scale
    #       theta is rotation
    #       tx, ty is translation
    # default RANSAC method is worse on some cases
    M, _ = cv2.estimateAffinePartial2D(keypoints, ARCFACE_KEYPOINTS, method=cv2.LMEDS)
    scale = (M[0, 0] ** 2 + M[0, 1] ** 2) ** 0.5
    M /= scale  # remove scale

    img_size = round(ARCFACE_IMG_SIZE / scale)
    return cv2.warpAffine(img, M, (img_size, img_size), borderValue=0.0)


class FaceEmbedder:
    def __init__(self):
        self.detector = YOLOv8Face()
        self.embedder = load_adaface_ir101().eval()

    @torch.no_grad()
    def __call__(self, img: Image.Image):
        _, _, kpts = self.detector(img)
        bsize = kpts.shape[0]

        img_np = np.asarray(img)
        faces = torch.empty(bsize, 3, 112, 112)
        for i in range(kpts.shape[0]):
            face = arcface_crop(img_np, kpts[i].numpy())
            face = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW
            faces[i] = F.interpolate(face, (112, 112), mode="bilinear", antialias=True).squeeze(0)

        feat = self.embedder.forward_features(faces)
        return feat.flatten(-2).transpose(1, 2)  # (bsize, 49, 512)
