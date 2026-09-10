import cv2
import numpy as np
import torch
from torchvision import models, transforms


class PedestrianFeatureExtractor:
    def __init__(self, device="cuda"):
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        # VGG16 pretrained backbone
        vgg = models.vgg16(pretrained=True)

        # 마지막 pooling까지 사용
        self.backbone = vgg.features.to(self.device)
        self.backbone.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),

            # ImageNet normalization
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def crop_person(self, frame, bbox, context_ratio=1.5):
        """
        pedestrian bbox 주변을 약간 포함해서 crop
        bbox = [x1, y1, x2, y2]
        """

        h, w = frame.shape[:2]

        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        bw = x2 - x1
        bh = y2 - y1

        new_w = bw * context_ratio
        new_h = bh * context_ratio

        nx1 = int(max(0, cx - new_w / 2))
        ny1 = int(max(0, cy - new_h / 2))

        nx2 = int(min(w, cx + new_w / 2))
        ny2 = int(min(h, cy + new_h / 2))

        crop = frame[ny1:ny2, nx1:nx2]

        return crop

    @torch.no_grad()
    def extract_frame_feature(self, frame, bbox):
        crop = self.crop_person(frame, bbox)

        if crop.size == 0:
            return None

        # OpenCV BGR → RGB
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        tensor = self.transform(crop)
        tensor = tensor.unsqueeze(0).to(self.device)

        feature = self.backbone(tensor)

        return feature.squeeze(0).cpu()

    @torch.no_grad()
    def extract_sequence(self, sequence):
        """
        sequence:
        [
            {
                "frame_idx": ...,
                "frame": ...,
                "bbox": [...]
            },
            ...
        ]

        반환:
            visual_features : [1, 15, 512, 7, 7]
            bbox_sequence   : [1, 15, 4]
        """

        features = []
        bboxes = []

        for item in sequence:
            feature = self.extract_frame_feature(
                item["frame"],
                item["bbox"]
            )

            if feature is None:
                return None, None

            features.append(feature)
            bboxes.append(item["bbox"])

        visual_features = torch.stack(features, dim=0)

        bbox_sequence = torch.tensor(
            bboxes,
            dtype=torch.float32
        )

        # batch dimension
        visual_features = visual_features.unsqueeze(0)
        bbox_sequence = bbox_sequence.unsqueeze(0)

        return visual_features, bbox_sequence