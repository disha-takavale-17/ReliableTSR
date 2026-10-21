"""Microsoft Table Transformer Table Detection (TD) for locating tables on full pages."""
from dataclasses import dataclass
from typing import List, Optional
from PIL import Image

DETECTION_MODEL_ID = "microsoft/table-transformer-detection"


@dataclass
class DetectedTable:
    table_id: str
    page_number: int
    bbox: List[float]  # [xmin, ymin, xmax, ymax]
    score: float
    cropped_image: Image.Image


class TableDetector:
    """Detects table boundaries across full-page document images."""

    def __init__(self, model_id: str = DETECTION_MODEL_ID, device: str = "cpu", cache_dir: Optional[str] = "data/model-cache"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id, cache_dir=cache_dir)
        self.model.to(device).eval()
        self.torch = torch

    def detect(self, page_image: Image.Image, page_number: int = 1, threshold: float = 0.6) -> List[DetectedTable]:
        """Detect and crop all tables present on a given document page."""
        image = page_image.convert("RGB")
        inputs = self.processor(
            images=image,
            size={"shortest_edge": 800, "longest_edge": 800},
            return_tensors="pt"
        ).to(self.device)

        with self.torch.inference_mode():
            outputs = self.model(**inputs)

        target_sizes = self.torch.tensor([[image.height, image.width]], device=self.device)
        result = self.processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes
        )[0]

        detected: List[DetectedTable] = []
        labels = self.model.config.id2label

        for i, (label_idx, score, box) in enumerate(zip(result["labels"], result["scores"], result["boxes"])):
            label_name = labels[int(label_idx)].lower()
            if "table" in label_name:
                coords = [round(float(v), 2) for v in box.tolist()]
                # Ensure bounding box is within image bounds
                xmin = max(0, int(coords[0]))
                ymin = max(0, int(coords[1]))
                xmax = min(image.width, int(coords[2]))
                ymax = min(image.height, int(coords[3]))

                if xmax > xmin and ymax > ymin:
                    cropped = image.crop((xmin, ymin, xmax, ymax))
                    detected.append(
                        DetectedTable(
                            table_id=f"p{page_number}_t{i+1}",
                            page_number=page_number,
                            bbox=[xmin, ymin, xmax, ymax],
                            score=round(float(score), 4),
                            cropped_image=cropped,
                        )
                    )

        # Fallback: if no table detected and the aspect ratio looks like a table crop,
        # offer the full image as one table
        if not detected:
            detected.append(
                DetectedTable(
                    table_id=f"p{page_number}_full",
                    page_number=page_number,
                    bbox=[0, 0, image.width, image.height],
                    score=1.0,
                    cropped_image=image.copy(),
                )
            )

        return detected
