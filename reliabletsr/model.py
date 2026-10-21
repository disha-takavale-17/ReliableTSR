"""Pinned Microsoft Table Transformer loading and object detection."""
from pathlib import Path

from . import MODEL_ID, MODEL_REVISION


class TATR:
    def __init__(self, local_model=None, cache_dir=None, device="cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.device = device
        source = str(Path(local_model).resolve()) if local_model else MODEL_ID
        kwargs = {} if local_model else {"revision": MODEL_REVISION,
                                         "cache_dir": cache_dir or "data/model-cache"}
        self.processor = AutoImageProcessor.from_pretrained(source, use_fast=False, **kwargs)
        self.model = AutoModelForObjectDetection.from_pretrained(source, **kwargs)
        self.model.to(device).eval()
        self.torch = torch
        self.source = source
        self.revision = "local-unverified" if local_model else MODEL_REVISION

    def detect(self, image, raw_threshold=0.05):
        image = image.convert("RGB")
        # The pinned checkpoint stores only longest_edge=800; Transformers 4.48
        # requires both keys. Equal limits preserve aspect ratio and cap the
        # longest image side at the checkpoint's configured 800 pixels.
        inputs = self.processor(images=image, size={"shortest_edge": 800, "longest_edge": 800},
                                return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = self.torch.tensor([[image.height, image.width]], device=self.device)
        result = self.processor.post_process_object_detection(outputs, threshold=raw_threshold,
                                                               target_sizes=target_sizes)[0]
        names = self.model.config.id2label
        return [{"id": f"d{i}", "label": names[int(label)],
                 "score": round(float(score), 6),
                 "bbox": [round(float(v), 3) for v in box.tolist()]}
                for i, (label, score, box) in enumerate(zip(result["labels"], result["scores"],
                                                              result["boxes"]))]
