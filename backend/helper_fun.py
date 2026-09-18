import io
from typing import List, Dict, Any
import numpy as np
from PIL import Image
from optimum.onnxruntime import ORTModelForZeroShotImageClassification
from transformers import AutoProcessor

class ClipZeroShotEngine:
    def __init__(self, model_dir="onnx_clip_model"):
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = ORTModelForZeroShotImageClassification.from_pretrained(
            model_dir,
            provider="CPUExecutionProvider"
        )

    def preprocess_image(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image

    def classify(self, image_bytes, candidate_labels):
        image = self.preprocess_image(image_bytes)

        inputs = self.processor(
            text=candidate_labels,
            images=image,
            return_tensors = "np",
            padding=True
        )

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image[0]

        exp_logits = np.exp(logits_per_image-np.max(logits_per_image))
        probabilities = exp_logits/np.sum(exp_logits)

        scores = [
            {"label":label, "confidence": round(float(prob),4)} 
            for  label, prob in zip(candidate_labels, probabilities)
        ]

        scores.sort(key=lambda x:x["confidence"], reverse=True)

        return {
            "predicted_category":scores[0]["label"],
            "top_confidence":scores[0]["confidence"],
            "all_predictions":scores
        }

if __name__ == "__main__":
    import sys
    eng = ClipZeroShotEngine()
    img = Image.open("images/bluecolor.png")
    buf = io.BytesIO()
    img.save(buf,format="PNG")
    test = buf.getvalue()

    r = eng.classify(test,["red","blue","green"])
    print(r)
