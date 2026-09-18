from optimum.onnxruntime import ORTModelForZeroShotImageClassification
from transformers import AutoProcessor

model_id = "openai/clip-vit-base-patch32"
save_dir = "onnx_clip_model"

model = ORTModelForZeroShotImageClassification.from_pretrained(
    model_id,
    export=True
)

processor = AutoProcessor.from_pretrained(model_id)

model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)
print("Saved sucessfully model")