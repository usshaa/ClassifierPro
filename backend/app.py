from flask import Flask, request, jsonify
import json
from backend.helper_fun import ClipZeroShotEngine

app = Flask(__name__)

eng = ClipZeroShotEngine(model_dir="onnx_clip_model")

@app.route("/",methods=["GET"])
def index():
    return "hi"

@app.route("/classify",methods=["POST"])
def classify_pro():
    img = request.files["image"]
    cat = request.form.get("categories")

    if cat.strip().startswith("["):
        candidate_labels = json.loads(cat)
    else:
        candidate_labels = [c.strip() for c in cat.split(",") if c.strip()]

    img_bytes = img.read()
    prediction = eng.classify(img_bytes,candidate_labels)
    return jsonify({
        "sucess":True,
        "filename":img.filename,
        "result":prediction
    })

if __name__ == "__main__":
    app.run(host="localhost",port=5000, debug=False)