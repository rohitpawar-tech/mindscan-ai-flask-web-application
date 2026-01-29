from flask import Flask, request, jsonify, render_template
from torchvision import models, transforms
from PIL import Image
import torch
import os
import base64
import random

# ------------------ Setup ------------------
app = Flask(__name__, template_folder="templates")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Model ------------------
image_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
image_model.eval()

# ------------------ Helper Function ------------------
def extract_image_features(image_paths):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    feats = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = image_model(tensor)
            feats.append(out.cpu().numpy().tolist())
        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")
            continue
    return feats

# ------------------ Routes ------------------
@app.route("/")
def home():
    return render_template("image_test.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        images_data = data.get("images", [])
        image_paths = []

        print(f"[DEBUG] Received {len(images_data)} images")

        for idx, img_data in enumerate(images_data):
            try:
                if "," in img_data:
                    img_data = img_data.split(",")[1]
                img_bytes = base64.b64decode(img_data)
                save_path = os.path.join(UPLOAD_DIR, f"img_{random.randint(1000,9999)}_{idx}.png")
                with open(save_path, "wb") as f:
                    f.write(img_bytes)
                print(f"[OK] Saved: {save_path}, {os.path.getsize(save_path)} bytes")
                image_paths.append(save_path)
            except Exception as e:
                print(f"[ERROR] Image save failed: {e}")

        if not image_paths:
            return jsonify({"success": False, "error": "No valid images received"})

        feats = extract_image_features(image_paths)
        print(f"[DEBUG] Extracted {len(feats)} feature sets")

        # ------------------ Dummy Prediction ------------------
        status = random.choice(["Healthy", "At-Risk", "Neutral", "Critical"])
        confidence = round(random.uniform(0.5, 0.99), 3)
        emotions = {
            "Happiness": random.randint(10, 90),
            "Sadness": random.randint(5, 70),
            "Anxiety": random.randint(5, 60),
            "Anger": random.randint(0, 40)
        }

        messages = {
            "Healthy": "The person appears emotionally stable and positive. Facial features reflect calmness and openness, indicating good mental well-being.",
            "Neutral": "The emotional tone seems balanced but slightly tense. Minor signs of stress or fatigue may be visible.",
            "At-Risk": "The person appears to be under emotional strain. Facial cues show sadness or anxiety, suggesting possible stress or burnout.",
            "Critical": "Signs of emotional fatigue and distress detected. The expression suggests deep concern or sadness. Emotional support or counseling is advised."
        }

        analysis_text = (
            f"🧠 AI Emotional Insight\n\n"
            f"Status: {status}\n"
            f"Confidence: {confidence}\n\n"
            f"Detected Emotional Levels:\n"
            f" - Happiness: {emotions['Happiness']}%\n"
            f" - Sadness: {emotions['Sadness']}%\n"
            f" - Anxiety: {emotions['Anxiety']}%\n"
            f" - Anger: {emotions['Anger']}%\n\n"
            f"📊 Summary: {messages.get(status)}"
        )

        return jsonify({
            "success": True,
            "status": status,
            "confidence": confidence,
            "emotions": emotions,
            "analysis_text": analysis_text
        })

    except Exception as e:
        print("[FATAL ERROR]", e)
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
