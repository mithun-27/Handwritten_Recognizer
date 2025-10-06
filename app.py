# app.py
import os
import glob
import random
import json
from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
from model import SmallCNN
from utils import base64_to_pil, pil_to_input_tensor, save_user_sample

app = Flask(__name__, static_folder="static", static_url_path="/static")

MODEL_PATH = "model.pt"
CLASS_MAP = "class_map.json"
USER_FOLDER = "user_data"
REPLAY_SAMPLES = 40  # number of replay samples to include in online train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallCNN(num_classes=36).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with open(CLASS_MAP, "r") as f:
    idx_to_char = json.load(f)

softmax = nn.Softmax(dim=1)

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    img_b64 = data.get("image")
    pil = base64_to_pil(img_b64)
    tensor = pil_to_input_tensor(pil).to(device)  # shape 1x1x28x28
    model.eval()
    with torch.no_grad():
        out = model(tensor)
        probs = softmax(out).cpu().numpy()[0]
        topk_idx = out.argmax(dim=1).item()
    # prepare top3
    topk = sorted([(i, float(p)) for i,p in enumerate(probs)], key=lambda x:-x[1])[:3]
    return jsonify({
        "pred_idx": int(topk_idx),
        "pred_char": idx_to_char[str(topk_idx)],
        "topk": [{"idx": i, "char": idx_to_char[str(i)], "prob": p} for i,p in topk]
    })

def get_replay_samples(k=32):
    files = glob.glob(os.path.join(USER_FOLDER, "*.png"))
    random.shuffle(files)
    samples = []
    for f in files[:k]:
        try:
            # filename: {idx}_{label}.png
            label = int(os.path.basename(f).split("_")[1].split(".")[0])
        except:
            continue
        from PIL import Image
        pil = Image.open(f)
        t = pil_to_input_tensor(pil)  # 1x1x28x28
        samples.append((t, torch.tensor([label], dtype=torch.long)))
    return samples

@app.route("/train", methods=["POST"])
def train_on_sample():
    payload = request.json
    label_raw = payload.get("label")
    img_b64 = payload.get("image")
    if not label_raw or not img_b64:
        return jsonify({"ok": False, "error": "missing label or image"}), 400

    pil = base64_to_pil(img_b64)

    # map label to index
    label_idx = None
    if isinstance(label_raw, int):
        label_idx = label_raw
    else:
        s = str(label_raw).strip().lower()
        if s.isdigit():
            label_idx = int(s)
        elif len(s) == 1 and 'a' <= s <= 'z':
            label_idx = 10 + (ord(s) - ord('a'))
        else:
            return jsonify({"ok": False, "error": "invalid label"}), 400

    # save sample
    save_user_sample(pil, label_idx, folder=USER_FOLDER)

    # online fine-tune: one small Adam step with new sample + replay
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    X = []
    Y = []
    # new
    X.append(pil_to_input_tensor(pil))
    Y.append(torch.tensor([label_idx], dtype=torch.long))
    # replay
    for t, lbl in get_replay_samples(k=REPLAY_SAMPLES):
        X.append(t)
        Y.append(lbl)

    X = torch.cat(X, dim=0).to(device)
    Y = torch.cat(Y, dim=0).to(device)

    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, Y)
    loss.backward()
    optimizer.step()
    # save updated model
    torch.save(model.state_dict(), MODEL_PATH)
    return jsonify({"ok": True, "loss": float(loss.item())})

if __name__ == "__main__":
    os.makedirs(USER_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
