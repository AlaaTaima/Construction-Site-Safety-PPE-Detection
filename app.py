"""
Site Guardian: PPE detection web app (Flask + ONNX Runtime).

Runs a YOLOv8 model exported to ONNX (best.onnx). The browser captures
camera frames and posts them to /api/detect, so no server camera is needed
(which is what makes it deployable on Render).
"""
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "models", "best.onnx"))
IMG_SIZE = 640
MAX_SIDE = 1280  # larger uploads are shrunk to this before inference

CLASSES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
]

# equipment key -> (class when worn, class when missing)
EQUIPMENT = {
    "hardhat": ("Hardhat", "NO-Hardhat"),
    "vest": ("Safety Vest", "NO-Safety Vest"),
    "mask": ("Mask", "NO-Mask"),
}
MISSING_TO_KEY = {neg: key for key, (_, neg) in EQUIPMENT.items()}

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

_session = None
_session_error = None


def get_session():
    """Load the ONNX model once, on first use."""
    global _session, _session_error
    if _session is not None:
        return _session
    if not os.path.exists(MODEL_PATH):
        _session_error = f"Model file not found: {MODEL_PATH}"
        return None
    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
        _session = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
        _session_error = None
    except Exception as exc:  # noqa: BLE001
        _session_error = str(exc)
    return _session


def letterbox(img, size=IMG_SIZE):
    """Resize keeping aspect ratio and pad to a square (same as YOLO)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def run_model(img, conf_thr, iou_thr=0.5):
    """Return a list of detections in the coordinates of `img`."""
    sess = get_session()
    h0, w0 = img.shape[:2]
    boxed, r, left, top = letterbox(img)
    blob = boxed[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])

    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    pred = np.squeeze(out, 0)
    if pred.shape[0] < pred.shape[1]:  # (4+nc, N) -> (N, 4+nc)
        pred = pred.T

    scores_all = pred[:, 4:]
    cls_ids = scores_all.argmax(1)
    confs = scores_all.max(1)
    keep = confs >= conf_thr
    if not keep.any():
        return []

    pred, cls_ids, confs = pred[keep], cls_ids[keep], confs[keep]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = np.clip((cx - bw / 2 - left) / r, 0, w0)
    y1 = np.clip((cy - bh / 2 - top) / r, 0, h0)
    x2 = np.clip((cx + bw / 2 - left) / r, 0, w0)
    y2 = np.clip((cy + bh / 2 - top) / r, 0, h0)

    # class-aware NMS: shift each class far apart so classes never suppress each other
    offset = cls_ids.astype(np.float32) * 10000.0
    nms_boxes = np.stack([x1 + offset, y1, x2 - x1, y2 - y1], 1).tolist()
    idxs = cv2.dnn.NMSBoxes(nms_boxes, confs.tolist(), float(conf_thr), float(iou_thr))
    idxs = np.array(idxs).flatten().astype(int) if len(idxs) else []

    dets = []
    for i in idxs:
        c = int(cls_ids[i])
        dets.append({
            "cls": c,
            "name": CLASSES[c] if c < len(CLASSES) else str(c),
            "conf": round(float(confs[i]), 3),
            "box": [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
        })
    return dets


def _center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _contains(outer, point, pad=0.05):
    pw, ph = (outer[2] - outer[0]) * pad, (outer[3] - outer[1]) * pad
    return (outer[0] - pw <= point[0] <= outer[2] + pw) and (outer[1] - ph <= point[1] <= outer[3] + ph)


def _area(box):
    return max(1, (box[2] - box[0]) * (box[3] - box[1]))


def analyse(dets, required):
    """
    Turn raw detections into per-worker compliance.

    A worker is a detected Person. A violation is a NO-* box (e.g. NO-Hardhat)
    whose center falls inside that person. NO-* boxes with no person around
    them still count, grouped by horizontal position so one worker is not
    counted twice.
    """
    workers = [
        {"box": d["box"], "violations": [], "source": "person"}
        for d in dets if d["name"] == "Person"
    ]
    orphans = []

    for d in dets:
        key = MISSING_TO_KEY.get(d["name"])
        if key is None or key not in required:
            continue
        c = _center(d["box"])
        hosts = [w for w in workers if w["source"] == "person" and _contains(w["box"], c)]
        if hosts:
            host = min(hosts, key=lambda w: _area(w["box"]))
            if key not in host["violations"]:
                host["violations"].append(key)
            continue

        merged = False
        for o in orphans:
            ow = o["box"][2] - o["box"][0]
            dw = d["box"][2] - d["box"][0]
            if abs(_center(o["box"])[0] - c[0]) < max(ow, dw) * 1.2:
                if key not in o["violations"]:
                    o["violations"].append(key)
                o["box"] = [min(o["box"][0], d["box"][0]), min(o["box"][1], d["box"][1]),
                            max(o["box"][2], d["box"][2]), max(o["box"][3], d["box"][3])]
                merged = True
                break
        if not merged:
            orphans.append({"box": list(d["box"]), "violations": [key], "source": "orphan"})

    workers.extend(orphans)
    for w in workers:
        w["ok"] = len(w["violations"]) == 0

    total = len(workers)
    violating = sum(1 for w in workers if not w["ok"])
    equipment = {}
    for key in EQUIPMENT:
        if key in required:
            bad = sum(1 for w in workers if key in w["violations"])
            equipment[key] = {"required": True, "violations": bad, "ok": total - bad}
        else:
            equipment[key] = {"required": False, "violations": 0, "ok": 0}

    return workers, {
        "workers": total,
        "violating": violating,
        "compliant": total - violating,
        "rate": round(100 * (total - violating) / total) if total else None,
        "equipment": equipment,
    }


@app.get("/")
def index():
    return send_from_directory(os.path.join(BASE_DIR, "static"), "index.html")


@app.get("/health")
def health():
    ready = get_session() is not None
    return jsonify(status="ok" if ready else "model_missing", model_ready=ready, detail=_session_error)


@app.post("/api/detect")
def api_detect():
    if get_session() is None:
        return jsonify(error="النموذج غير جاهز على الخادم. تأكد من وجود models/best.onnx."), 503

    file = request.files.get("image")
    if file is None:
        return jsonify(error="لم يتم إرسال صورة."), 400
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="تعذر قراءة الصورة. استخدم JPG أو PNG."), 400

    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    try:
        conf = min(0.95, max(0.1, float(request.form.get("conf", 0.45))))
    except ValueError:
        conf = 0.45
    raw = request.form.get("required")
    required = {"hardhat", "vest"} if raw is None else {k for k in raw.split(",") if k in EQUIPMENT}

    t0 = time.perf_counter()
    dets = run_model(img, conf)
    workers, summary = analyse(dets, required)
    ms = round((time.perf_counter() - t0) * 1000)

    return jsonify(width=w, height=h, detections=dets, workers=workers, summary=summary, ms=ms)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
