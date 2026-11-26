# Site Guardian: PPE Detection for Construction Sites

Real-time detection of personal protective equipment (hard hat, safety vest, mask) on construction-site workers.
The model is a **YOLOv8s** network fine-tuned with **Ultralytics / PyTorch**, exported to **ONNX**, and served by a
**Flask** web app that reads the camera **from the user's browser**, so it can run on a cloud host such as Render.

> This project is based on the open-source work of VoxDroid:
> [Construction-Site-Safety-PPE-Detection](https://github.com/VoxDroid/Construction-Site-Safety-PPE-Detection)
> and [ppe-guard](https://github.com/VoxDroid/ppe-guard). See [Credits and licences](#13-credits-and-licences).

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Key concepts](#2-key-concepts)
3. [Technology stack](#3-technology-stack)
4. [Dataset and classes](#4-dataset-and-classes)
5. [Training, step by step](#5-training-step-by-step)
6. [Results](#6-results)
7. [Exporting to ONNX](#7-exporting-to-onnx)
8. [How the web app works](#8-how-the-web-app-works)
9. [Project structure](#9-project-structure)
10. [Run locally](#10-run-locally)
11. [Deploy on Render](#11-deploy-on-render)
12. [Limitations and ideas](#12-limitations-and-ideas)
13. [Credits and licences](#13-credits-and-licences)

---

## 1. Project overview

**Problem.** Checking by eye that every worker wears a helmet and a vest, all day, across a whole site, is hard.

**Solution.** A camera feeds a detection model. The model finds workers and their equipment in each frame, and a small
rule engine decides which workers are not compliant. The web interface shows the live camera with boxes, the compliance
rate, and a violation log.

```
Browser camera ──► JPEG frame ──► Flask /api/detect ──► ONNX model ──► boxes + classes
                                                                          │
Browser draws boxes, gauge, log  ◄──────── JSON (workers, violations) ◄───┘
```

---

## 2. Key concepts

### 2.1 Object detection
Given an image, the model returns a list of objects. Each object has a **bounding box** (x1, y1, x2, y2), a **class**
(for example `Hardhat`) and a **confidence** between 0 and 1.

### 2.2 YOLO
YOLO means *You Only Look Once*: the network looks at the whole image in a single pass and predicts all boxes at once.
That makes it fast enough for video. We use **YOLOv8s** ("s" = small): about 11.1 million parameters, 129 layers.

### 2.3 COCO
**COCO** (Common Objects in Context) is a large public dataset from Microsoft with 80 everyday classes (person, car,
dog, chair, phone...). The pre-trained file `yolov8s.pt` was trained on COCO. COCO has **no hard hat and no safety vest**
class, so the pre-trained model only sees "person". It must be trained again for PPE.

### 2.4 Transfer learning (fine-tuning)
Instead of training from scratch (millions of images, days of compute), we start from the COCO weights, which already
know edges, shapes and people, and train further on 2,605 construction-site images. It is faster, needs less data, and
usually gives better accuracy. In our run, Ultralytics reported `Transferred 355/355 items from pretrained weights`.

---

## 3. Technology stack

| Library | Role in this project |
|---|---|
| **Ultralytics** | High-level YOLO API: load, train, validate, predict, export |
| **PyTorch** | Deep-learning engine underneath Ultralytics (training) |
| **ONNX / ONNX Runtime** | Portable model format and lightweight inference engine (deployment) |
| **OpenCV (cv2)** | Decode images, resize, letterbox, NMS in the web app |
| **NumPy** | Array maths: image tensors, box coordinates |
| **Pandas, Matplotlib, Seaborn** | Dataset statistics and training curves in the notebook |
| **Pillow** | Open and display images in the notebook |
| **PyYAML** | Write and read `data.yaml` |
| **Flask + Gunicorn** | Web server and API |
| **HTML / CSS / JavaScript** | Interface: camera access, drawing boxes, charts, log |

### 3.1 PyTorch in depth

**PyTorch** is an open-source deep-learning framework (created at Meta AI, now hosted by the PyTorch Foundation). It is the
engine that makes neural-network training practical. Its main ideas:

- **Tensors.** N-dimensional arrays, like NumPy arrays, that can live on a **GPU**. An image batch is a tensor of shape
  `(batch, 3, 640, 640)`.
- **Automatic differentiation (autograd).** Training means adjusting millions of numbers (weights) to reduce an error
  (the loss). PyTorch records every operation in the forward pass and then computes all gradients automatically in the
  backward pass. Nobody writes derivatives by hand.
- **`nn.Module`.** Layers (convolutions, batch-norm, activations) are Python classes that can be combined into a network.
  YOLOv8 is built this way.
- **Optimizers.** Algorithms such as SGD or AdamW that update the weights using the gradients. Ultralytics chose AdamW
  automatically in our run.
- **GPU acceleration (CUDA).** Heavy matrix maths runs on the graphics card, which is why training takes minutes on a
  Tesla T4 instead of days on a CPU.
- **Dynamic graphs.** The computation graph is built while the code runs, which makes debugging natural.

You normally do not write PyTorch code when using Ultralytics: `model.train(...)` runs a PyTorch training loop for you.
The `.pt` files (`yolov8s.pt`, `best.pt`) are PyTorch checkpoints.

### 3.2 Ultralytics

`ultralytics` is the official Python package for YOLO. A few lines replace what would otherwise be thousands of lines:

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")                 # load pre-trained weights
model.train(data="data.yaml", epochs=200)  # fine-tune
model.val()                                # evaluate (P, R, mAP)
model.predict("site.jpg")                  # detect
model.export(format="onnx")                # export for deployment
```

It also handles data loading, augmentation (flip, colour change, mosaic), loss functions (box, class, DFL), metrics and
plots such as the confusion matrix and `results.png`.

### 3.3 ONNX and ONNX Runtime: why export?

**ONNX** (Open Neural Network Exchange) is an open file format that describes a trained network as a graph of standard
operations. **ONNX Runtime** (from Microsoft) is an engine that executes that graph on CPU or GPU, with no PyTorch needed.

We train with PyTorch but **serve with ONNX Runtime** because:

1. **Small install.** PyTorch builds are large (hundreds of MB, often more than 1 GB with CUDA). ONNX Runtime for CPU is
   a small package. This matters on free hosting.
2. **Low memory.** Free Render instances have limited RAM (512 MB at the time of writing). PyTorch plus Ultralytics can
   exceed that; ONNX Runtime fits.
3. **Fast CPU inference.** ONNX Runtime applies graph optimisations (fusing layers, constant folding) that suit CPUs.
4. **No training code in production.** Inference needs only the graph and the weights: fewer dependencies, fewer
   things that can break, and a smaller attack surface.
5. **Portability.** The same `.onnx` file runs in Python, C++, C#, Java, mobile and the browser (ONNX Runtime Web).

Trade-off: the app must do the pre-processing and post-processing that Ultralytics normally does (letterbox resize,
confidence filter, NMS). This project implements them in `app.py`.

---

## 4. Dataset and classes

### 4.1 Dataset
**Construction Site Safety** dataset from Roboflow Universe (mirrored on Kaggle): **2,801 images** of 640x640 pixels.

| Split | Images | Used for |
|---|---|---|
| train | 2,605 | the model learns from these |
| valid | 114 | monitored after every epoch; the best weights (`best.pt`) are chosen with it |
| test | 82 | final, unbiased evaluation |

Labels are YOLO text files, one line per object: `class_id  x_center  y_center  width  height`, all normalised to 0..1.
`data.yaml` tells Ultralytics where the images are and lists the class names.

### 4.2 The 10 classes

The network is trained and evaluated on **exactly 10 classes**. Every prediction is one of them:

| ID | Class | Meaning |
|---|---|---|
| 0 | Hardhat | helmet worn |
| 1 | Mask | face mask worn |
| 2 | NO-Hardhat | head without a helmet |
| 3 | NO-Mask | face without a mask |
| 4 | NO-Safety Vest | worker without a vest |
| 5 | Person | a worker or person |
| 6 | Safety Cone | traffic cone |
| 7 | Safety Vest | high-visibility vest worn |
| 8 | machinery | heavy equipment |
| 9 | vehicle | construction vehicle |

The order must match `names` in `data.yaml` and `CLASSES` in `app.py`.

### 4.3 Logical groups (for explanation only)

The model does not know about groups. They only help to understand and to write the compliance rules:

| Group | Classes |
|---|---|
| Worn equipment | Hardhat, Mask, Safety Vest |
| Missing equipment (violation signals) | NO-Hardhat, NO-Mask, NO-Safety Vest |
| People | Person |
| Site objects | Safety Cone, machinery, vehicle |

(The original notebook merges "People" and "Site objects" into one "scene context" group, giving three groups.)

---

## 5. Training, step by step

### 5.1 Where and with what
Training ran in **Google Colab** on a **Tesla T4** GPU, using the notebook `notebooks/ppe_detection_colab.ipynb`
(an adapted copy of the original training notebook with Colab paths).

### 5.2 Which model
**YOLOv8s**, fine-tuned from `yolov8s.pt` (COCO). 11,139,470 parameters, 28.7 GFLOPs, input 640x640.
Output of the best weights: `best.pt` (PyTorch), then `best.onnx`.

### 5.3 Steps

1. **Enable GPU.** Runtime > Change runtime type > T4 GPU. Check with `!nvidia-smi`.
2. **Get the code and data.** `git clone` the repository (the dataset is inside it) and `pip install ultralytics`.
3. **Configure.** The `CFG` class holds epochs, batch size, learning rate, paths and the class list.
   Set `DEBUG = True` for a 3-epoch smoke test; set it to `False` for a real run.
4. **Create `data.yaml`** with absolute paths to the train, valid and test images and the 10 class names.
5. **Explore the data.** Show random images, count objects per class, and confirm all images are 640x640.
6. **Baseline.** Run the original `yolov8s.pt` on one image: it detects only "person", which shows why training is needed.
7. **Train.** `model.train(...)`. Each epoch passes over all training images once, then validates.
8. **Evaluate** `best.pt` on the test split with `model.val(split="test")`.
9. **Export** to ONNX and copy `best.pt` / `best.onnx` to Google Drive.

### 5.4 Training configuration (our 50-epoch run)

| Setting | Value |
|---|---|
| Ultralytics / Python / PyTorch | 8.4.218 / 3.13 / 2.14.1 (CUDA 13) |
| GPU | Tesla T4 |
| Epochs | 50 (the original project used 200) |
| Image size / batch | 640 / 16 |
| Optimizer | `auto`, which selected AdamW (lr 0.000714) |
| Early stopping patience | 25 epochs (not triggered) |
| Regularisation | dropout 0.025, weight decay 0.0005 |
| Augmentation | default: flip, HSV colour, scale, mosaic (switched off for the last 10 epochs) |
| Seed | 88 |
| Training time | 0.572 h (about 34 minutes) |

### 5.5 Reading the training output
Three losses are shown per epoch, and all should go down:

- `box_loss`: how well the box position and size match the truth.
- `cls_loss`: how well the class is predicted.
- `dfl_loss`: distribution focal loss, which sharpens the box edges.

After each epoch, validation prints `P`, `R`, `mAP50` and `mAP50-95` (see below). `best.pt` is the epoch with the best
fitness (a weighted mix of mAP50 and mAP50-95).

---

## 6. Results

### 6.1 Metrics explained

- **Precision (P):** of the objects the model reported, how many were correct.
- **Recall (R):** of the objects that really exist, how many the model found.
- **IoU:** overlap between a predicted box and the true box (intersection over union).
- **mAP50:** mean average precision when a box counts as correct at IoU >= 0.5.
- **mAP50-95:** the same, averaged over IoU 0.5 to 0.95: a stricter test of box accuracy.

The **test** split is the honest number: the model never trained on it and `best.pt` was not chosen with it.

### 6.2 Our run: 50 epochs (Colab, Tesla T4)

| Split | Images | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Validation | 114 | 0.901 | 0.794 | 0.845 | 0.544 |
| **Test** | 82 | **0.892** | **0.733** | **0.778** | **0.493** |

Speed on T4: 2.5 ms pre-processing, 11.7 ms inference, 3.6 ms post-processing per image.

Per-class results on the **test** split:

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Hardhat | 110 | 0.978 | 0.891 | 0.940 | 0.619 |
| Mask | 28 | 0.928 | 0.750 | 0.745 | 0.540 |
| NO-Hardhat | 41 | 0.812 | 0.561 | 0.532 | 0.308 |
| NO-Mask | 79 | 0.923 | 0.758 | 0.801 | 0.386 |
| NO-Safety Vest | 90 | 0.921 | 0.778 | 0.817 | 0.512 |
| Person | 174 | 0.897 | 0.810 | 0.888 | 0.583 |
| Safety Cone | 92 | 0.790 | 0.451 | 0.472 | 0.207 |
| Safety Vest | 61 | 0.904 | 0.885 | 0.917 | 0.634 |
| machinery | 44 | 0.836 | 0.841 | 0.886 | 0.668 |
| vehicle | 41 | 0.934 | 0.610 | 0.784 | 0.479 |

### 6.3 Original project: 200 epochs

Reported by the original author (200 epochs, RTX 5060 Ti, 1.925 hours):

| Split | Source | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Validation (114 images) | training log of the original notebook (`yolov8s`, 200 epochs) | 0.952 | 0.798 | 0.877 | 0.624 |
| Test (82 images) | `ppe-guard` README, model `ppe-yolov8s (v1)` | 0.923 | 0.782 | 0.814 | 0.556 |
| Test (82 images) | `ppe-guard` README, model `ppe-yolo26s` (newer architecture) | 0.940 | 0.765 | 0.822 | 0.574 |

The test rows are taken from the [ppe-guard README](https://github.com/VoxDroid/ppe-guard#how-accurate-is-it), which states they
were measured on the 82 held-out test images. The original repository also contains an 80-epoch run, which is not compared here.

### 6.4 Side by side (test split)

| Run | Epochs | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Ours (YOLOv8s) | 50 | 0.892 | 0.733 | 0.778 | 0.493 |
| Original `ppe-yolov8s (v1)` | 200 | 0.923 | 0.782 | 0.814 | 0.556 |
| Original `ppe-yolo26s` | 200 | 0.940 | 0.765 | 0.822 | 0.574 |
| **Ours (YOLOv8s)** | **200** | _fill in after retraining_ | _..._ | _..._ | _..._ |

**Reading the numbers**

- With one quarter of the epochs, our YOLOv8s reaches mAP50 0.778 against 0.814 for the 200-epoch run (about 3.6 points lower)
  and mAP50-95 0.493 against 0.556. Validation shows the same direction (0.845 vs 0.877 mAP50).
- Our training was **not finished**: the losses were still falling at epoch 50 and early stopping never fired, so more
  epochs are expected to help.
- The weakest classes are `NO-Hardhat` (recall 0.56) and `Safety Cone` (recall 0.45). Both have few examples
  (41 and 92 instances in only 8 images for cones), and a bare head or a small cone is visually hard.
- The validation and test sets are small (114 and 82 images), so differences of a few points can be noise.

---

## 7. Exporting to ONNX

In the notebook (cell "Export"):

```python
best_model = YOLO("runs/detect/<run>/weights/best.pt")
best_model.export(format="onnx", imgsz=640, half=False, int8=False, simplify=False, nms=False)
```

- `nms=False`: the network returns raw candidates; the app applies NMS itself.
- Expected output shape: **(1, 14, 8400)**. The log line looks like `output shape(s) (1, 14, 8400)`.
  - `14` = 4 box values (`cx, cy, w, h`) + 10 class scores.
  - `8400` = 80x80 + 40x40 + 20x20 candidate cells (small, medium and large objects).
- Recent Ultralytics versions export a recent ONNX opset, so `requirements.txt` asks for `onnxruntime>=1.22.0`
  (older versions fail with an "unsupported IR version" error).

---

## 8. How the web app works

### 8.1 Architecture

| Part | File | Job |
|---|---|---|
| Interface | `static/index.html` | camera, drawing, charts, settings, log (plain HTML/CSS/JS, no build step) |
| API | `app.py` (`/api/detect`, `/health`) | receives a frame, runs the model, applies the rules |
| Model | `models/best.onnx` | the exported network |

### 8.2 The camera

The server in the cloud has no camera, so the **browser** captures it:

1. `navigator.mediaDevices.getUserMedia` opens the user's camera (requires HTTPS, which Render provides).
2. Every 0.3 to 1.5 s (setting "scan speed") the page draws the current video frame to a canvas, scales it to at most 640 px,
   and encodes it as JPEG.
3. The frame is sent to `POST /api/detect` with the sensitivity and the list of required equipment.
4. The server answers with JSON; the page draws boxes over the video and updates the numbers.

A second tab accepts an uploaded photo (drag and drop) and runs the same pipeline once.

### 8.3 Inference pipeline (inside `app.py`)

1. **Letterbox:** resize to 640x640 keeping the aspect ratio, padding with grey.
2. **Normalise:** BGR to RGB, divide by 255, reorder to `(1, 3, 640, 640)`.
3. **Run** the ONNX model: output `(1, 14, 8400)`.
4. **Decode:** for each of the 8400 candidates, the best class score is its confidence; keep those above the threshold.
5. **NMS** (non-maximum suppression, class-aware): one object produces many overlapping boxes; keep the best, drop the rest.
6. **Map back** to the original image coordinates (remove the padding, undo the scale).

### 8.4 How a worker is judged

The model gives boxes only. Compliance comes from a rule on top (`analyse()` in `app.py`):

- Every `Person` box is a worker.
- A `NO-Hardhat`, `NO-Safety Vest` or `NO-Mask` box whose **centre falls inside a person box** (with 5 % tolerance)
  is a violation for that worker. If several people overlap, the smallest containing box wins.
- A `NO-*` box with no person around it still counts, grouped by horizontal position so one worker is not counted twice.
- The user chooses which equipment is required (hard hat and vest by default, mask optional).
- **Compliance rate** = compliant workers / all workers.

Known weakness: `NO-Hardhat` has the lowest recall of the violation classes, so some missing helmets are not flagged.
Lowering the sensitivity slider (for example to 30-35 %) finds more, at the cost of more false alarms.

### 8.5 API

`POST /api/detect` (multipart form): `image` (JPEG/PNG), `conf` (0.1 to 0.95), `required` (comma list of `hardhat,vest,mask`).

```json
{
  "width": 640, "height": 480, "ms": 59,
  "detections": [{"name": "Person", "conf": 0.9, "box": [100, 100, 300, 450]}],
  "workers": [{"box": [100, 100, 300, 450], "violations": ["hardhat"], "ok": false}],
  "summary": {"workers": 1, "violating": 1, "compliant": 0, "rate": 0, "equipment": {"hardhat": {"required": true, "violations": 1, "ok": 0}}}
}
```

`GET /health` reports whether the model file was found and loaded.

### 8.6 Interface features

Live camera and image upload, green boxes for compliant workers and dashed red boxes for violations, compliance gauge,
per-equipment status tiles, settings (required equipment, sensitivity, scan speed, optional alert sound), a violation
log with CSV export, and snapshot / result download.

---

## 9. Project structure

```
render-app/
  app.py                      Flask server, ONNX inference, compliance rules
  static/index.html           user interface
  models/best.onnx            put the exported model here
  notebooks/ppe_detection_colab.ipynb   training notebook (Colab)
  requirements.txt            Python dependencies
  .python-version             Python version for Render
  README.md
```

---

## 10. Run locally

```bash
pip install -r requirements.txt
# put best.onnx in models/
python app.py            # open http://localhost:5000
```

Browsers allow camera access on `http://localhost` and on HTTPS pages.

---

## 11. Deploy on Render

1. Put `render-app/` (with `models/best.onnx`, about 43 MB) in a GitHub repository.
2. Render: **New + > Web Service**, select the repository.
3. **Root Directory:** `render-app` (leave empty if this folder is the repository root).
4. **Build command:** `pip install -r requirements.txt`
5. **Start command:** `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`
6. **Instance type:** Free, then **Create Web Service**.
7. Open the URL. The "model ready" badge at the top of the page should be green.



---

