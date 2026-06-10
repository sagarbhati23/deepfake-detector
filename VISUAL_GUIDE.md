# 🎭 Deepfake Detector - Visual Guide & Diagrams

## **1. COMPLETE SYSTEM FLOW**

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER INTERACTION                            │
│  Opens http://localhost:7860 in browser                             │
│  or runs: python classify.py image.jpg                              │
│  or runs: python inference/video_inference.py                       │
└────────────────────┬────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        FILE INPUT LAYER                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐ │
│  │  Image Files     │  │  Video Files     │  │  Error Handling   │ │
│  │  .jpg, .png      │  │  .mp4, .mov      │  │  Unsupported →    │ │
│  │                  │  │                  │  │  Error Message    │ │
│  └────────┬─────────┘  └────────┬─────────┘  └───────────────────┘ │
│           │                     │                                    │
│           ▼                     ▼                                    │
│  ┌──────────────────┐  ┌──────────────────┐                        │
│  │ PIL.Image.open() │  │ cv2.VideoCapture │                        │
│  │ .convert("RGB")  │  │ extract frames   │                        │
│  └────────┬─────────┘  └────────┬─────────┘                        │
└──────────────────────────────────────────────────────────────────────┘
           │                      │
           ├──────────┬───────────┤
           ▼          ▼           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                               │
│  For IMAGE:               For VIDEO (10 frames):                    │
│  ┌──────────────────┐     ┌──────────────────────┐                 │
│  │ [H, W, 3]        │     │ [10, H, W, 3]        │                 │
│  │ → Resize         │     │ → For each frame:    │                 │
│  │ → [224, 224, 3]  │     │   → Resize [224,224] │                 │
│  │ → ToTensor       │     │   → ToTensor         │                 │
│  │ → [3, 224, 224]  │     │   → Normalize        │                 │
│  │ → Normalize      │     │ → Stack tensors      │                 │
│  │ → [3, 224, 224]  │     │                      │                 │
│  │   (normalized)   │     │                      │                 │
│  └────────┬─────────┘     └─────────┬────────────┘                 │
│           │                         │                               │
│           └─────────────┬───────────┘                               │
│                         ▼                                           │
│              ┌─────────────────────┐                                │
│              │  unsqueeze(0)       │                                │
│              │  Add batch dimension│                                │
│              │  [1, 3, 224, 224]   │                                │
│              │  (for image)        │                                │
│              │  [10, 3, 224, 224]  │                                │
│              │  (for video)        │                                │
│              └────────┬────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    NEURAL NETWORK LAYER                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Model: efficientnet_b0() [ImageNet pretrained]            │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │  Input: [B, 3, 224, 224]                            │   │   │
│  │  ├──────────────────────────────────────────────────────┤   │   │
│  │  │  Layer 1: Conv Stem                                 │   │   │
│  │  │    3 channels → 32 filters → [B, 32, 112, 112]     │   │   │
│  │  │                                                      │   │   │
│  │  │  Layer 2-8: MBConv Blocks (Mobile Inverted Conv)   │   │   │
│  │  │    • Depthwise separable convolutions               │   │   │
│  │  │    • Skip connections                               │   │   │
│  │  │    • Progressive channel expansion                  │   │   │
│  │  │    → [B, 1280, 7, 7] spatial features              │   │   │
│  │  │                                                      │   │   │
│  │  │  Layer 9: Adaptive Average Pooling                 │   │   │
│  │  │    [B, 1280, 7, 7] → [B, 1280] feature vector     │   │   │
│  │  ├──────────────────────────────────────────────────────┤   │   │
│  │  │  Output: [B, 1280] (1280-dimensional features)     │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └────────────┬──────────────────────────────────────────────┘   │
│               ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Custom Classification Head                                 │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │  Input: [B, 1280]                                   │   │   │
│  │  │  → Dropout(p=0.4)  [B, 1280]                        │   │   │
│  │  │    (Randomly zeros 40% of values during training)   │   │   │
│  │  │  → Linear(1280 → 2)  [B, 2]                         │   │   │
│  │  │    (Map to 2 classes: Real, Fake)                   │   │   │
│  │  │  → Output: [B, 2] raw logits                        │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └────────────┬──────────────────────────────────────────────┘   │
│               ▼                                                   │
│        [B, 2] = [[logit_real, logit_fake], ...]               │
└─────────────────────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   POST-PROCESSING LAYER                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  torch.softmax(logits, dim=1)                              │   │
│  │  Converts logits to probabilities [0, 1]                  │   │
│  │                                                             │   │
│  │  Example:                                                   │   │
│  │  Input logits:  [2.1, -1.5]                               │   │
│  │  After softmax: [0.89, 0.11]                              │   │
│  │  (89% Real, 11% Fake)                                     │   │
│  └────────────┬──────────────────────────────────────────────┘   │
│               ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  torch.max(probabilities)                                  │   │
│  │  Returns: (confidence_score, class_index)                 │   │
│  │                                                             │   │
│  │  Example: (0.89, 0)  ← 89% confidence, class 0 (Real)    │   │
│  └────────────┬──────────────────────────────────────────────┘   │
│               ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  For VIDEO only: Average across 10 frames                  │   │
│  │  torch.mean(all_frame_predictions)                         │   │
│  │  Reduces false positives/negatives                         │   │
│  └────────────┬──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐   │
│  │ class_index == 0 │  │ class_index == 1 │  │ confidence*100  │   │
│  │ → 🟢 Real        │  │ → 🔴 Deepfake    │  │ → "89%"         │   │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬────────┘   │
│           │                     │                     │             │
│           └─────────────┬───────────────────────────┘             │
│                         ▼                                          │
│         ┌────────────────────────────┐                            │
│         │  Display to user           │                            │
│         │  🟢 Real - 89%             │                            │
│         │  (+ image preview)         │                            │
│         └────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    USER OUTPUT                                       │
│  Web UI: Gradio interface shows result                              │
│  CLI: Terminal prints result                                        │
│  Batch: File with all results                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## **2. COMPONENT INTERACTION DIAGRAM**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE                                  │
│                                                                       │
│  config.yaml ───────────────────┐                                   │
│  (hyperparameters)              │                                   │
│                                 ▼                                   │
│  Dataset folder ──→ HybridDeepfakeDataset ──→ DataLoader           │
│  (real/fake)       (loads images)              (batch_size=4)      │
│                                                   │                 │
│                                                   ▼                 │
│                             DeepfakeDetector (PyTorch Lightning)    │
│                             ┌──────────────────────────────────┐    │
│                             │ EfficientNet-B0 + Classifier    │    │
│                             │ Loss: CrossEntropyLoss           │    │
│                             │ Optimizer: Adam(lr=0.0001)       │    │
│                             └────────────┬─────────────────────┘    │
│                                          │                          │
│                    ┌─────────────────────┼─────────────────────┐   │
│                    ▼                     ▼                     ▼    │
│              ModelCheckpoint        EarlyStopping         Logger   │
│              (saves best)           (stops if stuck)    (TensorBoard)
│                    │                     │                     │   │
│                    └─────────────────────┼─────────────────────┘   │
│                                          ▼                         │
│                                  pl.Trainer.fit()                  │
│                                  (orchestrates everything)         │
│                                          │                         │
│                                          ▼                         │
│                              models/best_model.ckpt                │
│                              (trained weights)                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PHASE                                 │
│                                                                       │
│  models/best_model.ckpt ──────────────────┐                        │
│  (saved weights)                          │                        │
│                                           ▼                        │
│                                      Load Model                    │
│                                      (eval mode)                   │
│                                           │                        │
│   ┌────────────┬────────────┐  ┌─────────┼──────────┐             │
│   ▼            ▼            ▼  ▼         ▼          ▼             │
│  app.py    classify.py  video_  realeval.py  ONNX   REST API       │
│  (Web UI)  (CLI tool)   inference (testing)  export  (future)      │
│   │            │         │        │           │       │            │
│   └────────────┼─────────┼────────┼───────────┴───────┘            │
│                │         │        │                                │
│                └─────────┼────────┴─ All use loaded model         │
│                          ▼                                         │
│                   Forward pass through EfficientNet                │
│                          │                                         │
│                          ▼                                         │
│                   Get predictions                                  │
│                          │                                         │
│                          ▼                                         │
│                   Return to user                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## **3. DATA STRUCTURES**

### **Training Data Structure**
```
Dataset Folder:
├── train/
│   ├── real/
│   │   ├── real_image_001.jpg
│   │   ├── real_image_002.png
│   │   └── ... (hundreds of real images)
│   └── fake/
│       ├── deepfake_001.jpg
│       ├── deepfake_002.png
│       └── ... (hundreds of fake images)
└── val/
    ├── real/
    │   └── ...
    └── fake/
        └── ...

After loading:
image_paths = [
    "C:/Datasets/train/real/real_image_001.jpg",
    "C:/Datasets/train/real/real_image_002.png",
    "C:/Datasets/train/fake/deepfake_001.jpg",
    ...
]
labels = [0, 0, 1, ...]  # 0=real, 1=fake
```

### **Batch Structure (During Training)**
```
One batch from DataLoader:
images: torch.Tensor([4, 3, 224, 224])
        │   │ ▲      │ ▲              ▲ ▲
        │   │ │      │ │              │ └─ Height (pixels)
        │   │ │      │ │              └─── Width (pixels)
        │   │ │      │ └────────────────── RGB channels (3)
        │   │ │      └──────────────────── Images in batch (4)
        │   └─────────────────────────────── Tensor data type

labels: torch.Tensor([4])
        Value: [0, 1, 1, 0]  # Which class each image is
```

### **Model Output Structure**
```
Forward pass output:
logits: torch.Tensor([4, 2])
        │   │   ▲ ▲
        │   │   │ └─ Number of classes (2: real, fake)
        │   │   └─── Batch size (4 images)
        │   └──────── Tensor type
        
Values: [[2.1, -1.5],      # Image 1: More likely real (2.1 > -1.5)
         [-0.8, 3.2],      # Image 2: More likely fake (3.2 > -0.8)
         [1.5, 0.2],       # Image 3: More likely real
         [-2.1, 1.8]]      # Image 4: More likely fake

After softmax:
probs: [[0.89, 0.11],      # 89% real, 11% fake
        [0.04, 0.96],      # 4% real, 96% fake
        [0.81, 0.19],      # 81% real, 19% fake
        [0.10, 0.90]]      # 10% real, 90% fake

Final prediction: [0, 1, 0, 1]  # argmax of each row
```

---

## **4. TRAINING LOOP DIAGRAM**

```
FOR EACH EPOCH:
│
├─ TRAINING PHASE
│  │
│  ├─ FOR EACH BATCH in train_loader:
│  │  │
│  │  ├─ Forward Pass
│  │  │  images [4, 3, 224, 224] ──→ EfficientNet ──→ logits [4, 2]
│  │  │
│  │  ├─ Loss Calculation
│  │  │  logits + labels ──→ CrossEntropyLoss ──→ loss (scalar)
│  │  │
│  │  ├─ Backward Pass (Backpropagation)
│  │  │  loss.backward() ──→ Calculate gradients for all weights
│  │  │
│  │  ├─ Weight Update
│  │  │  optimizer.step() ──→ Update all weights using gradients
│  │  │  Adam: w = w - lr * gradient
│  │  │
│  │  └─ Logging
│  │     Log: train_loss, train_accuracy
│  │
│  └─ END TRAINING PHASE
│
├─ VALIDATION PHASE
│  │
│  ├─ FOR EACH BATCH in val_loader:
│  │  │
│  │  ├─ Forward Pass (NO gradient tracking)
│  │  │  with torch.no_grad():
│  │  │      logits = model(images)
│  │  │
│  │  ├─ Loss Calculation
│  │  │  val_loss = loss_fn(logits, labels)
│  │  │
│  │  └─ Logging
│  │     Log: val_loss, val_accuracy
│  │
│  └─ END VALIDATION PHASE
│
├─ CALLBACKS CHECK
│  │
│  ├─ ModelCheckpoint
│  │  IF val_loss < best_val_loss:
│  │      Save current model ──→ models/best_model.ckpt
│  │
│  ├─ EarlyStopping
│  │  IF val_loss unchanged for 3 epochs:
│  │      STOP TRAINING (avoid overfitting)
│  │
│  └─ LR Scheduler
│     IF val_loss stuck for 2 epochs:
│         lr = lr * 0.5  (reduce learning rate)
│
└─ REPEAT FOR NEXT EPOCH
```

---

## **5. INFERENCE COMPARISON**

### **Image Inference (Simple)**
```
TIME: t=0ms
│
├─ Load image [H, W, 3]                    5ms
├─ Preprocess (Resize, Normalize)         10ms
├─ Model forward pass                    500ms
├─ Post-process (Softmax, argmax)         5ms
│
└─ TOTAL: ~520ms = 0.52 seconds
```

### **Video Inference (Complex)**
```
TIME: t=0ms
│
├─ Extract 10 frames                      500ms
│  (uniformly distributed across video)
│
├─ For frame 1:                          ~100ms
│  ├─ Preprocess
│  └─ Model forward pass
│
├─ For frame 2-10:                      ~900ms
│  (100ms × 9 more frames)
│
├─ Stack all probabilities               10ms
│  prob1, prob2, ..., prob10
│
├─ Average probabilities                 5ms
│  avg_prob = mean([prob1, prob2, ...])
│
├─ Get final prediction                  5ms
│  argmax(avg_prob)
│
└─ TOTAL: ~1420ms ≈ 1.4 seconds
```

---

## **6. MODEL ARCHITECTURE SIMPLIFIED**

```
INPUT [B, 3, 224, 224]
    ↓
CONV STEM (3→32)
    ↓
MBCONV BLOCK 1 (32→16)
    ├─ Depthwise Conv
    ├─ Pointwise Conv
    └─ Skip Connection
    ↓
MBCONV BLOCK 2-8 (progressive expansion)
    ├─ Block 2: 16→24
    ├─ Block 3: 24→40
    ├─ Block 4: 40→80
    ├─ Block 5: 80→112
    ├─ Block 6: 112→192
    ├─ Block 7: 192→320
    └─ Block 8: 320→1280
    ↓
ADAPTIVE AVG POOL [B, 1280]
    ↓
DROPOUT (p=0.4) [B, 1280]
    ↓
LINEAR (1280→2) [B, 2]
    ↓
OUTPUT [B, 2] (logits)

Total Parameters: ~5.3M
Model Size: ~20MB
```

---

## **7. PREDICTION CONFIDENCE VISUALIZATION**

```
If model outputs logits: [2.5, -1.2]

Step 1: Apply Softmax
┌──────────────────────────────────────────┐
│ e^2.5 / (e^2.5 + e^-1.2)  = 0.92        │  Real probability
│ e^-1.2 / (e^2.5 + e^-1.2) = 0.08        │  Fake probability
└──────────────────────────────────────────┘

Step 2: Visualize
Real: █████████████████████░░░░░░░░░░ 92%
Fake: ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 8%

Step 3: Make Prediction
Max probability: 0.92 (Real class)
Prediction: 🟢 REAL (92% confident)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If model outputs logits: [-0.8, 1.9]

Step 1: Apply Softmax
┌──────────────────────────────────────────┐
│ e^-0.8 / (e^-0.8 + e^1.9)  = 0.15       │  Real probability
│ e^1.9 / (e^-0.8 + e^1.9)   = 0.85       │  Fake probability
└──────────────────────────────────────────┘

Step 2: Visualize
Real: ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 15%
Fake: █████████████████░░░░░░░░░░░░░░░░░░ 85%

Step 3: Make Prediction
Max probability: 0.85 (Fake class)
Prediction: 🔴 DEEPFAKE (85% confident)
```

---

## **8. ERROR FLOW**

```
User Uploads File
    │
    ├─ Check: File exists? ──NO──→ ⚠️ Error: "No file selected"
    │
    └─ YES
        │
        ├─ Check: Valid MIME type? ──NO──→ ❌ Error: "Unsupported file type"
        │
        └─ YES
            │
            ├─ Try: Load image/video ──EXCEPTION──→ ❌ Error: "Failed to read file"
            │
            └─ SUCCESS
                │
                ├─ Try: Preprocess ──EXCEPTION──→ ❌ Error: "Preprocessing failed"
                │
                └─ SUCCESS
                    │
                    ├─ Try: Model inference ──EXCEPTION──→ ❌ Error: "Model inference failed"
                    │
                    └─ SUCCESS
                        │
                        └─ ✅ Return: Prediction + Confidence
```

---

## **9. CLASS DISTRIBUTION (Dataset)**

```
IDEAL BALANCED DATASET:
Real Images:    ████████████████████ 50% (e.g., 5000 images)
Fake Images:    ████████████████████ 50% (e.g., 5000 images)
                Total: 10,000 images

IMBALANCED DATASET (can cause bias):
Real Images:    ████████████████░░░░ 80% (e.g., 8000 images)
Fake Images:    ████░░░░░░░░░░░░░░░░ 20% (e.g., 2000 images)
                Total: 10,000 images
                Problem: Model biased towards predicting "Real"
```

---

## **10. PERFORMANCE TIMELINE**

```
BEFORE TRAINING
├─ Random weights
├─ train_loss ≈ 0.69 (random chance)
├─ train_acc ≈ 50% (random guessing)
└─ val_acc ≈ 50%

EPOCH 1
├─ train_loss ≈ 0.45 (decreasing)
├─ train_acc ≈ 78%
└─ val_acc ≈ 75%

EPOCH 2
├─ train_loss ≈ 0.32
├─ train_acc ≈ 85%
└─ val_acc ≈ 82%

EPOCH 3
├─ train_loss ≈ 0.28
├─ train_acc ≈ 88%
└─ val_acc ≈ 84% ← Best so far (CHECKPOINT SAVED)

EPOCH 4
├─ train_loss ≈ 0.25
├─ train_acc ≈ 90%
└─ val_acc ≈ 83% (WORSE than epoch 3)

EPOCH 5
├─ train_loss ≈ 0.22
├─ train_acc ≈ 91%
└─ val_acc ≈ 82% (EVEN WORSE)

TRAINING STOPS (EarlyStopping triggered)
├─ val_acc hasn't improved for 3 epochs
├─ Load: models/best_model.ckpt (from epoch 3)
└─ Final model: 84% validation accuracy
```

---

**Diagrams for visual clarity! 📊**

