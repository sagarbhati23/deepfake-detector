# 🎭 DEEPFAKE DETECTOR - Complete Project Explanation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Raw Idea & Problem Statement](#raw-idea--problem-statement)
3. [How It Works](#how-it-works)
4. [Tech Stack](#tech-stack)
5. [Project Architecture](#project-architecture)
6. [Component Breakdown](#component-breakdown)
7. [Data Flow](#data-flow)
8. [Key Features](#key-features)

---

## Project Overview

**Deepfake Detector** is an AI-powered application that detects manipulated/fake images and videos using deep learning. It analyzes media files and provides real-time predictions with confidence scores to determine whether content is authentic or artificially generated.

### Key Highlights:
- ✅ End-to-end deep learning pipeline
- ✅ Supports both images (JPG, PNG) and videos (MP4, MOV)
- ✅ Interactive web interface for easy usage
- ✅ Transfer learning from pretrained models
- ✅ Production-ready deployment

---

## Raw Idea & Problem Statement

### The Problem:
With AI advancements, deepfakes (AI-generated fake images/videos) are becoming increasingly sophisticated and harder to detect manually. This poses threats to:
- Trust and authenticity of media
- Misinformation spread
- Identity theft and impersonation
- Public safety

### The Solution:
Build an **automated detection system** using computer vision and deep learning that can:
1. Identify deepfake images quickly
2. Analyze video frames for manipulation
3. Provide confidence scores
4. Be accessible via a user-friendly interface

### Why This Matters:
Detection automation is faster, more scalable, and more consistent than manual inspection, enabling real-world deployment for social media platforms, news organizations, and security agencies.

---

## How It Works

### Step-by-Step Process:

```
1. User Uploads Media (Image/Video)
         ↓
2. System Detects File Type
         ↓
3. Media Preprocessing (Resizing, Normalization)
         ↓
4. Feature Extraction using EfficientNet-B0
         ↓
5. Binary Classification (Real vs. Fake)
         ↓
6. Generate Confidence Score
         ↓
7. Display Results to User
```

### For Images:
- Load image → Convert to RGB → Resize to 224×224
- Apply ImageNet normalization
- Pass through EfficientNet-B0 → Get 2 class outputs (Real/Fake)
- Apply softmax → Get probability scores
- Return highest probability as prediction

### For Videos:
- Extract N frames uniformly distributed across video duration
- Process each frame independently through the model
- Average predictions across all frames
- Return aggregated result

---

## Tech Stack

### 1. **Core ML Framework: PyTorch**
- **Why?** Industry-standard for deep learning, flexible, and production-ready
- **Role:** Neural network implementation, model training, inference

### 2. **Model Architecture: EfficientNet-B0**
- **What is it?** A lightweight, efficient CNN pretrained on ImageNet
- **Why?** 
  - Fast inference
  - Good accuracy with fewer parameters
  - Transfer learning ready
- **Details:**
  - Input: 224×224 RGB images
  - Output: 1280-dimensional feature vector
  - Training approach: Fine-tuning (weights from ImageNet)

### 3. **Training Framework: PyTorch Lightning**
- **What is it?** High-level wrapper around PyTorch for cleaner training code
- **Why?**
  - Automatic GPU/CPU handling
  - Built-in logging and checkpointing
  - Early stopping and learning rate scheduling
- **Features used:**
  - Trainer class for orchestration
  - LightningModule for model abstraction
  - Callbacks (ModelCheckpoint, EarlyStopping)

### 4. **Computer Vision: OpenCV + Pillow**
- **OpenCV:** Video processing, frame extraction, image degradation
- **Pillow (PIL):** Image loading and manipulation

### 5. **Web Framework: Gradio**
- **What is it?** Python library for building web UIs for ML models
- **Why?**
  - No web development needed
  - Auto-generates interactive interface
  - File upload/download support
  - Real-time predictions
- **Our Implementation:**
  - Drag-and-drop file upload
  - Live prediction output
  - Confidence score display
  - Image preview

### 6. **Configuration Management: YAML**
- Centralized config file for hyperparameters
- Easy experiment tracking

### 7. **Other Libraries:**
- **NumPy:** Numerical operations, frame sampling
- **scikit-learn:** Potential metrics calculation
- **TensorBoard:** Training visualization

---

## Project Architecture

### Overall System Architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
│  User uploads image/video via Gradio Web UI or CLI              │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING LAYER                            │
│  • File type detection (mimetypes)                              │
│  • Image: PIL.Image.open() → RGB conversion                     │
│  • Video: cv2.VideoCapture() → Frame extraction                 │
│  • Resize: 224×224 pixels                                       │
│  • Normalize: ImageNet mean/std                                 │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                  MODEL INFERENCE LAYER                           │
│  EfficientNet-B0:                                               │
│  • Conv Stem (3 channels → 32 filters)                          │
│  • MBConv Blocks (Mobile Inverted Residual)                     │
│  • Adaptive Average Pooling → 1280-D feature                    │
│  • Dropout(0.4) + Linear(1280 → 2)                              │
│  • Softmax → Confidence scores                                  │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                   OUTPUT LAYER                                   │
│  • Prediction: Real (class 0) or Deepfake (class 1)            │
│  • Confidence: Probability percentage                           │
│  • Display: Gradio UI, CLI, or Batch results                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### **A. TRAINING PIPELINE**

#### 1. **config.yaml** (Configuration File)
```yaml
Purpose: Centralized hyperparameter management
Contains:
  • lr: 0.0001 (Learning rate for Adam optimizer)
  • batch_size: 4 (Images per batch)
  • num_epochs: 1+ (Training iterations)
  • early_stopping_patience: 3 (Stop if no improvement)
  • train_paths: Path to training dataset
  • val_paths: Path to validation dataset
  • scheduler_patience: 2 (LR scheduler wait epochs)

Why structured this way?
  - Easy to modify hyperparameters without code changes
  - Reproducible experiments
  - Quick experiment tracking
```

#### 2. **datasets/hybrid_loader.py** (Data Loading)
```python
Purpose: Load images from disk and prepare for training

HybridDeepfakeDataset Class:
  ├── __init__()
  │   └── Scans folders for real/fake subdirectories
  │   └── Creates list of image paths and labels
  │       • real → label 0
  │       • fake → label 1
  │
  ├── __len__()
  │   └── Returns total number of images
  │
  └── __getitem__(idx)
      └── Loads single image at index
      └── Applies transforms (Resize, Normalize)
      └── Returns (image_tensor, label)

Why custom dataset class?
  - Flexibility to handle different data structures
  - Efficient batching
  - On-the-fly augmentation capability
```

#### 3. **lightning_modules/detector.py** (Model Training Logic)
```python
Purpose: Define training, validation, and optimization logic

DeepfakeDetector Class (extends pl.LightningModule):
  ├── __init__()
  │   └── Store model backbone
  │   └── Set learning rate
  │   └── Define loss function (CrossEntropyLoss)
  │
  ├── forward(x)
  │   └── Pass input through model → get logits
  │
  ├── training_step(batch, idx)
  │   ├── Unpack batch → images, labels
  │   ├── Forward pass → logits
  │   ├── Calculate loss
  │   ├── Calculate accuracy
  │   ├── Log metrics
  │   └── Return loss (PyTorch Lightning auto-backprop)
  │
  ├── validation_step(batch, idx)
  │   ├── Similar to training_step
  │   ├── No gradient computation
  │   └── Log validation metrics
  │
  └── configure_optimizers()
      └── Return Adam optimizer with specified LR

Why PyTorch Lightning?
  - Handles GPU/CPU automatically
  - Manages backward pass automatically
  - Structured, reproducible training
  - Built-in callbacks support
```

#### 4. **main_trainer.py** (Training Orchestration)
```python
Purpose: Main entry point for model training

Process:
  1. Load config.yaml
  2. Create data transforms (Resize, ToTensor, Normalize)
  3. Initialize datasets (train & validation)
  4. Create DataLoaders (batching, shuffling)
  5. Build EfficientNet-B0 backbone:
     - Load ImageNet pretrained weights
     - Modify classifier head:
       • Remove original 1000-class head
       • Add Dropout(0.4)
       • Add Linear(1280 → 2) for binary classification
  6. Wrap in DeepfakeDetector
  7. Configure callbacks:
     - ModelCheckpoint: Save best model (lowest val_loss)
     - EarlyStopping: Stop if val_loss doesn't improve for 3 epochs
  8. Initialize pl.Trainer with GPU/CPU auto-detect
  9. Start training: trainer.fit()

Key Features:
  • Early stopping prevents overfitting
  • Learning rate scheduling (factor=0.5 if stuck)
  • Checkpointing only saves best model
  • Automatic GPU detection
```

---

### **B. INFERENCE PIPELINE**

#### 1. **app.py** (Web Interface)
```python
Purpose: Interactive web UI for real-time predictions

Main Components:
  ├── load_model()
  │   └── Load EfficientNet-B0
  │   └── Modify classifier for 2 classes
  │   └── Load trained weights (best_model-v3.pt)
  │   └── Set to eval mode (no dropout during inference)
  │
  ├── predict_file(file_obj)
  │   ├── Detect file type using mimetypes
  │   ├── If IMAGE:
  │   │   ├── Load with PIL.Image
  │   │   ├── Preprocess (Resize, Normalize)
  │   │   ├── Forward pass
  │   │   ├── Get max probability
  │   │   └── Return label + confidence
  │   ├── If VIDEO:
  │   │   ├── Extract first frame with cv2
  │   │   ├── Same preprocessing + inference
  │   │   └── Return prediction based on first frame
  │   └── Else: Return error
  │
  └── Gradio UI:
      ├── File upload widget (gr.File)
      │   └── Accepts .jpg, .jpeg, .png, .mp4, .mov
      ├── Prediction output (gr.Textbox)
      │   └── Shows "🟢 Real" or "🔴 Deepfake"
      ├── Confidence output (gr.Textbox)
      │   └── Shows percentage
      └── Image preview (gr.Image)
          └── Displays processed image/first frame

Why Gradio?
  - Instant web UI without HTML/CSS/JS
  - Interactive in real-time
  - Professional-looking interface
  - Easy file handling

Run: python app.py → Opens http://localhost:7860
```

#### 2. **classify.py** (CLI Tool)
```python
Purpose: Command-line image classification

Usage: python classify.py path/to/image.jpg

Process:
  1. Parse command-line arguments (image path)
  2. Load model
  3. Preprocess image
  4. Inference
  5. Print results:
     • Prediction: REAL or FAKE
     • Real probability: 0.000 - 1.000
     • Fake probability: 0.000 - 1.000

Why?
  - Batch processing without UI
  - Script integration
  - Programmatic usage
  - Pipeline automation
```

#### 3. **inference/video_inference.py** (Multi-Frame Video Analysis)
```python
Purpose: Advanced video analysis with frame averaging

Key Process:
  1. extract_frames(video_path, num_frames=10)
     ├── Open video with cv2.VideoCapture
     ├── Calculate total frame count
     ├── Use np.linspace to uniformly sample N frames
     │   (E.g., 10 frames spread across entire video)
     ├── Convert each frame from BGR → RGB
     ├── Convert to PIL Image format
     └── Return list of processed frames
  
  2. predict_video(video_path)
     ├── Extract frames
     ├── For each frame:
     │   ├── Preprocess (Resize, Normalize)
     │   ├── Forward pass → get probabilities
     │   └── Store probabilities
     ├── Average probabilities across all frames
     │   (Reduces noise, improves robustness)
     ├── Get argmax → Final prediction
     └── Return prediction + probabilities
  
  3. Batch Processing
     └── Loop through videos_to_predict folder
     └── Process each .mp4 file
     └── Print results

Why Multi-Frame?
  - Single frame can be misleading
  - Video deepfakes have temporal patterns
  - Averaging reduces false positives
  - More robust than first-frame-only approach

Example Output:
  video1.mp4: REAL | Real: 0.891, Fake: 0.109
  video2.mp4: FAKE | Real: 0.234, Fake: 0.766
```

#### 4. **realeval.py** (Robustness Testing)
```python
Purpose: Test model robustness against image degradation

Degradation Simulations:
  1. Gaussian Blur (5×5 kernel)
     - Simulates video compression artifacts
  
  2. JPEG Compression (Quality=40)
     - Simulates lossy compression in social media
  
  3. Combined degradation (50% chance each)

Process:
  ├── Load all images/videos from test folder
  ├── For each file:
  │   ├── Apply random degradation
  │   ├── Inference
  │   ├── Print formatted result
  │   └── Handle errors gracefully
  │
  └── Output Format:
      filename (padded)  ➤  Prediction (padded)  (Confidence%)

Why?
  - Real-world media is degraded
  - Simulating compression = Testing real scenarios
  - Ensures model works on social media content
  - Measures robustness

Run: python realeval.py → Evaluates all images in realworld_samples/
```

---

### **C. UTILITY & EXPORT**

#### 5. **inference/export_onnx.py** (Model Export)
```python
Purpose: Export trained model to ONNX format

Why ONNX?
  - Platform-independent model format
  - Deploy to edge devices (mobile, IoT)
  - No PyTorch dependency needed at inference
  - Faster inference on some hardware
```

---

## Data Flow

### **Training Data Flow:**

```
Raw Dataset Folder Structure:
├── train/
│   ├── real/
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   └── fake/
│       ├── deepfake1.jpg
│       ├── deepfake2.png
│       └── ...
└── validation/
    ├── real/
    └── fake/

         ↓ (Path in config.yaml)

HybridDeepfakeDataset:
├── Scan real/ folder → assign label=0
├── Scan fake/ folder → assign label=1
├── Create image_paths list
├── Create labels list
└── __getitem__ returns (preprocessed_image, label)

         ↓ (DataLoader batching)

Batch of 4 images:
├── Shape: [4, 3, 224, 224] (4 images, 3 channels, 224×224)
├── Labels: tensor([0, 1, 1, 0])
└── Ready for training_step()

         ↓ (Training)

Loss Calculation:
├── Forward pass through EfficientNet-B0
├── Get logits: [4, 2] (4 images, 2 classes)
├── Apply CrossEntropyLoss
├── Backpropagation
└── Update weights with Adam optimizer

         ↓ (After each epoch)

Validation:
├── Forward pass on validation batch (no gradients)
├── Calculate val_loss and val_accuracy
├── Log to TensorBoard
├── ModelCheckpoint saves if best
└── EarlyStopping checks patience

         ↓ (Training ends)

Output:
└── models/best_model.ckpt (best performing model)
```

### **Inference Data Flow:**

```
User Input (Image/Video):
├── app.py receives file
├── mimetypes.guess_type() detects file type
│
├─ IF IMAGE:
│  ├── PIL.Image.open()
│  ├── Convert to RGB
│  └── Single image processing
│
└─ IF VIDEO:
   ├── cv2.VideoCapture()
   └── Extract first frame

         ↓

Preprocessing:
├── Resize to 224×224
├── Convert to tensor
├── Normalize with ImageNet stats
└── Add batch dimension [1, 3, 224, 224]

         ↓

Model Inference:
├── Load best_model-v3.pt
├── Forward pass
├── Get output logits [1, 2]
├── Apply softmax → probabilities
└── torch.max() → prediction + confidence

         ↓

Post-Processing:
├── Class 0 = "🟢 Real"
├── Class 1 = "🔴 Deepfake"
├── Confidence = max_probability × 100
└── Generate preview image

         ↓

Output to User:
├── Via Gradio UI:
│   ├── Prediction label
│   ├── Confidence percentage
│   └── Image preview
├── Via CLI (classify.py):
│   └── Printed to terminal
└── Via batch (realeval.py):
    └── Appended to results
```

---

## Key Features

### **1. Dual Input Support**
- **Images:** JPG, JPEG, PNG
- **Videos:** MP4, MOV
- Auto-detection via MIME types

### **2. Real-Time Processing**
- Instantaneous predictions
- No batch processing delays
- GPU acceleration (if available)

### **3. Confidence Scoring**
- Probability-based confidence
- Percentage display
- Helps users understand prediction certainty

### **4. Multiple Interfaces**
| Interface | Use Case |
|-----------|----------|
| **Gradio Web UI** | General users, no coding needed |
| **CLI (classify.py)** | Single image batch processing |
| **Video Inference** | Multi-frame video analysis |
| **Robustness Test (realeval.py)** | Model evaluation |

### **5. Transfer Learning**
- ImageNet pretrained weights
- Reduces training time
- Better accuracy with less data
- Only classifier head fine-tuned

### **6. Production Ready**
- ONNX export support
- Model checkpointing
- Error handling
- Robust preprocessing

### **7. Hyperparameter Control**
- YAML config file
- Easy experiment management
- Learning rate scheduling
- Early stopping

---

## Quick Reference: File Purposes

| File | Purpose | Type |
|------|---------|------|
| **config.yaml** | Hyperparameter configuration | Config |
| **main_trainer.py** | Model training orchestration | Training |
| **datasets/hybrid_loader.py** | Dataset loading & batching | Data |
| **lightning_modules/detector.py** | Training logic wrapper | Model |
| **app.py** | Web UI interface | Frontend |
| **classify.py** | CLI image classification | Inference |
| **inference/video_inference.py** | Multi-frame video analysis | Inference |
| **inference/export_onnx.py** | Model export to ONNX | Utility |
| **realeval.py** | Robustness evaluation | Testing |
| **requirements.txt** | Python dependencies | Setup |

---

## Presentation Tips for Ma'am

### **When Explaining the Project:**

1. **Start with Problem:** "Deepfakes are increasing, manual detection is slow. We need automation."

2. **Explain Solution:** "We built an AI system that automatically detects fake images/videos."

3. **Highlight Tech Choices:**
   - "We used EfficientNet-B0 because it's fast and accurate."
   - "PyTorch Lightning because it handles GPU automatically."
   - "Gradio because we need a web interface without frontend coding."

4. **Walk Through Process:**
   - "User uploads file → We preprocess → Pass through trained model → Get Real/Fake prediction."

5. **Show Capabilities:**
   - "It handles images, videos, and provides confidence scores."
   - "Multi-frame analysis makes video predictions more robust."

6. **Demo the UI:**
   - Run `python app.py` and show the web interface
   - Upload a test image
   - Show real-time prediction

7. **Technical Depth (if asked):**
   - Model: "EfficientNet-B0 with 1280 features, modified classifier for 2 classes"
   - Training: "CrossEntropyLoss, Adam optimizer, early stopping for overfitting prevention"
   - Inference: "FastTorch Lightning pipeline, GPU/CPU auto-detect"

8. **Challenges Overcome:**
   - "Handled both images and videos differently"
   - "Implemented multi-frame averaging for better video analysis"
   - "Tested robustness against image degradation"

---

## Summary

**Deepfake Detector** is a complete, production-ready deep learning application that:
- ✅ Solves a real-world problem (detecting AI-generated media)
- ✅ Uses modern tech stack (PyTorch, Gradio, Lightning)
- ✅ Provides multiple user interfaces (web, CLI, batch)
- ✅ Is robust and extensible
- ✅ Demonstrates end-to-end ML pipeline implementation

This project showcases your ability to build complete AI systems, from training to deployment! 🚀

