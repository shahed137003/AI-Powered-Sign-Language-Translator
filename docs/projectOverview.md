# Project Documentation

This document compiles all essential documentation for the AI-Powered Sign Language Translation System.

## 1. Overview

A multilingual sign language translation system supporting live translation, text-to-sign, voice-to-sign, and video translation, with accessibility features and AI-powered extensions.

---

## 2. System Architecture

### 2.1 Input Sources

* Camera (live sign capture)
* Microphone (speech input)
* Text input
* Uploaded videos or external links

### 2.2 AI & Processing

* **Preprocessing & Landmark Extraction**: MediaPipe + OpenCV
* **Gesture Recognition**: CNN, LSTM, Transformers
* **ASR (Speech Recognition)**: Whisper, Wav2Vec2.0, Google STT
* **Translation Engine**: Gloss generation, text processing
* **Rendering Engine**: Unity/Blender avatars, WebGL emoji animations

### 2.3 Output

* Text subtitles
* Voice synthesis
* Avatar animations
* Emoji/2D animations

### 2.4 Platforms

* React Web App (Tailwind CSS)
* Backend (Node.js / Python)
* AI Server (Python, PyTorch, TensorFlow)

---

## 3. Project Folder Structure

```
project-root/
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── context/
│   │   ├── utils/
│   │   └── assets/
│   └── package.json
│
├── backend/
│   ├── src/
│   │   ├── routes/
│   │   ├── controllers/
│   │   ├── services/
│   │   └── utils/
│   └── server.js
│
├── ai-engine/
│   ├── preprocessing/
│   │   ├── mediapipe_extraction.py
│   │   └── opencv_cleaning.py
│   ├── models/
│   │   ├── gesture_recognition/
│   │   ├── nlp_gloss/
│   │   └── asr/
│   ├── training/
│   ├── inference/
│   └── api/
│       └── ai_server.py
│
├── datasets/
│   ├── ASL/
│   ├── ArSL/
│   ├── MS-ASL/
│   └── youtube-asl/
│
├── docs/
│   ├── system_architecture.md
│   ├── ai_pipeline.md
│   ├── text_to_sign.md
│   ├── voice_to_sign.md
│   ├── video_translation.md
│   ├── preprocessing.md
│   ├── api_specs.md
│   ├── ui_ux_guidelines.md
│   └── future_work.md
│
├── requirements.txt
└── README.md
```

---

## 4. Documentation Files

This section contains all text that will be included in the `docs/` folder.

### 4.1 system_architecture.md

* Overview of input → processing → output
* Separation between AI engine & backend
* Components life cycle

### 4.2 ai_pipeline.md

* Landmark extraction
* Feature normalization
* Model architectures (CNN, LSTM, Transformers)
* Training pipeline
* Evaluation metrics

### 4.3 preprocessing.md

* Explanation of MediaPipe
* Why landmarks instead of raw images
* Steps:

  * Frame extraction
  * Hand tracking
  * Landmark cleaning & normalization
  * Sequence building

### 4.4 text_to_sign.md

* Lexical mapping
* Gloss generation
* Avatar rendering
* Limitations & future expansion

### 4.5 voice_to_sign.md

* ASR processing
* Error correction
* Integration with text-to-sign

### 4.6 video_translation.md

* Audio extraction
* Frame alignment
* Avatar synchronization

### 4.7 api_specs.md

* REST endpoints (e.g., `/translate/text`, `/translate/voice`, `/ai/gesture`)
* Request/response formats
* Error handling

### 4.8 ui_ux_guidelines.md

* Accessibility rules
* High-contrast mode
* Avatar display layout
* Responsive design rules

### 4.9 future_work.md

* Sentence-level translation
* BSL/ASL cross-sign translation
* Wearable integration
* Context-aware translation

---

## 5. AI & Backend Separation

### Why separate?

* Scalability
* Easy model updates
* Dedicated GPU server for AI

### Structure:

* Backend communicates with AI via HTTP/REST or gRPC
* AI server handles heavy inference

---

## 6. Where Preprocessing Happens

**All preprocessing & landmark extraction should be inside the `ai-engine/preprocessing/` folder.**

* Not in frontend
* Not in backend
* Only in AI engine

This ensures clean modularity.

---
