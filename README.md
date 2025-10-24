# 🤖✨ AI-Powered Sign Language Translator (Skeleton-Based)

> **Bridging the communication gap between the Deaf and hearing communities using Artificial Intelligence**

---

## 🧩 Overview  

Millions of deaf and hard-of-hearing individuals face daily communication barriers.  
This project proposes an **AI-powered multilingual platform** that performs **real-time, bidirectional translation** between **sign and spoken/written languages**, supporting **education, healthcare, and accessibility** across platforms.  

Unlike traditional avatar-based systems, our solution uses **AI-driven skeleton motion synthesis**, generating **realistic 2D/3D skeletal gestures** directly from text, speech, or video — creating a more lightweight, scalable, and research-focused approach.

---

## 🚀 Features  

### 👐 Core Features
- **🎥 Live Camera Translation** — Capture real-time gestures via camera → Translate into text or voice.  
- **📹 Video Upload Translation** — Translate pre-recorded or YouTube videos into skeleton-based sign motion.  
- **🌍 Multilingual Support** — Supports multiple sign (ASL, ArSL) and spoken (Arabic, English) languages.  
- **🗣️ Voice-to-Sign Conversion** — Convert speech input into AI-generated skeleton signing in real time.  

### 💡 Advanced & Innovative Features
- **💬 AI Sign Chatbot** — Chat or practice signing with an intelligent assistant that replies using animated skeleton poses.  
- **📶 Offline Mode** — TensorFlow Lite models allow translation in low-connectivity environments.  
- **♿ Accessibility Toolkit** — Adjustable sign speed, vibration feedback, high-contrast visualization, and text-to-speech options.  

---

## 🏗️ System Architecture  
<p align="center"> <img src="user Experience.png" alt="System Architecture Diagram" width="800"/> </p>
> The architecture integrates **gesture recognition**, **speech recognition**, **text-to-gloss translation**, and **AI motion synthesis** into one seamless skeleton-based pipeline.

**Flow Summary:**
1. 🎥 Input Sources (Camera, Microphone, Text, Video)
2. 🧠 AI Processing (Gesture Recognition, Speech-to-Text, NLP)
3. 🔡 Translation Engine (Seq2Seq Transformer for Text → Gloss)
4. 🦴 Motion Generator (Gloss → Pose Sequence using GAN/Transformer)
5. 📊 Visualization (2D/3D Skeleton Animation via MediaPipe or Matplotlib)
6. 🌐 Web Interface built with React & Tailwind CSS  

---

## 🧠 Methodology  

1. **Data Preprocessing** – Extract 3D body and hand landmarks using **MediaPipe** or **OpenPose** from sign language datasets.  
2. **Model Training** – Train **Transformers / GANs (Pose2Sign, SignGAN)** for **gloss-to-pose sequence generation**.  
3. **Text-to-Gloss Mapping** – Use **Seq2Seq models** (Transformer or BERT) to adapt spoken grammar to sign gloss.  
4. **Skeleton Motion Visualization** – Display AI-generated motion using **Matplotlib**, **Three.js**, or **MediaPipe rendering**.  

> The system bypasses the need for avatars or pre-recorded videos by directly generating motion trajectories for 2D/3D skeletons.

---

## 🧰 Tools & Frameworks  

| Category | Tools Used |
|-----------|-------------|
| Gesture Recognition | MediaPipe, OpenCV, OpenPose |
| AI Models | TensorFlow, PyTorch, Transformers, SignGAN, Pose2Sign |
| Speech Recognition | Whisper, Wav2Vec2.0, Google Speech-to-Text |
| Visualization | Matplotlib, Three.js, WebGL (skeleton rendering) |
| Front-End | React, Tailwind CSS |
| Back-End | Python, Flask/Django (optional) |

---

## 🧾 Datasets  

### 📘 Arabic Sign Language (ArSL)
- [Arabic Sign Language Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/birafaneimane/arabic-sign-language-alphabet-arsl-dataset)
- [Arabic Sign Language Data](https://www.kaggle.com/datasets/mohamedmostafa23334/arabic-sign-language-data)
- [Augmented ArSL Dataset](https://www.kaggle.com/datasets/sabribelmadoui/arabic-sign-language-augmented-dataset)

### 📗 American Sign Language (ASL)
- [MS-ASL: Microsoft Large Scale Dataset](https://microsoft.github.io/data-for-society/dataset?d=MS-ASL-American-Sign-Language-Dataset)
- [ASL Dataset (Kaggle)](https://www.kaggle.com/datasets/ayuraj/asl-dataset)

### 📙 Pose Extraction Datasets
- **PHOENIX-2014T** (German Sign Language, sentence-level with gloss and pose data)
- **YouTube-ASL** (ongoing dataset for real conversational signs)

---

## 🧭 User Journey  

1. **🏠 Home Screen** – Choose between *Live Translate*, *Video Upload*, or *Settings*.  
2. **🎥 Live Mode** – Sign in front of the camera → Real-time text/voice translation using pose detection.  
3. **🎙 Voice Mode** – Speak into the microphone → Watch generated skeleton motion representing signs.  
4. **📹 Upload Mode** – Upload a video → Extracts audio/text and visualizes corresponding skeleton signing.  
5. **⚙️ Accessibility Panel** – Customize skeleton color, motion speed, and feedback settings.  

---

## 🔮 Future Work  

- 🧬 Integrate **text-to-pose models** (Text2Pose, SignFlow) for smoother skeleton motion.  
- 🧠 Expand to **sentence-level translation** using PHOENIX-2014T.  
- 🔁 Enable **cross-sign translation** (e.g., ASL ↔ BSL).  
- 🕶️ Add **AR-based visualization** for skeleton signs in mixed reality.  
- 📈 Explore **self-supervised pose generation** using multimodal datasets.  

---

## 🧱 Project Vision  

> “To create a scalable, inclusive, and multilingual accessibility platform that uses **AI-driven skeleton motion synthesis** to eliminate communication barriers between Deaf and hearing individuals — advancing research in sign language translation and human motion generation.”

---

## 📜 License  

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  
