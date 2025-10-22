# 🤖✨ AI-Powered Sign Language Translator

> **Bridging the communication gap between the Deaf and hearing communities using Artificial Intelligence**

---

## 🧩 Overview  

Millions of deaf and hard-of-hearing individuals face daily communication barriers.  
This project proposes an **AI-powered multilingual platform** that performs **real-time, bidirectional translation** between **sign and spoken/written languages**, supporting **education, healthcare, and accessibility** across platforms.  

Built with cutting-edge technologies in **computer vision**, **speech processing**, and **natural language understanding**, our solution empowers inclusive communication for all.

---

## 🚀 Features  

### 👐 Core Features
- **🎥 Live Camera Translation** — Capture real-time gestures via camera → Translate into text or voice.  
- **📹 Video Upload Translation** — Translate pre-recorded or YouTube videos into sign animations.  
- **🌍 Multilingual Support** — Supports multiple sign (ASL, ArSL) and spoken (Arabic, English) languages.  
- **🗣️ Voice-to-Sign Conversion** — Convert speech input into 3D avatar sign animations in real time.  

### 💡 Advanced & Innovative Features
- **💬 AI Sign Chatbot** — Practice signing or chatting in sign language with an intelligent assistant.  
- **📶 Offline Mode** — Uses TensorFlow Lite for translation in low-connectivity areas.  
- **♿ Accessibility Toolkit** — Adjustable sign speed, text-to-speech, haptic notifications, high-contrast modes.  

---

## 🏗️ System Architecture  

<p align="center">
  <img src="use Experience.png" alt="System Architecture Diagram" width="800"/>
</p>

> The architecture integrates **gesture recognition**, **speech recognition**, **translation**, and **3D avatar rendering** into one seamless AI pipeline.

**Flow Summary:**
1. 🎥 Input sources (Camera, Mic, Text, Video)
2. 🧩 AI Processing (Gesture Recognition, Speech-to-Text, NLP)
3. 🔤 Translation Engine (Seq2Seq Transformer)
4. 🧍 Avatar Rendering & Output (3D animations, text, or voice)
5. 🌐 Web Interface built with React & Tailwind CSS

---

## 🧠 Methodology  

1. **Data Preprocessing** – Extract 3D hand landmarks using MediaPipe.  
2. **Model Training** – Train CNN/LSTM/Transformer models for gesture recognition.  
3. **Multilingual Mapping** – Map gestures to English/Arabic words and gloss sequences.  
4. **Avatar Animation** – Use Blender/Unity to visualize translated signs.  

---

## 🧰 Tools & Frameworks  

| Category | Tools Used |
|-----------|-------------|
| Gesture Recognition | MediaPipe, OpenCV |
| AI Models | TensorFlow, PyTorch, Transformers |
| Speech Recognition | Whisper, Wav2Vec2.0, Google Speech-to-Text |
| Avatar Rendering | Unity, Blender, WebGL |
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

---

## 🧭 User Journey  

1. **🏠 Home Screen** – Choose between *Live Translate*, *Video Upload*, or *Settings*.  
2. **🎥 Live Mode** – Sign in front of the camera → Real-time text/voice translation.  
3. **🎙 Voice Mode** – Speak into the microphone → See avatar signing your words.  
4. **📹 Upload Mode** – Upload video → Auto-translate with side-by-side sign output.  
5. **⚙️ Accessibility Panel** – Customize contrast, vibration alerts, and voice speed.

---

## 🔮 Future Work  

- 🧩 Expand to sentence-level translation with PHOENIX-2014T dataset.  
- 🤖 Integrate with Alexa/Google for voice-to-sign capabilities.  
- 🔁 Cross-sign translation (e.g., ASL ↔ BSL).  
- 🧤 Support wearable haptic feedback gloves.  
- 🕶️ Smart glasses integration for live subtitles and translation overlays.  

---

## 🧱 Project Vision  

> “To create a scalable, inclusive, and multilingual accessibility platform that uses AI to eliminate communication barriers between Deaf and hearing individuals — fostering equality in education, healthcare, and daily life.”

---

## 📜 License  

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  

---

