# AI-Powered Skeleton-Based Sign Language Translation System

## Overview

Millions of deaf and hard-of-hearing individuals face persistent communication barriers in education, healthcare, and daily life. This project presents an **AI-powered, multilingual sign language translation platform** that enables **real-time, bidirectional translation** between **sign languages** and **spoken/written languages**.

Unlike traditional avatar-based or video-retrieval systems, this solution adopts a **skeleton-based motion synthesis approach**, directly generating **2D/3D skeletal sign gestures** from text, speech, or video. This design significantly reduces computational overhead, improves scalability, and supports research-oriented extensions in human motion generation and sign language processing.

---

## Key Features

### Core Functionality
- **Real-Time Camera Translation**  
  Captures live sign language gestures via a camera and translates them into text or synthesized speech.

- **Video-Based Translation**  
  Translates pre-recorded or uploaded videos (including online sources) into skeleton-based sign motion.

- **Multilingual Support**  
  Supports multiple sign languages (e.g., ASL, ArSL) and spoken languages (e.g., Arabic, English).

- **Speech-to-Sign Translation**  
  Converts spoken language into AI-generated skeleton-based sign animations in real time.

### Advanced Capabilities
- **AI-Driven Sign Language Chatbot**  
  An interactive assistant that communicates using animated skeleton signing, enabling practice and learning.

- **Offline Mode**  
  Lightweight TensorFlow Lite models enable operation in low-connectivity environments.

- **Accessibility Toolkit**  
  Customizable sign speed, visual contrast, vibration feedback, and text-to-speech options for inclusive use.

---

## System Architecture

<p align="center">
  <img src="user Experience.png" alt="System Architecture Diagram" width="800"/>
</p>

The system integrates gesture recognition, speech recognition, natural language processing, and motion synthesis into a unified skeleton-based pipeline.

### Processing Pipeline
1. **Input Sources**: Camera, microphone, text, or video  
2. **AI Processing**: Gesture recognition, speech-to-text, and NLP  
3. **Translation Engine**: Seq2Seq Transformer for text-to-gloss conversion  
4. **Motion Generation**: Gloss-to-pose sequence using GANs or Transformers  
5. **Visualization**: 2D/3D skeleton animation (MediaPipe, Matplotlib, or Three.js)  
6. **Web Interface**: React and Tailwind CSS front-end  

---

## Methodology

1. **Data Preprocessing**  
   Extract 2D/3D body and hand landmarks using MediaPipe or OpenPose from sign language datasets.

2. **Model Training**  
   Train Transformer-based or GAN-based architectures (e.g., Pose2Sign, SignGAN) to generate pose sequences from gloss representations.

3. **Text-to-Gloss Translation**  
   Apply Seq2Seq or Transformer-based NLP models (e.g., BERT-based encoders) to adapt spoken language grammar into sign language gloss.

4. **Skeleton Motion Visualization**  
   Render generated pose sequences using Matplotlib, MediaPipe rendering, or WebGL-based frameworks such as Three.js.

This approach eliminates the dependency on pre-recorded videos or animated avatars by directly synthesizing skeletal motion trajectories.

---

## Tools and Technologies

| Category | Technologies |
|--------|-------------|
| Gesture Recognition | MediaPipe, OpenCV, OpenPose |
| Machine Learning | TensorFlow, PyTorch, Transformers, SignGAN, Pose2Sign |
| Speech Recognition | Whisper, Wav2Vec 2.0, Google Speech-to-Text |
| Visualization | Matplotlib, Three.js, WebGL |
| Front-End | React, Tailwind CSS |
| Back-End | Python, Flask or Django |

---

## Datasets

### American Sign Language (ASL)
- MS-ASL (Microsoft Large-Scale ASL Dataset)  
- ASL Dataset (Kaggle)

### Pose and Sentence-Level Datasets
- **PHOENIX-2014T** (Sentence-level sign language with gloss and pose data)  
- **YouTube-ASL** (Large-scale conversational sign language dataset)

---

## User Workflow

1. **Home Interface**  
   Users select between live translation, video upload, or accessibility settings.

2. **Live Translation Mode**  
   Users perform signs in front of a camera and receive real-time text or speech output.

3. **Speech-to-Sign Mode**  
   Spoken input is converted into skeleton-based sign language animation.

4. **Video Upload Mode**  
   Uploaded videos are processed to extract speech or text and generate corresponding sign motion.

5. **Accessibility Settings**  
   Users customize visualization parameters such as skeleton color, motion speed, and feedback options.

---

## Future Enhancements

- Integration of advanced text-to-pose models (e.g., Text2Pose, SignFlow) for smoother motion generation  
- Expansion to sentence-level translation using PHOENIX-2014T  
- Cross-sign-language translation (e.g., ASL to BSL)  
- Augmented reality-based visualization for immersive sign representation  
- Exploration of self-supervised and multimodal learning approaches for pose generation  

---

## Project Vision

The goal of this project is to develop a scalable, inclusive, and multilingual accessibility platform that leverages **AI-driven skeleton motion synthesis** to bridge communication gaps between deaf and hearing individuals, while advancing research in sign language translation and human motion generation.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.
