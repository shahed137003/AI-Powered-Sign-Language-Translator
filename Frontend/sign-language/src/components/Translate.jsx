import React, { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic, Camera, RefreshCw, Loader2, Video, Volume2, Trash2,
  Play, Square, PowerOff, Hand, Upload, StopCircle
} from "lucide-react";
import { getWsUrl } from "../lib/api";

// ─── Toast ────────────────────────────────────────────────────────────────────
const Toast = ({ message, type }) => {
  if (!message) return null;
  const base = "fixed bottom-5 left-1/2 transform -translate-x-1/2 p-4 rounded-xl shadow-2xl z-[100] flex items-center gap-3 font-semibold text-white";
  const color =
    type === "success" ? "bg-green-600 shadow-green-500/50" :
    type === "warning" ? "bg-yellow-600 shadow-yellow-500/50" :
    type === "error"   ? "bg-red-600 shadow-red-500/50"       :
                         "bg-purple-600 shadow-purple-500/50";
  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 50 }}
      className={`${base} ${color}`}
    >
      {type === "info" && <Volume2 className="w-5 h-5" />}
      {message}
    </motion.div>
  );
};

// ─── Keypoint helpers ─────────────────────────────────────────────────────────
const RELEVANT_FACE_INDICES = [
  0, 13, 14, 17, 37, 39, 40, 46, 52, 53, 55, 61, 63, 65, 66, 70, 78, 80, 81, 82,
  84, 87, 88, 91, 95, 105, 107, 146, 178, 181, 185, 191, 267, 269, 270, 276, 282,
  283, 285, 291, 293, 295, 296, 300, 308, 310, 311, 312, 314, 317, 318, 321, 324,
  334, 336, 375, 402, 405, 409, 415
];

const extractKeypoints = (results) => {
  const pose = results.poseLandmarks
    ? results.poseLandmarks.flatMap(lm => [lm.x, lm.y, lm.z, lm.visibility])
    : new Array(33 * 4).fill(0);
  const face = results.faceLandmarks
    ? RELEVANT_FACE_INDICES.flatMap(i => {
        const lm = results.faceLandmarks[i];
        return lm ? [lm.x, lm.y, lm.z] : [0, 0, 0];
      })
    : new Array(RELEVANT_FACE_INDICES.length * 3).fill(0);
  const lh = results.leftHandLandmarks
    ? results.leftHandLandmarks.flatMap(lm => [lm.x, lm.y, lm.z])
    : new Array(21 * 3).fill(0);
  const rh = results.rightHandLandmarks
    ? results.rightHandLandmarks.flatMap(lm => [lm.x, lm.y, lm.z])
    : new Array(21 * 3).fill(0);
  return [...pose, ...face, ...lh, ...rh];
};

const hasVisibleHands = (r) => !!(r.leftHandLandmarks || r.rightHandLandmarks);

// ─── WebSocket URL — Direct to AI service ────────────────────────────────────
// Dynamically resolved via getWsUrl to support local network and mobile devices

// ─────────────────────────────────────────────────────────────────────────────
export default function Translate() {
  const fadeUp = { hidden: { opacity: 0, y: 25 }, visible: { opacity: 1, y: 0 } };
  const options = ["Sign to Text", "Text / Audio to Sign"];

  // ── UI state ──────────────────────────────────────────────────────────────
  const [selected, setSelected]               = useState(options[0]);
  const [textInput, setTextInput]             = useState("");
  const [recognizedText, setRecognizedText]   = useState("");
  const [englishSentence, setEnglishSentence] = useState("");
  const [isTranslating, setIsTranslating]     = useState(false);
  const [message, setMessage]                 = useState(null);
  const [videoQueue, setVideoQueue]           = useState([]);
  const [currentVideo, setCurrentVideo]       = useState(null);

  const [isStreaming, setIsStreaming]               = useState(false);
  const [isProcessingVideo, setIsProcessingVideo]   = useState(false);
  const [isRecordingSign, setIsRecordingSign]       = useState(false);
  const [isAiLoading, setIsAiLoading]               = useState(false);
  const [framesCollected, setFramesCollected]       = useState(0);
  const [handDetected, setHandDetected]             = useState(false);
  const [mpReady, setMpReady]                       = useState(false);
  const [lastConfidence, setLastConfidence]         = useState(0);

  // ── Refs for deduplication ─────────────────────────────────────────────────
  const videoRef              = useRef(null);
  const canvasRef             = useRef(null);
  const socketRef             = useRef(null);
  const holisticRef           = useRef(null);
  const cameraStreamRef       = useRef(null);
  const frameRequestRef       = useRef(null);
  const isComponentMounted    = useRef(true);
  const isStartingRef         = useRef(false);
  const lastFrameTime         = useRef(0);
  const uploadedVideoUrlRef   = useRef(null);
  const isVideoProcessingRef  = useRef(false);
  const videoEndedHandlerRef  = useRef(null);
  const mpReadyRef            = useRef(false);
  const updateTimerRef        = useRef(null);
  const lastWordRef           = useRef("");
  const lastTranslationRef    = useRef("");
  const lastSignsRef          = useRef("");
  const processingSentenceRef = useRef(false);

  const FRAME_INTERVAL_MS = 50; // ~20 fps for better accuracy

  // ── Helpers ───────────────────────────────────────────────────────────────
  const showMessage = useCallback((text, type = "info") => {
    setMessage({ text, type });
    setTimeout(() => setMessage(null), 3500);
  }, []);

  const speakText = useCallback((text) => {
    const empty = ["Sign sequence will appear here...", "Grammatically correct sentence will appear here..."];
    if (!text || empty.includes(text)) { showMessage("Nothing to speak", "warning"); return; }
    if (!window.speechSynthesis) { showMessage("Speech synthesis not supported", "error"); return; }
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US"; u.rate = 0.9;
    window.speechSynthesis.speak(u);
  }, [showMessage]);

  // Update recognized text with deduplication
  const updateRecognizedText = useCallback((newText) => {
    if (newText === lastSignsRef.current) return;
    
    if (updateTimerRef.current) clearTimeout(updateTimerRef.current);
    
    updateTimerRef.current = setTimeout(() => {
      setRecognizedText(prev => {
        if (newText === prev) return prev;
        lastSignsRef.current = newText;
        return newText;
      });
    }, 50);
  }, []);

  // Update English translation with deduplication
  const updateEnglishTranslation = useCallback((newTranslation) => {
    if (!newTranslation || newTranslation === lastTranslationRef.current) return;
    if (newTranslation === "Waiting for more words...") return;
    if (processingSentenceRef.current) return;
    
    processingSentenceRef.current = true;
    
    setEnglishSentence(prev => {
      if (newTranslation === prev) return prev;
      lastTranslationRef.current = newTranslation;
      return newTranslation;
    });
    
    setTimeout(() => {
      processingSentenceRef.current = false;
    }, 500);
  }, []);

  // ── Lifecycle ─────────────────────────────────────────────────────────────
  useEffect(() => {
    isComponentMounted.current = true;
    return () => { 
      isComponentMounted.current = false; 
      turnOffSystem();
      if (updateTimerRef.current) clearTimeout(updateTimerRef.current);
    };
  }, []);

  useEffect(() => {
    const scripts = [
      "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js",
      "https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js"
    ];
    scripts.forEach(src => {
      if (!document.querySelector(`script[src="${src}"]`)) {
        const s = document.createElement("script"); 
        s.src = src; 
        s.async = true;
        document.body.appendChild(s);
      }
    });
  }, []);

  // ── WebSocket ─────────────────────────────────────────────────────────────
  const setupWebSocket = useCallback((onOpenCallback) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) { 
      onOpenCallback(); 
      return; 
    }
    
    const wsUrl = getWsUrl(8001, "/ws/sign-to-text");
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => { 
      if (!isComponentMounted.current) return; 
      socketRef.current = ws; 
      console.log("🔌 WebSocket connected");
      onOpenCallback(); 
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("📨 AI Response:", data.status, data.text || data.sentence);
        
        switch(data.status) {
          case "collecting":
            setIsRecordingSign(true);
            setFramesCollected(data.frames_collected || 0);
            break;
            
          case "success":
            setIsTranslating(false);
            setIsRecordingSign(false);
            
            if (data.sentence_buffer && data.sentence_buffer !== lastSignsRef.current) {
              updateRecognizedText(data.sentence_buffer);
            }
            
            if (data.english_sentence && data.english_sentence !== lastTranslationRef.current) {
              updateEnglishTranslation(data.english_sentence);
            }
            
            if (data.confidence > 0.3 && data.text) {
              setLastConfidence(data.confidence * 100);
            }
            break;
            
          case "sentence":
            if (data.sentence && data.sentence !== lastTranslationRef.current) {
              console.log("📝 New translation:", data.sentence);
              updateEnglishTranslation(data.sentence);
              showMessage("✨ Translation ready", "success");
            }
            break;
            
          case "error":
            setIsTranslating(false);
            setIsRecordingSign(false);
            showMessage(data.text || "AI error", "error");
            break;
            
          default:
            break;
        }
        
      } catch (e) { 
        console.error("WS parse error:", e); 
      }
    };
    
    ws.onclose = () => { 
      console.log("🔌 WebSocket disconnected");
      if (isComponentMounted.current && (isStreaming || isProcessingVideo)) turnOffSystem(); 
    };
    
    ws.onerror = (e) => { 
      console.error("WebSocket error:", e); 
      showMessage("AI connection error", "error"); 
      turnOffSystem(); 
    };
    
  }, [showMessage, updateRecognizedText, updateEnglishTranslation, isStreaming, isProcessingVideo]);

  // ── MediaPipe processing - OPTIMIZED FOR ACCURACY ──────────────────────────
  const startProcessing = useCallback((mediaElement, isMirrored = false, pausedForWarmup = false) => {
    if (!window.Holistic) { 
      showMessage("AI libraries still loading.", "warning"); 
      return false; 
    }
    if (!mediaElement) { 
      showMessage("Media element not ready.", "error"); 
      return false; 
    }

    const { Holistic, POSE_CONNECTIONS, HAND_CONNECTIONS, drawConnectors, drawLandmarks } = window;

    const holistic = new Holistic({
      locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`
    });
    
    // OPTIMIZED: Match debug script settings for better accuracy
    holistic.setOptions({
      modelComplexity: 1,           // Higher accuracy
      smoothLandmarks: true,        // Stable tracking
      enableSegmentation: false,
      refineFaceLandmarks: false,
      minDetectionConfidence: 0.5,  // Match debug script
      minTrackingConfidence: 0.5    // Match debug script
    });

    holistic.onResults((results) => {
      if (!mpReadyRef.current) {
        mpReadyRef.current = true;
        setMpReady(true);
        showMessage("✅ AI Ready!", "success");

        if (pausedForWarmup && mediaElement && mediaElement.paused) {
          mediaElement.currentTime = 0;
          mediaElement.play().catch(e => console.warn("Resume play failed:", e));
        }
      }

      if (!isComponentMounted.current) return;
      
      const canvas = canvasRef.current;
      if (!canvas || !mediaElement) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const handsVisible = hasVisibleHands(results);
      setHandDetected(handsVisible);

      if (canvas.width !== mediaElement.videoWidth) {
        canvas.width = mediaElement.videoWidth;
        canvas.height = mediaElement.videoHeight;
      }

      // Draw landmarks
      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (isMirrored) { 
        ctx.translate(canvas.width, 0); 
        ctx.scale(-1, 1); 
      }

      if (results.poseLandmarks) {
        drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: "#00FF00", lineWidth: 2 });
        drawLandmarks(ctx, results.poseLandmarks, { color: "#FF0000", lineWidth: 1 });
      }
      if (results.faceLandmarks) {
        RELEVANT_FACE_INDICES.forEach(i => {
          const lm = results.faceLandmarks[i]; 
          if (!lm) return;
          ctx.beginPath();
          ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 1.5, 0, 2 * Math.PI);
          ctx.fillStyle = "#00FFFF";
          ctx.fill();
        });
      }
      if (results.leftHandLandmarks) {
        drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: "#CC0000", lineWidth: 3 });
        drawLandmarks(ctx, results.leftHandLandmarks, { color: "#00FF00", lineWidth: 1.5 });
      }
      if (results.rightHandLandmarks) {
        drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: "#00CC00", lineWidth: 3 });
        drawLandmarks(ctx, results.rightHandLandmarks, { color: "#FF0000", lineWidth: 1.5 });
      }
      ctx.restore();

      // Send keypoints
      if (mpReadyRef.current && socketRef.current?.readyState === WebSocket.OPEN) {
        const keypoints = extractKeypoints(results);
        socketRef.current.send(JSON.stringify({
          type: "keypoints",
          data: keypoints,
          hands_visible: handsVisible
        }));
      }
    });

    holisticRef.current = holistic;

    // Frame processing loop - HIGHER FPS for better accuracy
    const processFrame = async (now) => {
      if (!isComponentMounted.current || !holisticRef.current) return;
      if (!mediaElement || mediaElement.ended) {
        frameRequestRef.current = requestAnimationFrame(processFrame);
        return;
      }

      const skip = mediaElement.paused && mpReadyRef.current;
      if (!skip && now - lastFrameTime.current >= FRAME_INTERVAL_MS) {
        lastFrameTime.current = now;
        if (mediaElement.readyState >= 2) {
          try { 
            await holisticRef.current.send({ image: mediaElement }); 
          } catch (e) { 
            if (!e.message?.includes("deleted object")) console.warn(e); 
          }
        }
      }
      frameRequestRef.current = requestAnimationFrame(processFrame);
    };
    
    frameRequestRef.current = requestAnimationFrame(processFrame);
    return true;
  }, [showMessage]);

  // ── Turn everything off ───────────────────────────────────────────────────
  const turnOffSystem = useCallback(() => {
    isStartingRef.current = false;
    isVideoProcessingRef.current = false;
    mpReadyRef.current = false;

    if (frameRequestRef.current) { 
      cancelAnimationFrame(frameRequestRef.current); 
      frameRequestRef.current = null; 
    }
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach(t => t.stop());
      cameraStreamRef.current = null;
    }
    if (holisticRef.current) { 
      holisticRef.current.close(); 
      holisticRef.current = null; 
    }
    if (socketRef.current) { 
      socketRef.current.close(); 
      socketRef.current = null; 
    }

    if (videoRef.current) {
      videoRef.current.pause();
      if (videoEndedHandlerRef.current) {
        videoRef.current.removeEventListener("ended", videoEndedHandlerRef.current);
        videoEndedHandlerRef.current = null;
      }
      videoRef.current.srcObject = null;
      videoRef.current.src = "";
    }
    if (uploadedVideoUrlRef.current) { 
      URL.revokeObjectURL(uploadedVideoUrlRef.current); 
      uploadedVideoUrlRef.current = null; 
    }

    setIsStreaming(false); 
    setIsProcessingVideo(false); 
    setIsRecordingSign(false);
    setIsAiLoading(false); 
    setFramesCollected(0); 
    setHandDetected(false); 
    setMpReady(false);
    setRecognizedText(""); 
    setEnglishSentence("");
    setLastConfidence(0);
    lastWordRef.current = "";
    lastTranslationRef.current = "";
    lastSignsRef.current = "";
  }, []);

  // ── Live camera ───────────────────────────────────────────────────────────
  const startCameraAndWS = useCallback(() => {
    if (isStartingRef.current) return;
    if (!window.Holistic) { 
      showMessage("AI libraries still loading.", "warning"); 
      return; 
    }
    if (!videoRef.current) { 
      showMessage("Camera element not ready.", "error"); 
      return; 
    }

    isStartingRef.current = true;
    mpReadyRef.current = false;
    setMpReady(false); 
    setIsAiLoading(true);
    lastTranslationRef.current = "";
    lastSignsRef.current = "";

    setupWebSocket(() => {
      // Higher resolution for better detection
      navigator.mediaDevices.getUserMedia({ 
        // 640x480 is a good balance for accuracy and performance; can go up to 1280x720 if needed
        video: { width: { ideal: 480 }, height: { ideal: 360 } } 
      })
        .then(stream => {
          if (!isComponentMounted.current) return;
          cameraStreamRef.current = stream;
          videoRef.current.srcObject = stream;
          videoRef.current.play();
          setIsStreaming(true); 
          setIsProcessingVideo(false);
          setIsAiLoading(false); 
          isStartingRef.current = false;
          showMessage("Camera on — warming up AI...", "info");
          startProcessing(videoRef.current, true, false);
        })
        .catch(err => {
          console.error(err); 
          showMessage("Camera access denied.", "error");
          turnOffSystem(); 
          isStartingRef.current = false;
        });
    });
  }, [showMessage, setupWebSocket, startProcessing, turnOffSystem]);

  // ── Video upload ──────────────────────────────────────────────────────────
  const handleVideoUpload = useCallback((event) => {
    const file = event.target.files[0];
    if (!file) return;

    turnOffSystem();
    mpReadyRef.current = false;
    setMpReady(false); 
    setIsAiLoading(true);
    lastTranslationRef.current = "";
    lastSignsRef.current = "";

    const url = URL.createObjectURL(file);
    uploadedVideoUrlRef.current = url;
    const vid = videoRef.current;
    if (!vid) return;

    vid.srcObject = null;
    vid.src = url;
    vid.muted = true;
    vid.load();

    vid.addEventListener("loadeddata", async () => {
      if (!isComponentMounted.current) return;

      vid.currentTime = 0;
      vid.pause();

      setupWebSocket(() => {
        isVideoProcessingRef.current = true;
        setIsProcessingVideo(true); 
        setIsStreaming(false); 
        setIsAiLoading(false);

        showMessage("⏳ Loading AI models — video will start automatically...", "info");

        startProcessing(vid, false, true);

        const onEnded = () => {
          console.log("📹 Video ended → sending 'end' to backend");
          if (socketRef.current?.readyState === WebSocket.OPEN) {
            socketRef.current.send(JSON.stringify({ type: "end" }));
          }
          setIsRecordingSign(false);
          setIsTranslating(true);
        };
        vid.addEventListener("ended", onEnded, { once: true });
        videoEndedHandlerRef.current = onEnded;
      });
    }, { once: true });
  }, [turnOffSystem, setupWebSocket, startProcessing, showMessage]);

  // ── Stop video ────────────────────────────────────────────────────────────
  const stopVideoProcessing = useCallback(() => {
    isVideoProcessingRef.current = false;
    mpReadyRef.current = false;
    if (frameRequestRef.current) { 
      cancelAnimationFrame(frameRequestRef.current); 
      frameRequestRef.current = null; 
    }
    if (holisticRef.current) { 
      holisticRef.current.close(); 
      holisticRef.current = null; 
    }
    if (socketRef.current) { 
      socketRef.current.close(); 
      socketRef.current = null; 
    }
    const vid = videoRef.current;
    if (vid) {
      vid.pause();
      if (videoEndedHandlerRef.current) { 
        vid.removeEventListener("ended", videoEndedHandlerRef.current); 
        videoEndedHandlerRef.current = null; 
      }
      vid.src = ""; 
      vid.srcObject = null;
    }
    if (uploadedVideoUrlRef.current) { 
      URL.revokeObjectURL(uploadedVideoUrlRef.current); 
      uploadedVideoUrlRef.current = null; 
    }
    setIsProcessingVideo(false); 
    setIsRecordingSign(false);
    setIsAiLoading(false); 
    setHandDetected(false); 
    setMpReady(false);
  }, []);

  // ── Action handlers ───────────────────────────────────────────────────────
  const handleManualStop = () => {
    if (isRecordingSign && socketRef.current?.readyState === WebSocket.OPEN) {
      setIsRecordingSign(false); 
      setIsTranslating(true);
      socketRef.current.send(JSON.stringify({ type: "end" }));
    }
  };

  const handleReset = () => {
    setRecognizedText(""); 
    setEnglishSentence("");
    lastWordRef.current = "";
    lastTranslationRef.current = "";
    lastSignsRef.current = "";
    if (socketRef.current?.readyState === WebSocket.OPEN)
      socketRef.current.send(JSON.stringify({ type: "reset" }));
    showMessage("Buffers cleared", "success");
  };

  const handleTranslateNow = () => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: "translate" }));
      showMessage("Translating...", "info");
    } else showMessage("WebSocket not connected", "error");
  };

  const handleTabChange = (option) => {
    setSelected(option); 
    turnOffSystem(); 
    setIsTranslating(false); 
    setTextInput("");
    setRecognizedText("");
    setEnglishSentence("");
    lastWordRef.current = "";
    lastTranslationRef.current = "";
    lastSignsRef.current = "";
  };

  const handleConvertText = () => {
    if (!textInput.trim()) return;
    const words = textInput.trim().toLowerCase().replace(/[^a-z0-9\s-]/g, "").split(/\s+/);
    const paths = words.map(w => `/videos/${w}.mp4`);
    setVideoQueue(paths); 
    setCurrentVideo(paths[0]); 
    setIsTranslating(true);
  };

  // ── Style constants ───────────────────────────────────────────────────────
  const box = "w-full bg-white dark:bg-[#0f0c29]/70 rounded-3xl p-6 sm:p-10 shadow-2xl shadow-purple-900/10 dark:shadow-purple-900/30 border border-purple-500/20 backdrop-blur-lg transition-colors duration-500";
  const btn = "px-6 py-3 rounded-full flex items-center justify-center gap-2 text-white font-bold shadow-lg transition-all duration-300";
  const dis = "opacity-50 cursor-not-allowed hover:scale-100";

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="w-full bg-gray-50 dark:bg-[#0f0c29] py-28 px-4 sm:px-6 lg:px-20 min-h-screen transition-colors duration-500">

      <motion.div initial="hidden" whileInView="visible" variants={fadeUp} className="max-w-6xl mx-auto text-center mb-16">
        <h2 className="text-5xl sm:text-6xl font-extrabold mb-6 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] bg-clip-text text-transparent">
          Translation Center
        </h2>
        <p className="text-gray-700 dark:text-gray-400 text-lg sm:text-xl max-w-2xl mx-auto">
          Ultra-Fast Local AI Inference. Show your hands to auto-record!
        </p>
      </motion.div>

      <div className="flex justify-center gap-4 mb-16 flex-wrap">
        {options.map(o => (
          <button key={o} onClick={() => handleTabChange(o)}
            className={`px-8 py-3 rounded-full font-bold transition-all duration-300 text-lg ${
              selected === o
                ? "bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] scale-105 text-white"
                : "bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200"
            }`}>{o}</button>
        ))}
      </div>

      <motion.div key={selected} initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} className="max-w-5xl mx-auto">

        {/* ══════════════════════ SIGN TO TEXT ══════════════════════ */}
        {selected === "Sign to Text" && (
          <div className="flex flex-col items-center gap-6">
            <div className={box}>

              <div className="flex justify-between items-center flex-wrap gap-4 mb-6">
                <h3 className="text-2xl font-extrabold bg-gradient-to-r from-[#6A3093] to-[#BF5AE0] bg-clip-text text-transparent">
                  Sign Language AI Camera / Video
                </h3>
                <div className="flex gap-3 flex-wrap">
                  {!isStreaming && !isProcessingVideo && (
                    <>
                      <button onClick={startCameraAndWS} disabled={isAiLoading}
                        className={`${btn} bg-purple-600 hover:bg-purple-700 hover:scale-105 ${isAiLoading ? dis : ""}`}>
                        {isAiLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Camera className="w-5 h-5" />}
                        {isAiLoading ? "Loading AI..." : "Turn On AI Camera"}
                      </button>
                      <label className={`${btn} bg-blue-600 hover:bg-blue-700 hover:scale-105 cursor-pointer`}>
                        <Upload className="w-5 h-5" /> Upload Video
                        <input type="file" accept="video/*" onChange={handleVideoUpload} className="hidden" />
                      </label>
                    </>
                  )}
                  {(isStreaming || isProcessingVideo) && (
                    <button onClick={turnOffSystem} className={`${btn} bg-gray-500 hover:bg-gray-600 hover:scale-105`}>
                      <PowerOff className="w-5 h-5" /> Turn Off
                    </button>
                  )}
                </div>
              </div>

              {/* Video area */}
              <div className="relative w-full aspect-video rounded-xl overflow-hidden bg-black border-4 border-gray-800 mb-6 shadow-inner">

                <video ref={videoRef}
                  className={`absolute inset-0 w-full h-full object-cover ${isStreaming ? "scale-x-[-1]" : ""}`}
                  playsInline muted={isStreaming} autoPlay
                />

                {(isStreaming || isProcessingVideo) && (
                  <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover pointer-events-none z-10" />
                )}

                {!isStreaming && !isProcessingVideo && !isAiLoading && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <Video className="w-12 h-12 text-gray-600 mb-4" />
                    <p className="text-gray-500 text-sm">No active source</p>
                  </div>
                )}

                {/* WebSocket / camera connecting */}
                {isAiLoading && (
                  <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center text-white z-20 backdrop-blur-sm">
                    <Loader2 className="w-12 h-12 animate-spin text-purple-500 mb-4" />
                    <h3 className="text-xl font-bold">Connecting to AI...</h3>
                    <p className="text-sm text-gray-400 mt-2">This may take a few seconds on first load.</p>
                  </div>
                )}

                {/* MediaPipe WASM loading */}
                {isProcessingVideo && !isAiLoading && !mpReady && (
                  <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center text-white z-20 backdrop-blur-sm">
                    <Loader2 className="w-12 h-12 animate-spin text-yellow-400 mb-4" />
                    <h3 className="text-xl font-bold text-yellow-300">Loading AI Models...</h3>
                    <p className="text-sm text-gray-300 mt-2">Video is paused — will start automatically</p>
                    <div className="mt-4 w-48 h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-yellow-400 rounded-full animate-pulse" style={{ width: "65%" }} />
                    </div>
                  </div>
                )}

                {/* Hand badge */}
                {(isStreaming || isProcessingVideo) && !isAiLoading && mpReady && (
                  <div className={`absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-bold shadow-xl z-20 transition-colors ${
                    handDetected ? "bg-green-600 text-white" : "bg-gray-600 text-gray-300"
                  }`}>
                    <Hand className="w-4 h-4" />
                    {handDetected ? "Hands Detected" : "No Hands"}
                  </div>
                )}

                {/* Recording indicator */}
                {isRecordingSign && (
                  <div className="absolute top-4 right-4 flex items-center gap-2 text-white bg-red-600 px-4 py-2 rounded-full text-sm font-bold shadow-xl animate-pulse z-20">
                    <span className="w-3 h-3 rounded-full bg-white" />
                    RECORDING ({framesCollected} frames)
                  </div>
                )}

                {/* Confidence indicator */}
                {lastConfidence > 0 && !isRecordingSign && (
                  <div className="absolute bottom-4 left-4 bg-black/60 text-white px-2 py-1 rounded text-xs z-20">
                    Conf: {Math.round(lastConfidence)}%
                  </div>
                )}

                {isTranslating && (
                  <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center text-white z-20">
                    <Loader2 className="w-12 h-12 animate-spin text-purple-500 mb-4" />
                    <h3 className="text-xl font-bold">AI is predicting your sign...</h3>
                  </div>
                )}
              </div>

              {(isStreaming || isProcessingVideo) && (
                <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-6 border border-gray-300 dark:border-gray-700 text-center mb-4">
                  <div className="flex items-center justify-center gap-3 text-gray-700 dark:text-gray-300">
                    <Hand className="w-5 h-5 text-purple-500" />
                    <p className="text-sm font-medium">
                      <span className="text-green-600 font-bold">Auto-sense active:</span>{" "}
                      Show hands to start recording, hide to predict
                    </p>
                  </div>
                  {isRecordingSign && (
                    <button onClick={handleManualStop}
                      className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-full text-sm font-bold flex items-center gap-2 mx-auto">
                      <Square className="w-4 h-4" /> Manual Stop
                    </button>
                  )}
                  {isProcessingVideo && !isRecordingSign && (
                    <button onClick={stopVideoProcessing}
                      className="mt-4 px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-full text-sm font-bold flex items-center gap-2 mx-auto">
                      <StopCircle className="w-4 h-4" /> Stop Video Processing
                    </button>
                  )}
                </div>
              )}

              <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Detected Signs Column */}
                <div>
                  <div className="flex justify-between items-end mb-2 border-b border-purple-500/50 pb-2">
                    <h4 className="text-xl font-semibold text-gray-800 dark:text-white">Detected Signs:</h4>
                    <div className="flex gap-2">
                      <button onClick={() => speakText(recognizedText)} title="Speak"
                        className="text-purple-600 hover:text-purple-800 dark:text-purple-400 dark:hover:text-purple-300 transition-colors">
                        <Volume2 className="w-5 h-5" />
                      </button>
                      <button onClick={handleReset}
                        className="text-sm text-red-500 hover:text-red-700 flex items-center gap-1 font-semibold">
                        <Trash2 className="w-4 h-4" /> Reset
                      </button>
                    </div>
                  </div>
                  <div className={`w-full min-h-32 p-4 rounded-xl border-2 border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800 shadow-inner text-xl tracking-wide transition-colors ${
                    recognizedText ? "text-purple-600 dark:text-pink-300 font-bold" : "text-gray-500 dark:text-gray-500 italic"
                  }`}>
                    {recognizedText || "Sign sequence will appear here..."}
                  </div>
                </div>

                {/* English Translation Column */}
                <div>
                  <div className="flex justify-between items-end mb-2 border-b border-purple-500/50 pb-2">
                    <h4 className="text-xl font-semibold text-gray-800 dark:text-white">English Translation:</h4>
                    <div className="flex gap-2">
                      <button onClick={() => speakText(englishSentence)} title="Speak"
                        className="text-purple-600 hover:text-purple-800 dark:text-purple-400 dark:hover:text-purple-300 transition-colors">
                        <Volume2 className="w-5 h-5" />
                      </button>
                      <button onClick={handleTranslateNow}
                        className="text-sm bg-purple-600 hover:bg-purple-700 text-white px-3 py-1 rounded-full flex items-center gap-1">
                        <Volume2 className="w-4 h-4" /> Translate
                      </button>
                    </div>
                  </div>
                  <div className={`w-full min-h-32 p-4 rounded-xl border-2 border-purple-400 dark:border-purple-600 bg-purple-50 dark:bg-purple-900/20 shadow-inner text-xl tracking-wide transition-colors ${
                    englishSentence ? "text-purple-700 dark:text-purple-300 font-bold" : "text-gray-500 dark:text-gray-500 italic"
                  }`}>
                    {englishSentence || "Grammatically correct sentence will appear here..."}
                  </div>
                </div>
              </div>

            </div>
          </div>
        )}

        {/* ══════════════════════ TEXT TO SIGN ══════════════════════ */}
        {selected === "Text / Audio to Sign" && (
          <div className="flex flex-col items-center gap-10">
            <div className={box}>
              <h3 className="text-3xl font-extrabold text-center mb-10 bg-gradient-to-r from-purple-500 to-pink-400 bg-clip-text text-transparent">
                Text / Audio → Sign Language Avatar
              </h3>
              <textarea rows={4} value={textInput} onChange={e => setTextInput(e.target.value)}
                placeholder="Type your message here to see it signed by the avatar..."
                disabled={isTranslating}
                className={`w-full p-5 rounded-xl border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-200 shadow-md focus:outline-none focus:ring-2 focus:ring-purple-500 transition resize-none text-lg ${isTranslating ? dis : ""}`}
              />
              <div className="flex justify-center items-center gap-8 mt-6">
                <button onClick={() => showMessage("Audio coming soon", "info")} disabled={isTranslating}
                  className={`w-16 h-16 rounded-full flex items-center justify-center text-white shadow-xl bg-purple-600 hover:bg-purple-700 ${isTranslating ? dis : ""}`}>
                  <Mic className="text-2xl" />
                </button>
                <div className="text-2xl font-bold text-gray-500 dark:text-gray-400">OR</div>
                <button onClick={handleConvertText} disabled={isTranslating || !textInput.trim()}
                  className={`${btn} bg-gradient-to-r from-[#6A3093] to-[#BF5AE0] ${(isTranslating || !textInput.trim()) ? dis : "hover:scale-105"}`}>
                  {isTranslating
                    ? <><Loader2 className="w-5 h-5 animate-spin" /> Generating Signs...</>
                    : <><RefreshCw className="w-5 h-5" /> Convert to Sign</>}
                </button>
              </div>
              <div className="mt-10 w-full h-64 rounded-2xl border border-gray-300 dark:border-purple-600/50 bg-gray-100 dark:bg-gray-800 shadow-inner flex flex-col items-center justify-center text-gray-500 dark:text-gray-400 text-center text-lg tracking-wide">
                {currentVideo ? (
                  <video key={currentVideo} src={currentVideo} autoPlay muted
                    className="rounded-lg w-full h-full object-contain mb-3"
                    onEnded={() => {
                      const next = videoQueue.indexOf(currentVideo) + 1;
                      next < videoQueue.length ? setCurrentVideo(videoQueue[next])
                        : (setCurrentVideo(null), setVideoQueue([]), setIsTranslating(false));
                    }}
                    onError={() => {
                      showMessage("No sign available for that word", "warning");
                      const next = videoQueue.indexOf(currentVideo) + 1;
                      next < videoQueue.length ? setCurrentVideo(videoQueue[next])
                        : (setCurrentVideo(null), setVideoQueue([]), setIsTranslating(false));
                    }}
                  />
                ) : (
                  <>
                    <img src="https://placehold.co/150x200/4c3093/ffffff?text=3D+Avatar"
                      alt="Sign Language Avatar" className="rounded-lg mb-3" />
                    <p>No sign available or translation finished</p>
                  </>
                )}
                {isTranslating && (
                  <p className="mt-2 text-purple-500 flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" /> Animating...
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

      </motion.div>

      <AnimatePresence>
        {message && <Toast message={message.text} type={message.type} />}
      </AnimatePresence>
    </div>
  );
}