import React, { useState, useEffect, useRef, useContext, useCallback, useMemo } from "react";
import { useTheme } from "../context/ThemeContext";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic, Camera, RefreshCw, Loader2, Video, Volume2, Trash2,
  Play, Square, PowerOff, Hand, Upload, StopCircle, Sparkles,
  Copy, Check, Send
} from "lucide-react";
import { FaCrown } from "react-icons/fa";
import { TbSparkles } from "react-icons/tb";

// ─── Toast ────────────────────────────────────────────────────────────────────
const Toast = ({ message, type }) => {
  if (!message) return null;
  const base = "fixed bottom-5 left-1/2 transform -translate-x-1/2 p-4 rounded-xl shadow-2xl z-[100] flex items-center gap-3 font-bold text-white backdrop-blur-xl";
  const color =
    type === "success" ? "bg-gradient-to-r from-green-600 to-emerald-600 shadow-green-500/50" :
    type === "warning" ? "bg-gradient-to-r from-yellow-600 to-amber-600 shadow-yellow-500/50" :
    type === "error"   ? "bg-gradient-to-r from-red-600 to-rose-600 shadow-red-500/50" :
                         "bg-gradient-to-r from-primary-600 to-pink-600 shadow-primary-500/50";
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
const WS_URL = "ws://localhost:8001/ws/sign-to-text";

// ─────────────────────────────────────────────────────────────────────────────
export default function Translate() {
  const { themeColor } = useTheme();
  const options = ["Sign to Text", "Text / Audio to Sign"];

  // ── UI state ──────────────────────────────────────────────────────────────
  const [selected, setSelected] = useState(options[0]);
  const [textInput, setTextInput] = useState("");
  const [recognizedText, setRecognizedText] = useState("");
  const [englishSentence, setEnglishSentence] = useState("");
  const [isTranslating, setIsTranslating] = useState(false);
  const [message, setMessage] = useState(null);
  const [videoQueue, setVideoQueue] = useState([]);
  const [currentVideo, setCurrentVideo] = useState(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [isProcessingVideo, setIsProcessingVideo] = useState(false);
  const [isRecordingSign, setIsRecordingSign] = useState(false);
  const [isAiLoading, setIsAiLoading] = useState(false);
  const [framesCollected, setFramesCollected] = useState(0);
  const [handDetected, setHandDetected] = useState(false);
  const [mpReady, setMpReady] = useState(false);
  const [lastConfidence, setLastConfidence] = useState(0);
  const [copied, setCopied] = useState(false);

  // Canvas ref for particle background
  const canvasRefBg = useRef(null);
  const particlesRef = useRef([]);
  const animationFrameRef = useRef(null);
  const [isDark, setIsDark] = useState(
    document.documentElement.classList.contains("dark")
  );

  // ── Refs for deduplication ─────────────────────────────────────────────────
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const socketRef = useRef(null);
  const holisticRef = useRef(null);
  const cameraStreamRef = useRef(null);
  const frameRequestRef = useRef(null);
  const isComponentMounted = useRef(true);
  const isStartingRef = useRef(false);
  const lastFrameTime = useRef(0);
  const uploadedVideoUrlRef = useRef(null);
  const isVideoProcessingRef = useRef(false);
  const videoEndedHandlerRef = useRef(null);
  const mpReadyRef = useRef(false);
  const updateTimerRef = useRef(null);
  const lastWordRef = useRef("");
  const lastTranslationRef = useRef("");
  const lastSignsRef = useRef("");
  const processingSentenceRef = useRef(false);

  const FRAME_INTERVAL_MS = 50;

  // ── Detect dark mode ──────────────────────────────────────────────────────
  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains("dark"));
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });

    return () => observer.disconnect();
  }, []);

  // ── Particle system background ───────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRefBg.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const themeColorsMap = {
      purple: ['#A855F7', '#9333EA', '#7C3AED', '#6D28D9', '#8B5CF6'],
      'midnight-blue': ['#6366F1', '#4F46E5', '#4338CA', '#3730A3', '#818CF8'],
    };
    const currentThemeColors = themeColorsMap[themeColor] || themeColorsMap['purple'];
    const colors = isDark ? currentThemeColors : currentThemeColors.slice().reverse();

    particlesRef.current = Array.from({ length: 80 }).map(() => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      size: Math.random() * 3 + 0.5,
      speedX: Math.random() * 0.3 - 0.15,
      speedY: Math.random() * 0.3 - 0.15,
      color: colors[Math.floor(Math.random() * colors.length)],
      opacity: Math.random() * 0.4 + 0.1,
      glow: Math.random() > 0.7,
    }));

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particlesRef.current.forEach(particle => {
        particle.x += particle.speedX;
        particle.y += particle.speedY;

        if (particle.x < -10) particle.x = canvas.width + 10;
        if (particle.x > canvas.width + 10) particle.x = -10;
        if (particle.y < -10) particle.y = canvas.height + 10;
        if (particle.y > canvas.height + 10) particle.y = -10;

        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        
        if (particle.glow) {
          const glowGradient = ctx.createRadialGradient(
            particle.x, particle.y, 0,
            particle.x, particle.y, particle.size * 3
          );
          glowGradient.addColorStop(0, particle.color + '99');
          glowGradient.addColorStop(1, particle.color + '00');
          ctx.fillStyle = glowGradient;
        } else {
          ctx.fillStyle = particle.color + Math.floor(particle.opacity * 255).toString(16).padStart(2, '0');
        }
        
        ctx.fill();

        particlesRef.current.forEach(otherParticle => {
          const dx = particle.x - otherParticle.x;
          const dy = particle.y - otherParticle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 80) {
            ctx.beginPath();
            ctx.strokeStyle = particle.color + '33';
            ctx.lineWidth = 0.4 * (1 - distance / 80);
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(otherParticle.x, otherParticle.y);
            ctx.stroke();
          }
        });
      });

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    const handleResize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };

    window.addEventListener('resize', handleResize);
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      window.removeEventListener('resize', handleResize);
    };
  }, [isDark, themeColor]);

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

  const copyToClipboard = useCallback((text) => {
    if (!text || text.includes("will appear here")) {
      showMessage("Nothing to copy", "warning");
      return;
    }
    navigator.clipboard.writeText(text);
    setCopied(true);
    showMessage("Copied to clipboard!", "success");
    setTimeout(() => setCopied(false), 2000);
  }, [showMessage]);

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
    setTimeout(() => { processingSentenceRef.current = false; }, 500);
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
    
    const ws = new WebSocket(WS_URL);
    
    ws.onopen = () => { 
      if (!isComponentMounted.current) return; 
      socketRef.current = ws; 
      console.log("🔌 WebSocket connected");
      onOpenCallback(); 
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("📨 AI Response:", data.status);
        
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
      } catch (e) { console.error("WS parse error:", e); }
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

  // ── MediaPipe processing ──────────────────────────────────────────────────
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
    
    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      refineFaceLandmarks: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
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

      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (isMirrored) { 
        ctx.translate(canvas.width, 0); 
        ctx.scale(-1, 1); 
      }

      if (results.poseLandmarks) {
        drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: "#A855F7", lineWidth: 2 });
        drawLandmarks(ctx, results.poseLandmarks, { color: "#C084FC", lineWidth: 1 });
      }
      if (results.faceLandmarks) {
        RELEVANT_FACE_INDICES.forEach(i => {
          const lm = results.faceLandmarks[i]; 
          if (!lm) return;
          ctx.beginPath();
          ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 1.5, 0, 2 * Math.PI);
          ctx.fillStyle = "#A855F7";
          ctx.fill();
        });
      }
      if (results.leftHandLandmarks) {
        drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: "#A855F7", lineWidth: 3 });
        drawLandmarks(ctx, results.leftHandLandmarks, { color: "#C084FC", lineWidth: 1.5 });
      }
      if (results.rightHandLandmarks) {
        drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: "#9333EA", lineWidth: 3 });
        drawLandmarks(ctx, results.rightHandLandmarks, { color: "#A855F7", lineWidth: 1.5 });
      }
      ctx.restore();

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
          try { await holisticRef.current.send({ image: mediaElement }); } 
          catch (e) { if (!e.message?.includes("deleted object")) console.warn(e); }
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
      navigator.mediaDevices.getUserMedia({ 
        video: { width: { ideal: 640 }, height: { ideal: 480 } } 
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

  // ── Style constants ──────────────────────────────────────────────────────
  const box = "w-full bg-white/80 dark:bg-white/5 rounded-3xl p-6 sm:p-10 shadow-2xl shadow-primary-900/10 dark:shadow-primary-900/30 border-2 border-primary-200/50 dark:border-primary-800/50 backdrop-blur-xl transition-all duration-500";
  const btn = "px-6 py-3 rounded-full flex items-center justify-center gap-2 text-white font-bold shadow-lg transition-all duration-300";
  const dis = "opacity-50 cursor-not-allowed hover:scale-100";

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="relative w-full bg-gradient-to-br from-gray-50 via-white to-primary-50/60 dark:from-primary-bg-1 dark:via-primary-bg-2 dark:to-primary-bg-3 py-28 px-4 sm:px-6 lg:px-20 min-h-screen transition-all duration-700">
      
      {/* Premium Canvas Particles */}
      <canvas
        ref={canvasRefBg}
        className="absolute inset-0 w-full h-full pointer-events-none opacity-60"
      />

      {/* Premium Geometric Grid */}
      <div className="absolute inset-0 opacity-40 dark:opacity-60 pointer-events-none">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(90deg, rgba(168, 85, 247, 0.1) 1px, transparent 1px),
            linear-gradient(180deg, rgba(168, 85, 247, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px'
        }} />
      </div>

      {/* Animated gradient orbs */}
      <motion.div
        className="absolute top-20 left-20 w-[500px] h-[500px] bg-primary-600/10 rounded-full blur-[120px]"
        animate={{
          x: [0, 60, 0],
          y: [0, -60, 0],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "linear",
        }}
      />
      <motion.div
        className="absolute bottom-20 right-20 w-[500px] h-[500px] bg-primary-400/10 rounded-full blur-[120px]"
        animate={{
          x: [0, -60, 0],
          y: [0, 60, 0],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "linear",
        }}
      />

      {/* Header */}
      <div className="relative z-10 max-w-6xl mx-auto text-center mb-16">
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          whileHover={{ scale: 1.02 }}
          className="inline-flex items-center gap-3 px-6 py-3 rounded-full bg-gradient-to-r from-primary-500/20 via-primary-400/10 to-primary-300/20 border-2 border-primary-300/40 dark:border-primary-600/40 backdrop-blur-xl shadow-2xl shadow-primary-500/20 relative overflow-hidden group mb-8"
        >
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
            className="p-1 rounded-full bg-gradient-to-r from-primary-500 to-primary-400"
          >
            <FaCrown className="text-white text-sm" />
          </motion.div>
          <span className="text-sm font-extrabold bg-gradient-to-r from-primary-700 via-primary-600 to-primary-500 dark:from-primary-400 dark:via-primary-300 dark:to-primary-200 bg-clip-text text-transparent">
            AI-POWERED TRANSLATION
          </span>
          <TbSparkles className="text-primary-500 text-lg" />
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
        </motion.div>

        <h2 className="text-5xl sm:text-6xl font-black mb-6">
          <span className="block text-gray-900 dark:text-white">Translation</span>
          <span className="block bg-gradient-to-r from-primary-700 via-primary-500 to-primary-400 bg-clip-text text-transparent animate-gradient">
            Center
          </span>
        </h2>
        <p className="text-gray-700 dark:text-gray-400 text-lg sm:text-xl max-w-2xl mx-auto">
          Ultra-Fast Local AI Inference. Show your hands to auto-record!
        </p>

        <div className="flex items-center justify-center gap-8 mt-10">
          <div className="w-12 h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent rounded-full" />
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            className="w-6 h-6 rounded-full border-2 border-primary-400/50"
          />
          <div className="w-12 h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent rounded-full" />
        </div>
      </div>

      {/* Tab Buttons */}
      <div className="relative z-10 flex justify-center gap-4 mb-16 flex-wrap">
        {options.map(o => (
          <button key={o} onClick={() => handleTabChange(o)}
            className={`px-8 py-3 rounded-full font-bold transition-all duration-300 text-lg ${
              selected === o
                ? "bg-gradient-to-r from-primary-custom-1 via-primary-custom-2 to-primary-custom-3 scale-105 text-white shadow-lg shadow-primary-500/40"
                : "bg-white/70 dark:bg-white/5 border-2 border-primary-200/50 dark:border-primary-800/50 text-gray-700 dark:text-gray-300 hover:border-primary-500 dark:hover:border-primary-600"
            }`}>{o}</button>
        ))}
      </div>

      <motion.div key={selected} initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} className="relative z-10 max-w-6xl mx-auto">

        {/* Sign to Text Tab */}
        {selected === "Sign to Text" && (
          <div className="flex flex-col items-center gap-6">
            <div className={box}>
              <div className="flex justify-between items-center flex-wrap gap-4 mb-6">
                <h3 className="text-2xl font-black bg-gradient-to-r from-primary-custom-1 to-primary-custom-3 bg-clip-text text-transparent">
                  Sign Language AI Camera / Video
                </h3>
                <div className="flex gap-3 flex-wrap">
                  {!isStreaming && !isProcessingVideo && (
                    <>
                      <button onClick={startCameraAndWS} disabled={isAiLoading}
                        className={`${btn} bg-gradient-to-r from-primary-600 to-primary-700 hover:from-primary-700 hover:to-primary-800 hover:scale-105 shadow-lg shadow-primary-500/40 ${isAiLoading ? dis : ""}`}>
                        {isAiLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Camera className="w-5 h-5" />}
                        {isAiLoading ? "Loading AI..." : "Turn On AI Camera"}
                      </button>
                      <label className={`${btn} bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 hover:scale-105 cursor-pointer shadow-lg shadow-blue-500/40`}>
                        <Upload className="w-5 h-5" /> Upload Video
                        <input type="file" accept="video/*" onChange={handleVideoUpload} className="hidden" />
                      </label>
                    </>
                  )}
                  {(isStreaming || isProcessingVideo) && (
                    <button onClick={turnOffSystem} className={`${btn} bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-700 hover:to-gray-800 hover:scale-105`}>
                      <PowerOff className="w-5 h-5" /> Turn Off
                    </button>
                  )}
                </div>
              </div>

              {/* Video Area */}
              <div className="relative w-full aspect-video rounded-xl overflow-hidden border-2 border-primary-500/30 bg-black shadow-2xl mb-6">
                <video ref={videoRef}
                  className={`absolute inset-0 w-full h-full object-cover ${isStreaming ? "scale-x-[-1]" : ""}`}
                  playsInline muted={isStreaming} autoPlay
                />
                {(isStreaming || isProcessingVideo) && (
                  <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover pointer-events-none z-10" />
                )}

                {!isStreaming && !isProcessingVideo && !isAiLoading && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-gradient-to-br from-gray-900/90 to-primary-900/90">
                    <Video className="w-16 h-16 text-primary-400 mb-4" />
                    <p className="text-gray-300 text-sm">No active source</p>
                  </div>
                )}

                {isAiLoading && (
                  <div className="absolute inset-0 bg-black/90 flex flex-col items-center justify-center text-white z-20">
                    <Loader2 className="w-16 h-16 animate-spin text-primary-500 mb-4" />
                    <h3 className="text-xl font-bold">Connecting to AI...</h3>
                  </div>
                )}

                {(isStreaming || isProcessingVideo) && !isAiLoading && mpReady && (
                  <div className={`absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-bold shadow-xl z-20 ${
                    handDetected ? "bg-green-600 text-white" : "bg-gray-600 text-gray-300"
                  }`}>
                    <Hand className="w-4 h-4" />
                    {handDetected ? "Hands Detected" : "No Hands"}
                  </div>
                )}

                {isRecordingSign && (
                  <div className="absolute top-4 right-4 flex items-center gap-2 text-white bg-red-600 px-4 py-2 rounded-full text-sm font-bold shadow-xl animate-pulse z-20">
                    <span className="w-3 h-3 rounded-full bg-white animate-ping" />
                    RECORDING ({framesCollected} frames)
                  </div>
                )}

                {isTranslating && (
                  <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center text-white z-20">
                    <Loader2 className="w-16 h-16 animate-spin text-primary-500 mb-4" />
                    <h3 className="text-xl font-bold">AI is predicting your sign...</h3>
                  </div>
                )}
              </div>

              {(isStreaming || isProcessingVideo) && (
                <div className="bg-primary-500/10 rounded-xl p-6 border-2 border-primary-500/20 text-center mb-4">
                  <p className="text-sm font-medium">
                    <span className="text-primary-600 font-bold">Auto-sense active:</span> Show hands to start recording, hide to predict
                  </p>
                  {isRecordingSign && (
                    <button onClick={handleManualStop} className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-full text-sm font-bold">
                      <Square className="w-4 h-4 inline mr-2" /> Manual Stop
                    </button>
                  )}
                </div>
              )}

              {/* Results Section - Side by Side Layout */}
              <div className="mt-8">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Detected Signs Box */}
                  <div className="bg-gradient-to-br from-gray-50/80 to-gray-100/50 dark:from-gray-800/30 dark:to-gray-900/20 rounded-2xl border border-primary-200/50 dark:border-primary-500/20 overflow-hidden transition-all duration-300 hover:shadow-xl">
                    <div className="bg-gradient-to-r from-primary-500/10 to-primary-600/5 px-6 py-4 border-b border-primary-200/50 dark:border-primary-500/20">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-lg">
                            <Hand className="w-5 h-5 text-white" />
                          </div>
                          <div>
                            <h4 className="text-lg font-bold text-gray-800 dark:text-white">Detected Signs</h4>
                            <p className="text-xs text-gray-500 dark:text-gray-400">Real-time sign sequence</p>
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <button 
                            onClick={() => speakText(recognizedText)} 
                            className="p-2 rounded-lg bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 hover:bg-primary-200 dark:hover:bg-primary-800/50 transition-all"
                            title="Speak"
                          >
                            <Volume2 className="w-4 h-4" />
                          </button>
                          <button 
                            onClick={() => copyToClipboard(recognizedText)} 
                            className="p-2 rounded-lg bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 hover:bg-primary-200 dark:hover:bg-primary-800/50 transition-all"
                            title="Copy"
                          >
                            {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                          </button>
                          <button 
                            onClick={handleReset}
                            className="p-2 rounded-lg bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 hover:bg-primary-200 dark:hover:bg-primary-800/50 transition-all"
                            title="Clear"
                          >
                            <RefreshCw className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                    <div className="p-6 min-h-[160px]">
                      <div className="text-gray-700 dark:text-gray-300 leading-relaxed break-words">
                        {recognizedText || (
                          <span className="text-gray-400 dark:text-gray-500 italic">
                            Sign sequence will appear here...
                          </span>
                        )}
                      </div>
                      {lastConfidence > 0 && recognizedText && (
                        <div className="mt-3 flex items-center gap-2">
                          <div className="flex-1 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-gradient-to-r from-green-500 to-emerald-500 rounded-full transition-all duration-300"
                              style={{ width: `${lastConfidence}%` }}
                            />
                          </div>
                          <span className="text-xs text-gray-500">{Math.round(lastConfidence)}% confidence</span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* English Translation Box */}
                  <div className="bg-gradient-to-br from-primary-50/80 to-primary-100/50 dark:from-primary-900/20 dark:to-primary-800/10 rounded-2xl border border-primary-300/50 dark:border-primary-500/30 overflow-hidden transition-all duration-300 hover:shadow-xl">
                    <div className="bg-gradient-to-r from-primary-600/10 to-primary-500/5 px-6 py-4 border-b border-primary-300/50 dark:border-primary-500/30">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-600 to-primary-500 flex items-center justify-center shadow-lg">
                            <Sparkles className="w-5 h-5 text-white" />
                          </div>
                          <div>
                            <h4 className="text-lg font-bold text-gray-800 dark:text-white">English Translation</h4>
                            <p className="text-xs text-gray-500 dark:text-gray-400">Grammatically corrected</p>
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <button 
                            onClick={() => speakText(englishSentence)} 
                            className="p-2 rounded-lg bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 hover:bg-primary-200 dark:hover:bg-primary-800/50 transition-all"
                            title="Speak"
                          >
                            <Volume2 className="w-4 h-4" />
                          </button>
                          <button 
                            onClick={() => copyToClipboard(englishSentence)} 
                            className="p-2 rounded-lg bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 hover:bg-primary-200 dark:hover:bg-primary-800/50 transition-all"
                            title="Copy"
                          >
                            <Copy className="w-4 h-4" />
                          </button>
                          <button 
                            onClick={handleTranslateNow}
                            className="p-2 rounded-lg bg-primary-500 text-white hover:bg-primary-600 transition-all"
                            title="Translate Now"
                          >
                            <Send className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                    <div className="p-6 min-h-[160px]">
                      <div className="text-gray-700 dark:text-gray-300 leading-relaxed break-words">
                        {englishSentence || (
                          <span className="text-gray-400 dark:text-gray-500 italic">
                            Grammatically correct sentence will appear here...
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Text to Sign Tab */}
        {selected === "Text / Audio to Sign" && (
          <div className="flex flex-col items-center gap-10">
            <div className={box}>
              <h3 className="text-3xl font-black text-center mb-10 bg-gradient-to-r from-primary-700 via-primary-500 to-primary-400 bg-clip-text text-transparent">
                Text / Audio → Sign Language Avatar
              </h3>
              
              {/* Input and Output Side by Side */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Input Section */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Enter your message:
                  </label>
                  <textarea 
                    rows={6} 
                    value={textInput} 
                    onChange={e => setTextInput(e.target.value)}
                    placeholder="Type your message here to see it signed by the avatar..."
                    className="w-full p-5 rounded-xl border-2 border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800/50 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none text-lg"
                  />
                  <div className="flex justify-center gap-4 mt-6">
                    <button onClick={() => showMessage("Audio coming soon", "info")}
                      className="w-16 h-16 rounded-full flex items-center justify-center text-white bg-gradient-to-r from-primary-600 to-pink-600 hover:scale-110 transition-all shadow-lg">
                      <Mic className="text-2xl" />
                    </button>
                    <div className="text-2xl font-bold text-gray-500 self-center">OR</div>
                    <button onClick={handleConvertText} disabled={!textInput.trim()}
                      className={`px-8 py-3 rounded-full font-bold text-white bg-gradient-to-r from-primary-custom-1 via-primary-custom-2 to-primary-custom-3 transition-all flex items-center gap-2 shadow-lg ${!textInput.trim() ? dis : "hover:scale-105"}`}>
                      <RefreshCw className="w-5 h-5" /> Convert to Sign
                    </button>
                  </div>
                </div>

                {/* Output Section - Avatar Display */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Sign Language Avatar:
                  </label>
                  <div className="w-full h-80 rounded-2xl border-2 border-primary-500/30 bg-gray-100 dark:bg-gray-800/50 flex flex-col items-center justify-center overflow-hidden shadow-inner">
                    {currentVideo ? (
                      <video key={currentVideo} src={currentVideo} autoPlay muted className="w-full h-full object-contain" />
                    ) : (
                      <div className="text-center">
                        <Hand className="w-16 h-16 text-primary-500 mx-auto mb-4 opacity-50" />
                        <p className="text-gray-500">No sign available or translation finished</p>
                        <p className="text-xs text-gray-400 mt-2">Type a message above and click "Convert to Sign"</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

      </motion.div>

      <AnimatePresence>
        {message && <Toast message={message.text} type={message.type} />}
      </AnimatePresence>

      <style jsx>{`
        @keyframes gradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .animate-gradient {
          background-size: 200% auto;
          animation: gradient 3s linear infinite;
        }
      `}</style>
    </div>
  );
}
