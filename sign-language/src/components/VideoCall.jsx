import React, { useRef, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  BsCameraVideo, 
  BsCameraVideoOff, 
  BsMic, 
  BsMicMute, 
  BsTelephoneX,
  BsArrowLeft,
  BsFullscreen,
  BsFullscreenExit,
  BsRecordCircle,
  BsStopCircle,
  BsTranslate,
  BsRobot,
  BsX,
  BsTrash,
  BsVolumeUp
} from "react-icons/bs";

const VideoCall = ({ 
  isOpen, 
  onClose, 
  remoteUsername, 
  localUsername,
  callType = "video",
  ws,
  isCaller = false,
  callId,
  incomingOffer
}) => {
  // Existing states (unchanged)
  const [localStream, setLocalStream] = useState(null);
  const [remoteStream, setRemoteStream] = useState(null);
  const [isVideoEnabled, setIsVideoEnabled] = useState(true);
  const [isAudioEnabled, setIsAudioEnabled] = useState(true);
  const [isConnecting, setIsConnecting] = useState(true);
  const [connectionError, setConnectionError] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [callDuration, setCallDuration] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [remoteVideoMuted, setRemoteVideoMuted] = useState(false);
  const [remoteAudioMuted, setRemoteAudioMuted] = useState(false);
  
  // AI Translation states
  const [isAiEnabled, setIsAiEnabled] = useState(false);
  const [aiGlossText, setAiGlossText] = useState("");
  const [aiEnglishText, setAiEnglishText] = useState("");
  const [aiConfidence, setAiConfidence] = useState(0);
  const [isAiRecording, setIsAiRecording] = useState(false);
  const [isAiLoading, setIsAiLoading] = useState(false);
  const isAiEnabledRef = useRef(false);
  
  // Refs for WebRTC
  const localVideoRef = useRef(null);
  const remoteVideoRef = useRef(null);
  const peerConnectionRef = useRef(null);
  const durationIntervalRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  
  // AI Refs – No separate aiVideoRef; reuse localVideoRef
  const aiSocketRef = useRef(null);
  const aiHolisticRef = useRef(null);
  const aiFrameRequestRef = useRef(null);
  const aiLastFrameTime = useRef(0);
  const AI_FRAME_INTERVAL_MS = 66; // ~15 fps for better performance
  const isAiComponentMounted = useRef(true);
  const isAiStarting = useRef(false);
  const mediaPipeLoaded = useRef(false);

  // ==========================================
  // AI KEYPOINT EXTRACTOR (same as Translate page)
  // ==========================================
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

    const leftHand = results.leftHandLandmarks
      ? results.leftHandLandmarks.flatMap(lm => [lm.x, lm.y, lm.z])
      : new Array(21 * 3).fill(0);

    const rightHand = results.rightHandLandmarks
      ? results.rightHandLandmarks.flatMap(lm => [lm.x, lm.y, lm.z])
      : new Array(21 * 3).fill(0);

    return [...pose, ...face, ...leftHand, ...rightHand];
  };

  const hasVisibleHands = (results) => !!(results.leftHandLandmarks || results.rightHandLandmarks);

  // ==========================================
  // Load MediaPipe scripts
  // ==========================================
  const loadMediaPipeScripts = () => {
    return new Promise((resolve) => {
      if (window.Holistic && window.drawConnectors) {
        mediaPipeLoaded.current = true;
        resolve();
        return;
      }
      const scripts = [
        "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js",
        "https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js"
      ];
      let loaded = 0;
      scripts.forEach(src => {
        if (!document.querySelector(`script[src="${src}"]`)) {
          const script = document.createElement("script");
          script.src = src;
          script.async = true;
          script.onload = () => {
            loaded++;
            if (loaded === scripts.length) {
              mediaPipeLoaded.current = true;
              resolve();
            }
          };
          document.body.appendChild(script);
        } else {
          loaded++;
          if (loaded === scripts.length) {
            mediaPipeLoaded.current = true;
            resolve();
          }
        }
      });
    });
  };

  // ==========================================
  // AI Translation Setup – reuses localVideoRef
  // ==========================================
  const startAICameraAndWS = async () => {
    if (isAiStarting.current) return;
    if (!mediaPipeLoaded.current) {
      await loadMediaPipeScripts();
    }
    if (!window.Holistic) {
      console.error("MediaPipe Holistic not loaded");
      return;
    }
    if (!localStream) {
      console.error("No local stream for AI");
      return;
    }
    if (!localVideoRef.current) {
      console.error("Local video element not ready");
      return;
    }

    isAiStarting.current = true;
    setIsAiLoading(true);

    // FIXED: Connect directly to AI service on port 8001
    const aiWs = new WebSocket("ws://localhost:8001/ws/sign-to-text");
    
    aiWs.onopen = () => {
      console.log("🤖 AI WebSocket connected");
      aiSocketRef.current = aiWs;
      initializeAIMediaPipe();
    };
    
    aiWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("📨 AI Response:", data);
        
        if (data.status === "collecting") {
          setIsAiRecording(true);
          if (data.sentence_buffer) setAiGlossText(data.sentence_buffer);
        } else if (data.status === "success") {
          setIsAiRecording(false);
          if (data.sentence_buffer !== undefined) {
            setAiGlossText(data.sentence_buffer);
          }
          if (data.english_sentence !== undefined) {
            setAiEnglishText(data.english_sentence);
          }
          if (data.confidence) {
            setAiConfidence(data.confidence * 100);
          }
          if (data.text && data.confidence > 0.3) {
            console.log(`✅ Word: ${data.text} (${(data.confidence*100).toFixed(1)}%)`);
          }
        } else if (data.status === "sentence") {
          setAiEnglishText(data.sentence);
          console.log(`✨ LLM: ${data.sentence}`);
        } else if (data.status === "error") {
          setIsAiRecording(false);
          console.warn("AI error:", data.text);
        }
      } catch (err) {
        console.error("AI WebSocket parse error:", err);
      }
    };
    
    aiWs.onclose = () => {
      console.log("🔌 AI WebSocket disconnected");
      aiSocketRef.current = null;
      setIsAiRecording(false);
      setIsAiLoading(false);
      isAiStarting.current = false;
    };
    
    aiWs.onerror = (err) => {
      console.error("AI WebSocket error:", err);
    };
  };

  const initializeAIMediaPipe = () => {
    if (!localStream || !localVideoRef.current) {
      console.log("❌ No local stream or video element for AI");
      return;
    }

    const { Holistic, POSE_CONNECTIONS, HAND_CONNECTIONS, drawConnectors, drawLandmarks } = window;
    const holistic = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });
    
    // OPTIMIZED: Better accuracy settings
    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      refineFaceLandmarks: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
    
    holistic.onResults((results) => {
      if (!isAiComponentMounted.current) return;
      if (aiSocketRef.current?.readyState === WebSocket.OPEN && isAiEnabledRef.current) {
        const handsVisible = hasVisibleHands(results);
        const keypoints = extractKeypoints(results);
        aiSocketRef.current.send(JSON.stringify({
          type: "keypoints",
          data: keypoints,
          hands_visible: handsVisible
        }));
      }
    });
    
    aiHolisticRef.current = holistic;
    
    // Ensure local video element has the stream and is playing
    if (localVideoRef.current) {
      localVideoRef.current.srcObject = localStream;
      localVideoRef.current.play().catch(e => console.warn("Local video play error:", e));
    }
    
    // Custom frame loop
    const processFrame = async (now) => {
      if (!isAiComponentMounted.current || !aiHolisticRef.current || !localVideoRef.current || !isAiEnabledRef.current) return;
      if (now - aiLastFrameTime.current >= AI_FRAME_INTERVAL_MS) {
        aiLastFrameTime.current = now;
        if (localVideoRef.current.readyState >= 2) {
          try {
            await aiHolisticRef.current.send({ image: localVideoRef.current });
          } catch (e) {
            if (!e.message?.includes("deleted object")) console.warn(e);
          }
        }
      }
      aiFrameRequestRef.current = requestAnimationFrame(processFrame);
    };
    
    aiFrameRequestRef.current = requestAnimationFrame(processFrame);
    
    setIsAiLoading(false);
    isAiStarting.current = false;
    console.log("🎥 AI camera started (reusing local video element)");
  };

  const stopAI = () => {
    console.log("🛑 Stopping AI...");
    setIsAiEnabled(false);
    isAiEnabledRef.current = false;
    isAiStarting.current = false;
    setIsAiRecording(false);
    
    if (aiFrameRequestRef.current) {
      cancelAnimationFrame(aiFrameRequestRef.current);
      aiFrameRequestRef.current = null;
    }
    if (aiHolisticRef.current) {
      aiHolisticRef.current.close();
      aiHolisticRef.current = null;
    }
    if (aiSocketRef.current) {
      aiSocketRef.current.close();
      aiSocketRef.current = null;
    }
  };

  const toggleAI = () => {
    if (!isAiEnabled) {
      setIsAiEnabled(true);
      isAiEnabledRef.current = true;
      startAICameraAndWS();
    } else {
      stopAI();
      setAiGlossText("");
      setAiEnglishText("");
      setAiConfidence(0);
    }
  };

  const resetAI = () => {
    if (aiSocketRef.current?.readyState === WebSocket.OPEN) {
      aiSocketRef.current.send(JSON.stringify({ type: "reset" }));
      setAiGlossText("");
      setAiEnglishText("");
      setAiConfidence(0);
      console.log("🔄 AI buffers reset");
    }
  };

  const speakText = (text) => {
    if (!text || text === "Sign sequence will appear here..." || text === "Translation will appear here...") {
      return;
    }
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'en-US';
      utterance.rate = 0.9;
      window.speechSynthesis.speak(utterance);
    }
  };

  // ==========================================
  // WebRTC logic
  // ==========================================
  const configuration = {
    iceServers: [
      { urls: "stun:stun.l.google.com:19302" },
      { urls: "stun:stun1.l.google.com:19302" },
      { urls: "stun:stun2.l.google.com:19302" },
      { urls: "stun:stun3.l.google.com:19302" },
      { urls: "stun:stun4.l.google.com:19302" }
    ]
  };

  useEffect(() => {
    if (isOpen) {
      initializeCall();
    }
    return () => {
      cleanupCall();
      stopAI();
    };
  }, [isOpen]);

  useEffect(() => {
    if (localVideoRef.current && localStream) {
      localVideoRef.current.srcObject = localStream;
    }
  }, [localStream]);

  useEffect(() => {
    window.handleRemoteAnswer = async (answer) => {
      if (peerConnectionRef.current) {
        try {
          await peerConnectionRef.current.setRemoteDescription(new RTCSessionDescription(answer));
          console.log("✅ Remote description set successfully");
        } catch (err) {
          console.error("❌ Error setting remote description:", err);
        }
      }
    };
    return () => { delete window.handleRemoteAnswer; };
  }, []);

  useEffect(() => {
    if (remoteVideoRef.current && remoteStream) {
      remoteVideoRef.current.srcObject = remoteStream;
      setIsConnecting(false);
      durationIntervalRef.current = setInterval(() => {
        setCallDuration(prev => prev + 1);
      }, 1000);
    }
  }, [remoteStream]);

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const initializeCall = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: callType === "video" ? { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" } : false,
        audio: true
      });
      setLocalStream(stream);
      
      const pc = new RTCPeerConnection(configuration);
      peerConnectionRef.current = pc;
      window.videoCallPeerConnection = pc;
      
      stream.getTracks().forEach(track => pc.addTrack(track, stream));
      
      pc.onicecandidate = (event) => {
        if (event.candidate && ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: "ice-candidate",
            target: remoteUsername,
            candidate: event.candidate,
            callId: callId
          }));
        }
      };
      
      pc.oniceconnectionstatechange = () => {
        console.log("ICE Connection State:", pc.iceConnectionState);
        if (pc.iceConnectionState === "failed" || pc.iceConnectionState === "disconnected") {
          setConnectionError("Connection lost");
          setIsConnecting(false);
        }
      };
      
      pc.onconnectionstatechange = () => {
        console.log("Connection state:", pc.connectionState);
        if (pc.connectionState === "connected") {
          setIsConnecting(false);
        } else if (pc.connectionState === "failed") {
          setConnectionError("Connection failed");
          setIsConnecting(false);
        }
      };
      
      pc.ontrack = (event) => {
        console.log("Received remote track:", event.track.kind);
        setRemoteStream(event.streams[0]);
        
        if (event.track.kind === 'video') {
          event.track.onmute = () => setRemoteVideoMuted(true);
          event.track.onunmute = () => setRemoteVideoMuted(false);
        }
        if (event.track.kind === 'audio') {
          event.track.onmute = () => setRemoteAudioMuted(true);
          event.track.onunmute = () => setRemoteAudioMuted(false);
        }
      };
      
      if (!isCaller && incomingOffer) {
        await pc.setRemoteDescription(new RTCSessionDescription(incomingOffer));
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        ws.send(JSON.stringify({
          type: "call-answer",
          target: remoteUsername,
          answer: pc.localDescription,
          callId: callId
        }));
      }
      
      if (isCaller && ws && ws.readyState === WebSocket.OPEN) {
        setTimeout(async () => {
          const offer = await pc.createOffer();
          await pc.setLocalDescription(offer);
          ws.send(JSON.stringify({
            type: "call-offer",
            target: remoteUsername,
            offer: pc.localDescription,
            callId: callId,
            callType: callType
          }));
        }, 1000);
      }
    } catch (err) {
      console.error("Error initializing call:", err);
      setConnectionError(err.message);
      setIsConnecting(false);
    }
  };

  const cleanupCall = () => {
    if (durationIntervalRef.current) clearInterval(durationIntervalRef.current);
    if (mediaRecorderRef.current && isRecording) mediaRecorderRef.current.stop();
    if (localStream) localStream.getTracks().forEach(track => track.stop());
    if (peerConnectionRef.current) peerConnectionRef.current.close();
    if (ws && ws.readyState === WebSocket.OPEN && callId) {
      ws.send(JSON.stringify({ type: "call-end", target: remoteUsername, callId: callId }));
    }
    delete window.videoCallPeerConnection;
  };

  const toggleVideo = () => {
    if (localStream) {
      const videoTrack = localStream.getVideoTracks()[0];
      if (videoTrack) videoTrack.enabled = !isVideoEnabled;
      setIsVideoEnabled(!isVideoEnabled);
    }
  };

  const toggleAudio = () => {
    if (localStream) {
      const audioTrack = localStream.getAudioTracks()[0];
      if (audioTrack) audioTrack.enabled = !isAudioEnabled;
      setIsAudioEnabled(!isAudioEnabled);
    }
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const startRecording = () => {
    if (!remoteStream) return;
    recordedChunksRef.current = [];
    const combinedStream = new MediaStream();
    if (localStream) localStream.getAudioTracks().forEach(track => combinedStream.addTrack(track));
    if (remoteStream) remoteStream.getTracks().forEach(track => combinedStream.addTrack(track));
    try {
      const mediaRecorder = new MediaRecorder(combinedStream, { mimeType: 'video/webm;codecs=vp9,opus' });
      mediaRecorder.ondataavailable = (event) => { if (event.data.size > 0) recordedChunksRef.current.push(event.data); };
      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `call-recording-${new Date().toISOString()}.webm`;
        a.click();
        URL.revokeObjectURL(url);
      };
      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
    } catch (err) { console.error("Error starting recording:", err); alert("Recording failed to start"); }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const endCall = () => {
    cleanupCall();
    stopAI();
    onClose();
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 bg-black flex items-center justify-center"
      >
        <div className="relative w-full h-full">
          {/* Remote Video */}
          {remoteStream ? (
            <video ref={remoteVideoRef} autoPlay playsInline className="w-full h-full object-cover" />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800">
              <div className="text-center text-white">
                {isConnecting ? (
                  <>
                    <div className="w-24 h-24 border-4 border-primary-500 border-t-transparent rounded-full animate-spin mx-auto mb-6"></div>
                    <p className="text-2xl font-semibold">Connecting to {remoteUsername}...</p>
                  </>
                ) : connectionError ? (
                  <>
                    <p className="text-2xl text-red-500 mb-4">{connectionError}</p>
                    <button onClick={onClose} className="px-8 py-3 bg-primary-500 text-white rounded-full">Close</button>
                  </>
                ) : (
                  <>
                    <div className="text-7xl mb-6 animate-pulse">📹</div>
                    <p className="text-3xl font-semibold mb-2">{remoteUsername}</p>
                    <p className="text-gray-400">Ringing...</p>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Local Video (Picture-in-Picture) */}
          {localStream && callType === "video" && (
            <div className="absolute bottom-28 right-4 w-56 h-72 bg-gray-800 rounded-xl overflow-hidden border-2 border-primary-500 shadow-2xl">
              <video ref={localVideoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
              {!isVideoEnabled && (
                <div className="absolute inset-0 bg-gray-900/80 flex items-center justify-center">
                  <BsCameraVideoOff className="text-white text-3xl" />
                </div>
              )}
              <div className="absolute bottom-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded">You</div>
              
              {/* AI Recording Indicator */}
              {isAiEnabled && isAiRecording && (
                <div className="absolute top-2 left-2 bg-red-500/80 text-white text-xs px-2 py-1 rounded-full flex items-center gap-1 animate-pulse">
                  <span className="w-2 h-2 bg-white rounded-full"></span> AI Recording
                </div>
              )}
              
              {/* AI Loading Indicator */}
              {isAiEnabled && isAiLoading && (
                <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                  <div className="text-white text-xs">Loading AI...</div>
                </div>
              )}
            </div>
          )}

          {/* AI Translation Overlay */}
          {isAiEnabled && (aiGlossText || aiEnglishText) && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="absolute bottom-40 left-1/2 transform -translate-x-1/2 bg-gradient-to-r from-primary-900/95 to-pink-900/95 backdrop-blur-md rounded-2xl p-4 border border-primary-500/50 shadow-2xl max-w-md w-full mx-4 z-30"
            >
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center gap-2">
                  <BsRobot className="text-primary-400 text-sm" />
                  <span className="text-primary-300 text-xs font-semibold">AI Translation</span>
                  {aiConfidence > 0 && <span className="text-green-400 text-xs">{aiConfidence.toFixed(0)}% conf</span>}
                </div>
                <div className="flex gap-2">
                  <button onClick={() => speakText(aiEnglishText || aiGlossText)} className="text-gray-400 hover:text-white">
                    <BsVolumeUp size={16} />
                  </button>
                  <button onClick={resetAI} className="text-gray-400 hover:text-white">
                    <BsTrash size={16} />
                  </button>
                  <button onClick={() => { setAiGlossText(""); setAiEnglishText(""); }} className="text-gray-400 hover:text-white">
                    <BsX size={16} />
                  </button>
                </div>
              </div>
              {aiGlossText && (
                <div className="mb-2">
                  <div className="text-primary-300 text-xs">Signs:</div>
                  <p className="text-white text-md font-medium break-words">{aiGlossText}</p>
                </div>
              )}
              {aiEnglishText && (
                <div>
                  <div className="text-primary-300 text-xs">English:</div>
                  <p className="text-white text-md font-medium break-words">{aiEnglishText}</p>
                </div>
              )}
            </motion.div>
          )}

          {/* AI Processing Indicator */}
          {isAiEnabled && isAiRecording && !aiGlossText && !aiEnglishText && !isAiLoading && (
            <div className="absolute top-20 left-1/2 transform -translate-x-1/2 bg-primary-500/80 text-white px-4 py-2 rounded-full text-sm font-semibold animate-pulse z-30">
              🤖 AI is processing your sign...
            </div>
          )}

          {/* Call Controls */}
          <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 flex items-center gap-4 bg-gray-900/90 backdrop-blur-md px-6 py-4 rounded-full border border-gray-700 shadow-2xl z-30">
            {/* AI Translation Button */}
            <button
              onClick={toggleAI}
              className={`p-4 rounded-full transition-all transform hover:scale-110 ${
                isAiEnabled ? 'bg-primary-500 hover:bg-primary-600 text-white animate-pulse' : 'bg-gray-700 hover:bg-gray-600 text-white'
              }`}
              title={isAiEnabled ? "Disable AI translation" : "Enable AI sign language translation"}
            >
              <BsTranslate size={22} />
            </button>

            {/* Audio Toggle */}
            <button
              onClick={toggleAudio}
              className={`p-4 rounded-full transition-all transform hover:scale-110 ${
                isAudioEnabled ? 'bg-gray-700 hover:bg-gray-600 text-white' : 'bg-red-500 hover:bg-red-600 text-white'
              }`}
            >
              {isAudioEnabled ? <BsMic size={22} /> : <BsMicMute size={22} />}
            </button>

            {/* Video Toggle */}
            {callType === "video" && (
              <button
                onClick={toggleVideo}
                className={`p-4 rounded-full transition-all transform hover:scale-110 ${
                  isVideoEnabled ? 'bg-gray-700 hover:bg-gray-600 text-white' : 'bg-red-500 hover:bg-red-600 text-white'
                }`}
              >
                {isVideoEnabled ? <BsCameraVideo size={22} /> : <BsCameraVideoOff size={22} />}
              </button>
            )}

            {/* Recording Toggle */}
            {remoteStream && (
              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={`p-4 rounded-full transition-all transform hover:scale-110 ${
                  isRecording ? 'bg-red-500 hover:bg-red-600 text-white animate-pulse' : 'bg-gray-700 hover:bg-gray-600 text-white'
                }`}
              >
                {isRecording ? <BsStopCircle size={22} /> : <BsRecordCircle size={22} />}
              </button>
            )}

            {/* End Call */}
            <button
              onClick={endCall}
              className="p-4 rounded-full bg-red-500 hover:bg-red-600 text-white transition-all transform hover:scale-110"
            >
              <BsTelephoneX size={22} />
            </button>

            {/* Fullscreen Toggle */}
            <button
              onClick={toggleFullscreen}
              className="p-4 rounded-full bg-gray-700 hover:bg-gray-600 text-white transition-all transform hover:scale-110"
            >
              {isFullscreen ? <BsFullscreenExit size={22} /> : <BsFullscreen size={22} />}
            </button>
          </div>

          {/* Call Info */}
          <div className="absolute top-4 left-4 bg-gray-900/80 backdrop-blur-sm px-5 py-2.5 rounded-full border border-gray-700 z-30">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${remoteStream ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'}`}></div>
                <span className="text-white font-medium">{remoteUsername}</span>
              </div>
              {remoteStream && (
                <>
                  <div className="w-px h-4 bg-gray-600"></div>
                  <span className="text-gray-300 font-mono">{formatDuration(callDuration)}</span>
                </>
              )}
              {isAiEnabled && (
                <>
                  <div className="w-px h-4 bg-gray-600"></div>
                  <span className="text-primary-400 text-sm flex items-center gap-1">
                    <BsRobot size={12} /> AI Active
                  </span>
                </>
              )}
            </div>
          </div>

          {/* Back Button */}
          <button
            onClick={onClose}
            className="absolute top-4 right-4 md:hidden p-3 bg-gray-900/80 backdrop-blur-sm rounded-full border border-gray-700 text-white hover:bg-gray-800 transition-colors z-30"
          >
            <BsArrowLeft size={22} />
          </button>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default VideoCall;