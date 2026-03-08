import React, { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, Camera, RefreshCw, Loader2, Video, Volume2, Trash2, Lightbulb, Play, Square, PowerOff, Hand } from "lucide-react";

// --- Toast Component ---
const Toast = ({ message, type }) => {
  if (!message) return null;
  const baseClasses = "fixed bottom-5 left-1/2 transform -translate-x-1/2 p-4 rounded-xl shadow-2xl z-[100] flex items-center gap-3 font-semibold text-white";
  let colorClasses = type === 'success' ? "bg-green-600 shadow-green-500/50" : 
                     type === 'warning' ? "bg-yellow-600 shadow-yellow-500/50" : 
                     type === 'error' ? "bg-red-600 shadow-red-500/50" : "bg-purple-600 shadow-purple-500/50";

  return (
    <motion.div initial={{ opacity: 0, y: 50 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 50 }} className={`${baseClasses} ${colorClasses}`}>
      {type === 'info' && <Volume2 className="w-5 h-5" />}
      {message}
    </motion.div>
  );
};

// ==========================================
// AI KEYPOINT EXTRACTOR (Matches Python exactly)
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

// Check if any hand is visible
const hasVisibleHands = (results) => {
  return !!(results.leftHandLandmarks || results.rightHandLandmarks);
};

export default function Translate() {
  const fadeUp = { hidden: { opacity: 0, y: 25 }, visible: { opacity: 1, y: 0 } };
  const options = ["Sign to Text", "Text / Audio to Sign"];
  
  const [selected, setSelected] = useState(options[0]);
  const [textInput, setTextInput] = useState("");
  const [recognizedText, setRecognizedText] = useState(""); 
  const [isTranslating, setIsTranslating] = useState(false);
  const [message, setMessage] = useState(null);
  
  const [videoQueue, setVideoQueue] = useState([]);
  const [currentVideo, setCurrentVideo] = useState(null); 
  
  // ==========================================
  // CONTINUOUS AI STATES & UX
  // ==========================================
  const [isStreaming, setIsStreaming] = useState(false); 
  const [isRecordingSign, setIsRecordingSign] = useState(false); 
  const [isAiLoading, setIsAiLoading] = useState(false);
  const [framesCollected, setFramesCollected] = useState(0);
  const [handDetected, setHandDetected] = useState(false); // New state for hand detection
  const [gracePeriod, setGracePeriod] = useState(0); // Countdown for auto-stop

  const videoRef = useRef(null); 
  const canvasRef = useRef(null);
  const socketRef = useRef(null);
  const holisticRef = useRef(null);
  const cameraRef = useRef(null);
  const isRecordingRef = useRef(false);
  const framesSinceHandSeen = useRef(0); // Counter for frames without hands
  const HAND_MISSING_THRESHOLD = 10; // Frames to wait before auto-stopping

  const showMessage = useCallback((text, type = 'info') => {
    setMessage({ text, type });
    setTimeout(() => setMessage(null), 3000);
  }, []);

  useEffect(() => {
    isRecordingRef.current = isRecordingSign;
  }, [isRecordingSign]);

  // ==========================================
  // DYNAMICALLY INJECT MEDIAPIPE CDN
  // ==========================================
  useEffect(() => {
    const scripts = [
      "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js",
      "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js",
      "https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js"
    ];

    scripts.forEach(src => {
      if (!document.querySelector(`script[src="${src}"]`)) {
        const script = document.createElement("script");
        script.src = src;
        script.async = true;
        document.body.appendChild(script);
      }
    });

    return () => turnOffSystem();
  }, []);

  // ==========================================
  // WEBSOCKET & MEDIAPIPE INITIALIZATION
  // ==========================================
  const startCameraAndWS = () => {
    if (!window.Holistic || !window.Camera) {
      showMessage("AI libraries are still downloading (approx 15MB). Please wait a moment and try again.", "warning");
      return;
    }

    setIsAiLoading(true);

    try {
      const ws = new WebSocket("ws://localhost:8000/ws/translate/sign-to-text");
      
      ws.onopen = () => {
        socketRef.current = ws;
        setIsStreaming(true);
        initializeMediaPipe();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.status === "collecting") {
              if (data.frames_collected % 5 === 0) {
                setFramesCollected(data.frames_collected);
              }
          }

          if (data.text && data.text !== "collecting" && data.text !== "Too short") {
              setIsTranslating(false);
              setIsRecordingSign(false); // Ensure recording is off
              framesSinceHandSeen.current = 0; // Reset counter
              
              if (data.confidence > 0) {
                 setRecognizedText(prev => prev === "" ? data.text : prev + " " + data.text);
                 showMessage(`Result: ${data.text} (${(data.confidence*100).toFixed(1)}%)`, "success");
              } else {
                 showMessage(data.text, "warning");
              }
          } else if (data.text === "Too short") {
              setIsTranslating(false);
              setIsRecordingSign(false);
              showMessage("Gesture was too short. Try again.", "warning");
          }
        } catch (err) { console.error("WS parse error:", err); }
      };

      ws.onclose = () => { 
        socketRef.current = null;
        turnOffSystem();
      }; 

    } catch (error) {
      setIsAiLoading(false);
      showMessage("Server is offline.", "error");
    }
  };

  const initializeMediaPipe = () => {
    const { Holistic, POSE_CONNECTIONS, HAND_CONNECTIONS, drawConnectors, drawLandmarks, Camera } = window;

    const holistic = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });

    holistic.setOptions({
      modelComplexity: 0,
      smoothLandmarks: true,
      refineFaceLandmarks: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    holistic.onResults((results) => {
      const canvasCtx = canvasRef.current?.getContext('2d');
      if (!canvasCtx || !canvasRef.current || !videoRef.current) return;

      // Check for hand visibility
      const handsVisible = hasVisibleHands(results);
      setHandDetected(handsVisible);
      
      if (canvasRef.current.width !== videoRef.current.videoWidth) {
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
      }

      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      canvasCtx.translate(canvasRef.current.width, 0);
      canvasCtx.scale(-1, 1);

      // Draw landmarks
      if (results.poseLandmarks) {
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
        drawLandmarks(canvasCtx, results.poseLandmarks, { color: '#FF0000', lineWidth: 2 });
      }
      
      // Draw face landmarks (lips/eyebrows only)
      if (results.faceLandmarks) {
        RELEVANT_FACE_INDICES.forEach(i => {
          const lm = results.faceLandmarks[i];
          if (!lm) return;
          canvasCtx.beginPath();
          canvasCtx.arc(
            lm.x * canvasRef.current.width,
            lm.y * canvasRef.current.height,
            1,  // size of landmarks instead of 2 make it 1
            0,
            2 * Math.PI
          );
          canvasCtx.fillStyle = "#00FFFF";
          canvasCtx.fill();
        });
      }
      
      if (results.leftHandLandmarks) {
        drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: '#CC0000', lineWidth: 5 });
        drawLandmarks(canvasCtx, results.leftHandLandmarks, { color: '#00FF00', lineWidth: 2 });
      }
      if (results.rightHandLandmarks) {
        drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: '#00CC00', lineWidth: 5 });
        drawLandmarks(canvasCtx, results.rightHandLandmarks, { color: '#FF0000', lineWidth: 2 });
      }
      canvasCtx.restore();
      
      // ==========================================
      // AUTO-SENSE LOGIC (Replaces R and E keys)
      // ==========================================
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        
        // Auto-start recording when hands appear
        if (handsVisible) {
          if (!isRecordingRef.current && !isTranslating) {
            console.log("Hands detected! Auto-starting recording...");
            setIsRecordingSign(true);
            setFramesCollected(0);
            framesSinceHandSeen.current = 0;
          } else if (isRecordingRef.current) {
            // Hands still visible while recording, reset counter
            framesSinceHandSeen.current = 0;
            
            // Send keypoints during recording
            const keypoints = extractKeypoints(results);
            socketRef.current.send(JSON.stringify({
              type: "keypoints",
              data: keypoints
            }));
          }
        } 
        // Hands not visible, but we were recording
        else if (isRecordingRef.current && !isTranslating) {
          framesSinceHandSeen.current += 1;
          setGracePeriod(HAND_MISSING_THRESHOLD - framesSinceHandSeen.current);
          
          // Keep sending keypoints during grace period
          const keypoints = extractKeypoints(results);
          socketRef.current.send(JSON.stringify({
            type: "keypoints",
            data: keypoints
          }));
          
          // Auto-stop after threshold
          if (framesSinceHandSeen.current >= HAND_MISSING_THRESHOLD) {
            console.log("Hands gone too long! Auto-predicting...");
            setIsRecordingSign(false);
            setIsTranslating(true);
            socketRef.current.send(JSON.stringify({ end: true }));
            framesSinceHandSeen.current = 0;
            setGracePeriod(0);
          }
        }
      }
    });

    holisticRef.current = holistic;

    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        if (videoRef.current && videoRef.current.readyState >= 2) {
           await holistic.send({ image: videoRef.current });
        }
      },
      width: 480,
      height: 360
    });

    camera.start().then(() => {
      setIsAiLoading(false);
      showMessage("AI System Ready! Show hands to auto-record.", "success");
    });
    
    cameraRef.current = camera;
  };

  const turnOffSystem = () => {
    if (cameraRef.current) { cameraRef.current.stop(); cameraRef.current = null; }
    if (holisticRef.current) { holisticRef.current.close(); holisticRef.current = null; }
    if (socketRef.current) { socketRef.current.close(); socketRef.current = null; }
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    
    setIsStreaming(false);
    setIsRecordingSign(false);
    setIsAiLoading(false);
    setFramesCollected(0);
    setHandDetected(false);
    setGracePeriod(0);
    framesSinceHandSeen.current = 0;
  };

  // Manual stop button (optional, as a fallback)
  const handleManualStop = () => {
    if (isRecordingSign && socketRef.current?.readyState === WebSocket.OPEN) {
      setIsRecordingSign(false);
      setIsTranslating(true);
      socketRef.current.send(JSON.stringify({ end: true }));
      framesSinceHandSeen.current = 0;
      setGracePeriod(0);
    }
  };

  const handleTabChange = (option) => {
    setSelected(option);
    turnOffSystem();
    setIsTranslating(false);
    setRecognizedText("");
    setTextInput("");
  }

  const handleConvertText = () => {
    if (!textInput.trim()) return;
    const words = textInput.trim().toLowerCase().replace(/[^a-z0-9\s-]/g, "").split(/\s+/);
    const paths = words.map(word => `/videos/${word}.mp4`);
    setVideoQueue(paths);
    setCurrentVideo(paths[0]);
    setIsTranslating(true);
  };

  const contentBoxClasses = "w-full bg-white dark:bg-[#0f0c29]/70 rounded-3xl p-6 sm:p-10 shadow-2xl shadow-purple-900/10 dark:shadow-purple-900/30 border border-purple-500/20 backdrop-blur-lg transition-colors duration-500";
  const buttonBaseClasses = "px-6 py-3 rounded-full flex items-center justify-center gap-2 text-white font-bold shadow-lg transition-all duration-300";
  const disabledClasses = "opacity-50 cursor-not-allowed hover:scale-100";

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
        {options.map((option) => (
          <button
            key={option}
            onClick={() => handleTabChange(option)}
            className={`px-8 py-3 rounded-full font-bold transition-all duration-300 text-lg ${selected === option ? `bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] scale-105 text-white` : "bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200"}`}
          >
            {option}
          </button>
        ))}
      </div>

      <motion.div key={selected} initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} className="max-w-5xl mx-auto">
        
        {/* ===================== SIGN TO TEXT ===================== */}
        {selected === "Sign to Text" && (
          <div className="flex flex-col items-center gap-6">
            <div className={contentBoxClasses}>
              
              <div className="flex justify-between items-center mb-6">
                 <h3 className="text-2xl font-extrabold bg-gradient-to-r from-[#6A3093] to-[#BF5AE0] bg-clip-text text-transparent">
                   Sign Language AI Camera
                 </h3>
                 {!isStreaming ? (
                    <button onClick={startCameraAndWS} disabled={isAiLoading} className={`${buttonBaseClasses} bg-purple-600 hover:bg-purple-700 ${isAiLoading && disabledClasses}`}>
                       {isAiLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Camera className="w-5 h-5" />}
                       {isAiLoading ? "Loading AI..." : "Turn On AI Camera"}
                    </button>
                 ) : (
                    <button onClick={turnOffSystem} className={`${buttonBaseClasses} bg-gray-500 hover:bg-gray-600`}>
                       <PowerOff className="w-5 h-5" /> Turn Off Camera
                    </button>
                 )}
              </div>

              {/* VIDEO AREA */}
              <div className="relative w-full aspect-video rounded-xl overflow-hidden bg-black border-4 border-gray-800 mb-6 shadow-inner">
                
                <video 
                    ref={videoRef} 
                    className={`absolute inset-0 w-full h-full object-cover transform scale-x-[-1] ${isStreaming ? 'block' : 'hidden'}`} 
                    playsInline 
                    muted 
                    autoPlay 
                />
                
                {isStreaming ? (
                    <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover pointer-events-none z-10" />
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <Video className="w-12 h-12 text-gray-600 mb-4" />
                    <p className="text-gray-500 text-sm">Camera is offline</p>
                  </div>
                )}

                {/* AI LOADING OVERLAY */}
                {isAiLoading && (
                  <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center text-white z-20 backdrop-blur-sm">
                     <Loader2 className="w-12 h-12 animate-spin text-purple-500 mb-4" />
                     <h3 className="text-xl font-bold">Warming up AI Engine...</h3>
                     <p className="text-sm text-gray-400 mt-2">This may take a few seconds on first load.</p>
                  </div>
                )}

                {/* HAND DETECTION STATUS */}
                {isStreaming && !isAiLoading && (
                  <div className={`absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-bold shadow-xl z-20 ${
                    handDetected ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-300'
                  }`}>
                    <Hand className="w-4 h-4" />
                    {handDetected ? 'Hands Detected' : 'No Hands'}
                  </div>
                )}

                {/* RECORDING OVERLAY */}
                {isRecordingSign && (
                  <div className="absolute top-4 right-4 flex items-center gap-2 text-white bg-red-600 px-4 py-2 rounded-full text-sm font-bold shadow-xl animate-pulse z-20">
                    <span className="w-3 h-3 rounded-full bg-white" />
                    RECORDING ({framesCollected} frames)
                  </div>
                )}
                
                {/* GRACE PERIOD COUNTDOWN */}
                {isRecordingSign && gracePeriod > 0 && !handDetected && (
                  <div className="absolute top-20 right-4 flex items-center gap-2 text-white bg-yellow-600 px-4 py-2 rounded-full text-sm font-bold shadow-xl z-20">
                    Finishing in {gracePeriod}
                  </div>
                )}

                {isTranslating && (
                  <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center text-white z-20">
                     <Loader2 className="w-12 h-12 animate-spin text-purple-500 mb-4" />
                     <h3 className="text-xl font-bold">AI is predicting your sign...</h3>
                  </div>
                )}
              </div>

              {/* AUTO-SENSE INFO */}
              {isStreaming && (
                <div className="bg-gray-100 dark:bg-gray-800 rounded-xl p-6 border border-gray-300 dark:border-gray-700 text-center mb-4">
                    <div className="flex items-center justify-center gap-3 text-gray-700 dark:text-gray-300">
                      <Hand className="w-5 h-5 text-purple-500" />
                      <p className="text-sm font-medium">
                        <span className="text-green-600 font-bold">Auto-sense active:</span> Show hands to start recording, hide to predict
                      </p>
                    </div>
                    
                    {/* Optional manual stop button as fallback */}
                    {isRecordingSign && (
                      <button
                        onClick={handleManualStop}
                        className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-full text-sm font-bold flex items-center gap-2 mx-auto"
                      >
                        <Square className="w-4 h-4" /> Manual Stop
                      </button>
                    )}
                </div>
              )}

              {/* RESULT TEXT BOX */}
              <div className="mt-8">
                 <div className="flex justify-between items-end mb-4 border-b border-purple-500/50 pb-2">
                    <h4 className="text-xl font-semibold text-gray-800 dark:text-white">Constructed Sentence:</h4>
                    <button onClick={() => setRecognizedText("")} className="text-sm text-red-500 hover:text-red-700 flex items-center gap-1 font-semibold">
                        <Trash2 className="w-4 h-4" /> Clear Text
                    </button>
                 </div>
                 
                 <div className={`w-full min-h-32 p-5 rounded-xl border-2 border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800 shadow-inner text-gray-900 dark:text-gray-200 text-2xl tracking-wide transition-colors ${recognizedText ? 'text-purple-600 dark:text-pink-300 font-bold' : 'text-gray-500 dark:text-gray-500 italic'}`}>
                    {recognizedText || (
                      <div className="flex items-center gap-3 text-lg font-normal">
                        <Lightbulb className="w-6 h-6"/>
                        Show your hands to start recording automatically. Hide hands to translate.
                      </div>
                    )}
                 </div>
              </div>

            </div>
          </div>
        )}

        {/* ===================== TEXT TO SIGN ===================== */}
        {selected === "Text / Audio to Sign" && (
            <div className="flex flex-col items-center gap-10">
            <div className={contentBoxClasses}>
              <h3 className="text-3xl font-extrabold text-center mb-10 bg-gradient-to-r from-purple-500 to-pink-400 bg-clip-text text-transparent">
                Text / Audio → Sign Language Avatar
              </h3>
              <textarea
                rows={4} value={textInput} onChange={(e) => setTextInput(e.target.value)}
                placeholder="Type your message here to see it signed by the avatar..."
                disabled={isTranslating}
                className={`w-full p-5 rounded-xl border border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-200 shadow-md focus:outline-none focus:ring-2 focus:ring-purple-500 transition resize-none text-lg ${isTranslating && disabledClasses}`}
              />
              <div className="flex justify-center items-center gap-8 mt-6">
                <button onClick={() => showMessage("Audio coming soon", "info")} disabled={isTranslating} className={`w-16 h-16 rounded-full flex items-center justify-center text-white shadow-xl bg-purple-600 hover:bg-purple-700 ${isTranslating && disabledClasses}`}>
                  <Mic className="text-2xl" />
                </button>
                <div className="text-2xl font-bold text-gray-500 dark:text-gray-400">OR</div>
                <button onClick={handleConvertText} disabled={isTranslating || !textInput.trim()} className={`${buttonBaseClasses} bg-gradient-to-r from-[#6A3093] to-[#BF5AE0] ${(isTranslating || !textInput.trim()) && disabledClasses}`}>
                  {isTranslating ? <><Loader2 className="w-5 h-5 animate-spin" /> Generating Signs...</> : <><RefreshCw className="w-5 h-5" /> Convert to Sign</>}
                </button>
              </div>
              <div className="mt-10 w-full h-64 rounded-2xl border border-gray-300 dark:border-purple-600/50 bg-gray-100 dark:bg-gray-800 shadow-inner flex flex-col items-center justify-center text-gray-500 dark:text-gray-400 text-center text-lg tracking-wide">
                {currentVideo ? (
                <video key={currentVideo} src={currentVideo} autoPlay muted className="rounded-lg w-full h-full object-contain mb-3"
                  onEnded={() => {
                    const nextIndex = videoQueue.indexOf(currentVideo) + 1;
                    if (nextIndex < videoQueue.length) setCurrentVideo(videoQueue[nextIndex]);
                    else { setCurrentVideo(null); setVideoQueue([]); setIsTranslating(false); }
                  }}
                  onError={() => {
                    showMessage(`No sign available for that word`, 'warning');
                    const nextIndex = videoQueue.indexOf(currentVideo) + 1;
                    if (nextIndex < videoQueue.length) setCurrentVideo(videoQueue[nextIndex]);
                    else { setCurrentVideo(null); setVideoQueue([]); setIsTranslating(false); }
                  }}
                />
              ) : (
                <>
                  <img src={`https://placehold.co/150x200/4c3093/ffffff?text=3D+Avatar`} alt="Sign Language Avatar Placeholder" className="rounded-lg mb-3" />
                  <p>No sign available or translation finished</p>
                </>
              )}
                {isTranslating && <p className="mt-2 text-purple-500 flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin"/> Animating...</p>}
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