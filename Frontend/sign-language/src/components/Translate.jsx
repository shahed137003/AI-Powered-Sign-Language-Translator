import React, { useState, useRef, useCallback ,useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FaMicrophone, FaCamera, FaSyncAlt, FaSpinner, FaStopCircle, FaVideo, FaVolumeUp } from "react-icons/fa";
import { HiOutlineLightBulb } from "react-icons/hi"; // For the recognized text box placeholder

// --- Toast/Message Component (Replaces alert()) ---
const Toast = ({ message, type, onClose }) => {
  if (!message) return null;

  const baseClasses = "fixed bottom-5 left-1/2 transform -translate-x-1/2 p-4 rounded-xl shadow-2xl z-[100] flex items-center gap-3 font-semibold text-white";
  let colorClasses = "";

  switch (type) {
    case 'success':
      colorClasses = "bg-green-600 shadow-green-500/50";
      break;
    case 'warning':
      colorClasses = "bg-yellow-600 shadow-yellow-500/50";
      break;
    case 'error':
      colorClasses = "bg-red-600 shadow-red-500/50";
      break;
    case 'info':
    default:
      colorClasses = "bg-purple-600 shadow-purple-500/50";
      break;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 50 }}
      className={`${baseClasses} ${colorClasses}`}
    >
      {type === 'info' && <FaVolumeUp className="text-xl" />}
      {message}
    </motion.div>
  );
};
// ----------------------------------------------------

export default function Translate() {
  const fadeUp = { hidden: { opacity: 0, y: 25 }, visible: { opacity: 1, y: 0 } };
  const options = ["Sign to Text", "Text / Audio to Sign"];
  
  const [selected, setSelected] = useState(options[0]);
  const [textInput, setTextInput] = useState("");
  const [recognizedText, setRecognizedText] = useState("This is my dummy test text. The translation results will appear here.");
  const [isRecording, setIsRecording] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [message, setMessage] = useState(null);
  const [recordingType, setRecordingType] = useState(null); // 'camera' or 'mic'
  const [videoSrc, setVideoSrc] = useState(null); // ← New state for video source
  const [videoQueue, setVideoQueue] = useState([]);
  const [currentVideo, setCurrentVideo] = useState(null); 
  // Ref for video element (simulated video feed)
  const videoRef = useRef(null); 
  const canvasRef = useRef(null);
  const socketRef = useRef(null);
  const streamRef = useRef(null);
  const recordingInterval = useRef(null);
  const isProcessingRef = useRef(false); // Acts as a "Kill Switch" for the loop

  const showMessage = useCallback((text, type = 'info') => {
    setMessage({ text, type });
    setTimeout(() => setMessage(null), 3000);
  }, []);

  // --- Unified Recording Handler ---
  const handleRecord = (type) => {
    if (isRecording) {
      showMessage(`${type} recording stopped. Processing sign recognition...`, 'info');
      setIsRecording(false);
      setRecordingType(null);
      setIsTranslating(true);

      // Simulate recognition time
      setTimeout(() => {
        setIsTranslating(false);
        if (type === 'Camera') {
          setRecognizedText("Hello, how are you doing today? I love LinguaSign!");
          showMessage("Sign-to-Text translation complete!", 'success');
        } else {
          showMessage("Audio recording processed. Ready for Sign conversion!", 'success');
        }
      }, 3000);

    } else {
      if (selected === "Sign to Text") {
        setRecognizedText(""); // Clear previous result
      }
      showMessage(`${type} recording started. Say or sign clearly!`, 'warning');
      setIsRecording(true);
      setRecordingType(type === 'Camera' ? 'camera' : 'mic');
      
      // Simulate max recording time (15 seconds)
      setTimeout(() => {
        if (isRecording) { // Only auto-stop if still recording
          handleRecord(type);
        }
      }, 15000);
    }
  };
  const startCamera = async () => {
    if (streamRef.current) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 }, 
          height: { ideal: 480 } 
        } 
      });
      console.log("✅ Camera access granted");

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          {
            console.log("Video metadata loaded", videoRef.current.videoWidth, videoRef.current.videoHeight);
          };
        }
        // Force video to play
        videoRef.current.play().catch(e => {
            console.error("Video play failed:", e);
            showMessage("Camera error. Please refresh page.", "error");
        });
      }
      
    } catch (error) {
      console.error("Camera access error:", error);
      showMessage("Camera permission denied. Please allow camera access.", "error");
    }
  };
  const stopCamera = () => {
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
  };
  const startWebSocket = () => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      return;
    }

    console.log("🔄 Connecting WebSocket...");
    const ws = new WebSocket("ws://localhost:8000/ws/translate/sign-to-text");
    
    ws.onopen = () => {
      console.log("✅ Connected");
      socketRef.current = ws;
      setRecognizedText("🟢 Connected – sending frames...");
      showMessage("AI connection established", "success");
      
      // RESET KILL SWITCH
      isProcessingRef.current = true; 
      
      // Start the loop
      recordingInterval.current = sendFrames();
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // CASE 1: Final Prediction Received
        if (data.text) {
          // 🛑 KILL SWITCH: Stop sending frames IMMEDIATELY
          isProcessingRef.current = false;
          clearInterval(recordingInterval.current);

          setRecognizedText(data.text);
          showMessage(`Sign recognized: ${data.text}`, "success");

          // Graceful shutdown
          setIsRecording(false);
          // stopCamera();
          
          // Close socket after a tiny delay to allow backend to finish
          setTimeout(() => {
             socketRef.current?.close();
          }, 100);
        } 
        
        // CASE 2: Progress Update
        else if (data.progress) {
           setPredictionProgress(Math.floor(data.progress));
           setRecognizedText(`Collecting frames... ${Math.floor(data.progress)}%`);
        }
      } catch (err) {
        console.error("WS parse error:", err);
      }
    };

    ws.onerror = (err) => {
      console.error("WS error:", err);
      // Don't show alert, just log it to avoid UI spam
    };

    ws.onclose = () => {
      console.log("WS closed");
      socketRef.current = null;
    }; 
  };

  // const startWebSocket = () => {
  //   if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
  //     console.log("WebSocket already connected");
  //     return;
  //   }

  //   console.log("🔄 Connecting WebSocket to: ws://localhost:8000/ws/translate/sign-to-text");
    
  //   const ws = new WebSocket("ws://localhost:8000/ws/translate/sign-to-text");
    
  //   ws.onopen = () => {
  //     console.log("✅ WebSocket connected successfully");
  //     socketRef.current = ws;

  //     setRecognizedText("🟢 Connected – sending frames...");
  //     showMessage("AI connection established", "success");

  //     // RESET KILL SWITCH
  //     isProcessingRef.current = true;

  //     // ✅ Start sending frames ONLY now
  //     recordingInterval.current = sendFrames();
  //   };
  //    ws.onmessage = (event) => {
  //     try {
  //       const data = JSON.parse(event.data);

  //       if (data.text) {
  //         setRecognizedText(data.text);
  //         showMessage("Sign recognized successfully!", "success");

  //         // Stop sending frames after prediction
  //         clearInterval(recordingInterval.current);
  //         setIsRecording(false);
  //         stopCamera();
  //         socketRef.current?.close();
  //       } else if (data.progress) {
  //         setPredictionProgress(Math.floor(data.progress));
  //         setRecognizedText(`Collecting frames... ${Math.floor(data.progress)}%`);
  //       }
  //     } catch (err) {
  //       console.error("WS parse error:", err);
  //     }
  //   };

  //   ws.onerror = (err) => {
  //     console.error("WS error:", err);
  //     showMessage("WebSocket error", "error");
  //   };

  //   ws.onclose = () => {
  //     console.log("WS closed");
  //     socketRef.current = null;
  //   }; 
  // };
  // Add to your state
  const [frameCount, setFrameCount] = useState(0);
  const [predictionProgress, setPredictionProgress] = useState(0);

  const sendFrames = () => {
      let count = 0;
      const FRAME_LEN = 96;

      const send = () => {
        // FIX: Removed "!isRecording" check. 
        // We stop the loop using clearInterval, so we don't need to check state here.
        if (!videoRef.current || !canvasRef.current) return;

        // Check if video is actually playing/ready
        if (videoRef.current.readyState < 2) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        // Ensure canvas matches video size
        if (canvas.width !== videoRef.current.videoWidth) {
            canvas.width = videoRef.current.videoWidth;
            canvas.height = videoRef.current.videoHeight;
        }

        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        
        // Use 0.6 quality for faster transmission
        const base64 = canvas.toDataURL("image/jpeg", 0.6); 

        if (socketRef.current?.readyState === WebSocket.OPEN) {
          socketRef.current.send(base64);
          
          // Only log every 10th frame to avoid flooding the console
          if (count % 10 === 0) {
              console.log(`📸 Sent frame ${count}/${FRAME_LEN}`);
          }
          
          count++;
          setFrameCount(count);

          // Optional: Auto-stop logic if you want strict 96 frame batches
          // if (count >= FRAME_LEN) {
          //   console.log("✅ Batch complete");
          //   clearInterval(recordingInterval.current);
          // }
        }
      };

      // Set to 50ms (approx 20 FPS) for smoother capture
      return setInterval(send, 50); 
    };

  const handleVideoRecord = async () => {
    if (!isRecording) {
      setRecognizedText("");
      setFrameCount(0);
      setPredictionProgress(0);

      showMessage("Camera started", "info");

      // ✅ 1. set recording FIRST
      setIsRecording(true);
      setRecordingType("camera");

      // ✅ 2. start camera
      await startCamera();

      // ✅ 3. connect websocket
      startWebSocket();

    } else {
      showMessage("Camera stopped", "info");

      clearInterval(recordingInterval.current);
      socketRef.current?.close();

      stopCamera();
      setIsRecording(false);
      setRecordingType(null);
    }
  };
useEffect(() => {
  return () => {
    clearInterval(recordingInterval.current);
    socketRef.current?.close();
    stopCamera();
  };
}, []);
  const handleAudioRecord = () => handleRecord('Audio');

  // --- Text Conversion Handler ---
  const handleConvertText = () => {
    if (!textInput.trim()) return;

  // Split and map words
  const words = textInput
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .split(/\s+/);
  const paths = words.map(word => `/videos/${word}.mp4`);

    setVideoQueue(paths);       // Queue of videos
    setCurrentVideo(paths[0]);  // Start with the first video
    setIsTranslating(true);
    // if (!textInput.trim()) {
    //   showMessage("Please enter text or record audio first.", 'error');
    //   return;
    // }
    // setIsTranslating(true);
    // showMessage(`Translating "${textInput.substring(0, 20)}..." to sign language...`, 'info');
    // const word = textInput.trim().toLowerCase(); 
    // const videoPath = `/videos/${word}.mp4`; // videos folder in public
    // setVideoSrc(videoPath); // set video to play
    // // Simulate conversion time
    // setTimeout(() => {
    //   setIsTranslating(false);
    //   showMessage("Text-to-Sign conversion complete! Watch the avatar.", 'success');
    // }, 1000);
  };

  // Switch tab cleanup
  const handleTabChange = (option) => {
    setSelected(option);
    setIsRecording(false);
    setIsTranslating(false);
    setRecognizedText("");
    setTextInput("");
    setRecordingType(null);
    setMessage(null);
  }

  // Common styling for content containers
  const contentBoxClasses = "w-full bg-white dark:bg-[#0f0c29]/70 rounded-3xl p-6 sm:p-10 shadow-2xl shadow-purple-900/10 dark:shadow-purple-900/30 border border-purple-500/20 backdrop-blur-lg transition-colors duration-500";
  const buttonBaseClasses = "px-8 py-4 rounded-full flex items-center gap-3 text-white font-bold shadow-lg transition-all duration-300";
  const gradientButtonClasses = "bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] hover:scale-[1.03] active:scale-95 shadow-purple-700/50 dark:shadow-pink-500/30";
  const iconButtonClasses = "w-20 h-20 rounded-full flex items-center justify-center text-white shadow-xl hover:scale-105 active:scale-90 transition-all duration-300";
  const focusRing = "focus:outline-none focus:ring-4 focus:ring-purple-500/50";
  const disabledClasses = "opacity-50 cursor-not-allowed";

  return (
    <div className="w-full bg-gray-50 dark:bg-[#0f0c29] py-28 px-4 sm:px-6 lg:px-20 min-h-screen transition-colors duration-500">

      {/* PAGE TITLE */}
      <motion.div
        initial="hidden"
        whileInView="visible"
        variants={fadeUp}
        transition={{ duration: 0.7 }}
        viewport={{ once: true }}
        className="max-w-6xl mx-auto text-center mb-16"
      >
        <h2 className="
          text-5xl sm:text-6xl font-extrabold mb-6 
          bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
          bg-clip-text text-transparent
        ">
          Translation Center
        </h2>

        <p className="text-gray-700 dark:text-gray-400 text-lg sm:text-xl max-w-2xl mx-auto">
          Convert between <span className="font-semibold text-purple-600 dark:text-[#A044FF]">Sign Language</span>, <span className="font-semibold text-purple-600 dark:text-[#A044FF]">Text</span>, and <span className="font-semibold text-purple-600 dark:text-[#A044FF]">Audio</span> — all in real time.
        </p>
      </motion.div>

      {/* OPTION SWITCH */}
      <div className="flex justify-center gap-4 mb-16 flex-wrap">
        {options.map((option) => (
          <button
            key={option}
            onClick={() => handleTabChange(option)}
            disabled={isRecording || isTranslating}
            className={`
              px-8 py-3 rounded-full font-bold transition-all duration-300 text-lg
              ${focusRing}
              ${selected === option
                ? `${gradientButtonClasses} scale-105`
                : "bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600"}
              ${(isRecording || isTranslating) && disabledClasses}
            `}
          >
            {option}
          </button>
        ))}
      </div>

      {/* MAIN CONTENT */}
      <motion.div
        key={selected}
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
        className="max-w-5xl mx-auto"
      >
        {/* ----------------------- SIGN TO TEXT ----------------------- */}
        {selected === "Sign to Text" && (
          <div className="flex flex-col items-center gap-10">

            <div className={contentBoxClasses}>
              <h3 className="text-3xl font-extrabold text-center mb-10 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] bg-clip-text text-transparent">
                Sign Language → Text Recognition
              </h3>

              {/* Video Feed and Status */}
              <div className="relative w-full aspect-video rounded-xl overflow-hidden bg-gray-200 dark:bg-gray-800 border-4 border-gray-300 dark:border-purple-600/50 mb-8 shadow-inner">
                {/* ALWAYS show video element, but control visibility */}
                <video
                  ref={videoRef}
                  className={`w-full h-full object-cover ${isRecording && recordingType === 'camera' ? 'block' : 'hidden'}`}
                  playsInline
                  muted
                  autoPlay
                />
                
                {/* Show placeholder ONLY when not recording */}
                {!isRecording || recordingType !== 'camera' ? (
                  <div className="w-full h-full flex flex-col items-center justify-center">
                    <FaVideo className="text-5xl text-gray-400 dark:text-purple-700/50 mb-4" />
                    <p className="text-gray-600 dark:text-gray-400 text-sm">
                      Ready to receive sign language video input
                    </p>
                  </div>
                ) : null}
                
                {/* Hidden canvas for AI processing */}
                <canvas
                  ref={canvasRef}
                  className="hidden"
                />
                
                {/* Recording Indicator */}
                {isRecording && recordingType === 'camera' && (
                  <div className="absolute top-4 right-4 flex items-center gap-2 text-white bg-red-600 px-3 py-1 rounded-full text-sm font-semibold shadow-xl">
                    <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
                    Recording...
                  </div>
                )}
              </div>

              {/* 🔄 AI Progress Indicator */}
              {predictionProgress > 0 && predictionProgress < 100 && (
                <div className="mt-4 text-center">
                  <p className="text-sm text-purple-500 font-semibold">
                    Processing: {predictionProgress}%
                  </p>

                  <div className="w-full bg-gray-300 rounded-full h-2 mt-2">
                    <div
                      className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${predictionProgress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Action Button */}
              <div className="flex justify-center mt-6">
                <button
                  onClick={handleVideoRecord}
                  disabled={isTranslating || (isRecording && recordingType !== 'camera')}
                  className={`${buttonBaseClasses} ${gradientButtonClasses} ${isRecording && recordingType === 'camera' ? 'from-red-600 to-red-800' : ''} ${isTranslating && disabledClasses} ${focusRing}`}
                >
                  {isRecording && recordingType === 'camera' ? (
                    <>
                      <FaStopCircle className="text-xl" /> Stop Recording
                    </>
                  ) : isTranslating ? (
                    <>
                      <FaSpinner className="text-xl animate-spin" /> Analyzing Sign...
                    </>
                  ) : (
                    <>
                      <FaCamera className="text-xl" /> Start Camera
                    </>
                  )}
                </button>
              </div>

              {/* Result Area */}
              <div className="mt-10">
                 <h4 className="text-xl font-semibold mb-4 text-gray-800 dark:text-white border-b border-purple-500/50 pb-2">Recognized Text:</h4>
                 <div className={`
                    w-full min-h-32 p-5 rounded-xl border-2 border-gray-300 dark:border-gray-700
                    bg-gray-100 dark:bg-gray-800 shadow-inner
                    text-gray-900 dark:text-gray-200 text-lg tracking-wide transition-colors
                    ${recognizedText? 'text-purple-600 dark:text-pink-300 font-medium' : 'text-gray-500 dark:text-gray-500 italic'}
                 `}>
                    {recognizedText || (
                      <div className="flex items-center gap-3">
                        <HiOutlineLightBulb className="text-2xl"/>
                        Detected signs will be converted to text here.
                      </div>
                    )}
                 </div>
              </div>
            </div>
          </div>
        )}

        {/* ------------------- TEXT / AUDIO TO SIGN ------------------- */}
        {selected === "Text / Audio to Sign" && (
          <div className="flex flex-col items-center gap-10">
            <div className={contentBoxClasses}>
              <h3 className="text-3xl font-extrabold text-center mb-10 bg-gradient-to-r from-purple-500 to-pink-400 bg-clip-text text-transparent">
                Text / Audio → Sign Language Avatar
              </h3>

              {/* TEXT INPUT */}
              <textarea
                rows={4}
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                placeholder="Type your message here to see it signed by the avatar..."
                disabled={isRecording || isTranslating}
                className={`
                  w-full p-5 rounded-xl border border-gray-300 dark:border-gray-700
                  bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-200
                  shadow-md focus:outline-none focus:ring-2 focus:ring-purple-500 transition resize-none
                  text-lg
                  ${(isRecording || isTranslating) && disabledClasses}
                `}
              />

              {/* AUDIO RECORD / CONVERT BUTTONS */}
              <div className="flex justify-center items-center gap-8 mt-6">
                 {/* Audio Record Button */}
                <button
                  onClick={handleAudioRecord}
                  disabled={isTranslating || (isRecording && recordingType !== 'mic')}
                  className={`${iconButtonClasses} bg-purple-600 hover:bg-purple-700 ${focusRing} ${isTranslating && disabledClasses} ${isRecording && recordingType === 'mic' ? 'bg-red-600 hover:bg-red-700' : ''}`}
                  aria-label={isRecording && recordingType === 'mic' ? "Stop Audio Recording" : "Start Audio Recording"}
                >
                  {isRecording && recordingType === 'mic' ? (
                    <motion.span 
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                    >
                        <FaStopCircle className="text-3xl" />
                    </motion.span>
                  ) : (
                    <FaMicrophone className="text-3xl" />
                  )}
                </button>

                <div className="text-2xl font-bold text-gray-500 dark:text-gray-400">OR</div>

                {/* Convert Button */}
                <button 
                    onClick={handleConvertText}
                    disabled={isTranslating || isRecording || !textInput.trim()}
                    className={`${buttonBaseClasses} ${gradientButtonClasses} ${(isTranslating || isRecording || !textInput.trim()) && disabledClasses} ${focusRing}`}
                >
                  {isTranslating ? (
                    <>
                      <FaSpinner className="text-xl animate-spin" /> Generating Signs...
                    </>
                  ) : (
                    <>
                      <FaSyncAlt className="text-xl" /> Convert to Sign
                    </>
                  )}
                </button>
              </div>

              {/* AVATAR PREVIEW */}
              <div className="
                mt-10 w-full h-64 rounded-2xl border border-gray-300 dark:border-purple-600/50
                bg-gray-100 dark:bg-gray-800 shadow-inner flex flex-col items-center justify-center
                text-gray-500 dark:text-gray-400 text-center text-lg tracking-wide
              ">
                {/* <img 
                    src={`https://placehold.co/150x200/4c3093/ffffff?text=3D+Avatar`} 
                    alt="Sign Language Avatar Placeholder" 
                    className="rounded-lg mb-3"
                /> */}
                {/* <p>Sign Language Avatar Preview</p> */}
                {currentVideo ? (
                <video
                  key={currentVideo}
                  src={currentVideo}
                  autoPlay
                  muted
                  className="rounded-lg w-full h-full object-contain mb-3"
                  onEnded={() => {
                    const nextIndex = videoQueue.indexOf(currentVideo) + 1;
                    if (nextIndex < videoQueue.length) {
                      setCurrentVideo(videoQueue[nextIndex]);
                    } else {
                      setCurrentVideo(null);   // finished all videos
                      setVideoQueue([]);   
                      setIsTranslating(false); // stop translation animation
                    }
                  }}
                  onError={() => {
                    // Video not found → show placeholder message
                    console.warn(`Video not found: ${currentVideo}`);
                    showMessage(`No sign available for "${textInput.split(" ")[videoQueue.indexOf(currentVideo)]}"`, 'warning');
                    
                    // Move to next video or stop
                    const nextIndex = videoQueue.indexOf(currentVideo) + 1;
                    if (nextIndex < videoQueue.length) {
                      setCurrentVideo(videoQueue[nextIndex]);
                    } else {
                      setCurrentVideo(null);
                      setVideoQueue([]); 
                      setIsTranslating(false);
                    }
                  }}
                />
              ) : (
                <>
                  <img 
                    src={`https://placehold.co/150x200/4c3093/ffffff?text=3D+Avatar`} 
                    alt="Sign Language Avatar Placeholder" 
                    className="rounded-lg mb-3"
                  />
                  <p>No sign available or translation finished</p>
                </>
              )}
                              
                {isTranslating && <p className="mt-2 text-purple-500 flex items-center gap-2"><FaSpinner className="animate-spin"/> Animating...</p>}
              </div>
            </div>
          </div>
        )}
      </motion.div>
      
      {/* GLOBAL MESSAGE TOAST */}
      <AnimatePresence>
        {message && <Toast message={message.text} type={message.type} />}
      </AnimatePresence>
      
    </div>
  );
}