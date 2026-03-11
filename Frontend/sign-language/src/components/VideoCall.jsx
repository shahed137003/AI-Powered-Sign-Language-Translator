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
  BsStopCircle
} from "react-icons/bs";

const VideoCall = ({ 
  isOpen, 
  onClose, 
  remoteUsername, 
  localUsername,
  callType = "video",
  ws,
  isCaller = false,
  callId
}) => {
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
  
  const localVideoRef = useRef(null);
  const remoteVideoRef = useRef(null);
  const peerConnectionRef = useRef(null);
  const durationIntervalRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);

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
    };
  }, [isOpen]);

  useEffect(() => {
    if (localVideoRef.current && localStream) {
      localVideoRef.current.srcObject = localStream;
    }
  }, [localStream]);

  useEffect(() => {
    if (remoteVideoRef.current && remoteStream) {
      remoteVideoRef.current.srcObject = remoteStream;
      setIsConnecting(false);
      
      // Start call duration timer
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
      // Get user media
      const stream = await navigator.mediaDevices.getUserMedia({
        video: callType === "video",
        audio: true
      });
      
      setLocalStream(stream);
      
      // Create peer connection
      const pc = new RTCPeerConnection(configuration);
      peerConnectionRef.current = pc;
      
      // Expose peer connection for ICE candidate handling
      window.videoCallPeerConnection = pc;
      
      // Add local tracks
      stream.getTracks().forEach(track => {
        pc.addTrack(track, stream);
      });
      
      // Handle ICE candidates
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
      
      // Handle connection state changes
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
      
      // Handle remote stream
      pc.ontrack = (event) => {
        console.log("Received remote track:", event.track.kind);
        setRemoteStream(event.streams[0]);
        
        // Monitor remote track enabled states
        if (event.track.kind === 'video') {
          event.track.onmute = () => setRemoteVideoMuted(true);
          event.track.onunmute = () => setRemoteVideoMuted(false);
        }
        if (event.track.kind === 'audio') {
          event.track.onmute = () => setRemoteAudioMuted(true);
          event.track.onunmute = () => setRemoteAudioMuted(false);
        }
      };
      
      // Handle negotiation needed
      pc.onnegotiationneeded = async () => {
        try {
          if (isCaller && ws && ws.readyState === WebSocket.OPEN) {
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            
            ws.send(JSON.stringify({
              type: "call-offer",
              target: remoteUsername,
              offer: pc.localDescription,
              callId: callId,
              callType: callType
            }));
          }
        } catch (err) {
          console.error("Negotiation error:", err);
        }
      };
      
      // If caller, create and send offer after a short delay to ensure everything is ready
      if (isCaller && ws && ws.readyState === WebSocket.OPEN) {
        setTimeout(async () => {
          try {
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            
            ws.send(JSON.stringify({
              type: "call-offer",
              target: remoteUsername,
              offer: pc.localDescription,
              callId: callId,
              callType: callType
            }));
          } catch (err) {
            console.error("Error creating offer:", err);
          }
        }, 1000);
      }
      
    } catch (err) {
      console.error("Error initializing call:", err);
      setConnectionError(err.message);
      setIsConnecting(false);
    }
  };

  const cleanupCall = () => {
    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current);
    }
    
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
    }
    
    if (localStream) {
      localStream.getTracks().forEach(track => {
        track.stop();
        console.log(`Stopped ${track.kind} track`);
      });
    }
    
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
    }
    
    if (ws && ws.readyState === WebSocket.OPEN && callId) {
      ws.send(JSON.stringify({
        type: "call-end",
        target: remoteUsername,
        callId: callId
      }));
    }
    
    // Clean up window reference
    delete window.videoCallPeerConnection;
  };

  const toggleVideo = () => {
    if (localStream) {
      const videoTrack = localStream.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.enabled = !isVideoEnabled;
        setIsVideoEnabled(!isVideoEnabled);
      }
    }
  };

  const toggleAudio = () => {
    if (localStream) {
      const audioTrack = localStream.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = !isAudioEnabled;
        setIsAudioEnabled(!isAudioEnabled);
      }
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
    
    // Add local audio if available
    if (localStream) {
      localStream.getAudioTracks().forEach(track => {
        combinedStream.addTrack(track);
      });
    }
    
    // Add remote audio and video
    if (remoteStream) {
      remoteStream.getTracks().forEach(track => {
        combinedStream.addTrack(track);
      });
    }
    
    try {
      const mediaRecorder = new MediaRecorder(combinedStream, {
        mimeType: 'video/webm;codecs=vp9,opus'
      });
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };
      
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
    } catch (err) {
      console.error("Error starting recording:", err);
      alert("Recording failed to start");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const endCall = () => {
    cleanupCall();
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
        {/* Main Remote Video */}
        <div className="relative w-full h-full">
          {remoteStream ? (
            <>
              <video
                ref={remoteVideoRef}
                autoPlay
                playsInline
                className="w-full h-full object-cover"
              />
              {remoteVideoMuted && (
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-black/50 text-white px-4 py-2 rounded-full">
                  Remote video muted
                </div>
              )}
            </>
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800">
              <div className="text-center text-white">
                {isConnecting ? (
                  <>
                    <div className="w-24 h-24 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-6"></div>
                    <p className="text-2xl font-semibold">Connecting to {remoteUsername}...</p>
                    <p className="text-gray-400 mt-3">Please wait</p>
                  </>
                ) : connectionError ? (
                  <>
                    <div className="text-7xl mb-6">📞</div>
                    <p className="text-2xl text-red-500 mb-4">{connectionError}</p>
                    <button
                      onClick={onClose}
                      className="px-8 py-3 bg-purple-500 text-white rounded-full hover:bg-purple-600 transition-colors font-semibold"
                    >
                      Close
                    </button>
                  </>
                ) : (
                  <>
                    <div className="text-7xl mb-6 animate-pulse">
                      {callType === "video" ? "📹" : "📞"}
                    </div>
                    <p className="text-3xl font-semibold mb-2">{remoteUsername}</p>
                    <p className="text-gray-400 text-lg">Ringing...</p>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Local Video (Picture-in-Picture) */}
          {localStream && callType === "video" && (
            <motion.div 
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="absolute bottom-28 right-4 w-56 h-72 bg-gray-800 rounded-xl overflow-hidden border-2 border-purple-500 shadow-2xl"
            >
              <video
                ref={localVideoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              {!isVideoEnabled && (
                <div className="absolute inset-0 bg-gray-900/80 flex items-center justify-center">
                  <BsCameraVideoOff className="text-white text-3xl" />
                </div>
              )}
              <div className="absolute bottom-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded">
                You
              </div>
            </motion.div>
          )}

          {/* Call Controls */}
          <motion.div 
            initial={{ y: 100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="absolute bottom-8 left-1/2 transform -translate-x-1/2 flex items-center gap-4 bg-gray-900/90 backdrop-blur-md px-6 py-4 rounded-full border border-gray-700 shadow-2xl"
          >
            {/* Audio Toggle */}
            <button
              onClick={toggleAudio}
              className={`p-4 rounded-full transition-all transform hover:scale-110 ${
                isAudioEnabled 
                  ? 'bg-gray-700 hover:bg-gray-600 text-white' 
                  : 'bg-red-500 hover:bg-red-600 text-white'
              }`}
              title={isAudioEnabled ? "Mute microphone" : "Unmute microphone"}
            >
              {isAudioEnabled ? <BsMic size={22} /> : <BsMicMute size={22} />}
            </button>

            {/* Video Toggle (only for video calls) */}
            {callType === "video" && (
              <button
                onClick={toggleVideo}
                className={`p-4 rounded-full transition-all transform hover:scale-110 ${
                  isVideoEnabled 
                    ? 'bg-gray-700 hover:bg-gray-600 text-white' 
                    : 'bg-red-500 hover:bg-red-600 text-white'
                }`}
                title={isVideoEnabled ? "Turn off camera" : "Turn on camera"}
              >
                {isVideoEnabled ? <BsCameraVideo size={22} /> : <BsCameraVideoOff size={22} />}
              </button>
            )}

            {/* Recording Toggle (only when connected) */}
            {remoteStream && (
              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={`p-4 rounded-full transition-all transform hover:scale-110 ${
                  isRecording 
                    ? 'bg-red-500 hover:bg-red-600 text-white animate-pulse' 
                    : 'bg-gray-700 hover:bg-gray-600 text-white'
                }`}
                title={isRecording ? "Stop recording" : "Start recording"}
              >
                {isRecording ? <BsStopCircle size={22} /> : <BsRecordCircle size={22} />}
              </button>
            )}

            {/* End Call */}
            <button
              onClick={endCall}
              className="p-4 rounded-full bg-red-500 hover:bg-red-600 text-white transition-all transform hover:scale-110"
              title="End call"
            >
              <BsTelephoneX size={22} />
            </button>

            {/* Fullscreen Toggle */}
            <button
              onClick={toggleFullscreen}
              className="p-4 rounded-full bg-gray-700 hover:bg-gray-600 text-white transition-all transform hover:scale-110"
              title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            >
              {isFullscreen ? <BsFullscreenExit size={22} /> : <BsFullscreen size={22} />}
            </button>
          </motion.div>

          {/* Call Info */}
          <motion.div 
            initial={{ y: -100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="absolute top-4 left-4 bg-gray-900/80 backdrop-blur-sm px-5 py-2.5 rounded-full border border-gray-700"
          >
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
              {remoteAudioMuted && (
                <>
                  <div className="w-px h-4 bg-gray-600"></div>
                  <span className="text-yellow-500 text-sm">Remote audio muted</span>
                </>
              )}
            </div>
          </motion.div>

          {/* Back Button (for mobile) */}
          <button
            onClick={onClose}
            className="absolute top-4 right-4 md:hidden p-3 bg-gray-900/80 backdrop-blur-sm rounded-full border border-gray-700 text-white hover:bg-gray-800 transition-colors"
          >
            <BsArrowLeft size={22} />
          </button>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default VideoCall;