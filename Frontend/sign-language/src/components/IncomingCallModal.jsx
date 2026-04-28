import React, { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BsCameraVideo, BsTelephone, BsTelephoneX, BsVolumeUp } from "react-icons/bs";

const IncomingCallModal = ({ isOpen, caller, callType, onAccept, onReject }) => {
  const audioRef = useRef(null);

useEffect(() => {
  if (isOpen) {
    // Simple beep using Web Audio API with proper cleanup
    let audioContext = null;
    try {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      
      oscillator.type = 'sine';
      oscillator.frequency.value = 440;
      gainNode.gain.value = 0.1;
      
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      
      oscillator.start();
      
      const stopSound = setTimeout(() => {
        if (audioContext) {
          audioContext.close().catch(console.error);
          audioContext = null;
        }
      }, 800);
      
      return () => {
        clearTimeout(stopSound);
        if (audioContext) {
          audioContext.close().catch(console.error);
        }
      };
    } catch (err) {
      console.log("Could not play ringtone:", err);
    }
  }
}, [isOpen]);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center"
      >
        <motion.div
          initial={{ scale: 0.9, y: 30 }}
          animate={{ scale: 1, y: 0 }}
          exit={{ scale: 0.9, y: 30 }}
          className="bg-gradient-to-br from-gray-900 via-purple-900/20 to-gray-900 rounded-3xl p-8 max-w-md w-full mx-4 border border-purple-500/30 shadow-2xl"
        >
          <div className="text-center">
            {/* Animated rings */}
            <div className="relative mb-6">
              <motion.div
                animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0, 0.5] }}
                transition={{ duration: 2, repeat: Infinity }}
                className="absolute inset-0 rounded-full bg-purple-500/20"
              ></motion.div>
              <motion.div
                animate={{ scale: [1, 1.3, 1], opacity: [0.3, 0, 0.3] }}
                transition={{ duration: 2, delay: 0.5, repeat: Infinity }}
                className="absolute inset-0 rounded-full bg-purple-500/30"
              ></motion.div>
              
              {/* Caller Avatar */}
              <div className="relative w-28 h-28 rounded-full bg-gradient-to-br from-purple-500 to-purple-600 flex items-center justify-center mx-auto">
                {callType === "video" ? (
                  <BsCameraVideo className="text-white text-5xl" />
                ) : (
                  <BsTelephone className="text-white text-5xl" />
                )}
              </div>
            </div>

            {/* Call Info */}
            <h3 className="text-3xl font-bold text-white mb-3">
              Incoming {callType === "video" ? "Video" : "Audio"} Call
            </h3>
            <p className="text-gray-300 text-xl mb-2">
              <span className="font-semibold text-purple-400">@{caller}</span>
            </p>
            <p className="text-gray-400 mb-8 flex items-center justify-center gap-2">
              <BsVolumeUp className="animate-pulse" />
              is calling you...
            </p>

            {/* Action Buttons */}
            <div className="flex gap-4">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={onReject}
                className="flex-1 py-4 px-4 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-xl font-semibold transition-all flex items-center justify-center gap-3 hover:shadow-lg hover:shadow-red-500/30"
              >
                <BsTelephoneX className="text-xl" />
                <span>Decline</span>
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={onAccept}
                className="flex-1 py-4 px-4 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-xl font-semibold transition-all flex items-center justify-center gap-3 hover:shadow-lg hover:shadow-green-500/30"
              >
                {callType === "video" ? <BsCameraVideo className="text-xl" /> : <BsTelephone className="text-xl" />}
                <span>Accept</span>
              </motion.button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default IncomingCallModal;