import React, { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  FaMicrophone, 
  FaCamera, 
  FaSyncAlt, 
  FaSpinner, 
  FaStopCircle, 
  FaVideo, 
  FaVolumeUp,
  FaPlay,
  FaPause,
  FaDownload,
  FaHands,
  FaKeyboard,
  FaRobot,
  FaMagic,
  FaWaveSquare,
  FaBrain
} from "react-icons/fa";
import { TbSparkles, TbHandClick, TbMessageLanguage, TbWaveSine } from "react-icons/tb";
import { HiOutlineLightBulb } from "react-icons/hi";

// --- Toast/Message Component ---
const Toast = ({ message, type, onClose }) => {
  if (!message) return null;

  const baseClasses = "fixed top-6 left-1/2 transform -translate-x-1/2 p-4 px-6 rounded-xl shadow-2xl z-[100] flex items-center gap-3 font-semibold text-white backdrop-blur-xl border border-white/20";
  let colorClasses = "";

  switch (type) {
    case 'success':
      colorClasses = "bg-gradient-to-r from-green-500/90 to-emerald-500/90 shadow-green-500/50";
      break;
    case 'warning':
      colorClasses = "bg-gradient-to-r from-yellow-500/90 to-amber-500/90 shadow-yellow-500/50";
      break;
    case 'error':
      colorClasses = "bg-gradient-to-r from-red-500/90 to-pink-500/90 shadow-red-500/50";
      break;
    case 'info':
    default:
      colorClasses = "bg-gradient-to-r from-[#6A3093]/90 via-[#A044FF]/90 to-[#BF5AE0]/90 shadow-purple-500/50";
      break;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -50 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -50 }}
      className={`${baseClasses} ${colorClasses}`}
    >
      {type === 'info' && <TbSparkles className="text-xl" />}
      <span>{message}</span>
      <div className="w-32 h-1 bg-white/30 rounded-full overflow-hidden ml-3">
        <motion.div
          initial={{ width: "100%" }}
          animate={{ width: "0%" }}
          transition={{ duration: 3, ease: "linear" }}
          className="h-full bg-white"
        />
      </div>
    </motion.div>
  );
};

export default function Translate() {
  const fadeUp = { 
    hidden: { opacity: 0, y: 25 }, 
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] }
    } 
  };
  
  const options = ["Sign to Text", "Text / Audio to Sign"];
  
  const [selected, setSelected] = useState(options[0]);
  const [textInput, setTextInput] = useState("");
  const [recognizedText, setRecognizedText] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [message, setMessage] = useState(null);
  const [recordingType, setRecordingType] = useState(null);
  const [avatarAnimation, setAvatarAnimation] = useState("idle");
  
  const showMessage = useCallback((text, type = 'info') => {
    setMessage({ text, type });
    setTimeout(() => setMessage(null), 3000);
  }, []);

  const handleRecord = (type) => {
    if (isRecording) {
      showMessage(`${type} recording stopped. Processing ${selected === 'Sign to Text' ? 'sign recognition' : 'audio transcription'}...`, 'info');
      setIsRecording(false);
      setRecordingType(null);
      setIsTranslating(true);

      setTimeout(() => {
        setIsTranslating(false);
        if (type === 'Camera') {
          setRecognizedText("Hello, how are you doing today? I love using LinguaSign for seamless communication!");
          showMessage("Sign-to-Text translation complete! Accuracy: 97.3%", 'success');
        } else {
          showMessage("Audio processed successfully! Ready for sign language conversion.", 'success');
        }
      }, 3000);

    } else {
      if (selected === "Sign to Text") {
        setRecognizedText("");
      }
      showMessage(`${type} recording started. ${type === 'Camera' ? 'Show clear hand signs!' : 'Speak clearly!'}`, 'warning');
      setIsRecording(true);
      setRecordingType(type === 'Camera' ? 'camera' : 'mic');
      
      setTimeout(() => {
        if (isRecording) {
          handleRecord(type);
        }
      }, 15000);
    }
  };

  const handleVideoRecord = () => handleRecord('Camera');
  const handleAudioRecord = () => handleRecord('Audio');

  const handleConvertText = () => {
    if (!textInput.trim()) {
      showMessage("Please enter text or record audio first.", 'error');
      return;
    }
    
    setAvatarAnimation("animating");
    setIsTranslating(true);
    showMessage(`Translating "${textInput.substring(0, 20)}..." to sign language...`, 'info');

    setTimeout(() => {
      setIsTranslating(false);
      setAvatarAnimation("completed");
      showMessage("Text-to-Sign conversion complete! Avatar animation ready.", 'success');
    }, 4000);
  };

  const handleTabChange = (option) => {
    setSelected(option);
    setIsRecording(false);
    setIsTranslating(false);
    setRecognizedText("");
    setTextInput("");
    setRecordingType(null);
    setAvatarAnimation("idle");
    setMessage(null);
  }

  // Common styling
  const contentBoxClasses = `
    w-full rounded-3xl p-6 sm:p-10 shadow-2xl backdrop-blur-xl
    bg-gradient-to-br from-white/80 to-white/60 dark:from-white/10 dark:to-white/5
    border border-purple-200/50 dark:border-purple-500/20
    shadow-purple-100/20 dark:shadow-purple-900/20
    transition-all duration-500
  `;

  const buttonBaseClasses = "px-8 py-4 rounded-full flex items-center gap-3 text-white font-bold shadow-lg transition-all duration-300 hover:scale-[1.03] active:scale-95";
  const gradientButtonClasses = "bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] shadow-purple-500/30 hover:shadow-purple-500/50";
  
  const iconButtonClasses = `
    w-20 h-20 rounded-full flex items-center justify-center text-white shadow-xl 
    hover:scale-105 active:scale-90 transition-all duration-300
    border border-white/20 backdrop-blur-sm
  `;

  return (
    <div className="w-full min-h-screen py-28 px-4 sm:px-6 lg:px-8 relative overflow-hidden bg-gradient-to-br from-gray-50 via-white to-purple-50/60 dark:from-[#0a0518] dark:via-[#110a2e] dark:to-[#1e0f5c]">
      
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

      {/* Animated glows */}
      <motion.div 
        animate={{ 
          x: [0, 50, 0],
          y: [0, 30, 0]
        }}
        transition={{ 
          duration: 20, 
          repeat: Infinity, 
          ease: "linear" 
        }}
        className="absolute top-1/4 left-1/4 w-[600px] h-[600px] bg-gradient-to-r from-purple-600/10 via-pink-600/10 to-indigo-600/10 rounded-full blur-[120px]"
      />

      {/* Floating particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(15)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-purple-500/20 rounded-full"
            initial={{ 
              x: Math.random() * 100 + 'vw', 
              y: Math.random() * 100 + 'vh',
              scale: 0 
            }}
            animate={{ 
              y: [null, -20, 20, -15],
              x: [null, 15, -15, 10],
              scale: [0, 1, 1, 0],
              opacity: [0, 0.5, 0.5, 0]
            }}
            transition={{ 
              duration: Math.random() * 8 + 15,
              repeat: Infinity,
              ease: "linear",
              delay: Math.random() * 3
            }}
          />
        ))}
      </div>

      {/* PAGE HEADER */}
      <div className="relative z-10 max-w-7xl mx-auto">
        <motion.div
          initial="hidden"
          whileInView="visible"
          variants={fadeUp}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          {/* Premium Badge */}
          <motion.div
            whileHover={{ scale: 1.05, rotate: 1 }}
            className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-gradient-to-r from-purple-500/15 via-purple-400/10 to-purple-300/10 border border-purple-200/60 dark:border-purple-700/60 backdrop-blur-xl shadow-lg shadow-purple-500/10 relative overflow-hidden group mb-8"
          >
            <div className="relative">
              <span className="absolute animate-ping inline-flex h-3.5 w-3.5 rounded-full bg-purple-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-gradient-to-r from-purple-500 to-purple-400" />
            </div>
            <span className="text-sm font-bold bg-gradient-to-r from-purple-600 via-purple-500 to-purple-400 bg-clip-text text-transparent">
              Real-time Translation
            </span>
            <TbSparkles className="text-purple-500 ml-1" />
            <div className="absolute -inset-1 bg-gradient-to-r from-purple-500/0 via-purple-400/10 to-purple-500/0 group-hover:via-purple-400/20 transition-all duration-500" />
          </motion.div>

          <motion.h1
            variants={fadeUp}
            className="font-extrabold text-4xl sm:text-5xl lg:text-[53px] leading-tight mb-6"
          >
            <span className="block text-gray-900 dark:text-white">
              AI-Powered
            </span>
            <span className="block bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
              Translation Center
            </span>
          </motion.h1>

          <motion.p
            variants={fadeUp}
            className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed"
          >
            Convert between <span className="font-semibold text-purple-600 dark:text-[#A044FF]">Sign Language</span>, 
            <span className="font-semibold text-purple-600 dark:text-[#A044FF]"> Text</span>, and 
            <span className="font-semibold text-purple-600 dark:text-[#A044FF]"> Speech</span> with cutting-edge AI technology.
          </motion.p>

          {/* Decorative Elements */}
          <motion.div
            variants={fadeUp}
            className="flex items-center justify-center gap-8 mt-10"
          >
            <div className="w-12 h-1 bg-gradient-to-r from-transparent via-purple-500 to-transparent rounded-full" />
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="w-6 h-6 rounded-full border-2 border-purple-400/50"
            />
            <div className="w-12 h-1 bg-gradient-to-r from-transparent via-purple-500 to-transparent rounded-full" />
          </motion.div>
        </motion.div>

        {/* OPTION SWITCH */}
        <div className="flex justify-center gap-6 mb-16 flex-wrap">
          {options.map((option) => (
            <motion.button
              key={option}
              onClick={() => handleTabChange(option)}
              disabled={isRecording || isTranslating}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className={`
                relative px-10 py-4 rounded-full font-bold transition-all duration-300 text-lg
                ${selected === option
                  ? `${gradientButtonClasses} text-white shadow-2xl`
                  : 'bg-gradient-to-r from-white/80 to-white/60 dark:from-white/10 dark:to-white/5 text-gray-700 dark:text-gray-200 border border-purple-200/50 dark:border-purple-500/20 hover:bg-white/90 dark:hover:bg-white/15'
                }
                ${(isRecording || isTranslating) && 'opacity-50 cursor-not-allowed'}
                backdrop-blur-sm
              `}
            >
              {selected === option && (
                <motion.div
                  layoutId="optionTab"
                  className="absolute inset-0 bg-gradient-to-r from-[#6A3093]/20 via-[#A044FF]/20 to-[#BF5AE0]/20 rounded-full"
                />
              )}
              <span className="relative z-10 flex items-center gap-3">
                {option === "Sign to Text" ? <FaHands /> : <TbMessageLanguage />}
                {option}
              </span>
            </motion.button>
          ))}
        </div>

        {/* MAIN CONTENT */}
        <motion.div
          key={selected}
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="max-w-6xl mx-auto"
        >
          {/* SIGN TO TEXT SECTION */}
          {selected === "Sign to Text" && (
            <div className="flex flex-col items-center gap-10">
              <div className={contentBoxClasses}>
                <div className="flex items-center justify-center gap-4 mb-10">
                  <div className="p-3 rounded-xl bg-gradient-to-br from-purple-100 to-purple-50 dark:from-[#6A3093]/20 dark:to-[#A044FF]/20">
                    <FaHands className="text-3xl text-purple-600 dark:text-purple-400" />
                  </div>
                  <h3 className="text-3xl font-extrabold text-center bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] bg-clip-text text-transparent">
                    Sign Language → Text Recognition
                  </h3>
                </div>

                {/* Video Feed and Status */}
                <div className="relative w-full aspect-video rounded-2xl overflow-hidden bg-gradient-to-br from-gray-200 to-gray-300 dark:from-gray-800 dark:to-gray-900 border-2 border-dashed border-gray-400/50 dark:border-purple-600/30 flex items-center justify-center mb-8 shadow-inner group">
                  {/* Simulated Video/Camera Area */}
                  <div className="relative z-10 text-center">
                    <FaVideo className="text-6xl text-gray-400 dark:text-purple-700/50 mb-4" />
                    <p className="text-gray-600 dark:text-gray-400 text-lg">
                      {isRecording && recordingType === 'camera' 
                        ? "Live Feed Active" 
                        : "Ready to receive sign language input"}
                    </p>
                  </div>
                  
                  {/* Animated background */}
                  <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 via-transparent to-pink-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                  
                  {/* Recording Indicator */}
                  <AnimatePresence>
                    {isRecording && recordingType === 'camera' && (
                      <motion.div 
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        transition={{ duration: 0.3 }}
                        className="absolute top-6 right-6 flex items-center gap-2 text-white bg-gradient-to-r from-red-600 to-pink-600 px-4 py-2 rounded-full text-sm font-semibold shadow-xl backdrop-blur-sm border border-red-400/30"
                      >
                        <motion.span 
                          animate={{ opacity: [0, 1, 0] }}
                          transition={{ duration: 1, repeat: Infinity }}
                          className="w-2 h-2 rounded-full bg-white"
                        />
                        Recording Live...
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Status indicator */}
                  <div className="absolute bottom-4 left-4 flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {isRecording ? 'AI Analyzing...' : 'Standby'}
                    </span>
                  </div>
                </div>

                {/* Action Button */}
                <div className="flex justify-center mt-6">
                  <motion.button
                    onClick={handleVideoRecord}
                    disabled={isTranslating || (isRecording && recordingType !== 'camera')}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className={`${buttonBaseClasses} ${gradientButtonClasses} ${isRecording && recordingType === 'camera' ? 'from-red-600 via-pink-600 to-red-800' : ''} ${isTranslating && 'opacity-50 cursor-not-allowed'} relative overflow-hidden group`}
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-white/5 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
                    <span className="relative z-10 flex items-center gap-3">
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
                    </span>
                  </motion.button>
                </div>

                {/* Result Area */}
                <div className="mt-10">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-xl font-semibold text-gray-800 dark:text-white flex items-center gap-2">
                      <TbWaveSine className="text-purple-600" />
                      Recognized Text
                    </h4>
                    {recognizedText && (
                      <button className="text-sm text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 transition-colors flex items-center gap-2">
                        <FaDownload size={14} /> Copy Result
                      </button>
                    )}
                  </div>
                  
                  <div className={`
                    w-full min-h-40 p-6 rounded-2xl border-2
                    bg-gradient-to-br from-gray-100 to-gray-50 dark:from-gray-800/50 dark:to-gray-900/50
                    shadow-inner
                    transition-all duration-500
                    ${recognizedText 
                      ? 'border-purple-300/50 dark:border-purple-500/30 text-purple-700 dark:text-purple-300' 
                      : 'border-gray-300/50 dark:border-gray-700/50 text-gray-500 dark:text-gray-500 italic'
                    }
                  `}>
                    {recognizedText ? (
                      <motion.p
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-lg leading-relaxed tracking-wide"
                      >
                        {recognizedText}
                      </motion.p>
                    ) : (
                      <div className="flex flex-col items-center justify-center h-full gap-3">
                        <HiOutlineLightBulb className="text-3xl text-gray-400 dark:text-gray-600" />
                        <p>Detected signs will be converted to text here</p>
                        <p className="text-sm text-gray-400 dark:text-gray-600">AI accuracy: 97.3%</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TEXT / AUDIO TO SIGN SECTION */}
          {selected === "Text / Audio to Sign" && (
            <div className="flex flex-col items-center gap-10">
              <div className={contentBoxClasses}>
                <div className="flex items-center justify-center gap-4 mb-10">
                  <div className="p-3 rounded-xl bg-gradient-to-br from-purple-100 to-purple-50 dark:from-[#6A3093]/20 dark:to-[#A044FF]/20">
                    <FaKeyboard className="text-3xl text-purple-600 dark:text-purple-400" />
                  </div>
                  <h3 className="text-3xl font-extrabold text-center bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] bg-clip-text text-transparent">
                    Text / Audio → Sign Language
                  </h3>
                </div>

                {/* TEXT INPUT */}
                <div className="relative group">
                  <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500/0 via-purple-400/0 to-purple-300/0 group-hover:via-purple-400/10 group-hover:opacity-100 opacity-0 rounded-2xl transition-all duration-300" />
                  <textarea
                    rows={4}
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    placeholder="Type your message here to see it signed by the avatar..."
                    disabled={isRecording || isTranslating}
                    className={`
                      relative w-full p-5 rounded-xl border 
                      bg-gradient-to-br from-white/70 to-white/50 dark:from-gray-800/50 dark:to-gray-900/50
                      text-gray-900 dark:text-gray-200
                      shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-500/50 transition resize-none
                      text-lg placeholder-gray-400 dark:placeholder-gray-600
                      border-gray-300/50 dark:border-gray-700/50
                      ${(isRecording || isTranslating) && 'opacity-50 cursor-not-allowed'}
                    `}
                  />
                </div>

                {/* AUDIO RECORD / CONVERT BUTTONS */}
                <div className="flex flex-col sm:flex-row justify-center items-center gap-8 mt-10">
                  {/* Audio Record Button */}
                  <div className="flex flex-col items-center gap-4">
                    <motion.button
                      onClick={handleAudioRecord}
                      disabled={isTranslating || (isRecording && recordingType !== 'mic')}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className={`
                        ${iconButtonClasses} 
                        ${isRecording && recordingType === 'mic' 
                          ? 'bg-gradient-to-r from-red-600 to-pink-600' 
                          : 'bg-gradient-to-r from-purple-600 to-pink-600'
                        }
                        ${isTranslating && 'opacity-50 cursor-not-allowed'}
                      `}
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
                    </motion.button>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {isRecording && recordingType === 'mic' ? 'Recording...' : 'Audio Input'}
                    </span>
                  </div>

                  <div className="text-2xl font-bold text-gray-500 dark:text-gray-400">OR</div>

                  {/* Convert Button */}
                  <div className="flex flex-col items-center gap-4">
                    <motion.button 
                      onClick={handleConvertText}
                      disabled={isTranslating || isRecording || !textInput.trim()}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      className={`${buttonBaseClasses} ${gradientButtonClasses} ${(isTranslating || isRecording || !textInput.trim()) && 'opacity-50 cursor-not-allowed'} relative overflow-hidden group`}
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-white/5 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
                      <span className="relative z-10 flex items-center gap-3">
                        {isTranslating ? (
                          <>
                            <FaSpinner className="text-xl animate-spin" /> Generating Signs...
                          </>
                        ) : (
                          <>
                            <FaMagic className="text-xl" /> Convert to Sign
                          </>
                        )}
                      </span>
                    </motion.button>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      AI-Powered Translation
                    </span>
                  </div>
                </div>

                {/* AVATAR PREVIEW */}
                <div className={`
                  mt-12 w-full h-72 rounded-2xl border
                  bg-gradient-to-br from-gray-100 to-gray-50 dark:from-gray-800/50 dark:to-gray-900/50
                  shadow-inner flex flex-col items-center justify-center
                  text-gray-500 dark:text-gray-400 text-center
                  transition-all duration-500
                  ${avatarAnimation === 'completed' 
                    ? 'border-purple-400/50 dark:border-purple-500/30 shadow-purple-500/10' 
                    : 'border-gray-300/50 dark:border-gray-700/50'
                  }
                `}>
                  <div className="relative">
                    {/* Avatar Container */}
                    <div className="relative w-40 h-40 rounded-full overflow-hidden border-4 border-white dark:border-gray-700 shadow-2xl mb-4">
                      <img 
                        src={`https://placehold.co/200x200/4c3093/ffffff?text=AI+Avatar`} 
                        alt="Sign Language Avatar" 
                        className="w-full h-full object-cover"
                      />
                      {/* Animation indicator */}
                      {avatarAnimation === 'animating' && (
                        <div className="absolute inset-0 bg-gradient-to-t from-purple-600/20 to-transparent flex items-center justify-center">
                          <div className="text-white text-sm font-semibold bg-purple-600/80 px-3 py-1 rounded-full">
                            Animating...
                          </div>
                        </div>
                      )}
                    </div>
                    
                    {/* Status Indicator */}
                    <div className="absolute -top-2 -right-2">
                      <div className={`
                        w-8 h-8 rounded-full flex items-center justify-center shadow-lg
                        ${avatarAnimation === 'completed' 
                          ? 'bg-green-500' 
                          : avatarAnimation === 'animating' 
                          ? 'bg-yellow-500 animate-pulse' 
                          : 'bg-gray-400'
                        }
                      `}>
                        {avatarAnimation === 'completed' && <FaPlay className="text-white text-xs" />}
                        {avatarAnimation === 'animating' && <FaSpinner className="text-white text-xs animate-spin" />}
                      </div>
                    </div>
                  </div>
                  
                  <p className="text-lg font-medium mt-4">Sign Language Avatar</p>
                  <p className="text-sm text-gray-400 dark:text-gray-600 mt-2">
                    {avatarAnimation === 'completed' 
                      ? 'Translation complete! Watch the animation.' 
                      : avatarAnimation === 'animating' 
                      ? 'Generating sign language animation...' 
                      : 'Ready for translation'
                    }
                  </p>
                  
                  {avatarAnimation === 'completed' && (
                    <div className="flex gap-3 mt-4">
                      <button className="px-4 py-2 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 text-white text-sm font-medium hover:opacity-90 transition-opacity">
                        <FaPlay className="inline mr-2" /> Play Animation
                      </button>
                      <button className="px-4 py-2 rounded-lg bg-gradient-to-r from-gray-600 to-gray-700 text-white text-sm font-medium hover:opacity-90 transition-opacity">
                        <FaDownload className="inline mr-2" /> Download
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </motion.div>
      </div>
      
      {/* GLOBAL MESSAGE TOAST */}
      <AnimatePresence>
        {message && <Toast message={message.text} type={message.type} />}
      </AnimatePresence>
      
    </div>
  );
}