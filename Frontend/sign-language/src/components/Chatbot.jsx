import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  BsRobot, 
  BsSendFill, 
  BsMicFill, 
  BsMicMuteFill,
  BsStars,
  BsLightningFill,
  BsArrowLeft,
  BsTrash,
  BsDownload
} from 'react-icons/bs';
import { 
  FaUser, 
  FaRegCopy, 
  FaCheck, 
  FaRegThumbsUp, 
  FaRegThumbsDown,
  FaExpand,
  FaCompress
} from 'react-icons/fa';
import { TbHandLoveYou } from 'react-icons/tb';
import { GiArtificialIntelligence } from 'react-icons/gi';
import { useNavigate } from 'react-router-dom';

export default function Chatbot() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState([
    { 
      id: 1, 
      text: "Hello! I'm your AI sign language assistant. How can I help you today?", 
      sender: 'ai', 
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) 
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Sample AI responses
  const aiResponses = [
    "That's an interesting question about sign language!",
    "I can help you translate that gesture. The sign for that is...",
    "Based on American Sign Language (ASL), that gesture means...",
    "Would you like me to show you the proper hand positioning?",
    "That's a common sign! It's done by...",
    "Remember to keep your fingers relaxed when signing that.",
    "Great question! In sign language, that's expressed by..."
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = {
      id: messages.length + 1,
      text: input,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Simulate AI response delay
    setTimeout(() => {
      const aiResponse = aiResponses[Math.floor(Math.random() * aiResponses.length)];
      const aiMessage = {
        id: messages.length + 2,
        text: aiResponse,
        sender: 'ai',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, aiMessage]);
      setIsLoading(false);
    }, 1000);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleVoiceInput = () => {
    if (!isRecording) {
      setIsRecording(true);
      // Simulate voice recording
      setTimeout(() => {
        setIsRecording(false);
        setInput("How do I sign 'thank you' in American Sign Language?");
      }, 2000);
    } else {
      setIsRecording(false);
    }
  };

  const handleClearChat = () => {
    setMessages([
      { 
        id: 1, 
        text: "Hello! I'm your AI sign language assistant. How can I help you today?", 
        sender: 'ai', 
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) 
      }
    ]);
  };

  const handleCopyText = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      chatContainerRef.current?.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const handleFeedback = (messageId, type) => {
    // In a real app, you would send this feedback to your backend
    console.log(`Feedback ${type} for message ${messageId}`);
  };

  const fadeUp = {
    hidden: { opacity: 0, y: 40 },
    show: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.8,
        ease: [0.22, 1, 0.36, 1]
      }
    }
  };

  const fade = {
    hidden: { opacity: 0 },
    show: { 
      opacity: 1,
      transition: {
        duration: 1,
        ease: "easeOut"
      }
    }
  };

  const scaleIn = {
    hidden: { opacity: 0, scale: 0.8 },
    show: { 
      opacity: 1, 
      scale: 1,
      transition: {
        duration: 0.6,
        ease: "backOut"
      }
    }
  };

  return (
    <div className="relative w-full min-h-screen bg-gradient-to-br from-gray-50 via-white to-purple-50/60 dark:from-[#0a0518] dark:via-[#110a2e] dark:to-[#1e0f5c] overflow-hidden selection:bg-purple-500 selection:text-white transition-all duration-700">
      
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
      <div className="absolute top-1/4 left-1/4 w-[400px] h-[400px] bg-gradient-to-r from-purple-600/20 via-purple-500/10 to-pink-500/10 rounded-full blur-[120px] pointer-events-none animate-pulse-slow" />
      <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-gradient-to-r from-pink-600/15 via-purple-400/10 to-blue-500/10 rounded-full blur-[120px] pointer-events-none" />

      <div className="relative z-10 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-20">
        
        {/* Header Section */}
        <motion.div
          initial="hidden"
          animate="show"
          variants={{
            hidden: { opacity: 0 },
            show: {
              opacity: 1,
              transition: {
                staggerChildren: 0.1,
                delayChildren: 0.2
              }
            }
          }}
          className="mb-8"
        >
          {/* Back Button & Premium Badge */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8">
            <motion.button
              variants={fadeUp}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => navigate('/')}
              className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-gradient-to-r from-purple-500/15 via-purple-400/10 to-purple-300/10 border border-purple-200/60 dark:border-purple-700/60 backdrop-blur-xl shadow-lg shadow-purple-500/10 hover:shadow-purple-500/20 transition-all"
            >
              <BsArrowLeft className="text-purple-500" />
              <span className="text-sm font-bold bg-gradient-to-r from-purple-600 via-purple-500 to-purple-400 bg-clip-text text-transparent">
                Back to Home
              </span>
            </motion.button>

            <motion.div
              variants={fadeUp}
              whileHover={{ scale: 1.05, rotate: 1 }}
              className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-gradient-to-r from-purple-500/15 via-purple-400/10 to-purple-300/10 border border-purple-200/60 dark:border-purple-700/60 backdrop-blur-xl shadow-lg shadow-purple-500/10"
            >
              <div className="relative">
                <span className="absolute animate-ping inline-flex h-3.5 w-3.5 rounded-full bg-purple-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-gradient-to-r from-purple-500 to-purple-400" />
              </div>
              <span className="text-sm font-bold bg-gradient-to-r from-purple-600 via-purple-500 to-purple-400 bg-clip-text text-transparent">
                AI Assistant v3.0 
              </span>
            </motion.div>
          </div>

          {/* Main Header */}
          <motion.h1
            variants={fadeUp}
            className="font-extrabold text-4xl sm:text-5xl lg:text-[56px] leading-tight mb-6 text-center"
          >
            <motion.span
              variants={fadeUp}
              className="block text-gray-900 dark:text-white"
            >
              AI Sign Language
            </motion.span>
            <motion.span
              variants={fadeUp}
              transition={{ delay: 0.1 }}
              className="block bg-gradient-to-r from-purple-600 to-purple-400 bg-clip-text text-transparent"
            >
              Assistant
            </motion.span>
          </motion.h1>

          <motion.p
            variants={fadeUp}
            transition={{ delay: 0.2 }}
            className="text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto text-center mb-8"
          >
            Your intelligent companion for sign language translation, learning, and communication assistance
          </motion.p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Left Sidebar - Features & Tools */}
          <motion.div
            variants={fade}
            initial="hidden"
            animate="show"
            className="lg:col-span-1 space-y-6"
          >
            {/* AI Capabilities Card */}
            <div className="p-6 rounded-2xl backdrop-blur-xl bg-white/90 dark:bg-white/10 border border-white/30 dark:border-white/10 shadow-xl shadow-purple-500/10">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <GiArtificialIntelligence className="text-purple-500" />
                AI Capabilities
              </h3>
              <div className="space-y-3">
                {[
                  { icon: <BsLightningFill />, text: "Real-time Translation", color: "text-green-500" },
                  { icon: <TbHandLoveYou />, text: "Gesture Recognition", color: "text-blue-500" },
                  { icon: <BsStars />, text: "Learning Assistance", color: "text-pink-500" },
                  { icon: <BsRobot />, text: "Context Understanding", color: "text-purple-500" }
                ].map((feature, i) => (
                  <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-gray-50/50 dark:bg-gray-900/30 hover:bg-gray-100/50 dark:hover:bg-gray-800/50 transition-colors">
                    <div className={`text-lg ${feature.color}`}>
                      {feature.icon}
                    </div>
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      {feature.text}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Actions */}
            <div className="p-6 rounded-2xl backdrop-blur-xl bg-gradient-to-br from-purple-50/80 to-pink-50/50 dark:from-purple-900/20 dark:to-pink-900/20 border border-purple-200/50 dark:border-purple-500/20">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                Quick Actions
              </h3>
              <div className="space-y-3">
                <button
                  onClick={handleClearChat}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-white/80 dark:bg-gray-800/80 border border-gray-300 dark:border-gray-700 rounded-xl text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-all"
                >
                  <BsTrash />
                  Clear Chat
                </button>
                <button
                  onClick={toggleFullscreen}
                  className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-purple-300/50 dark:border-purple-500/50 rounded-xl text-purple-600 dark:text-purple-400 hover:from-purple-500/30 hover:to-pink-500/30 transition-all"
                >
                  {isFullscreen ? <FaCompress /> : <FaExpand />}
                  {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
                </button>
                <button className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-purple-300/50 dark:border-purple-500/50 rounded-xl text-purple-600 dark:text-purple-400 hover:from-purple-500/30 hover:to-pink-500/30 transition-all">
                  <BsDownload />
                  Export Chat
                </button>
              </div>
            </div>
          </motion.div>

          {/* Main Chat Interface */}
          <motion.div
            variants={fade}
            initial="hidden"
            animate="show"
            ref={chatContainerRef}
            className="lg:col-span-3"
          >
            <div className="h-[600px] flex flex-col rounded-2xl backdrop-blur-xl bg-white/90 dark:bg-white/10 border border-white/30 dark:border-white/10 shadow-2xl shadow-purple-500/20 overflow-hidden">
              
              {/* Chat Header */}
              <div className="p-6 border-b border-gray-200/50 dark:border-gray-800/50 bg-gradient-to-r from-purple-50/50 to-pink-50/50 dark:from-purple-900/10 dark:to-pink-900/10">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] flex items-center justify-center">
                    <BsRobot className="text-2xl text-white" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                      Sign Language AI Assistant
                    </h2>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Online • Powered by Deep Learning
                    </p>
                  </div>
                </div>
              </div>

              {/* Messages Container */}
              <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {messages.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`max-w-[80%] ${message.sender === 'user' ? 'order-2' : 'order-1'}`}>
                      <div className="flex items-center gap-2 mb-1">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                          message.sender === 'user' 
                            ? 'bg-gradient-to-r from-blue-500 to-blue-600' 
                            : 'bg-gradient-to-r from-purple-500 to-purple-600'
                        }`}>
                          {message.sender === 'user' ? 
                            <FaUser className="text-sm text-white" /> : 
                            <BsRobot className="text-sm text-white" />
                          }
                        </div>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {message.sender === 'user' ? 'You' : 'AI Assistant'} • {message.timestamp}
                        </span>
                      </div>
                      <div className={`rounded-2xl p-4 ${
                        message.sender === 'user'
                          ? 'bg-gradient-to-r from-blue-500/10 to-blue-600/10 border border-blue-200/50 dark:border-blue-800/50'
                          : 'bg-gradient-to-r from-purple-500/10 to-purple-600/10 border border-purple-200/50 dark:border-purple-800/50'
                      }`}>
                        <p className="text-gray-800 dark:text-gray-200 whitespace-pre-wrap">
                          {message.text}
                        </p>
                        <div className="flex items-center justify-end gap-2 mt-3">
                          {message.sender === 'ai' && (
                            <>
                              <button
                                onClick={() => handleCopyText(message.text, message.id)}
                                className="p-2 rounded-lg hover:bg-white/20 dark:hover:bg-black/20 transition-colors"
                                title="Copy text"
                              >
                                {copiedId === message.id ? (
                                  <FaCheck className="text-green-500" />
                                ) : (
                                  <FaRegCopy className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" />
                                )}
                              </button>
                              <button
                                onClick={() => handleFeedback(message.id, 'like')}
                                className="p-2 rounded-lg hover:bg-white/20 dark:hover:bg-black/20 transition-colors"
                                title="Helpful"
                              >
                                <FaRegThumbsUp className="text-gray-400 hover:text-green-500" />
                              </button>
                              <button
                                onClick={() => handleFeedback(message.id, 'dislike')}
                                className="p-2 rounded-lg hover:bg-white/20 dark:hover:bg-black/20 transition-colors"
                                title="Not helpful"
                              >
                                <FaRegThumbsDown className="text-gray-400 hover:text-red-500" />
                              </button>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
                
                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex justify-start"
                  >
                    <div className="max-w-[80%]">
                      <div className="flex items-center gap-2 mb-1">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-purple-600 flex items-center justify-center">
                          <BsRobot className="text-sm text-white" />
                        </div>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          AI Assistant is typing...
                        </span>
                      </div>
                      <div className="rounded-2xl p-4 bg-gradient-to-r from-purple-500/10 to-purple-600/10 border border-purple-200/50 dark:border-purple-800/50">
                        <div className="flex space-x-2">
                          <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" />
                          <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce delay-100" />
                          <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce delay-200" />
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="p-6 border-t border-gray-200/50 dark:border-gray-800/50 bg-gradient-to-r from-gray-50/50 to-purple-50/50 dark:from-gray-900/10 dark:to-purple-900/10">
                <div className="flex items-end gap-3">
                  <div className="flex-1 relative">
                    <textarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Type your message here... Ask about sign language, translations, or learning tips"
                      className="w-full px-4 py-3 pl-12 pr-24 bg-white/80 dark:bg-gray-900/80 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 resize-none transition-all"
                      rows="2"
                    />
                    <div className="absolute left-4 top-3.5 text-gray-400 dark:text-gray-500">
                      <FaUser />
                    </div>
                    <div className="absolute right-4 top-3.5 flex items-center gap-2">
                      <button
                        onClick={handleVoiceInput}
                        className={`p-2 rounded-lg transition-all ${
                          isRecording
                            ? 'bg-red-500 text-white animate-pulse'
                            : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-300 dark:hover:bg-gray-600'
                        }`}
                        title={isRecording ? 'Stop recording' : 'Voice input'}
                      >
                        {isRecording ? <BsMicMuteFill /> : <BsMicFill />}
                      </button>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        Press Enter to send
                      </span>
                    </div>
                  </div>
                  <motion.button
                    onClick={handleSend}
                    disabled={!input.trim() || isLoading}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="p-4 rounded-xl bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] text-white shadow-lg shadow-purple-500/40 hover:shadow-purple-500/60 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <BsSendFill className="text-xl" />
                  </motion.button>
                </div>
                
                {/* Quick Prompts */}
                <div className="mt-4 flex flex-wrap gap-2">
                  {[
                    "How do I sign 'hello'?",
                    "Show me numbers in ASL",
                    "What's the sign for thank you?",
                    "Explain hand positioning"
                  ].map((prompt, i) => (
                    <button
                      key={i}
                      onClick={() => setInput(prompt)}
                      className="px-3 py-1.5 text-sm rounded-full bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-300/30 dark:border-purple-500/30 text-purple-600 dark:text-purple-400 hover:from-purple-500/20 hover:to-pink-500/20 transition-all"
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Tips & Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6"
        >
          <div className="p-6 rounded-2xl backdrop-blur-xl bg-gradient-to-r from-purple-500/5 to-transparent border border-purple-200/30 dark:border-purple-500/20">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-purple-500 to-purple-600 flex items-center justify-center">
                <BsLightningFill className="text-white" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">24/7</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Always Available</div>
              </div>
            </div>
          </div>
          <div className="p-6 rounded-2xl backdrop-blur-xl bg-gradient-to-r from-pink-500/5 to-transparent border border-pink-200/30 dark:border-pink-500/20">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-pink-500 to-pink-600 flex items-center justify-center">
                <BsStars className="text-white" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">50+</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Sign Language Variants</div>
              </div>
            </div>
          </div>
          <div className="p-6 rounded-2xl backdrop-blur-xl bg-gradient-to-r from-blue-500/5 to-transparent border border-blue-200/30 dark:border-blue-500/20">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-blue-500 to-blue-600 flex items-center justify-center">
                <TbHandLoveYou className="text-white" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">99%</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Accuracy Rate</div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Custom CSS for animations */}
      <style jsx>{`
        @keyframes pulse-slow {
          0%, 100% { opacity: 0.5; }
          50% { opacity: 0.8; }
        }
        .animate-pulse-slow {
          animation: pulse-slow 4s ease-in-out infinite;
        }
        
        /* Custom scrollbar */
        .overflow-y-auto {
          scrollbar-width: thin;
          scrollbar-color: rgba(168, 85, 247, 0.3) transparent;
        }
        
        .overflow-y-auto::-webkit-scrollbar {
          width: 6px;
        }
        
        .overflow-y-auto::-webkit-scrollbar-track {
          background: transparent;
        }
        
        .overflow-y-auto::-webkit-scrollbar-thumb {
          background-color: rgba(168, 85, 247, 0.3);
          border-radius: 20px;
        }
        
        .overflow-y-auto::-webkit-scrollbar-thumb:hover {
          background-color: rgba(168, 85, 247, 0.5);
        }
      `}</style>
    </div>
  );
}