import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BsRobot,
  BsSendFill,
  BsTrash,
  BsStars,
  BsPlus,
  BsLayoutSidebar,
  BsArrowLeft,
  BsLightningFill,
  BsDownload,
} from 'react-icons/bs';
import {
  FaUser,
  FaRegCopy,
  FaCheck,
  FaRegThumbsUp,
  FaRegThumbsDown,
  FaExpand,
  FaCompress,
} from 'react-icons/fa';
import { TbHandLoveYou, TbMessage2 } from 'react-icons/tb';
import { GiArtificialIntelligence } from 'react-icons/gi';
import { getApiUrl } from '../lib/api';
import axios from 'axios';

/* ─── helpers ─────────────────────────────────────────── */
const timestamp = () =>
  new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

const WELCOME = {
  id: 'welcome',
  role: 'ai',
  text: "Hi! I'm your AI sign language assistant 👋\n\nAsk me anything about sign language — translations, gestures, learning tips, or how the app works.",
  time: timestamp(),
};

const QUICK_PROMPTS = [
  "How do I sign 'hello'?",
  "What's ASL for 'thank you'?",
  "Explain hand positioning basics",
  "How accurate is the translation?",
  "What sign languages do you support?",
];

/* ─── component ───────────────────────────────────────── */
export default function Chatbot() {
  const [messages, setMessages] = useState([WELCOME]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [chatHistory, setChatHistory] = useState([
    { id: 1, title: 'Current conversation', active: true },
  ]);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const bottomRef = useRef(null);
  const textareaRef = useRef(null);
  const chatContainerRef = useRef(null);
  const API_URL = getApiUrl();

  /* auto-scroll */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  /* auto-grow textarea */
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  }, [input]);

  /* ── fullscreen state sync ── */
  useEffect(() => {
    const onFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', onFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', onFullscreenChange);
  }, []);

  /* ── export chat ── */
  const exportChat = useCallback(() => {
    if (messages.length === 0) return;
    const lines = messages.map(m =>
      `[${m.time}] ${m.role === 'user' ? 'You' : 'AI Assistant'}:\n${m.text}`
    );
    const content = lines.join('\n\n');
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${new Date().toISOString().slice(0,10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }, [messages]);

  /* send message */
  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    const userMsg = { id: Date.now(), role: 'user', text, time: timestamp() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await axios.post(`${API_URL}/chatbot/chat`, { message: text });
      const reply = res.data?.reply || res.data?.response || res.data?.message || 'Sorry, I could not process that.';
      setMessages(prev => [
        ...prev,
        { id: Date.now() + 1, role: 'ai', text: reply, time: timestamp() },
      ]);
    } catch (err) {
      const fallbackReplies = [
        "That's a great question about sign language! The AI backend is currently processing your request. Please make sure the backend server is running.",
        "I understand your query. For the best experience, ensure the backend server is connected via the Server Settings button in the navbar.",
        "Great question! The backend AI service handles your requests. If you see this, check your server connection in the app settings.",
      ];
      setMessages(prev => [
        ...prev,
        {
          id: Date.now() + 1,
          role: 'ai',
          text: fallbackReplies[Math.floor(Math.random() * fallbackReplies.length)],
          time: timestamp(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, API_URL]);

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleCopy = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleNewChat = () => {
    setMessages([WELCOME]);
    setInput('');
    setSidebarOpen(false);
  };

  const handleClearChat = () => {
    setMessages([WELCOME]);
    setInput('');
    setSidebarOpen(false);
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      chatContainerRef.current?.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  };

  const handleFeedback = (messageId, type) => {
    console.log(`Feedback ${type} for message ${messageId}`);
  };

  /* ─── render ─────────────────────────────────────────── */
  return (
    <div className="relative w-full min-h-screen bg-gradient-to-br from-gray-50 via-white to-primary-50/60 dark:from-primary-bg-1 dark:via-primary-bg-2 dark:to-primary-bg-3 overflow-hidden selection:bg-primary-500 selection:text-white transition-all duration-700">
      {/* Animated background orbs */}
      <div className="absolute inset-0 opacity-40 dark:opacity-60 pointer-events-none">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(90deg, var(--theme-grid-color) 1px, transparent 1px),
            linear-gradient(180deg, var(--theme-grid-color) 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px'
        }} />
      </div>
      <div className="absolute top-1/4 left-1/4 w-[400px] h-[400px] bg-gradient-to-r from-primary-600/20 via-primary-500/10 to-pink-500/10 rounded-full blur-[120px] pointer-events-none animate-pulse-slow" />
      <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-gradient-to-r from-pink-600/15 via-primary-400/10 to-blue-500/10 rounded-full blur-[120px] pointer-events-none" />

      <div className="relative z-10 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-20">
        {/* Header Section */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8">
          <button
            onClick={() => window.history.back()}
            className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-gradient-to-r from-primary-500/15 via-primary-400/10 to-primary-300/10 border border-primary-200/60 dark:border-primary-700/60 backdrop-blur-xl shadow-lg shadow-primary-500/10 hover:shadow-primary-500/20 transition-all"
          >
            <BsArrowLeft className="text-primary-500" />
            <span className="text-sm font-bold bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 bg-clip-text text-transparent">
              Back to Home
            </span>
          </button>
          <div className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-gradient-to-r from-primary-500/15 via-primary-400/10 to-primary-300/10 border border-primary-200/60 dark:border-primary-700/60 backdrop-blur-xl shadow-lg shadow-primary-500/10">
            <div className="relative">
              <span className="absolute animate-ping inline-flex h-3.5 w-3.5 rounded-full bg-primary-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-gradient-to-r from-primary-500 to-primary-400" />
            </div>
            <span className="text-sm font-bold bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 bg-clip-text text-transparent">
              AI Assistant v3.0
            </span>
          </div>
        </div>

        {/* Main grid: sidebar + chat */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-6">
            {/* Slide-out sidebar for mobile */}
            <AnimatePresence>
              {sidebarOpen && (
                <>
                  <motion.aside
                    initial={{ x: -280, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    exit={{ x: -280, opacity: 0 }}
                    transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                    className="fixed left-0 top-[64px] bottom-0 w-[260px] z-30
                      bg-gray-50 dark:bg-[#171717]
                      border-r border-gray-200 dark:border-gray-800
                      flex flex-col shadow-2xl lg:hidden"
                  >
                    {/* Sidebar content (same as before) */}
                    <div className="p-3 border-b border-gray-200 dark:border-gray-800">
                      <button
                        onClick={handleNewChat}
                        className="w-full flex items-center gap-3 px-4 py-3 rounded-xl
                          border border-gray-300 dark:border-gray-700
                          text-gray-700 dark:text-gray-300 text-sm font-medium
                          hover:bg-white dark:hover:bg-gray-800 transition-all group"
                      >
                        <BsPlus className="text-xl group-hover:text-primary-500 transition-colors" />
                        New chat
                      </button>
                    </div>
                    <div className="flex-1 overflow-y-auto p-3 space-y-1">
                      <p className="text-xs font-semibold text-gray-400 dark:text-gray-500 px-3 mb-2 uppercase tracking-wider">
                        Today
                      </p>
                      {chatHistory.map(ch => (
                        <button
                          key={ch.id}
                          className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-all
                            ${ch.active
                              ? 'bg-white dark:bg-gray-800 text-gray-900 dark:text-white shadow-sm'
                              : 'text-gray-600 dark:text-gray-400 hover:bg-white dark:hover:bg-gray-800'
                            }`}
                        >
                          <TbMessage2 className="inline mr-2 opacity-60" />
                          {ch.title}
                        </button>
                      ))}
                    </div>
                    <div className="p-3 border-t border-gray-200 dark:border-gray-800">
                      <button
                        onClick={handleClearChat}
                        className="w-full flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm
                          text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 transition-all"
                      >
                        <BsTrash />
                        Clear conversations
                      </button>
                    </div>
                  </motion.aside>
                  <div
                    className="fixed inset-0 z-20 bg-black/30 lg:hidden"
                    onClick={() => setSidebarOpen(false)}
                  />
                </>
              )}
            </AnimatePresence>

            {/* Always-visible sidebar on large screens */}
            <div className="hidden lg:block space-y-6">
              {/* AI Capabilities Card */}
              <div className="p-6 rounded-2xl backdrop-blur-xl bg-white/90 dark:bg-white/10 border border-white/30 dark:border-white/10 shadow-xl shadow-primary-500/10">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                  <GiArtificialIntelligence className="text-primary-500" />
                  AI Capabilities
                </h3>
                <div className="space-y-3">
                  {[
                    { icon: <BsLightningFill />, text: "Real-time Translation", color: "text-green-500" },
                    { icon: <TbHandLoveYou />, text: "Gesture Recognition", color: "text-blue-500" },
                    { icon: <BsStars />, text: "Learning Assistance", color: "text-pink-500" },
                    { icon: <BsRobot />, text: "Context Understanding", color: "text-primary-500" }
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
              <div className="p-6 rounded-2xl backdrop-blur-xl bg-gradient-to-br from-primary-50/80 to-pink-50/50 dark:from-primary-900/20 dark:to-pink-900/20 border border-primary-200/50 dark:border-primary-500/20">
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
                    className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-primary-500/20 to-pink-500/20 border border-primary-300/50 dark:border-primary-500/50 rounded-xl text-primary-600 dark:text-primary-400 hover:from-primary-500/30 hover:to-pink-500/30 transition-all"
                  >
                    {isFullscreen ? <FaCompress /> : <FaExpand />}
                    {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
                  </button>
                  <button
                    onClick={exportChat}
                    className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-primary-500/20 to-pink-500/20 border border-primary-300/50 dark:border-primary-500/50 rounded-xl text-primary-600 dark:text-primary-400 hover:from-primary-500/30 hover:to-pink-500/30 transition-all"
                  >
                    <BsDownload />
                    Export Chat
                  </button>
                </div>
              </div>
            </div>

            {/* Mobile toggle button for sidebar */}
            <button
              onClick={() => setSidebarOpen(s => !s)}
              className="lg:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors text-gray-500 dark:text-gray-400"
              title="Toggle sidebar"
            >
              <BsLayoutSidebar className="text-lg" />
            </button>
          </div>

          {/* Main Chat Interface */}
          <motion.div
            ref={chatContainerRef}
            className={`lg:col-span-3 ${isFullscreen ? 'fixed inset-0 z-50 h-screen w-screen' : ''}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
          >
            <div className={`flex flex-col rounded-2xl backdrop-blur-xl bg-white/90 dark:bg-white/10 border border-white/30 dark:border-white/10 shadow-2xl shadow-primary-500/20 overflow-hidden ${isFullscreen ? 'h-full rounded-none border-0' : 'h-[600px]'}`}>
              {/* Chat Header */}
              <div className="p-6 border-b border-gray-200/50 dark:border-gray-800/50 bg-gradient-to-r from-primary-50/50 to-pink-50/50 dark:from-primary-900/10 dark:to-pink-900/10">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-primary-custom-1 via-primary-custom-2 to-primary-custom-3 flex items-center justify-center">
                    <BsRobot className="text-2xl text-white" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                      LinguaSign AI
                    </h2>
                    <p className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-green-500 inline-block" />
                      Online • Powered by Deep Learning
                    </p>
                  </div>
                  <div className="ml-auto flex items-center gap-2">
                    <button
                      onClick={handleNewChat}
                      className="p-2 rounded-lg hover:bg-white/20 dark:hover:bg-black/20 transition-colors text-gray-500 dark:text-gray-400"
                      title="New chat"
                    >
                      <BsPlus className="text-xl" />
                    </button>
                  </div>
                </div>
              </div>

              {/* Messages Container */}
              <div className="flex-1 overflow-y-auto p-6 space-y-6" style={{ scrollbarWidth: 'thin', scrollbarColor: 'var(--theme-scrollbar-thumb) transparent' }}>
                {/* Welcome / empty state */}
                {messages.length === 1 && messages[0].id === 'welcome' && (
                  <div className="flex flex-col items-center justify-center h-full text-center">
                    <motion.div
                      initial={{ scale: 0.8, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ type: 'spring', stiffness: 200 }}
                      className="w-20 h-20 rounded-3xl bg-gradient-to-br from-primary-custom-1 via-primary-custom-2 to-primary-custom-3 flex items-center justify-center mb-6 shadow-2xl shadow-primary-500/40"
                    >
                      <TbHandLoveYou className="text-4xl text-white" />
                    </motion.div>
                    <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white mb-2">
                      How can I help you today?
                    </h2>
                    <p className="text-gray-500 dark:text-gray-400 max-w-sm">
                      Ask anything about sign language — I'm here to help!
                    </p>
                  </div>
                )}

                {/* Messages */}
                {messages.map((msg) => (
                  <motion.div
                    key={msg.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`max-w-[80%] ${msg.role === 'user' ? 'order-2' : 'order-1'}`}>
                      <div className="flex items-center gap-2 mb-1">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                          msg.role === 'user'
                            ? 'bg-gradient-to-r from-blue-500 to-blue-600'
                            : 'bg-gradient-to-r from-primary-500 to-primary-600'
                        }`}>
                          {msg.role === 'user' ?
                            <FaUser className="text-sm text-white" /> :
                            <BsRobot className="text-sm text-white" />
                          }
                        </div>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {msg.role === 'user' ? 'You' : 'AI Assistant'} • {msg.time}
                        </span>
                      </div>
                      <div className={`rounded-2xl p-4 ${
                        msg.role === 'user'
                          ? 'bg-gradient-to-r from-blue-500/10 to-blue-600/10 border border-blue-200/50 dark:border-blue-800/50'
                          : 'bg-gradient-to-r from-primary-500/10 to-primary-600/10 border border-primary-200/50 dark:border-primary-800/50'
                      }`}>
                        <p className="text-gray-800 dark:text-gray-200 whitespace-pre-wrap">
                          {msg.text}
                        </p>
                        {msg.role === 'ai' && (
                          <div className="flex items-center justify-end gap-2 mt-3">
                            <button
                              onClick={() => handleCopy(msg.text, msg.id)}
                              className="p-2 rounded-lg hover:bg-white/20 dark:hover:bg-black/20 transition-colors"
                              title="Copy text"
                            >
                              {copiedId === msg.id ? (
                                <FaCheck className="text-green-500" />
                              ) : (
                                <FaRegCopy className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" />
                              )}
                            </button>
                            <button
                              onClick={() => handleFeedback(msg.id, 'like')}
                              className="p-2 rounded-lg hover:bg-white/20 dark:hover:bg-black/20 transition-colors"
                              title="Helpful"
                            >
                              <FaRegThumbsUp className="text-gray-400 hover:text-green-500" />
                            </button>
                            <button
                              onClick={() => handleFeedback(msg.id, 'dislike')}
                              className="p-2 rounded-lg hover:bg-white/20 dark:hover:bg-black/20 transition-colors"
                              title="Not helpful"
                            >
                              <FaRegThumbsDown className="text-gray-400 hover:text-red-500" />
                            </button>
                          </div>
                        )}
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
                        <div className="w-8 h-8 rounded-full bg-gradient-to-r from-primary-500 to-primary-600 flex items-center justify-center">
                          <BsRobot className="text-sm text-white" />
                        </div>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          AI Assistant is typing...
                        </span>
                      </div>
                      <div className="rounded-2xl p-4 bg-gradient-to-r from-primary-500/10 to-primary-600/10 border border-primary-200/50 dark:border-primary-800/50">
                        <div className="flex space-x-2">
                          {[0, 1, 2].map(i => (
                            <motion.div
                              key={i}
                              className="w-2 h-2 bg-primary-500 rounded-full"
                              animate={{ y: [0, -6, 0] }}
                              transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.15 }}
                            />
                          ))}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
                <div ref={bottomRef} />
              </div>

              {/* Input Area */}
              <div className="p-6 border-t border-gray-200/50 dark:border-gray-800/50 bg-gradient-to-r from-gray-50/50 to-primary-50/50 dark:from-gray-900/10 dark:to-primary-900/10">
                <div className="flex items-end gap-3">
                  <div className="flex-1 relative">
                    <textarea
                      ref={textareaRef}
                      value={input}
                      onChange={e => setInput(e.target.value)}
                      onKeyDown={handleKey}
                      placeholder="Type your message here... Ask about sign language, translations, or learning tips"
                      className="w-full px-4 py-3 pl-12 pr-24 bg-white/80 dark:bg-gray-900/80 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 resize-none transition-all"
                      rows="2"
                      style={{ scrollbarWidth: 'none' }}
                    />
                    <div className="absolute left-4 top-3.5 text-gray-400 dark:text-gray-500">
                      <FaUser />
                    </div>
                    <div className="absolute right-4 top-3.5 flex items-center gap-2">
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
                    className="p-4 rounded-xl bg-gradient-to-r from-primary-custom-1 via-primary-custom-2 to-primary-custom-3 text-white shadow-lg shadow-primary-500/40 hover:shadow-primary-500/60 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <BsSendFill className="text-xl" />
                  </motion.button>
                </div>

                {/* Quick Prompts */}
                {messages.length <= 1 && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {QUICK_PROMPTS.map((p, i) => (
                      <motion.button
                        key={i}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.05 }}
                        onClick={() => setInput(p)}
                        className="px-3 py-1.5 text-sm rounded-full bg-gradient-to-r from-primary-500/10 to-pink-500/10 border border-primary-300/30 dark:border-primary-500/30 text-primary-600 dark:text-primary-400 hover:from-primary-500/20 hover:to-pink-500/20 transition-all"
                      >
                        {p}
                      </motion.button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        </div>

        {/* Stats Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6"
        >
          <div className="p-6 rounded-2xl backdrop-blur-xl bg-gradient-to-r from-primary-500/5 to-transparent border border-primary-200/30 dark:border-primary-500/20">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-primary-500 to-primary-600 flex items-center justify-center">
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

      {/* Custom CSS for animations and scrollbar */}
      <style jsx>{`
        @keyframes pulse-slow {
          0%, 100% { opacity: 0.5; }
          50% { opacity: 0.8; }
        }
        .animate-pulse-slow {
          animation: pulse-slow 4s ease-in-out infinite;
        }
        .overflow-y-auto {
          scrollbar-width: thin;
          scrollbar-color: var(--theme-scrollbar-thumb) transparent;
        }
        .overflow-y-auto::-webkit-scrollbar {
          width: 6px;
        }
        .overflow-y-auto::-webkit-scrollbar-track {
          background: transparent;
        }
        .overflow-y-auto::-webkit-scrollbar-thumb {
          background-color: var(--theme-scrollbar-thumb);
          border-radius: 20px;
        }
        .overflow-y-auto::-webkit-scrollbar-thumb:hover {
          background-color: var(--theme-scrollbar-thumb-hover);
        }
      `}</style>
    </div>
  );
}