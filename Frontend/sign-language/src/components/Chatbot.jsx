import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BsRobot,
  BsSendFill,
  BsTrash,
  BsStars,
  BsPlus,
  BsLayoutSidebar,
} from 'react-icons/bs';
import {
  FaUser,
  FaRegCopy,
  FaCheck,
  FaRegThumbsUp,
  FaRegThumbsDown,
} from 'react-icons/fa';
import { TbHandLoveYou, TbMessage2 } from 'react-icons/tb';
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

  const bottomRef = useRef(null);
  const textareaRef = useRef(null);
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
      // Fallback: show a smart contextual reply if backend isn't available
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

  /* ─── render ─────────────────────────────────────────── */
  return (
    <div className="flex h-screen bg-white dark:bg-[#0d0d0d] pt-[64px]">

      {/* ── Sidebar ───────────────────────────────────────── */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ x: -280, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -280, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            className="fixed left-0 top-[64px] bottom-0 w-[260px] z-30
              bg-gray-50 dark:bg-[#171717]
              border-r border-gray-200 dark:border-gray-800
              flex flex-col shadow-2xl"
          >
            {/* New chat */}
            <div className="p-3 border-b border-gray-200 dark:border-gray-800">
              <button
                onClick={handleNewChat}
                className="w-full flex items-center gap-3 px-4 py-3 rounded-xl
                  border border-gray-300 dark:border-gray-700
                  text-gray-700 dark:text-gray-300 text-sm font-medium
                  hover:bg-white dark:hover:bg-gray-800 transition-all group"
              >
                <BsPlus className="text-xl group-hover:text-purple-500 transition-colors" />
                New chat
              </button>
            </div>

            {/* History */}
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

            {/* Clear */}
            <div className="p-3 border-t border-gray-200 dark:border-gray-800">
              <button
                onClick={handleNewChat}
                className="w-full flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm
                  text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 transition-all"
              >
                <BsTrash />
                Clear conversations
              </button>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Sidebar backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/30 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ── Main chat area ─────────────────────────────────── */}
      <div className="flex flex-col flex-1 min-w-0 relative">

        {/* Top bar */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-[#0d0d0d]">
          <button
            onClick={() => setSidebarOpen(s => !s)}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors text-gray-500 dark:text-gray-400"
            title="Toggle sidebar"
          >
            <BsLayoutSidebar className="text-lg" />
          </button>

          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#6A3093] to-[#BF5AE0] flex items-center justify-center shadow-md">
              <BsRobot className="text-white text-sm" />
            </div>
            <div>
              <p className="text-sm font-semibold text-gray-900 dark:text-white leading-none">
                LinguaSign AI
              </p>
              <p className="text-xs text-green-500 leading-none mt-0.5 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500 inline-block" />
                Online
              </p>
            </div>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-full
                bg-purple-50 dark:bg-purple-900/20
                border border-purple-200/60 dark:border-purple-700/40 text-xs font-medium
                text-purple-600 dark:text-purple-400"
            >
              <BsStars className="text-purple-500" />
              AI Assistant
            </motion.div>
            <button
              onClick={handleNewChat}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors text-gray-500 dark:text-gray-400"
              title="New chat"
            >
              <BsPlus className="text-xl" />
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto scroll-smooth" style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(168,85,247,0.3) transparent' }}>

          {/* Empty state / welcome */}
          {messages.length === 1 && messages[0].id === 'welcome' && (
            <div className="flex flex-col items-center justify-center min-h-[40vh] px-4 pt-12 pb-4 text-center">
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: 'spring', stiffness: 200 }}
                className="w-20 h-20 rounded-3xl bg-gradient-to-br from-[#6A3093] via-[#A044FF] to-[#BF5AE0] flex items-center justify-center mb-6 shadow-2xl shadow-purple-500/40"
              >
                <TbHandLoveYou className="text-4xl text-white" />
              </motion.div>
              <motion.h2
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.1 }}
                className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white mb-2"
              >
                How can I help you today?
              </motion.h2>
              <motion.p
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="text-gray-500 dark:text-gray-400 max-w-sm"
              >
                Ask anything about sign language — I'm here to help!
              </motion.p>
            </div>
          )}

          <div className="max-w-3xl mx-auto px-4 py-6 space-y-2">
            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <MessageBubble
                  key={msg.id}
                  msg={msg}
                  copiedId={copiedId}
                  onCopy={handleCopy}
                />
              ))}
            </AnimatePresence>

            {/* Typing indicator */}
            <AnimatePresence>
              {isLoading && (
                <motion.div
                  key="typing"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="flex items-start gap-3 py-2"
                >
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#6A3093] to-[#BF5AE0] flex items-center justify-center flex-shrink-0 shadow-md">
                    <BsRobot className="text-white text-sm" />
                  </div>
                  <div className="px-4 py-3 rounded-2xl rounded-tl-sm bg-gray-100 dark:bg-gray-800 flex items-center gap-1.5">
                    {[0, 1, 2].map(i => (
                      <motion.div
                        key={i}
                        className="w-2 h-2 rounded-full bg-purple-500"
                        animate={{ y: [0, -6, 0] }}
                        transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.15 }}
                      />
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div ref={bottomRef} />
          </div>
        </div>

        {/* Input area */}
        <div className="border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-[#0d0d0d] px-4 py-4">
          <div className="max-w-3xl mx-auto">

            {/* Quick prompts — only when no user messages yet */}
            {messages.length <= 1 && (
              <div className="flex flex-wrap gap-2 mb-3">
                {QUICK_PROMPTS.map((p, i) => (
                  <motion.button
                    key={i}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.05 }}
                    onClick={() => setInput(p)}
                    className="px-3.5 py-2 text-sm rounded-full
                      border border-gray-200 dark:border-gray-700
                      text-gray-600 dark:text-gray-300
                      bg-white dark:bg-gray-900
                      hover:border-purple-400 dark:hover:border-purple-500
                      hover:text-purple-600 dark:hover:text-purple-400
                      hover:bg-purple-50 dark:hover:bg-purple-900/20
                      transition-all"
                  >
                    {p}
                  </motion.button>
                ))}
              </div>
            )}

            {/* Input box */}
            <div className="relative flex items-end gap-2
              bg-white dark:bg-[#1a1a1a]
              border border-gray-300 dark:border-gray-700
              rounded-2xl shadow-sm focus-within:border-purple-500 focus-within:ring-2 focus-within:ring-purple-500/20
              transition-all overflow-hidden"
            >
              <textarea
                ref={textareaRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="Message LinguaSign AI…"
                rows={1}
                className="flex-1 resize-none bg-transparent px-4 py-3.5
                  text-gray-900 dark:text-gray-100
                  placeholder-gray-400 dark:placeholder-gray-500
                  text-sm leading-relaxed outline-none max-h-[200px] overflow-y-auto"
                style={{ scrollbarWidth: 'none' }}
              />
              <motion.button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="m-2 p-2.5 rounded-xl
                  bg-gradient-to-br from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
                  text-white shadow-lg shadow-purple-500/30
                  disabled:opacity-40 disabled:cursor-not-allowed
                  hover:shadow-purple-500/50 transition-all flex-shrink-0"
              >
                <BsSendFill className="text-base" />
              </motion.button>
            </div>

            <p className="text-center text-xs text-gray-400 dark:text-gray-600 mt-2">
              Press <kbd className="px-1 py-0.5 rounded bg-gray-100 dark:bg-gray-800 font-mono text-xs">Enter</kbd> to send &nbsp;·&nbsp; <kbd className="px-1 py-0.5 rounded bg-gray-100 dark:bg-gray-800 font-mono text-xs">Shift+Enter</kbd> for new line
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─── Message bubble component ────────────────────────── */
function MessageBubble({ msg, copiedId, onCopy }) {
  const isUser = msg.role === 'user';

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.25 }}
      className={`flex items-start gap-3 py-2 group ${isUser ? 'flex-row-reverse' : ''}`}
    >
      {/* Avatar */}
      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 shadow-md
        ${isUser
          ? 'bg-gradient-to-br from-blue-500 to-blue-600'
          : 'bg-gradient-to-br from-[#6A3093] to-[#BF5AE0]'
        }`}
      >
        {isUser
          ? <FaUser className="text-white text-xs" />
          : <BsRobot className="text-white text-sm" />
        }
      </div>

      {/* Bubble */}
      <div className={`flex flex-col max-w-[75%] sm:max-w-[70%] ${isUser ? 'items-end' : 'items-start'}`}>
        <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap break-words
          ${isUser
            ? 'rounded-tr-sm bg-gradient-to-br from-[#6A3093] to-[#A044FF] text-white shadow-lg shadow-purple-500/20'
            : 'rounded-tl-sm bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-100 shadow-sm'
          }`}
        >
          {msg.text}
        </div>

        {/* Action bar for AI messages */}
        {!isUser && (
          <div className="flex items-center gap-0.5 mt-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
            <ActionBtn
              onClick={() => onCopy(msg.text, msg.id)}
              title="Copy"
              icon={copiedId === msg.id
                ? <FaCheck className="text-green-500" />
                : <FaRegCopy />
              }
            />
            <ActionBtn icon={<FaRegThumbsUp />} title="Good response" />
            <ActionBtn icon={<FaRegThumbsDown />} title="Bad response" />
            <span className="text-xs text-gray-400 dark:text-gray-600 ml-2">{msg.time}</span>
          </div>
        )}

        {isUser && (
          <span className="text-xs text-gray-400 dark:text-gray-600 mt-1">{msg.time}</span>
        )}
      </div>
    </motion.div>
  );
}

function ActionBtn({ onClick, icon, title }) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition-all"
    >
      <span className="text-sm">{icon}</span>
    </button>
  );
}