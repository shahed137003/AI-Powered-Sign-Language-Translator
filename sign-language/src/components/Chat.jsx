import React, { useState, useEffect, useRef, useContext, useCallback, useMemo } from "react";
import { useTheme } from "../context/ThemeContext";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { 
  BsSendFill, 
  BsRobot, 
  BsPersonCircle, 
  BsThreeDotsVertical,
  BsEmojiSmile,
  BsImage,
  BsMicFill,
  BsCheckAll,
  BsClock,
  BsChevronLeft,
  BsSearch,
  BsFilter,
  BsPlusCircleFill,
  BsPhone,
  BsCameraVideo,
  BsInfoCircle,
  BsCheckCircle
} from "react-icons/bs";
import { 
  TbHandLoveYou, 
  TbMoodSmile,
  TbMessages,
  TbSend,
  TbSparkles
} from "react-icons/tb";
import { 
  FiPaperclip, 
  FiMoreVertical 
} from "react-icons/fi";
import { 
  RiAttachment2, 
  RiCloseFill 
} from "react-icons/ri";
import { 
  AiOutlineLoading3Quarters 
} from "react-icons/ai";
import { 
  FaRegSmile,
  FaUserFriends,
  FaCog,
  FaUser,
  FaSearch,
  FaPlus,
  FaCrown,
  FaSignInAlt,
  FaUserPlus,
  FaShieldAlt
} from "react-icons/fa";
import axios from "axios";
import { AuthContext } from "../context/AuthContext";
import VideoCall from "../components/VideoCall";
import IncomingCallModal from "../components/IncomingCallModal";
import { v4 as uuidv4 } from 'uuid';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Emoji picker component
const EmojiPicker = ({ onEmojiSelect, onClose }) => {
  const emojis = ["😀", "😊", "👍", "👋", "❤️", "🎉", "🤔", "👏", "🙏", "🤝", "💬", "🎯", "✨", "🚀", "💡"];
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      className="absolute bottom-16 left-4 bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border-2 border-primary-200/50 dark:border-primary-800/50 p-4 w-72 z-50 backdrop-blur-xl"
    >
      <div className="flex justify-between items-center mb-3">
        <span className="font-bold text-gray-800 dark:text-white">Emojis</span>
        <button 
          onClick={onClose}
          className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
        >
          <RiCloseFill className="text-gray-500 dark:text-gray-400" />
        </button>
      </div>
      <div className="grid grid-cols-8 gap-2">
        {emojis.map((emoji, idx) => (
          <button
            key={idx}
            onClick={() => onEmojiSelect(emoji)}
            className="text-2xl hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg p-2 transition-colors"
          >
            {emoji}
          </button>
        ))}
      </div>
    </motion.div>
  );
};

// Message component
const Message = ({ message, isOwn, time, status, type = "text" }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex ${isOwn ? 'justify-end' : 'justify-start'} mb-4`}
    >
      <div className={`flex max-w-[80%] ${isOwn ? 'flex-row-reverse' : ''}`}>
        {/* Avatar */}
        <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${isOwn ? 'ml-3' : 'mr-3'}`}>
          {isOwn ? (
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-lg">
              <BsPersonCircle className="text-white text-xl" />
            </div>
          ) : (
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center shadow-lg">
              <FaUser className="text-white text-xl" />
            </div>
          )}
        </div>
        
        {/* Message Content */}
        <div>
          <div className={`px-4 py-3 rounded-2xl ${isOwn 
            ? 'bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-br-none shadow-lg shadow-primary-500/20' 
            : 'bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-2 border-primary-200/50 dark:border-primary-800/50 text-gray-800 dark:text-white rounded-bl-none shadow-lg'
          }`}>
            {type === "text" ? (
              <p className="text-sm font-medium">{message}</p>
            ) : (
              <div className="flex items-center gap-2">
                <TbHandLoveYou className="text-lg" />
                <span className="text-sm">Gesture detected and translated</span>
              </div>
            )}
          </div>
          
          {/* Time and Status */}
          <div className={`flex items-center gap-2 mt-1 text-xs ${isOwn ? 'justify-end' : 'justify-start'}`}>
            <span className="text-gray-500 dark:text-gray-400">{time}</span>
            {isOwn && (
              <div className="text-primary-500">
                {status === 'sent' && <BsCheckAll />}
                {status === 'delivered' && <BsCheckAll className="text-blue-500" />}
                {status === 'read' && <BsCheckCircle className="text-green-500" />}
                {status === 'sending' && <AiOutlineLoading3Quarters className="animate-spin" />}
              </div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

// Search User Modal Component
const SearchUserModal = ({ isOpen, onClose, onUserFound, currentUsername }) => {
  const [username, setUsername] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!username.trim()) {
      setError("Please enter a username");
      return;
    }

    if (username === currentUsername) {
      setError("You cannot chat with yourself");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await axios.get(
        `${API_URL}/users/check-username/${username}`
      );

      if (response.data.exists) {
        onUserFound(username);
        setUsername("");
        onClose();
      } else {
        setError(`User "${username}" not found`);
      }
    } catch (err) {
      console.error("Search error:", err);
      setError("Error searching for user");
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="bg-white/95 dark:bg-gray-900/95 backdrop-blur-xl rounded-2xl p-6 max-w-md w-full mx-4 border-2 border-primary-200/50 dark:border-primary-800/50 shadow-2xl shadow-primary-500/20"
      >
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-bold bg-gradient-to-r from-primary-700 to-primary-500 bg-clip-text text-transparent">New Conversation</h3>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
          >
            <RiCloseFill className="text-gray-600 dark:text-gray-400 text-xl" />
          </button>
        </div>

        <form onSubmit={handleSearch}>
          <div className="mb-4">
            <label className="block text-sm font-bold text-gray-700 dark:text-gray-300 mb-2">
              Enter username
            </label>
            <div className="relative">
              <BsSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="e.g., john_doe"
                className="w-full pl-10 pr-4 py-3 bg-gray-100/80 dark:bg-gray-800/80 border-2 border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-gray-800 dark:text-white transition-all"
                autoFocus
              />
            </div>
            {error && (
              <p className="text-sm text-red-500 mt-2 flex items-center gap-1">
                <span>⚠️</span> {error}
              </p>
            )}
          </div>

          <div className="flex gap-3">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 py-3 px-4 bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-white rounded-xl font-bold hover:bg-gray-300 dark:hover:bg-gray-700 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex-1 py-3 px-4 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-xl font-bold hover:shadow-lg hover:shadow-primary-500/30 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {loading ? (
                <AiOutlineLoading3Quarters className="animate-spin" />
              ) : (
                <>
                  <FaSearch />
                  Search
                </>
              )}
            </button>
          </div>
        </form>
      </motion.div>
    </div>
  );
};

// Chat input component
const ChatInput = ({ onSendMessage, isTyping, onTyping }) => {
  const [message, setMessage] = useState("");
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);
  const [recording, setRecording] = useState(false);
  const typingTimeoutRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim()) {
      onSendMessage(message);
      setMessage("");
      setShowEmojiPicker(false);
      
      if (onTyping) {
        onTyping(false);
      }
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    }
  };

  const handleChange = (e) => {
    const value = e.target.value;
    setMessage(value);
    
    if (onTyping) {
      onTyping(true);
      
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
      
      typingTimeoutRef.current = setTimeout(() => {
        onTyping(false);
      }, 2000);
    }
  };

  const handleEmojiSelect = (emoji) => {
    setMessage(prev => prev + emoji);
    if (onTyping) {
      onTyping(true);
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
      typingTimeoutRef.current = setTimeout(() => {
        onTyping(false);
      }, 2000);
    }
  };

  const toggleRecording = () => {
    setRecording(!recording);
  };

  useEffect(() => {
    return () => {
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    };
  }, []);

  return (
    <div className="relative px-4 py-3 border-t-2 border-primary-200/50 dark:border-primary-800/50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl">
      {isTyping && (
        <div className="absolute -top-8 left-4 bg-white/90 dark:bg-gray-800/90 backdrop-blur-sm px-3 py-2 rounded-full shadow-md border-2 border-primary-200/50 dark:border-primary-800/50">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
              <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
              <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
            </div>
            <span className="text-sm text-gray-600 dark:text-gray-400 font-medium">Typing...</span>
          </div>
        </div>
      )}

      {showEmojiPicker && (
        <EmojiPicker 
          onEmojiSelect={handleEmojiSelect}
          onClose={() => setShowEmojiPicker(false)}
        />
      )}

      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <button
          type="button"
          className="p-3 rounded-full hover:bg-gray-100/50 dark:hover:bg-gray-800/50 transition-colors"
        >
          <BsPlusCircleFill className="text-primary-500 text-xl" />
        </button>

        <button
          type="button"
          onClick={() => setShowEmojiPicker(!showEmojiPicker)}
          className="p-3 rounded-full hover:bg-gray-100/50 dark:hover:bg-gray-800/50 transition-colors"
        >
          <BsEmojiSmile className="text-gray-600 dark:text-gray-400 text-xl" />
        </button>

        <div className="flex-1 relative">
          <input
            type="text"
            value={message}
            onChange={handleChange}
            placeholder="Type your message..."
            className="w-full px-4 py-3 bg-gray-100/80 dark:bg-gray-800/80 border-2 border-gray-300 dark:border-gray-700 rounded-full focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-gray-800 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 transition-all"
          />
          
          <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex items-center gap-2">
            <button
              type="button"
              className="p-2 hover:bg-gray-200/50 dark:hover:bg-gray-700/50 rounded-full transition-colors"
            >
              <BsImage className="text-gray-600 dark:text-gray-400" />
            </button>
            <button
              type="button"
              className="p-2 hover:bg-gray-200/50 dark:hover:bg-gray-700/50 rounded-full transition-colors"
            >
              <FiPaperclip className="text-gray-600 dark:text-gray-400" />
            </button>
          </div>
        </div>

        {message.trim() ? (
          <button
            type="submit"
            className="p-3 rounded-full bg-gradient-to-r from-primary-500 to-primary-600 text-white hover:shadow-lg hover:shadow-primary-500/30 transition-all"
          >
            <TbSend className="text-xl" />
          </button>
        ) : (
          <button
            type="button"
            onClick={toggleRecording}
            className={`p-3 rounded-full transition-all ${recording 
              ? 'bg-red-500 text-white animate-pulse' 
              : 'bg-gray-200/80 dark:bg-gray-800/80 text-gray-600 dark:text-gray-400 hover:bg-gray-300/80 dark:hover:bg-gray-700/80'
            }`}
          >
            <BsMicFill className="text-xl" />
          </button>
        )}
      </form>
    </div>
  );
};

// Chat sidebar component
const ChatSidebar = ({ conversations, onSelectConversation, activeConversation, currentUser, onNewChat }) => {
  const formatTime = (dateString) => {
    if (!dateString) return "";
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / (1000 * 60));
    
    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="w-full md:w-80 border-r-2 border-primary-200/50 dark:border-primary-800/50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl h-full overflow-hidden flex flex-col">
      {/* Header */}
      <div className="p-4 border-b-2 border-primary-200/50 dark:border-primary-800/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-lg">
              <TbMessages className="text-white text-2xl" />
            </div>
            <div>
              <h2 className="font-black text-lg bg-gradient-to-r from-primary-700 to-primary-500 bg-clip-text text-transparent">Messages</h2>
              <p className="text-sm text-primary-500 font-semibold">@{currentUser?.email?.split('@')[0] || 'user'}</p>
            </div>
          </div>
        </div>

        {/* New Conversation Button */}
        <button 
          onClick={onNewChat}
          className="mt-4 w-full py-3 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-full font-bold hover:shadow-lg hover:shadow-primary-500/30 transition-all flex items-center justify-center gap-2"
        >
          <FaPlus className="text-lg" />
          New Conversation
        </button>

        {/* Search */}
        <div className="mt-4 relative">
          <BsSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search conversations..."
            className="w-full pl-10 pr-4 py-2 bg-gray-100/80 dark:bg-gray-800/80 border-2 border-gray-300 dark:border-gray-700 rounded-full focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-gray-800 dark:text-white transition-all"
          />
        </div>
      </div>

      {/* Conversation List */}
      <div className="flex-1 overflow-y-auto">
        {conversations.length === 0 ? (
          <div className="p-8 text-center text-gray-500 dark:text-gray-400">
            <FaUserFriends className="text-4xl mx-auto mb-3 opacity-50" />
            <p className="font-medium">No conversations yet</p>
            <p className="text-sm mt-2">Click "New Conversation" to start chatting</p>
          </div>
        ) : (
          conversations.map((conversation) => {
            const otherUser = conversation.other_username || conversation.username;
            const lastMessage = conversation.last_message || "No messages yet";
            const isActive = activeConversation?.other_username === otherUser || activeConversation?.username === otherUser;
            
            return (
              <motion.button
                key={otherUser}
                whileHover={{ scale: 0.98 }}
                onClick={() => onSelectConversation(conversation)}
                className={`w-full p-4 text-left border-b border-primary-200/30 dark:border-primary-800/30 transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-primary-50/80 to-primary-100/50 dark:from-primary-900/30 dark:to-primary-800/20 border-l-4 border-primary-500' 
                    : 'hover:bg-gray-50/50 dark:hover:bg-gray-800/50'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center shadow-lg">
                      <FaUser className="text-white text-xl" />
                    </div>
                    {conversation.online && (
                      <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-white dark:border-gray-900"></div>
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between items-start">
                      <h3 className="font-bold text-gray-800 dark:text-white">
                        {otherUser}
                      </h3>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {formatTime(conversation.last_message_time)}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 truncate">
                      {lastMessage.length > 30 ? lastMessage.substring(0, 30) + '...' : lastMessage}
                    </p>
                  </div>
                </div>
              </motion.button>
            );
          })
        )}
      </div>
    </div>
  );
};

// Main Chat Page Component
export default function Chat() {
  const { themeColor } = useTheme();
  const { user, isAuthenticated } = useContext(AuthContext);
  const [messages, setMessages] = useState([]);
  const [conversations, setConversations] = useState([]);
  const [activeConversation, setActiveConversation] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const [otherUserTyping, setOtherUserTyping] = useState(false);
  const [loading, setLoading] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [showSearchModal, setShowSearchModal] = useState(false);
  const [currentUserId, setCurrentUserId] = useState(null);
  const [isDark, setIsDark] = useState(false);

  // Video call states
  const [showVideoCall, setShowVideoCall] = useState(false);
  const [showIncomingCall, setShowIncomingCall] = useState(false);
  const [incomingCall, setIncomingCall] = useState(null);
  const [activeCall, setActiveCall] = useState(null);
  const [callWs, setCallWs] = useState(null);

  // Particle background refs
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationFrameRef = useRef(null);

  const messagesEndRef = useRef(null);
  const wsRef = useRef(null);

  const currentUsername = user?.email?.split('@')[0] || '';
  const token = localStorage.getItem('token');

  // Detect dark mode
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

  // Particle system - NO connections between particles
  useEffect(() => {
    const canvas = canvasRef.current;
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
      size: Math.random() * 3 + 1,
      speedX: Math.random() * 0.2 - 0.1,
      speedY: Math.random() * 0.2 - 0.1,
      color: colors[Math.floor(Math.random() * colors.length)],
      opacity: Math.random() * 0.4 + 0.1,
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

        // Draw individual particles - NO connections
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fillStyle = particle.color + Math.floor(particle.opacity * 255).toString(16).padStart(2, '0');
        ctx.fill();
        
        // Add subtle glow to larger particles
        if (particle.size > 2) {
          ctx.beginPath();
          ctx.arc(particle.x, particle.y, particle.size * 2, 0, Math.PI * 2);
          ctx.fillStyle = particle.color + '20';
          ctx.fill();
        }
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

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Save username to localStorage
  useEffect(() => {
    if (currentUsername) {
      localStorage.setItem('username', currentUsername);
    }
  }, [currentUsername]);

  // Fetch current user ID
  useEffect(() => {
    const fetchCurrentUser = async () => {
      try {
        const response = await axios.get(`${API_URL}/users/me`, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
        setCurrentUserId(response.data.id);
      } catch (error) {
        console.error("Error fetching current user:", error);
      }
    };

    if (isAuthenticated && token) {
      fetchCurrentUser();
    }
  }, [isAuthenticated, token]);

  // Load conversations from localStorage on mount
  useEffect(() => {
    if (currentUsername && isAuthenticated && currentUserId) {
      const savedConversations = JSON.parse(
        localStorage.getItem(`conversations_${currentUsername}`) || '[]'
      );

      const sortedConversations = savedConversations.sort(
        (a, b) => new Date(b.last_message_time) - new Date(a.last_message_time)
      );

      setConversations(sortedConversations);

      if (sortedConversations.length > 0 && !activeConversation) {
        const mostRecent = sortedConversations[0];
        setActiveConversation(mostRecent);
        loadChatHistory(mostRecent.other_username || mostRecent.username);
      }
    }
  }, [currentUsername, isAuthenticated, currentUserId]);

  // WebSocket connection
  useEffect(() => {
    if (currentUsername && isAuthenticated && token) {
      connectWebSocket();
      return () => {
        if (wsRef.current) {
          wsRef.current.close();
        }
      };
    }
  }, [currentUsername, isAuthenticated, token]);

  const connectWebSocket = () => {
    if (!token) return;

    const ws = new WebSocket(
      `ws://localhost:8000/ws/chat?token=${token}`
    );

    ws.onopen = () => {
      console.log("✅ WebSocket connected");
      setWsConnected(true);
      wsRef.current = ws;
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("📩 Received:", data);

      if (data.error) {
        console.error("WebSocket error:", data.error);
        return;
      }

      setMessages((prev) => {
        const exists = prev.some(msg => msg.id === data.id);
        if (exists) return prev;
        
        return [
          ...prev,
          {
            id: data.id,
            text: data.content,
            isOwn: data.sender_id === currentUserId,
            time: new Date(data.created_at).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            }),
            status: 'delivered',
            type: 'text'
          },
        ];
      });

      if (data.sender_id !== currentUserId) {
        updateConversationFromMessage("User", data.content);
      }
    };

    ws.onclose = () => {
      console.log("❌ WebSocket closed");
      setWsConnected(false);
      wsRef.current = null;
      setTimeout(() => {
        if (currentUsername && isAuthenticated) {
          connectWebSocket();
        }
      }, 3000);
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      setWsConnected(false);
    };
  };

  // Connect to video call WebSocket
  useEffect(() => {
    if (currentUsername && isAuthenticated && token) {
      connectCallWebSocket();
    }
    return () => {
      if (callWs) {
        callWs.close();
      }
    };
  }, [currentUsername, isAuthenticated, token]);

  const connectCallWebSocket = () => {
    if (!token) return;

    const ws = new WebSocket(
      `ws://localhost:8000/ws/video-call?token=${token}`
    );

    ws.onopen = () => {
      console.log("✅ Video call WebSocket connected");
      setCallWs(ws);
      
      const pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" }));
        }
      }, 30000);
      
      ws.pingInterval = pingInterval;
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleCallSignaling(data);
    };

    ws.onclose = () => {
      console.log("❌ Video call WebSocket closed");
      if (ws.pingInterval) {
        clearInterval(ws.pingInterval);
      }
      setTimeout(connectCallWebSocket, 3000);
    };

    ws.onerror = (err) => {
      console.error("Video call WebSocket error:", err);
    };
  };

  const handleCallSignaling = (data) => {
    console.log("📞 Call signal:", data);

    switch (data.type) {
      case "incoming-call":
        setIncomingCall({
          callId: data.callId,
          caller: data.caller,
          callType: data.callType,
          offer: data.offer
        });
        setShowIncomingCall(true);
        break;
      case "call-answered":
        if (window.handleRemoteAnswer) {
          window.handleRemoteAnswer(data.answer);
        }
        break;
      case "call-rejected":
        alert(`${data.from} rejected the call`);
        setShowVideoCall(false);
        setActiveCall(null);
        break;
      case "call-ended":
        if (data.reason === "peer-disconnected") {
          alert(`${data.from || "The other user"} disconnected`);
        } else {
          alert(`Call ended by ${data.from || "the other user"}`);
        }
        setShowVideoCall(false);
        setActiveCall(null);
        break;
      case "user-offline":
        alert(`${data.target} is offline`);
        break;
      case "error":
        alert(data.message);
        break;
      case "pong":
        console.log("🏓 Pong received");
        break;
      case "ice-candidate":
        if (window.videoCallPeerConnection) {
          window.videoCallPeerConnection.addIceCandidate(new RTCIceCandidate(data.candidate))
            .catch(err => console.error("Error adding ICE candidate:", err));
        }
        break;
      default:
        console.log("Unknown message type:", data.type);
        break;
    }
  };

  const initiateCall = async (type = "video") => {
    if (!activeConversation) {
      alert("Please select a user to call");
      return;
    }

    const receiverUsername = activeConversation.other_username || activeConversation.username;
    
    if (receiverUsername === currentUsername) {
      alert("You cannot call yourself");
      return;
    }

    if (!callWs || callWs.readyState !== WebSocket.OPEN) {
      alert("Video call service is not connected. Please try again.");
      return;
    }

    const callId = uuidv4();

    setActiveCall({
      callId,
      remoteUsername: receiverUsername,
      callType: type,
      isCaller: true
    });

    setShowVideoCall(true);
  };

  const updateConversationFromMessage = (otherUsername, message) => {
    setConversations(prev => {
      const updated = [...prev];
      const existingIndex = updated.findIndex(c => 
        c.other_username === otherUsername || c.username === otherUsername
      );
      
      const conversationData = {
        other_username: otherUsername,
        last_message: message,
        last_message_time: new Date().toISOString(),
        unread_count: 0,
        online: false
      };
      
      if (existingIndex >= 0) {
        updated.splice(existingIndex, 1);
        updated.unshift(conversationData);
      } else {
        updated.unshift(conversationData);
      }
      
      localStorage.setItem(`conversations_${currentUsername}`, JSON.stringify(updated));
      return updated;
    });
  };

  const saveConversation = (otherUsername, lastMessage, lastMessageTime) => {
    setConversations(prev => {
      const updated = [...prev];
      const existingIndex = updated.findIndex(c => 
        c.other_username === otherUsername || c.username === otherUsername
      );
      
      const conversationData = {
        other_username: otherUsername,
        last_message: lastMessage,
        last_message_time: lastMessageTime,
        unread_count: 0,
        online: false
      };
      
      if (existingIndex >= 0) {
        updated.splice(existingIndex, 1);
        updated.unshift(conversationData);
      } else {
        updated.unshift(conversationData);
      }
      
      localStorage.setItem(`conversations_${currentUsername}`, JSON.stringify(updated));
      return updated;
    });
  };

  const loadChatHistory = async (otherUsername) => {
    if (!currentUsername || !otherUsername || !currentUserId || !token) return;
    
    setLoading(true);
    try {
      const response = await axios.get(
        `${API_URL}/chat/history/${otherUsername}`,
        {
          headers: {
            Authorization: `Bearer ${token}`
          }
        }
      );
      
      const formattedMessages = response.data.map(msg => ({
        id: msg.id,
        text: msg.content,
        isOwn: msg.sender_id === currentUserId,
        time: new Date(msg.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        status: 'read',
        type: 'text'
      }));

      setMessages(formattedMessages);
      
      if (formattedMessages.length > 0) {
        const lastMsg = formattedMessages[formattedMessages.length - 1];
        saveConversation(otherUsername, lastMsg.text, response.data[response.data.length - 1].created_at);
      }
    } catch (error) {
      console.error('Error loading chat history:', error);
      setMessages([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSendMessage = (text) => {
    if (!wsRef.current || !activeConversation || !wsConnected) {
      alert('Cannot send message: Not connected');
      return;
    }

    const receiverUsername = activeConversation.other_username || activeConversation.username;

    if (receiverUsername === currentUsername) {
      alert('Cannot send message to yourself');
      return;
    }

    const newMessage = {
      id: Date.now(),
      text: text,
      isOwn: true,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      status: 'sending',
      type: 'text'
    };
    
    setMessages(prev => [...prev, newMessage]);

    wsRef.current.send(JSON.stringify({
      to: receiverUsername,
      message: text
    }));
    
    saveConversation(receiverUsername, text, new Date().toISOString());
  };

  const handleSelectConversation = async (conversation) => {
    setActiveConversation(conversation);
    await loadChatHistory(conversation.other_username || conversation.username);
  };

  const handleUserFound = async (username) => {
    if (username === currentUsername) {
      alert('You cannot chat with yourself');
      return;
    }

    const existingConv = conversations.find(c => 
      c.other_username === username || c.username === username
    );
    
    if (existingConv) {
      setActiveConversation(existingConv);
      await loadChatHistory(username);
    } else {
      const newConv = {
        other_username: username,
        last_message: "",
        last_message_time: new Date().toISOString(),
        unread_count: 0,
        online: false
      };
      setActiveConversation(newConv);
      setMessages([]);
      
      setConversations(prev => {
        const updated = [newConv, ...prev];
        localStorage.setItem(`conversations_${currentUsername}`, JSON.stringify(updated));
        return updated;
      });
    }
    
    setShowSearchModal(false);
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

  if (!isAuthenticated) {
    return (
      <div className="relative w-full h-screen bg-gradient-to-br from-gray-50 via-white to-primary-50/60 dark:from-primary-bg-1 dark:via-primary-bg-2 dark:to-primary-bg-3 overflow-hidden">
        
        {/* Premium Canvas Particles */}
        <canvas
          ref={canvasRef}
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
          className="absolute top-20 left-20 w-[600px] h-[600px] bg-primary-600/10 rounded-full blur-[120px]"
          animate={{
            x: [0, 100, 0],
            y: [0, -100, 0],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: "linear",
          }}
        />
        <motion.div
          className="absolute bottom-20 right-20 w-[600px] h-[600px] bg-primary-400/10 rounded-full blur-[120px]"
          animate={{
            x: [0, -100, 0],
            y: [0, 100, 0],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: "linear",
          }}
        />

        {/* Content Container */}
        <div className="relative z-10 w-full h-full flex items-center justify-center px-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ 
              duration: 0.6,
              ease: [0.22, 1, 0.36, 1],
              delay: 0.2
            }}
            className="max-w-md w-full"
          >
            {/* Premium Badge */}
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.5 }}
              className="flex justify-center mb-6"
            >
              <div className="inline-flex items-center gap-3 px-6 py-3 rounded-full bg-gradient-to-r from-primary-500/20 via-primary-400/10 to-primary-300/20 border-2 border-primary-300/40 dark:border-primary-600/40 backdrop-blur-xl shadow-2xl shadow-primary-500/20 relative overflow-hidden group">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                  className="p-1 rounded-full bg-gradient-to-r from-primary-500 to-primary-400"
                >
                  <FaCrown className="text-white text-sm" />
                </motion.div>
                <span className="text-sm font-extrabold bg-gradient-to-r from-primary-700 via-primary-600 to-primary-500 dark:from-primary-400 dark:via-primary-300 dark:to-primary-200 bg-clip-text text-transparent">
                  ACCESS REQUIRED
                </span>
                <TbSparkles className="text-primary-500 text-lg" />
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
              </div>
            </motion.div>

            {/* Main Card */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3, duration: 0.5 }}
              className="bg-white/90 dark:bg-white/10 backdrop-blur-xl rounded-3xl p-8 md:p-10 border-2 border-primary-200/50 dark:border-primary-800/50 shadow-2xl shadow-primary-500/20 text-center"
            >
              {/* Icon Container */}
              <motion.div
                animate={{ 
                  scale: [1, 1.1, 1],
                  rotate: [0, 5, -5, 0]
                }}
                transition={{ 
                  duration: 3, 
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
                className="w-24 h-24 rounded-full bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center mx-auto mb-6 shadow-lg shadow-primary-500/40"
              >
                <TbMessages className="text-white text-4xl" />
              </motion.div>

              {/* Title */}
              <h2 className="text-3xl md:text-4xl font-black mb-3">
                <span className="block text-gray-900 dark:text-white">Welcome to</span>
                <span className="block bg-gradient-to-r from-primary-700 via-primary-500 to-primary-400 bg-clip-text text-transparent">
                  LinguaSign Chat
                </span>
              </h2>

              {/* Description */}
              <p className="text-gray-600 dark:text-gray-300 mb-8 leading-relaxed">
                Connect with friends, share sign language translations, and experience 
                real-time AI-powered conversations.
              </p>



              {/* Action Buttons */}
<div className="space-y-4">
  <Link to="/login" className="block">
    <motion.button
      whileHover={{ scale: 1.03, boxShadow: "0 0 25px rgba(160, 68, 255, 0.6)" }}
      whileTap={{ scale: 0.98 }}
      className="w-full relative overflow-hidden px-6 py-4 bg-gradient-to-r from-primary-custom-1 via-primary-custom-2 to-primary-custom-3 text-white font-bold rounded-xl shadow-lg shadow-primary-500/40 hover:shadow-primary-500/60 transition-all duration-300 group"
    >
      <span className="relative z-10 flex items-center justify-center gap-3">
        <FaSignInAlt className="group-hover:scale-110 transition-transform" />
        Sign In to Your Account
      </span>
      <div className="absolute top-0 left-0 w-full h-full bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 z-0 rounded-xl" />
    </motion.button>
  </Link>

  <Link to="/register" className="block">
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className="w-full px-6 py-4 border-2 border-primary-600/50 dark:border-primary-500/50 text-primary-600 dark:text-primary-400 hover:bg-primary-50/50 dark:hover:bg-primary-900/20 hover:border-primary-700/50 dark:hover:border-primary-400/50 rounded-xl font-bold transition-all duration-300 flex items-center justify-center gap-3 group"
    >
      <FaUserPlus className="group-hover:scale-110 transition-transform" />
      Create New Account
    </motion.button>
  </Link>
</div>

              {/* Guest Access Note */}
              <p className="text-xs text-center text-gray-500 dark:text-gray-400 mt-6">
                Don't have an account? Sign up in just a few clicks!
              </p>
            </motion.div>

            {/* Security Footer */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
              className="mt-8 text-center"
            >
              <div className="inline-flex items-center gap-3 text-sm text-gray-500 dark:text-gray-400">
                <FaShieldAlt className="text-primary-500" />
                <span>Secure authentication • Your data is protected with 256-bit SSL</span>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-screen bg-gradient-to-br from-gray-50 via-white to-primary-50/60 dark:from-primary-bg-1 dark:via-primary-bg-2 dark:to-primary-bg-3 overflow-hidden relative">
      
      {/* Particle Canvas - NO connections */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none opacity-40"
      />

      {/* Geometric Grid Background */}
      <div className="absolute inset-0 opacity-20 dark:opacity-30 pointer-events-none">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(90deg, rgba(168, 85, 247, 0.08) 1px, transparent 1px),
            linear-gradient(180deg, rgba(168, 85, 247, 0.08) 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px'
        }} />
      </div>

      {/* Animated gradient orbs */}
      <motion.div
        className="absolute top-20 left-20 w-[400px] h-[400px] bg-primary-600/10 rounded-full blur-[120px] pointer-events-none"
        animate={{
          x: [0, 50, 0],
          y: [0, -50, 0],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "linear",
        }}
      />
      <motion.div
        className="absolute bottom-20 right-20 w-[400px] h-[400px] bg-primary-400/10 rounded-full blur-[120px] pointer-events-none"
        animate={{
          x: [0, -50, 0],
          y: [0, 50, 0],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "linear",
        }}
      />

      {/* Search User Modal */}
      <SearchUserModal 
        isOpen={showSearchModal}
        onClose={() => setShowSearchModal(false)}
        onUserFound={handleUserFound}
        currentUsername={currentUsername}
      />

      {/* Connection Status */}
      <div className="absolute top-4 right-4 z-20 flex items-center gap-2 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm px-3 py-1.5 rounded-full border-2 border-primary-200/50 dark:border-primary-800/50 shadow-lg">
        <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`}></div>
        <span className="text-xs font-semibold text-gray-600 dark:text-gray-400">
          {wsConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>

      <div className="relative z-10 w-full h-full flex flex-col md:flex-row">
        {/* Sidebar */}
        <div className="hidden md:block">
          <ChatSidebar 
            conversations={conversations}
            onSelectConversation={handleSelectConversation}
            activeConversation={activeConversation}
            currentUser={user}
            onNewChat={() => setShowSearchModal(true)}
          />
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col h-full">
          {/* Chat Header */}
          <motion.div 
            initial="hidden"
            animate="show"
            variants={fadeUp}
            className="px-6 py-4 border-b-2 border-primary-200/50 dark:border-primary-800/50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl flex items-center justify-between"
          >
            <div className="flex items-center gap-3">
              <button className="md:hidden p-2 hover:bg-gray-100/50 dark:hover:bg-gray-800/50 rounded-lg">
                <BsChevronLeft className="text-gray-600 dark:text-gray-400" />
              </button>
              <div className="relative">
                <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center shadow-lg">
                  <FaUser className="text-white text-2xl" />
                </div>
                {activeConversation?.online && (
                  <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-white dark:border-gray-900"></div>
                )}
              </div>
              <div>
                <h2 className="font-black text-lg text-gray-800 dark:text-white">
                  {activeConversation ? (activeConversation.other_username || activeConversation.username) : 'Select a user'}
                </h2>
                <p className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-1 font-medium">
                  {activeConversation ? (
                    <>
                      <span className={`w-2 h-2 ${activeConversation.online ? 'bg-green-500' : 'bg-gray-400'} rounded-full`}></span>
                      {activeConversation.online ? 'Online' : 'Offline'}
                    </>
                  ) : (
                    'Start a new conversation'
                  )}
                </p>
              </div>
            </div>

            {/* Mobile New Chat Button */}
            <button 
              onClick={() => setShowSearchModal(true)}
              className="md:hidden p-3 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-full hover:shadow-lg hover:shadow-primary-500/30 transition-all"
            >
              <FaPlus className="text-xl" />
            </button>

            {/* Header Actions */}
            {activeConversation && (
              <div className="hidden md:flex items-center gap-2">
                <button 
                  onClick={() => initiateCall("audio")}
                  className={`p-3 rounded-full hover:bg-gray-100/50 dark:hover:bg-gray-800/50 transition-colors ${!wsConnected ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={!wsConnected}
                  title="Start audio call"
                >
                  <BsPhone className={`text-xl ${wsConnected ? 'text-gray-600 dark:text-gray-400' : 'text-gray-400'}`} />
                </button>
                <button 
                  onClick={() => initiateCall("video")}
                  className={`p-3 rounded-full hover:bg-gray-100/50 dark:hover:bg-gray-800/50 transition-colors ${!wsConnected ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={!wsConnected}
                  title="Start video call"
                >
                  <BsCameraVideo className={`text-xl ${wsConnected ? 'text-gray-600 dark:text-gray-400' : 'text-gray-400'}`} />
                </button>
                <button className="p-3 rounded-full hover:bg-gray-100/50 dark:hover:bg-gray-800/50 transition-colors">
                  <BsInfoCircle className="text-gray-600 dark:text-gray-400 text-xl" />
                </button>
                <button className="p-2 hover:bg-gray-100/50 dark:hover:bg-gray-800/50 rounded-lg">
                  <BsThreeDotsVertical className="text-gray-600 dark:text-gray-400" />
                </button>
              </div>
            )}
          </motion.div>

          {/* Messages Container */}
          <div className="flex-1 overflow-y-auto p-4 md:p-6 bg-gradient-to-b from-transparent to-primary-50/20 dark:to-primary-900/5">
            {!activeConversation ? (
              <motion.div 
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-center mt-20"
              >
                <div className="inline-block bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm px-8 py-8 rounded-2xl border-2 border-primary-200/50 dark:border-primary-800/50 shadow-2xl shadow-primary-500/20">
                  <div className="w-20 h-20 rounded-full bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center mx-auto mb-4 shadow-lg">
                    <TbMessages className="text-white text-3xl" />
                  </div>
                  <span className="font-black text-gray-800 dark:text-white text-xl mb-2 block">
                    Welcome, @{currentUsername}!
                  </span>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    Click "New Conversation" to start chatting
                  </p>
                  <button
                    onClick={() => setShowSearchModal(true)}
                    className="px-6 py-3 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-full font-bold hover:shadow-lg hover:shadow-primary-500/30 transition-all"
                  >
                    Start New Chat
                  </button>
                </div>
              </motion.div>
            ) : loading ? (
              <div className="flex justify-center items-center h-full">
                <AiOutlineLoading3Quarters className="animate-spin text-primary-500 text-3xl" />
              </div>
            ) : (
              <>
                {messages.map((message) => (
                  <Message
                    key={message.id}
                    message={message.text}
                    isOwn={message.isOwn}
                    time={message.time}
                    status={message.status}
                    type={message.type}
                  />
                ))}
                
                {otherUserTyping && (
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center shadow-lg">
                      <FaUser className="text-white text-xl" />
                    </div>
                    <div className="px-4 py-3 rounded-2xl bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-2 border-primary-200/50 dark:border-primary-800/50">
                      <div className="flex items-center gap-1">
                        <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Actions Bar */}
          {activeConversation && (
            <div className="px-4 py-3 border-t-2 border-primary-200/50 dark:border-primary-800/50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl">
              <div className="flex items-center justify-center gap-4">
                <button className="flex flex-col items-center gap-1 px-4 py-2 rounded-xl hover:bg-gray-100/50 dark:hover:bg-gray-800/50 transition-colors">
                  <div className="p-2 rounded-full bg-gradient-to-br from-primary-500/20 to-primary-400/20">
                    <TbHandLoveYou className="text-primary-500 text-xl" />
                  </div>
                  <span className="text-xs font-semibold text-gray-600 dark:text-gray-400">Gesture</span>
                </button>
                <button className="flex flex-col items-center gap-1 px-4 py-2 rounded-xl hover:bg-gray-100/50 dark:hover:bg-gray-800/50 transition-colors">
                  <div className="p-2 rounded-full bg-gradient-to-br from-blue-500/20 to-cyan-400/20">
                    <BsMicFill className="text-blue-500 text-xl" />
                  </div>
                  <span className="text-xs font-semibold text-gray-600 dark:text-gray-400">Voice</span>
                </button>
                <button 
                  onClick={() => initiateCall("video")}
                  className={`flex flex-col items-center gap-1 px-4 py-2 rounded-xl hover:bg-gray-100/50 dark:hover:bg-gray-800/50 transition-colors ${!wsConnected ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={!wsConnected}
                >
                  <div className="p-2 rounded-full bg-gradient-to-br from-green-500/20 to-emerald-400/20">
                    <BsCameraVideo className={`text-xl ${wsConnected ? 'text-green-500' : 'text-gray-400'}`} />
                  </div>
                  <span className="text-xs font-semibold text-gray-600 dark:text-gray-400">Video</span>
                </button>
                <button 
                  onClick={() => initiateCall("audio")}
                  className={`flex flex-col items-center gap-1 px-4 py-2 rounded-xl hover:bg-gray-100/50 dark:hover:bg-gray-800/50 transition-colors ${!wsConnected ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={!wsConnected}
                >
                  <div className="p-2 rounded-full bg-gradient-to-br from-orange-500/20 to-amber-400/20">
                    <BsPhone className={`text-xl ${wsConnected ? 'text-orange-500' : 'text-gray-400'}`} />
                  </div>
                  <span className="text-xs font-semibold text-gray-600 dark:text-gray-400">Audio</span>
                </button>
              </div>
            </div>
          )}

          {/* Chat Input */}
          {activeConversation && (
            <ChatInput 
              onSendMessage={handleSendMessage}
              isTyping={isTyping}
              onTyping={setIsTyping}
            />
          )}
        </div>
      </div>

      {/* Video Call Components */}
      <VideoCall
        isOpen={showVideoCall}
        onClose={() => {
          setShowVideoCall(false);
          setActiveCall(null);
        }}
        remoteUsername={activeCall?.remoteUsername}
        localUsername={currentUsername}
        callType={activeCall?.callType}
        ws={callWs}
        isCaller={activeCall?.isCaller}
        callId={activeCall?.callId}
        incomingOffer={activeCall?.offer}
      />

      <IncomingCallModal
        isOpen={showIncomingCall}
        caller={incomingCall?.caller}
        callType={incomingCall?.callType}
        onAccept={() => {
          setShowIncomingCall(false);
          setActiveCall({
            callId: incomingCall.callId,
            remoteUsername: incomingCall.caller,
            callType: incomingCall.callType,
            isCaller: false,
            offer: incomingCall.offer
          });
          setShowVideoCall(true);
        }}
        onReject={() => {
          setShowIncomingCall(false);
          if (callWs && callWs.readyState === WebSocket.OPEN) {
            callWs.send(JSON.stringify({
              type: "call-reject",
              target: incomingCall.caller,
              callId: incomingCall.callId
            }));
          }
          setIncomingCall(null);
        }}
      />
    </div>
  );
}
