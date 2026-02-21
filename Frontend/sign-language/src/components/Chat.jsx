import React, { useState, useRef, useEffect, useContext } from "react";
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
  TbSend
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
  FaPlus
} from "react-icons/fa";
import axios from "axios";
import { AuthContext } from "../context/AuthContext";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Emoji picker component
const EmojiPicker = ({ onEmojiSelect, onClose }) => {
  const emojis = ["😀", "😊", "👍", "👋", "❤️", "🎉", "🤔", "👏", "🙏", "🤝", "💬", "🎯", "✨", "🚀", "💡"];
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      className="absolute bottom-16 left-4 bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 p-4 w-72 z-50"
    >
      <div className="flex justify-between items-center mb-3">
        <span className="font-semibold text-gray-800 dark:text-white">Emojis</span>
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
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-purple-600 flex items-center justify-center">
              <BsPersonCircle className="text-white text-xl" />
            </div>
          ) : (
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center">
              <FaUser className="text-white text-xl" />
            </div>
          )}
        </div>
        
        {/* Message Content */}
        <div>
          <div className={`px-4 py-3 rounded-2xl ${isOwn 
            ? 'bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-br-none' 
            : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-800 dark:text-white rounded-bl-none'
          }`}>
            {type === "text" ? (
              <p className="text-sm">{message}</p>
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
              <div className="text-purple-500">
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
      // Check if user exists using your backend endpoint
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
        className="bg-white dark:bg-gray-800 rounded-2xl p-6 max-w-md w-full mx-4 border border-gray-200 dark:border-gray-700 shadow-2xl"
      >
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-bold text-gray-800 dark:text-white">New Conversation</h3>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
          >
            <RiCloseFill className="text-gray-600 dark:text-gray-400 text-xl" />
          </button>
        </div>

        <form onSubmit={handleSearch}>
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Enter username
            </label>
            <div className="relative">
              <FaSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="e.g., john_doe"
                className="w-full pl-10 pr-4 py-3 bg-gray-100 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 text-gray-800 dark:text-white"
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
              className="flex-1 py-3 px-4 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-white rounded-xl font-semibold hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex-1 py-3 px-4 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-xl font-semibold hover:shadow-lg hover:shadow-purple-500/30 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
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
    <div className="relative px-4 py-3 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
      {isTyping && (
        <div className="absolute -top-8 left-4 bg-white dark:bg-gray-800 px-3 py-2 rounded-full shadow-md border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
              <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
              <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
            </div>
            <span className="text-sm text-gray-600 dark:text-gray-400">Typing...</span>
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
          className="p-3 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        >
          <BsPlusCircleFill className="text-purple-500 text-xl" />
        </button>

        <button
          type="button"
          onClick={() => setShowEmojiPicker(!showEmojiPicker)}
          className="p-3 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        >
          <BsEmojiSmile className="text-gray-600 dark:text-gray-400 text-xl" />
        </button>

        <div className="flex-1 relative">
          <input
            type="text"
            value={message}
            onChange={handleChange}
            placeholder="Type your message..."
            className="w-full px-4 py-3 bg-gray-100 dark:bg-gray-800 border-0 rounded-full focus:outline-none focus:ring-2 focus:ring-purple-500 text-gray-800 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
          />
          
          <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex items-center gap-2">
            <button
              type="button"
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-full transition-colors"
            >
              <BsImage className="text-gray-600 dark:text-gray-400" />
            </button>
            <button
              type="button"
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-full transition-colors"
            >
              <FiPaperclip className="text-gray-600 dark:text-gray-400" />
            </button>
          </div>
        </div>

        {message.trim() ? (
          <button
            type="submit"
            className="p-3 rounded-full bg-gradient-to-r from-purple-500 to-purple-600 text-white hover:shadow-lg hover:shadow-purple-500/30 transition-all"
          >
            <TbSend className="text-xl" />
          </button>
        ) : (
          <button
            type="button"
            onClick={toggleRecording}
            className={`p-3 rounded-full transition-all ${recording 
              ? 'bg-red-500 text-white animate-pulse' 
              : 'bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-300 dark:hover:bg-gray-700'
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
    <div className="w-full md:w-80 border-r border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 h-full overflow-hidden flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-500 to-purple-600 flex items-center justify-center">
              <TbMessages className="text-white text-2xl" />
            </div>
            <div>
              <h2 className="font-bold text-lg text-gray-800 dark:text-white">Messages</h2>
              <p className="text-sm text-purple-500">@{currentUser?.email?.split('@')[0] || 'user'}</p>
            </div>
          </div>
        </div>

        {/* New Conversation Button - ALWAYS VISIBLE */}
        <button 
          onClick={onNewChat}
          className="mt-4 w-full py-3 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-full font-semibold hover:shadow-lg hover:shadow-purple-500/30 transition-all flex items-center justify-center gap-2"
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
            className="w-full pl-10 pr-4 py-2 bg-gray-100 dark:bg-gray-800 border-0 rounded-full focus:outline-none focus:ring-2 focus:ring-purple-500 text-gray-800 dark:text-white"
          />
        </div>
      </div>

      {/* Conversation List */}
      <div className="flex-1 overflow-y-auto">
        {conversations.length === 0 ? (
          <div className="p-8 text-center text-gray-500 dark:text-gray-400">
            <FaUserFriends className="text-4xl mx-auto mb-3 opacity-50" />
            <p>No conversations yet</p>
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
                className={`w-full p-4 text-left border-b border-gray-100 dark:border-gray-800 transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 border-l-4 border-purple-500' 
                    : 'hover:bg-gray-50 dark:hover:bg-gray-800'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center">
                      <FaUser className="text-white text-xl" />
                    </div>
                    {conversation.online && (
                      <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-white dark:border-gray-900"></div>
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between items-start">
                      <h3 className="font-semibold text-gray-800 dark:text-white">
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
export default function ChatPage() {
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
  
  const messagesEndRef = useRef(null);
  const wsRef = useRef(null);

  const currentUsername = user?.email?.split('@')[0] || '';

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

  // Load conversations from localStorage on mount
  useEffect(() => {
    if (currentUsername && isAuthenticated) {
      const savedConversations = JSON.parse(localStorage.getItem(`conversations_${currentUsername}`) || '[]');
      
      // Sort by last message time (most recent first)
      const sortedConversations = savedConversations.sort((a, b) => 
        new Date(b.last_message_time) - new Date(a.last_message_time)
      );
      
      setConversations(sortedConversations);
      
      // If there are conversations, load the most recent one
      if (sortedConversations.length > 0 && !activeConversation) {
        const mostRecent = sortedConversations[0];
        setActiveConversation(mostRecent);
        loadChatHistory(mostRecent.other_username || mostRecent.username);
      }
    }
  }, [currentUsername, isAuthenticated]);

  // WebSocket connection
  useEffect(() => {
    if (currentUsername && isAuthenticated) {
      connectWebSocket();
      return () => {
        if (wsRef.current) {
          wsRef.current.close();
        }
      };
    }
  }, [currentUsername, isAuthenticated]);

  const connectWebSocket = () => {
    try {
      const ws = new WebSocket(`ws://localhost:8000/ws/chat/${currentUsername}`);
      
      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setWsConnected(true);
        wsRef.current = ws;
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('📩 Received:', data);
        
        // Handle incoming message
        if (data.from && data.from !== currentUsername) {
          // Only add if we're in the correct conversation
          if (activeConversation && 
              (data.from === activeConversation.other_username || 
               data.from === activeConversation.username)) {
            
            const newMessage = {
              id: Date.now(),
              text: data.message,
              isOwn: false,
              time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
              status: 'delivered',
              type: 'text'
            };
            
            setMessages(prev => [...prev, newMessage]);
          }
          
          // Update conversations list
          updateConversationFromMessage(data.from, data.message);
        }
        
        // Handle typing indicator
        if (data.typing !== undefined && data.from && data.from !== currentUsername) {
          if (activeConversation && 
              (data.from === activeConversation.other_username || 
               data.from === activeConversation.username)) {
            setOtherUserTyping(data.typing);
          }
        }
      };

      ws.onclose = () => {
        console.log('❌ WebSocket disconnected');
        setWsConnected(false);
        wsRef.current = null;
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setWsConnected(false);
      };
    } catch (error) {
      console.error('WebSocket connection error:', error);
    }
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
    if (!currentUsername || !otherUsername) return;
    
    setLoading(true);
    try {
      const response = await axios.get(
        `${API_URL}/chat/history/${currentUsername}/${otherUsername}`
      );
      
      console.log('Chat history:', response.data);
      
      // If we don't have the user ID yet and there are messages,
      // try to find our ID from the messages where sender is current user
      let userId = currentUserId;
      
      if (!userId && response.data.length > 0) {
        // Look for a message where the sender is the current user
        // Since we don't have the ID, we need to make an assumption
        // The first message in history might be from either user
        // Let's try to get the ID from the first message where the sender username matches
        // But since we don't have sender username in the response, we'll use a different approach
        
        // For now, let's get the IDs of both users from the first message
        const firstMsg = response.data[0];
        const secondMsg = response.data.length > 1 ? response.data[1] : null;
        
        // The current user ID is either firstMsg.sender_id or firstMsg.receiver_id
        // We need to determine which one is us
        // We can store both possibilities and try to determine from context
        const possibleIds = [firstMsg.sender_id, firstMsg.receiver_id];
        
        // Store both IDs and we'll determine later based on message patterns
        localStorage.setItem('possible_user_ids', JSON.stringify(possibleIds));
        
        // For now, let's assume the first message's sender is the other user
        // and we are the receiver (common in chat history)
        userId = firstMsg.receiver_id;
        setCurrentUserId(userId);
        localStorage.setItem('userId', userId);
      }
      
      const formattedMessages = response.data.map(msg => {
        // If we have a user ID, use it to determine ownership
        const isOwn = userId ? msg.sender_id === userId : false;
        
        return {
          id: msg.id,
          text: msg.content,
          isOwn: isOwn,
          time: new Date(msg.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          status: 'read',
          type: 'text'
        };
      });
      
      setMessages(formattedMessages);
      
      // Save to conversations
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

    // Add message to UI immediately
    const newMessage = {
      id: Date.now(),
      text: text,
      isOwn: true,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      status: 'sent',
      type: 'text'
    };
    
    setMessages(prev => [...prev, newMessage]);

    // Send via WebSocket
    wsRef.current.send(JSON.stringify({
      to: receiverUsername,
      message: text
    }));
    
    // Update conversation
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
      <div className="w-full h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 via-white to-purple-50/60 dark:from-[#0a0518] dark:via-[#110a2e] dark:to-[#1e0f5c]">
        <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm p-8 rounded-2xl border border-gray-200 dark:border-gray-700 text-center">
          <TbMessages className="text-5xl text-purple-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-2">Please Login</h2>
          <p className="text-gray-600 dark:text-gray-400">You need to be logged in to chat</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-screen bg-gradient-to-br from-gray-50 via-white to-purple-50/60 dark:from-[#0a0518] dark:via-[#110a2e] dark:to-[#1e0f5c] overflow-hidden">
      {/* Geometric Grid Background */}
      <div className="absolute inset-0 opacity-20 dark:opacity-30 pointer-events-none">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(90deg, rgba(168, 85, 247, 0.1) 1px, transparent 1px),
            linear-gradient(180deg, rgba(168, 85, 247, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '60px 60px'
        }} />
      </div>

      {/* Search User Modal */}
      <SearchUserModal 
        isOpen={showSearchModal}
        onClose={() => setShowSearchModal(false)}
        onUserFound={handleUserFound}
        currentUsername={currentUsername}
      />

      {/* Connection Status */}
      <div className="absolute top-4 right-4 z-20 flex items-center gap-2 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm px-3 py-1.5 rounded-full border border-gray-200 dark:border-gray-700">
        <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`}></div>
        <span className="text-xs text-gray-600 dark:text-gray-400">
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
            className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl flex items-center justify-between"
          >
            <div className="flex items-center gap-3">
              <button className="md:hidden p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg">
                <BsChevronLeft className="text-gray-600 dark:text-gray-400" />
              </button>
              <div className="relative">
                <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center">
                  <FaUser className="text-white text-2xl" />
                </div>
                {activeConversation?.online && (
                  <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-white dark:border-gray-900"></div>
                )}
              </div>
              <div>
                <h2 className="font-bold text-lg text-gray-800 dark:text-white">
                  {activeConversation ? (activeConversation.other_username || activeConversation.username) : 'Select a user'}
                </h2>
                <p className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-1">
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
              className="md:hidden p-3 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-full hover:shadow-lg hover:shadow-purple-500/30 transition-all"
            >
              <FaPlus className="text-xl" />
            </button>

            {/* Header Actions */}
            {activeConversation && (
              <div className="hidden md:flex items-center gap-2">
                <button className="p-3 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                  <BsPhone className="text-gray-600 dark:text-gray-400" />
                </button>
                <button className="p-3 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                  <BsCameraVideo className="text-gray-600 dark:text-gray-400" />
                </button>
                <button className="p-3 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                  <BsInfoCircle className="text-gray-600 dark:text-gray-400" />
                </button>
                <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg">
                  <BsThreeDotsVertical className="text-gray-600 dark:text-gray-400" />
                </button>
              </div>
            )}
          </motion.div>

          {/* Messages Container */}
          <div className="flex-1 overflow-y-auto p-4 md:p-6 bg-gradient-to-b from-transparent to-purple-50/20 dark:to-purple-900/5">
            {!activeConversation ? (
              <motion.div 
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-center mt-20"
              >
                <div className="inline-block bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm px-8 py-8 rounded-2xl border border-gray-200 dark:border-gray-700">
                  <div className="w-20 h-20 rounded-full bg-gradient-to-br from-purple-500 to-purple-600 flex items-center justify-center mx-auto mb-4">
                    <TbMessages className="text-white text-3xl" />
                  </div>
                  <span className="font-semibold text-gray-800 dark:text-white text-xl mb-2 block">
                    Welcome, @{currentUsername}!
                  </span>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    Click "New Conversation" to start chatting
                  </p>
                  <button
                    onClick={() => setShowSearchModal(true)}
                    className="px-6 py-3 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-full font-semibold hover:shadow-lg hover:shadow-purple-500/30 transition-all"
                  >
                    Start New Chat
                  </button>
                </div>
              </motion.div>
            ) : loading ? (
              <div className="flex justify-center items-center h-full">
                <AiOutlineLoading3Quarters className="animate-spin text-purple-500 text-3xl" />
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
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center">
                      <FaUser className="text-white text-xl" />
                    </div>
                    <div className="px-4 py-3 rounded-2xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
                      <div className="flex items-center gap-1">
                        <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
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
            <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl">
              <div className="flex items-center justify-center gap-4">
                <button className="flex flex-col items-center gap-1 px-4 py-2 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                  <div className="p-2 rounded-full bg-gradient-to-br from-purple-500/20 to-purple-400/20">
                    <TbHandLoveYou className="text-purple-500 text-xl" />
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400">Gesture</span>
                </button>
                <button className="flex flex-col items-center gap-1 px-4 py-2 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                  <div className="p-2 rounded-full bg-gradient-to-br from-blue-500/20 to-cyan-400/20">
                    <BsMicFill className="text-blue-500 text-xl" />
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400">Voice</span>
                </button>
                <button className="flex flex-col items-center gap-1 px-4 py-2 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                  <div className="p-2 rounded-full bg-gradient-to-br from-green-500/20 to-emerald-400/20">
                    <BsCameraVideo className="text-green-500 text-xl" />
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400">Video</span>
                </button>
                <button className="flex flex-col items-center gap-1 px-4 py-2 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                  <div className="p-2 rounded-full bg-gradient-to-br from-orange-500/20 to-amber-400/20">
                    <FaCog className="text-orange-500 text-xl" />
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400">Settings</span>
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
    </div>
  );
}