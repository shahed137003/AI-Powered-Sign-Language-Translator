import React, { useEffect, useState, useRef } from "react";
import { FaHandBackFist, FaBars, FaXmark } from "react-icons/fa6";
import { FaSun, FaMoon, FaSignInAlt, FaSignOutAlt, FaUser, FaCaretDown, FaCrown, FaPalette } from "react-icons/fa";
import { BsTranslate, BsFillChatDotsFill, BsStars, BsLightningCharge } from "react-icons/bs";
import { TbHome, TbMail, TbUser, TbSparkles,TbHelp, TbMessageChatbot } from "react-icons/tb";
import { NavLink, useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../context/useAuth";
import { useTheme } from "../context/ThemeContext";
import { motion, AnimatePresence } from "framer-motion";
export default function Navbar() {
  const [darkMode, setDarkMode] = useState(() => {
    if (
      localStorage.getItem("theme") === "dark" ||
      (!("theme" in localStorage) &&
        window.matchMedia("(prefers-color-scheme: dark)").matches)
    ) {
      return true;
    }
    return false;
  });

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [appearanceMenuOpen, setAppearanceMenuOpen] = useState(false);
  const [hoveredItem, setHoveredItem] = useState(null);

  const { user, logout, isAuthenticated } = useAuth();
  const { themeColor, setThemeColor } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();

  // Apply dark mode to document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
    document.body.classList.toggle("dark", darkMode);
  }, [darkMode]);

  // Scroll effect
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Close mobile menu when route changes
  useEffect(() => {
    setMobileMenuOpen(false);
    setUserMenuOpen(false);
  }, [location]);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const toggleMobileMenu = () => setMobileMenuOpen(!mobileMenuOpen);
  const toggleUserMenu = () => {
    setUserMenuOpen(!userMenuOpen);
    setAppearanceMenuOpen(false);
  };

  const toggleAppearanceMenu = () => {
    setAppearanceMenuOpen(!appearanceMenuOpen);
    setUserMenuOpen(false);
  };
  const handleLogout = () => {
    logout();
    navigate('/');
    setUserMenuOpen(false);
  };

  const navItems = [
    { name: "Home", to: "/", icon: <TbHome />, color: "from-primary-500 to-primary-400" },
    
    { name: "Guide", to: "/guide", icon: <TbHelp /> },
    { name: "Translate", to: "/translate", icon: <BsTranslate />, color: "from-primary-500 to-primary-300" },
    { name: "Profile", to: "/profile", icon: <BsStars />, color: "from-primary-600 to-primary-400" },
    { name: "Contact", to: "/contactus", icon: <TbMail />, color: "from-primary-500 to-primary-400" },
    
  ];

  const userMenuItems = [
    { name: "Dashboard", to: "/", icon: <TbHome />, color: "from-primary-500 to-primary-400" },
    { name: "Profile", to: "/profile", icon: <FaUser />, color: "from-primary-600 to-primary-300" },
    { name: "Chatbot", to: "/chat", icon: <TbMessageChatbot />, color: "from-primary-500 to-primary-400" },
    { name: "Settings", to: "/profile", icon: <BsLightningCharge />, color: "from-primary-600 to-primary-400" },
  ];

  // Nav animations
  const containerVariants = {
    hidden: { opacity: 0, y: -20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5,
        ease: "easeOut",
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: (i) => ({
      opacity: 1,
      x: 0,
      transition: {
        delay: i * 0.1,
        duration: 0.3,
        ease: "easeOut"
      }
    })
  };

  const mobileMenuVariants = {
    hidden: { opacity: 0, height: 0, y: -20 },
    visible: { 
      opacity: 1,
      height: "auto",
      y: 0,
      transition: { duration: 0.3, ease: "easeInOut" }
    },
    exit: { 
      opacity: 0,
      height: 0,
      y: -20,
      transition: { duration: 0.2, ease: "easeIn" }
    }
  };

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={containerVariants}
      className={`
        fixed top-0 left-0 w-full px-4 sm:px-6 lg:px-8 py-3 flex items-center justify-between
        bg-gradient-to-r from-white/80 via-white/90 to-white/80 
        dark:from-primary-bg-1/95 dark:via-primary-bg-2/95 dark:to-primary-bg-1/95
        backdrop-blur-xl
        border-b-2 border-primary-200/50 dark:border-primary-900/40
        ${scrolled 
          ? "shadow-2xl shadow-primary-500/15 dark:shadow-primary-900/30" 
          : "shadow-lg shadow-primary-500/5 dark:shadow-primary-900/20"
        }
        transition-all duration-500 z-50
      `}
    >
      {/* Logo with enhanced animation */}
      <motion.div
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        className="flex items-center gap-2 sm:gap-3"
      >
        <NavLink
          to="/"
          className="flex items-center gap-2 sm:gap-3 group"
        >
          <div className="relative">
            <motion.div
              animate={{ 
                rotate: [0, 10, 0, -5, 0],
                scale: [1, 1.05, 1]
              }}
              transition={{ 
                duration: 4, 
                repeat: Infinity, 
                ease: "easeInOut",
                repeatDelay: 2
              }}
              className="relative"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-primary-custom-2 to-primary-custom-3 rounded-full blur-xl opacity-60 group-hover:opacity-100 transition-opacity duration-500" />
              <FaHandBackFist className="text-3xl sm:text-4xl text-primary-custom-3 relative z-10" />
            </motion.div>
          </div>
          <span className="
            text-2xl sm:text-3xl font-black italic
            bg-gradient-to-r from-primary-custom-1 via-primary-custom-2 to-primary-custom-3 
            bg-clip-text text-transparent 
            drop-shadow-lg
            group-hover:drop-shadow-xl
            transition-all duration-300
          ">
            LinguaSign
          </span>
        </NavLink>
      </motion.div>

      {/* Desktop Navigation - Enhanced */}
      <div className="hidden md:flex items-center gap-1 lg:gap-2">
        {navItems.map((item, i) => (
          <motion.div
            key={item.to}
            custom={i}
            variants={itemVariants}
            initial="hidden"
            animate="visible"
            onMouseEnter={() => setHoveredItem(i)}
            onMouseLeave={() => setHoveredItem(null)}
          >
            <NavLink
              to={item.to}
              className={({ isActive }) => `
                relative px-5 py-2.5 mx-1 rounded-xl flex items-center gap-2
                text-gray-700 dark:text-gray-200 font-semibold text-sm
                transition-all duration-300 group
                ${isActive 
                  ? "text-primary-700 dark:text-primary-300 bg-gradient-to-r from-primary-50/80 to-primary-100/50 dark:from-primary-900/40 dark:to-primary-800/20 shadow-inner border border-primary-200/50 dark:border-primary-700/30" 
                  : "hover:text-primary-600 dark:hover:text-primary-300 hover:bg-white/60 dark:hover:bg-white/10"
                }
              `}
            >
              {({ isActive }) => (
                <>
                  <motion.span 
                    className="text-lg opacity-80"
                    animate={hoveredItem === i || isActive ? { scale: 1.1, rotate: 5 } : { scale: 1, rotate: 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    {item.icon}
                  </motion.span>
                  {item.name}
                  
                  {/* Active indicator with gradient */}
                  {isActive && (
                    <motion.div
                      layoutId="activeIndicator"
                      className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-8 h-1 rounded-full bg-gradient-to-r from-primary-custom-1 via-primary-custom-2 to-primary-custom-3"
                      initial={{ width: 0, opacity: 0 }}
                      animate={{ width: 32, opacity: 1 }}
                      transition={{ duration: 0.3 }}
                    />
                  )}
                  
                  {/* Hover glow effect */}
                  <motion.div 
                    className="absolute inset-0 bg-gradient-to-r from-primary-500/0 via-primary-400/10 to-primary-500/0 rounded-xl opacity-0 group-hover:opacity-100 transition-all duration-300"
                    animate={hoveredItem === i ? { opacity: 1 } : { opacity: 0 }}
                  />
                </>
              )}
            </NavLink>
          </motion.div>
        ))}
      </div>

      {/* Right side controls */}
      <div className="flex items-center gap-3 lg:gap-4">
        {/* User Profile Section - Enhanced */}
        {isAuthenticated ? (
          <div className="relative">
            <motion.button
              whileHover={{ scale: 1.03, y: -2 }}
              whileTap={{ scale: 0.98 }}
              onClick={toggleUserMenu}
              className="
                hidden sm:flex items-center gap-3 px-4 py-2
                bg-white/80 dark:bg-white/5
                backdrop-blur-xl
                border-2 border-primary-200/50 dark:border-primary-800/50
                rounded-2xl
                shadow-lg shadow-primary-500/10 dark:shadow-primary-900/20
                hover:shadow-xl hover:shadow-primary-500/20 dark:hover:shadow-primary-900/30
                hover:border-primary-300 dark:hover:border-primary-600
                transition-all duration-300 group
              "
            >
              <div className="relative">
                <div className="
                  w-10 h-10 rounded-full 
                  bg-gradient-to-br from-primary-custom-1 via-primary-custom-2 to-primary-custom-3
                  flex items-center justify-center
                  shadow-inner
                ">
                  <FaUser className="text-white text-sm" />
                </div>
                <motion.div 
                  className="
                    absolute -inset-1 rounded-full 
                    bg-gradient-to-r from-primary-custom-1 via-primary-custom-2 to-primary-custom-3 
                    blur-md transition-opacity duration-300
                  "
                  animate={{ opacity: [0, 0.3, 0] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              </div>
              
              <div className="text-left">
                <p className="text-sm font-black text-gray-800 dark:text-white">
                  {user?.email?.split('@')[0] || 'User'}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400 truncate max-w-[120px]">
                  {user?.email}
                </p>
              </div>
              
              <FaCaretDown className={`
                text-primary-500 dark:text-primary-400 
                transition-all duration-300
                ${userMenuOpen ? "rotate-180" : "group-hover:rotate-180"}
              `} />
            </motion.button>

            {/* User Dropdown Menu - Enhanced */}
            <AnimatePresence>
              {userMenuOpen && (
                <motion.div
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.95 }}
                  transition={{ duration: 0.2 }}
                  className="
                    absolute right-0 mt-3 w-72
                    bg-white/98 dark:bg-primary-bg-1/98
                    backdrop-blur-xl
                    border-2 border-primary-200/50 dark:border-primary-800/50
                    rounded-2xl
                    shadow-2xl shadow-primary-500/20 dark:shadow-primary-900/40
                    overflow-hidden
                  "
                >
                  {/* Premium Badge */}
                  <div className="absolute top-2 right-2">
                    <div className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-gradient-to-r from-primary-500/20 to-primary-400/10 text-[10px] font-bold text-primary-600 dark:text-primary-400">
                      <FaCrown className="text-[8px]" />
                      Premium
                    </div>
                  </div>

                  {/* User info header */}
                  <div className="p-5 border-b-2 border-primary-200/30 dark:border-primary-800/30 bg-gradient-to-br from-primary-50/30 to-transparent dark:from-primary-900/10">
                    <div className="flex items-center gap-4">
                      <div className="
                        w-14 h-14 rounded-full 
                        bg-gradient-to-br from-primary-custom-1 via-primary-custom-2 to-primary-custom-3
                        flex items-center justify-center
                        shadow-lg
                      ">
                        <FaUser className="text-white text-xl" />
                      </div>
                      <div>
                        <p className="font-black text-gray-800 dark:text-white text-lg">
                          {user?.email?.split('@')[0] || 'User'}
                        </p>
                        <p className="text-sm text-gray-500 dark:text-gray-400 truncate max-w-[180px]">
                          {user?.email}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Menu items */}
                  <div className="p-3">
                    {userMenuItems.map((item) => (
                      <NavLink
                        key={item.to}
                        to={item.to}
                        onClick={() => setUserMenuOpen(false)}
                        className="
                          flex items-center gap-4 px-4 py-3 rounded-xl
                          text-gray-700 dark:text-gray-300 font-medium
                          hover:bg-gradient-to-r hover:from-primary-50 hover:to-primary-100/50 dark:hover:from-primary-900/30 dark:hover:to-primary-800/20
                          hover:text-primary-700 dark:hover:text-primary-300
                          transition-all duration-200 group
                        "
                      >
                        <span className="text-lg text-primary-500 group-hover:scale-110 transition-transform">
                          {item.icon}
                        </span>
                        {item.name}
                      </NavLink>
                    ))}
                  </div>

                  {/* Logout button */}
                  <div className="p-4 border-t-2 border-primary-200/30 dark:border-primary-800/30 bg-gradient-to-t from-primary-50/20 to-transparent dark:from-primary-900/5">
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={handleLogout}
                      className="
                        w-full flex items-center justify-center gap-2
                        px-4 py-3 rounded-xl
                        bg-gradient-to-r from-primary-custom-1 via-primary-custom-2 to-primary-custom-3
                        text-white font-bold
                        shadow-lg shadow-primary-500/30
                        hover:shadow-xl hover:shadow-primary-500/40
                        transition-all duration-300
                        group
                      "
                    >
                      <FaSignOutAlt className="group-hover:rotate-180 transition-transform duration-300" />
                      Logout
                    </motion.button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ) : (
          <NavLink
            to="/login"
            className="
              hidden sm:flex items-center gap-2 px-6 py-2.5
              bg-gradient-to-r from-primary-custom-1 via-primary-custom-2 to-primary-custom-3
              text-white font-bold rounded-full
              shadow-lg shadow-primary-500/40
              hover:shadow-xl hover:shadow-primary-500/60
              hover:scale-105 active:scale-95
              transition-all duration-300
              group relative overflow-hidden
            "
          >
            <span className="
              absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent 
              -translate-x-full group-hover:translate-x-full 
              transition-transform duration-1000
            " />
            <FaSignInAlt className="group-hover:translate-x-1 transition-transform duration-300" />
            Sign In
          </NavLink>
        )}

        {/* Professional Appearance Dropdown */}
        <div className="relative hidden sm:block">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={toggleAppearanceMenu}
            aria-label="Appearance Settings"
            className="
              relative w-10 h-10 flex items-center justify-center
              bg-white/80 dark:bg-white/5
              backdrop-blur-xl
              rounded-full
              border border-primary-200/50 dark:border-primary-800/50
              shadow-md hover:shadow-lg transition-all duration-300
              text-gray-600 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400
            "
          >
            <FaPalette size={16} />
          </motion.button>
          
          <AnimatePresence>
            {appearanceMenuOpen && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.95 }}
                transition={{ duration: 0.2 }}
                className="
                  absolute right-0 mt-3 p-4 w-64
                  bg-white/95 dark:bg-[#0f0920]/95
                  backdrop-blur-xl
                  border border-primary-200/50 dark:border-primary-800/50
                  rounded-2xl
                  shadow-2xl shadow-primary-500/10 dark:shadow-primary-900/40
                  flex flex-col gap-4 z-50
                "
              >
                {/* Theme Mode Segment */}
                <div>
                  <div className="text-xs font-bold text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wider">
                    Appearance
                  </div>
                  <div className="flex bg-gray-100 dark:bg-gray-800/50 p-1 rounded-xl">
                    <button
                      onClick={() => { if(darkMode) toggleDarkMode(); }}
                      className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-semibold transition-all ${
                        !darkMode 
                          ? 'bg-white dark:bg-gray-700 shadow text-primary-600 dark:text-primary-400' 
                          : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                      }`}
                    >
                      <FaSun size={14} /> Light
                    </button>
                    <button
                      onClick={() => { if(!darkMode) toggleDarkMode(); }}
                      className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-semibold transition-all ${
                        darkMode 
                          ? 'bg-gray-800 shadow text-primary-400' 
                          : 'text-gray-500 hover:text-gray-700'
                      }`}
                    >
                      <FaMoon size={14} /> Dark
                    </button>
                  </div>
                </div>

                {/* Accent Color Segment */}
                <div>
                  <div className="text-xs font-bold text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wider">
                    Accent Color
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => setThemeColor('purple')}
                      className={`flex items-center gap-2 p-2 rounded-xl border transition-all ${
                        themeColor === 'purple' 
                          ? 'border-purple-500 bg-purple-50 dark:bg-purple-500/10' 
                          : 'border-transparent hover:bg-gray-50 dark:hover:bg-white/5'
                      }`}
                    >
                      <div className="w-4 h-4 rounded-full bg-purple-500 shadow-sm"></div>
                      <span className={`text-sm font-medium ${themeColor === 'purple' ? 'text-purple-700 dark:text-purple-300' : 'text-gray-600 dark:text-gray-400'}`}>Purple</span>
                    </button>
                    <button
                      onClick={() => setThemeColor('midnight-blue')}
                      className={`flex items-center gap-2 p-2 rounded-xl border transition-all ${
                        themeColor === 'midnight-blue' 
                          ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-500/10' 
                          : 'border-transparent hover:bg-gray-50 dark:hover:bg-white/5'
                      }`}
                    >
                      <div className="w-4 h-4 rounded-full bg-indigo-600 shadow-sm"></div>
                      <span className={`text-sm font-medium ${themeColor === 'midnight-blue' ? 'text-indigo-700 dark:text-indigo-300' : 'text-gray-600 dark:text-gray-400'}`}>Blue</span>
                    </button>
                  </div>
                </div>

              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Mobile Menu Button - Enhanced */}
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={toggleMobileMenu}
          className="
            md:hidden w-10 h-10 flex items-center justify-center
            bg-white/80 dark:bg-white/5
            backdrop-blur-xl
            border-2 border-primary-200/50 dark:border-primary-800/50
            rounded-xl
            shadow-lg shadow-primary-500/10 dark:shadow-primary-900/20
            hover:shadow-xl hover:shadow-primary-500/20 dark:hover:shadow-primary-900/30
            transition-all duration-300
          "
        >
          {mobileMenuOpen ? (
            <FaXmark className="text-gray-700 dark:text-gray-300 text-xl" />
          ) : (
            <FaBars className="text-gray-700 dark:text-gray-300 text-xl" />
          )}
        </motion.button>
      </div>

      {/* Mobile Menu - Enhanced */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            variants={mobileMenuVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            className="
              absolute top-full left-0 w-full md:hidden
              bg-gradient-to-b from-white/98 via-white/96 to-white/94
              dark:from-primary-bg-1/98 dark:via-primary-bg-2/96 dark:to-primary-bg-1/94
              backdrop-blur-2xl
              border-b-2 border-primary-200/50 dark:border-primary-800/50
              shadow-2xl shadow-primary-500/20 dark:shadow-primary-900/40
              overflow-hidden
            "
          >
            <div className="px-6 py-8 space-y-4">
              {/* Navigation Links */}
              {navItems.map((item, i) => (
                <motion.div
                  key={item.to}
                  custom={i}
                  variants={itemVariants}
                  initial="hidden"
                  animate="visible"
                >
                  <NavLink
                    to={item.to}
                    className={({ isActive }) => `
                      flex items-center gap-4 px-5 py-4 rounded-2xl
                      text-lg font-bold
                      transition-all duration-300 group
                      ${isActive 
                        ? "bg-gradient-to-r from-primary-50 to-primary-100/50 dark:from-primary-900/30 dark:to-primary-800/20 text-primary-700 dark:text-primary-300 shadow-inner border border-primary-200/50 dark:border-primary-700/30" 
                        : "text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-300 hover:bg-white/60 dark:hover:bg-white/10"
                      }
                    `}
                  >
                    {({ isActive }) => (
                      <>
                        <span className={`
                          text-2xl transition-all duration-300
                          group-hover:scale-110 group-hover:rotate-6
                          ${isActive ? "text-primary-600 dark:text-primary-400" : ""}
                        `}>
                          {item.icon}
                        </span>
                        {item.name}
                        {isActive && (
                          <motion.div 
                            layoutId="mobileActiveIndicator"
                            className="ml-auto w-2 h-2 rounded-full bg-gradient-to-r from-primary-custom-1 to-primary-custom-2" 
                          />
                        )}
                      </>
                    )}
                  </NavLink>
                </motion.div>
              ))}

              {/* User Section in Mobile - Enhanced */}
              <div className="pt-6 border-t-2 border-primary-200/30 dark:border-primary-800/30">
                {isAuthenticated ? (
                  <>
                    <div className="flex items-center gap-4 px-5 py-4 rounded-2xl bg-gradient-to-r from-primary-50/50 to-transparent dark:from-primary-900/20 mb-4 border border-primary-200/30 dark:border-primary-800/30">
                      <div className="
                        w-12 h-12 rounded-full 
                        bg-gradient-to-br from-primary-custom-1 via-primary-custom-2 to-primary-custom-3
                        flex items-center justify-center
                        shadow-lg
                      ">
                        <FaUser className="text-white text-lg" />
                      </div>
                      <div className="flex-1">
                        <p className="font-black text-gray-800 dark:text-white">
                          {user?.email?.split('@')[0] || 'User'}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400 truncate max-w-[180px]">
                          {user?.email}
                        </p>
                      </div>
                      <div className="flex items-center gap-1 px-2 py-1 rounded-full bg-gradient-to-r from-primary-500/20 to-primary-400/10">
                        <FaCrown className="text-primary-500 text-[10px]" />
                        <span className="text-[10px] font-bold text-primary-600 dark:text-primary-400">Pro</span>
                      </div>
                    </div>

                    {/* Mobile User Menu Items */}
                    <div className="space-y-2 mb-6">
                      {userMenuItems.map((item) => (
                        <NavLink
                          key={item.to}
                          to={item.to}
                          onClick={() => setMobileMenuOpen(false)}
                          className="
                            flex items-center gap-4 px-5 py-3 rounded-xl
                            text-gray-700 dark:text-gray-300 font-medium
                            hover:bg-gradient-to-r hover:from-primary-50 hover:to-primary-100/50 dark:hover:from-primary-900/30 dark:hover:to-primary-800/20
                            hover:text-primary-700 dark:hover:text-primary-300
                            transition-all duration-200 group
                          "
                        >
                          <span className="text-xl text-primary-500 group-hover:scale-110 transition-transform">
                            {item.icon}
                          </span>
                          {item.name}
                        </NavLink>
                      ))}
                    </div>

                    <motion.button
                      whileTap={{ scale: 0.98 }}
                      onClick={handleLogout}
                      className="
                        w-full flex items-center justify-center gap-3
                        px-5 py-4 rounded-2xl
                        bg-gradient-to-r from-primary-custom-1 via-primary-custom-2 to-primary-custom-3
                        text-white font-bold
                        shadow-lg shadow-primary-500/30
                        hover:shadow-xl hover:shadow-primary-500/40
                        transition-all duration-300
                        group
                      "
                    >
                      <FaSignOutAlt className="text-lg group-hover:rotate-180 transition-transform duration-300" />
                      Logout
                    </motion.button>
                  </>
                ) : (
                  <NavLink
                    to="/login"
                    onClick={() => setMobileMenuOpen(false)}
                    className="
                      flex items-center justify-center gap-3
                      px-5 py-4 rounded-2xl
                      bg-gradient-to-r from-primary-custom-1 via-primary-custom-2 to-primary-custom-3
                      text-white font-bold
                      shadow-lg shadow-primary-500/40
                      hover:shadow-xl hover:shadow-primary-500/60
                      hover:scale-105 active:scale-95
                      transition-all duration-300
                      group
                    "
                  >
                    <FaSignInAlt className="text-lg group-hover:translate-x-1 transition-transform duration-300" />
                    Sign In
                  </NavLink>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Blur effect for backdrop */}
      {mobileMenuOpen && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/20 dark:bg-black/40 backdrop-blur-sm z-40 md:hidden"
          onClick={toggleMobileMenu}
        />
      )}
    </motion.div>
  );
}