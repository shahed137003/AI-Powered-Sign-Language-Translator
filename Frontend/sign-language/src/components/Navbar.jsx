import React, { useEffect, useState } from "react";
import { FaHandBackFist, FaBars, FaXmark } from "react-icons/fa6";
import { FaSun, FaMoon, FaSignInAlt, FaSignOutAlt, FaUser, FaCaretDown } from "react-icons/fa";
import { BsTranslate, BsFillChatDotsFill } from "react-icons/bs";
import { TbHome, TbMail, TbUser } from "react-icons/tb";
import { NavLink, useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../context/useAuth";
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

  const { user, logout, isAuthenticated } = useAuth();
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
  const toggleUserMenu = () => setUserMenuOpen(!userMenuOpen);

  // Logout function
  const handleLogout = () => {
    logout();
    navigate('/');
    setUserMenuOpen(false);
  };

  const navItems = [
    { name: "Home", to: "/", icon: <TbHome /> },
    { name: "Translate", to: "/translate", icon: <BsTranslate /> },
    { name: "Profile", to: "/profile", icon: <TbUser /> },
    { name: "Contact", to: "/contactus", icon: <TbMail /> },
  ];

  const userMenuItems = [
    { name: "Dashboard", to: "/dashboard", icon: <TbHome /> },
    { name: "Settings", to: "/settings", icon: <FaUser /> },
    { name: "Help", to: "/help", icon: <BsFillChatDotsFill /> },
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
    hidden: { 
      opacity: 0,
      height: 0,
      y: -20
    },
    visible: { 
      opacity: 1,
      height: "auto",
      y: 0,
      transition: {
        duration: 0.3,
        ease: "easeInOut"
      }
    },
    exit: { 
      opacity: 0,
      height: 0,
      y: -20,
      transition: {
        duration: 0.2,
        ease: "easeIn"
      }
    }
  };

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={containerVariants}
      className={`
        fixed top-0 left-0 w-full px-4 sm:px-6 lg:px-16 py-3 flex items-center justify-between
        bg-gradient-to-r from-white/90 via-white/95 to-white/90 
        dark:from-[#0f0c29]/95 dark:via-[#0f0c29]/95 dark:to-[#0f0c29]/95
        backdrop-blur-xl
        border-b border-gray-200/60 dark:border-purple-900/50
        ${scrolled ? "shadow-2xl shadow-purple-500/10 dark:shadow-purple-900/30" : "shadow-lg shadow-purple-500/5 dark:shadow-purple-900/20"}
        transition-all duration-500 z-50
      `}
    >
      {/* Logo with animation */}
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
              animate={{ rotate: [0, 10, 0] }}
              transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
              className="relative"
            >
              <FaHandBackFist className="text-3xl sm:text-4xl text-[#BF5AE0] " />
              <div className="absolute inset-0 bg-gradient-to-r from-[#A044FF] to-[#BF5AE0] blur-md opacity-60 dark:opacity-40 -z-10" />
            </motion.div>
          </div>
          <span className="
            text-2xl sm:text-3xl font-bold italic
            bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] 
           
            bg-clip-text text-transparent 
            drop-shadow-sm
            group-hover:drop-shadow-lg
            transition-all duration-300
          ">
            LinguaSign
          </span>
        </NavLink>
      </motion.div>

      {/* Desktop Navigation */}
      <div className="hidden md:flex items-center gap-1 lg:gap-2">
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
                relative px-4 py-2 mx-1 rounded-xl flex items-center gap-2
                text-gray-700 dark:text-gray-200 font-medium
                transition-all duration-300 group
                ${isActive 
                  ? "text-purple-700 dark:text-purple-300 bg-gradient-to-r from-purple-50 to-purple-100/50 dark:from-purple-900/30 dark:to-purple-800/20 shadow-inner" 
                  : "hover:text-purple-600 dark:hover:text-purple-300 hover:bg-white/50 dark:hover:bg-white/5"
                }
              `}
            >
              <span className="text-lg opacity-80">{item.icon}</span>
              {item.name}
              
              {/* Active indicator */}
              {({ isActive }) => isActive && (
                <motion.div
                  layoutId="activeIndicator"
                  className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-8 h-1 rounded-full bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]"
                  initial={{ width: 0 }}
                  animate={{ width: 32 }}
                  transition={{ duration: 0.3 }}
                />
              )}
              
              {/* Hover glow effect */}
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500/0 via-purple-400/0 to-purple-500/0 rounded-xl group-hover:via-purple-400/5 group-hover:opacity-100 opacity-0 transition-all duration-300" />
            </NavLink>
          </motion.div>
        ))}
      </div>

      {/* Right side controls */}
      <div className="flex items-center gap-3 lg:gap-4">
        {/* User Profile Section */}
        {isAuthenticated ? (
          <div className="relative">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={toggleUserMenu}
              className="
                hidden sm:flex items-center gap-3 px-4 py-2
                bg-gradient-to-r from-white/80 to-white/60 
                dark:from-gray-800/80 dark:to-gray-900/60
                backdrop-blur-xl
                border border-gray-200/60 dark:border-purple-900/50
                rounded-2xl
                shadow-lg shadow-purple-500/10 dark:shadow-purple-900/20
                hover:shadow-xl hover:shadow-purple-500/20 dark:hover:shadow-purple-900/30
                transition-all duration-300 group
              "
            >
              <div className="relative">
                <div className="
                  w-10 h-10 rounded-full 
                  bg-gradient-to-br from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
                  flex items-center justify-center
                  shadow-inner
                ">
                  <FaUser className="text-white text-sm" />
                </div>
                <div className="
                  absolute -inset-1 rounded-full 
                  bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] 
                  opacity-0 group-hover:opacity-20
                  blur-md transition-opacity duration-300
                " />
              </div>
              
              <div className="text-left">
                <p className="text-sm font-semibold text-gray-800 dark:text-white">
                  {user?.email?.split('@')[0] || 'User'}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400 truncate max-w-[120px]">
                  {user?.email}
                </p>
              </div>
              
              <FaCaretDown className={`
                text-gray-400 dark:text-gray-500 
                transition-transform duration-300
                ${userMenuOpen ? "rotate-180" : ""}
              `} />
            </motion.button>

            {/* User Dropdown Menu */}
            <AnimatePresence>
              {userMenuOpen && (
                <motion.div
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.95 }}
                  transition={{ duration: 0.2 }}
                  className="
                    absolute right-0 mt-2 w-64
                    bg-white/95 dark:bg-gray-900/95
                    backdrop-blur-xl
                    border border-gray-200/60 dark:border-purple-900/50
                    rounded-2xl
                    shadow-2xl shadow-purple-500/20 dark:shadow-purple-900/40
                    overflow-hidden
                  "
                >
                  {/* User info header */}
                  <div className="p-4 border-b border-gray-100 dark:border-gray-800">
                    <div className="flex items-center gap-3">
                      <div className="
                        w-12 h-12 rounded-full 
                        bg-gradient-to-br from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
                        flex items-center justify-center
                      ">
                        <FaUser className="text-white text-lg" />
                      </div>
                      <div>
                        <p className="font-semibold text-gray-800 dark:text-white">
                          {user?.email?.split('@')[0] || 'User'}
                        </p>
                        <p className="text-sm text-gray-500 dark:text-gray-400 truncate">
                          {user?.email}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Menu items */}
                  <div className="p-2">
                    {userMenuItems.map((item) => (
                      <NavLink
                        key={item.to}
                        to={item.to}
                        className="
                          flex items-center gap-4 px-4 py-3 rounded-xl
                          text-gray-700 dark:text-gray-300
                          hover:bg-purple-50 dark:hover:bg-purple-900/30
                          hover:text-purple-700 dark:hover:text-purple-300
                          transition-all duration-200
                        "
                      >
                        <span className="text-lg">{item.icon}</span>
                        {item.name}
                      </NavLink>
                    ))}
                  </div>

                  {/* Logout button */}
                  <div className="p-4 border-t border-gray-100 dark:border-gray-800">
                    <button
                      onClick={handleLogout}
                      className="
                        w-full flex items-center justify-center gap-2
                        px-4 py-3 rounded-xl
                        bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
                        text-white font-medium
                        shadow-lg shadow-purple-500/30
                        hover:shadow-xl hover:shadow-purple-500/40
                        active:scale-[0.98]
                        transition-all duration-300
                        group
                      "
                    >
                      <FaSignOutAlt className="group-hover:rotate-180 transition-transform duration-300" />
                      Logout
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ) : (
          <NavLink
            to="/login"
            className="
              hidden sm:flex items-center gap-2 px-5 py-2.5
              bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
              text-white font-medium rounded-4xl
              shadow-lg shadow-purple-500/30
              hover:shadow-xl hover:shadow-purple-500/40
              hover:scale-[1.02] active:scale-[0.98]
              transition-all duration-300
              group relative overflow-hidden
            "
          >
            <span className="
              absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent 
              -translate-x-full group-hover:translate-x-full 
              transition-transform duration-1000
            " />
            <FaSignInAlt className="group-hover:translate-x-1 transition-transform duration-300" />
            Login
          </NavLink>
        )}

        {/* Dark Mode Toggle - Enhanced */}
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={toggleDarkMode}
          aria-label="Toggle Dark Mode"
          className="
            relative w-16 h-8 flex items-center
            bg-gradient-to-r from-gray-200 via-gray-300 to-gray-200
            dark:from-gray-800 dark:via-gray-900 dark:to-gray-800
            rounded-full p-1
            shadow-inner shadow-gray-400/50 dark:shadow-gray-900
            border border-gray-300/50 dark:border-gray-700/50
            transition-all duration-500
            group
          "
        >
          <motion.div
            layout
            transition={{ type: "spring", stiffness: 500, damping: 30 }}
            className={`
              absolute w-6 h-6 rounded-full flex items-center justify-center
              shadow-lg
              ${darkMode 
                ? "translate-x-8 bg-gradient-to-br from-yellow-300 to-orange-400" 
                : "translate-x-0 bg-gradient-to-br from-white to-gray-100"
              }
            `}
          >
            {darkMode ? (
              <FaSun className="text-yellow-600 text-xs" />
            ) : (
              <FaMoon className="text-purple-600 text-xs" />
            )}
          </motion.div>
          
          {/* Background gradients */}
          <div className={`
            absolute inset-0 rounded-full transition-opacity duration-500
            ${darkMode ? "opacity-100" : "opacity-0"}
          `}>
            <div className="absolute inset-0 bg-gradient-to-r from-yellow-500/10 to-orange-500/10 rounded-full" />
          </div>
          <div className={`
            absolute inset-0 rounded-full transition-opacity duration-500
            ${darkMode ? "opacity-0" : "opacity-100"}
          `}>
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 to-blue-500/10 rounded-full" />
          </div>
        </motion.button>

        {/* Mobile Menu Button */}
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={toggleMobileMenu}
          className="
            md:hidden w-10 h-10 flex items-center justify-center
            bg-gradient-to-r from-white/80 to-white/60
            dark:from-gray-800/80 dark:to-gray-900/60
            backdrop-blur-xl
            border border-gray-200/60 dark:border-purple-900/50
            rounded-xl
            shadow-lg shadow-purple-500/10 dark:shadow-purple-900/20
            hover:shadow-xl hover:shadow-purple-500/20 dark:hover:shadow-purple-900/30
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

      {/* Mobile Menu */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            variants={mobileMenuVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            className="
              absolute top-full left-0 w-full md:hidden
              bg-gradient-to-b from-white/98 via-white/95 to-white/92
              dark:from-[#0f0c29]/98 dark:via-[#0f0c29]/96 dark:to-[#0f0c29]/94
              backdrop-blur-2xl
              border-b border-gray-200/60 dark:border-purple-900/50
              shadow-2xl shadow-purple-500/20 dark:shadow-purple-900/40
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
                      flex items-center gap-4 px-4 py-4 rounded-2xl
                      text-lg font-medium
                      transition-all duration-300 group
                      ${isActive 
                        ? "bg-gradient-to-r from-purple-50 to-purple-100/50 dark:from-purple-900/30 dark:to-purple-800/20 text-purple-700 dark:text-purple-300 shadow-inner" 
                        : "text-gray-700 dark:text-gray-300 hover:text-purple-600 dark:hover:text-purple-300 hover:bg-white/50 dark:hover:bg-white/5"
                      }
                    `}
                  >
                    <span className={`
                      text-xl transition-transform duration-300
                      group-hover:scale-110
                      ${isActive ? "text-purple-600 dark:text-purple-400" : ""}
                    `}>
                      {item.icon}
                    </span>
                    {item.name}
                    {({ isActive }) => isActive && (
                      <div className="ml-auto w-2 h-2 rounded-full bg-gradient-to-r from-[#6A3093] to-[#A044FF]" />
                    )}
                  </NavLink>
                </motion.div>
              ))}

              {/* User Section in Mobile */}
              <div className="pt-4 border-t border-gray-200/50 dark:border-gray-800/50">
                {isAuthenticated ? (
                  <>
                    <div className="flex items-center gap-4 px-4 py-3 rounded-2xl bg-white/50 dark:bg-gray-800/30 mb-4">
                      <div className="
                        w-12 h-12 rounded-full 
                        bg-gradient-to-br from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
                        flex items-center justify-center
                      ">
                        <FaUser className="text-white text-lg" />
                      </div>
                      <div className="flex-1">
                        <p className="font-semibold text-gray-800 dark:text-white">
                          {user?.email?.split('@')[0] || 'User'}
                        </p>
                        <p className="text-sm text-gray-500 dark:text-gray-400 truncate">
                          {user?.email}
                        </p>
                      </div>
                    </div>

                    {/* Mobile User Menu Items */}
                    <div className="space-y-2 mb-4">
                      {userMenuItems.map((item) => (
                        <NavLink
                          key={item.to}
                          to={item.to}
                          className="
                            flex items-center gap-4 px-4 py-3 rounded-xl
                            text-gray-700 dark:text-gray-300
                            hover:bg-purple-50 dark:hover:bg-purple-900/30
                            hover:text-purple-700 dark:hover:text-purple-300
                            transition-all duration-200
                          "
                        >
                          <span className="text-lg">{item.icon}</span>
                          {item.name}
                        </NavLink>
                      ))}
                    </div>

                    <button
                      onClick={handleLogout}
                      className="
                        w-full flex items-center justify-center gap-3
                        px-4 py-4 rounded-2xl
                        bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
                        text-white font-semibold
                        shadow-lg shadow-purple-500/30
                        hover:shadow-xl hover:shadow-purple-500/40
                        active:scale-[0.98]
                        transition-all duration-300
                        group
                      "
                    >
                      <FaSignOutAlt className="text-lg group-hover:rotate-180 transition-transform duration-300" />
                      Logout
                    </button>
                  </>
                ) : (
                  <NavLink
                    to="/login"
                    className="
                      flex items-center justify-center gap-3
                      px-4 py-4 rounded-2xl
                      bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
                      text-white font-semibold
                      shadow-lg shadow-purple-500/30
                      hover:shadow-xl hover:shadow-purple-500/40
                      active:scale-[0.98]
                      transition-all duration-300
                      group
                    "
                  >
                    <FaSignInAlt className="text-lg group-hover:translate-x-1 transition-transform duration-300" />
                    Login to Your Account
                  </NavLink>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Blur effect for backdrop */}
      {mobileMenuOpen && (
        <div 
          className="fixed inset-0 bg-black/20 dark:bg-black/40 backdrop-blur-sm z-40 md:hidden"
          onClick={toggleMobileMenu}
        />
      )}
    </motion.div>
  );
}