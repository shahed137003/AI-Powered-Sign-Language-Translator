import React, { useState, useEffect, useRef } from "react";
import { useTheme } from "../context/ThemeContext";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Save, 
  CheckCircle, 
  User, 
  Lock, 
  Mail, 
  Globe, 
  Palette, 
  Bell, 
  Shield,
  Eye,
  EyeOff,
  Camera,
  BarChart3,
  Activity,
  Award,
  Settings,
  LogOut,
  Zap,
  Sparkles
} from 'lucide-react';
import { TbSparkles, TbHandLoveYou } from "react-icons/tb";
import { BsArrowRight } from "react-icons/bs";

const PROFILE_PLACEHOLDER_URL = "https://placehold.co/300x300/A044FF/ffffff?text=User";

export default function Profile() {
  const { themeColor } = useTheme();
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationFrameRef = useRef(null);
  const [isDark, setIsDark] = useState(false);
  const [showToast, setShowToast] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  
  // Theme and Particle Effect
  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains("dark"));
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    setIsDark(document.documentElement.classList.contains("dark"));
    return () => observer.disconnect();
  }, []);

  // Particle system matching Home page
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

    particlesRef.current = Array.from({ length: 100 }).map(() => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      size: Math.random() * 3 + 1,
      speedX: Math.random() * 0.5 - 0.25,
      speedY: Math.random() * 0.5 - 0.25,
      color: colors[Math.floor(Math.random() * colors.length)],
      opacity: Math.random() * 0.5 + 0.1,
      glow: Math.random() > 0.8,
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

        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        
        if (particle.glow) {
          const glowGradient = ctx.createRadialGradient(
            particle.x, particle.y, 0,
            particle.x, particle.y, particle.size * 3
          );
          glowGradient.addColorStop(0, particle.color + '88');
          glowGradient.addColorStop(1, particle.color + '00');
          ctx.fillStyle = glowGradient;
        } else {
          ctx.fillStyle = particle.color + Math.floor(particle.opacity * 255).toString(16).padStart(2, '0');
        }
        
        ctx.fill();

        particlesRef.current.forEach(otherParticle => {
          const dx = particle.x - otherParticle.x;
          const dy = particle.y - otherParticle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 80) {
            ctx.beginPath();
            ctx.strokeStyle = particle.color + '33';
            ctx.lineWidth = 0.5 * (1 - distance / 80);
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(otherParticle.x, otherParticle.y);
            ctx.stroke();
          }
        });
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
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      window.removeEventListener('resize', handleResize);
    };
  }, [isDark, themeColor]);

  const [user, setUser] = useState({
    name: "Shahd Mohamed",
    email: "shahd@linguasign.com",
    password: "",
    preferredLanguage: "ASL",
    notifications: true,
    twoFactor: false,
    autoSave: true
  });

  const [stats] = useState({
    translations: "1.2K",
    accuracy: "98.7%",
    streak: "42",
    level: "Expert"
  });

  // Animation variants matching Home page
  const fadeUp = {
    hidden: { opacity: 0, y: 40 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.8,
        ease: [0.22, 1, 0.36, 1]
      }
    }
  };

  const scaleIn = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: { 
      opacity: 1, 
      scale: 1,
      transition: {
        duration: 0.6,
        ease: "backOut"
      }
    }
  };

  const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setUser((prevUser) => ({ 
      ...prevUser, 
      [name]: type === 'checkbox' ? checked : value 
    }));
  };

  const handleSave = () => {
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000);
  };

  return (
    <div className="relative w-full min-h-screen bg-gradient-to-br from-gray-50 via-white to-primary-50/60 dark:from-[#0a0518] dark:via-[#110a2e] dark:to-[#1e0f5c] overflow-hidden selection:bg-primary-500 selection:text-white transition-all duration-700">
      
      {/* Premium Canvas Particles */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
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
        className="absolute top-0 left-0 w-[600px] h-[600px] bg-primary-600/20 rounded-full blur-[120px]"
        animate={{
          x: [0, 200, 0],
          y: [0, -200, 0],
        }}
        transition={{
          duration: 25,
          repeat: Infinity,
          ease: "linear",
        }}
      />
      <motion.div
        className="absolute bottom-0 right-0 w-[600px] h-[600px] bg-primary-400/20 rounded-full blur-[120px]"
        animate={{
          x: [0, -200, 0],
          y: [0, 200, 0],
        }}
        transition={{
          duration: 25,
          repeat: Infinity,
          ease: "linear",
        }}
      />

      {/* Save Toast Notification */}
      <AnimatePresence>
        {showToast && (
          <motion.div
            initial={{ x: "100%", opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: "100%", opacity: 0 }}
            transition={{ duration: 0.5, type: "spring", stiffness: 100 }}
            className="fixed top-24 right-6 p-4 rounded-xl shadow-2xl bg-gradient-to-r from-green-500/90 to-emerald-500/90 backdrop-blur-md text-white font-semibold flex items-center gap-2 z-50"
          >
            <CheckCircle size={20} />
            <span>Changes saved successfully!</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          className="text-center mb-16"
        >
          {/* Premium Badge */}
          <motion.div
            whileHover={{ scale: 1.05, rotate: 1 }}
            className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-gradient-to-r from-primary-500/15 via-primary-400/10 to-primary-300/10 border border-primary-200/60 dark:border-primary-700/60 backdrop-blur-xl shadow-lg shadow-primary-500/10 relative overflow-hidden group mb-8"
          >
            <div className="relative">
              <span className="absolute animate-ping inline-flex h-3.5 w-3.5 rounded-full bg-primary-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-gradient-to-r from-primary-500 to-primary-400" />
            </div>
            <span className="text-sm font-bold bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 bg-clip-text text-transparent">
              Personal Dashboard
            </span>
            <TbSparkles className="text-primary-500 ml-1" />
            <div className="absolute -inset-1 bg-gradient-to-r from-primary-500/0 via-primary-400/10 to-primary-500/0 group-hover:via-primary-400/20 transition-all duration-500" />
          </motion.div>

          <motion.h1
            variants={fadeUp}
            className="font-extrabold text-4xl sm:text-5xl lg:text-[53px] leading-tight mb-6"
          >
            <span className="block text-gray-900 dark:text-white">
              Welcome Back,
            </span>
            <span className="block bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 bg-clip-text text-transparent">
              {user.name}
            </span>
          </motion.h1>
          
          <motion.p
            variants={fadeUp}
            className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto"
          >
            Manage your personal details, preferences, and track your AI translation journey all in one place.
          </motion.p>

          {/* Decorative Elements */}
          <motion.div
            variants={fadeUp}
            className="flex items-center justify-center gap-8 mt-10"
          >
            <div className="w-12 h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent rounded-full" />
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="w-6 h-6 rounded-full border-2 border-primary-400/50"
            />
            <div className="w-12 h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent rounded-full" />
          </motion.div>
        </motion.div>


  

        {/* Main Content */}
        <div className="flex flex-col lg:flex-row gap-8">
          {/* Left Column - Profile Card */}
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeUp}
            className="lg:w-2/5"
          >
            <div className="relative group h-full">
              <div className="relative h-full p-8 rounded-2xl bg-white/70 dark:bg-white/5 backdrop-blur-xl border border-primary-200/50 dark:border-primary-500/20 shadow-xl hover:shadow-2xl transition-all duration-300 text-center">
                {/* Profile Image */}
                <div className="relative mb-6">
                  <div className="w-32 h-32 mx-auto rounded-full overflow-hidden border-4 border-white dark:border-gray-800 shadow-xl relative group/image">
                    <img
                      src={PROFILE_PLACEHOLDER_URL}
                      alt="Profile"
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-primary-600/50 to-transparent opacity-0 group-hover/image:opacity-100 transition-opacity duration-500" />
                    <button className="absolute bottom-2 right-2 p-2 rounded-full bg-white dark:bg-gray-800 shadow-lg opacity-0 group-hover/image:opacity-100 transition-opacity duration-300">
                      <Camera size={16} className="text-primary-600" />
                    </button>
                  </div>
                  <div className="absolute -top-2 -right-2">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-r from-primary-500 to-pink-500 flex items-center justify-center shadow-lg">
                      <TbSparkles className="text-white" size={14} />
                    </div>
                  </div>
                </div>
                
                <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">{user.name}</h2>
                <p className="text-primary-600 dark:text-primary-400 font-medium text-sm mb-4 flex items-center justify-center gap-2">
                  <Mail size={14} />
                  {user.email}
                </p>
                
                <div className="flex flex-wrap gap-2 justify-center mb-6">
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-gradient-to-r from-primary-500/10 to-pink-500/10 text-primary-700 dark:text-primary-300 border border-primary-300/30">
                    Premium
                  </span>
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-gradient-to-r from-blue-500/10 to-cyan-500/10 text-blue-700 dark:text-blue-300 border border-blue-300/30">
                    AI Translator
                  </span>
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-gradient-to-r from-green-500/10 to-emerald-500/10 text-green-700 dark:text-green-300 border border-green-300/30">
                    Verified
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-4 mt-8 mb-8 text-left">
                  <div className="bg-gradient-to-br from-white/40 to-white/10 dark:from-gray-800/40 dark:to-gray-800/10 p-4 rounded-xl border border-primary-200/30 dark:border-primary-700/30 shadow-sm backdrop-blur-md hover:scale-105 transition-transform duration-300">
                    <div className="text-gray-500 dark:text-gray-400 text-xs font-semibold mb-1 flex items-center gap-1.5"><BarChart3 size={14} className="text-primary-500"/> Translations</div>
                    <div className="text-2xl font-black text-gray-900 dark:text-white">{stats.translations}</div>
                  </div>
                  <div className="bg-gradient-to-br from-white/40 to-white/10 dark:from-gray-800/40 dark:to-gray-800/10 p-4 rounded-xl border border-pink-200/30 dark:border-pink-700/30 shadow-sm backdrop-blur-md hover:scale-105 transition-transform duration-300">
                    <div className="text-gray-500 dark:text-gray-400 text-xs font-semibold mb-1 flex items-center gap-1.5"><Activity size={14} className="text-pink-500"/> Accuracy</div>
                    <div className="text-2xl font-black text-gray-900 dark:text-white">{stats.accuracy}</div>
                  </div>
                  <div className="bg-gradient-to-br from-white/40 to-white/10 dark:from-gray-800/40 dark:to-gray-800/10 p-4 rounded-xl border border-orange-200/30 dark:border-orange-700/30 shadow-sm backdrop-blur-md hover:scale-105 transition-transform duration-300">
                    <div className="text-gray-500 dark:text-gray-400 text-xs font-semibold mb-1 flex items-center gap-1.5"><Zap size={14} className="text-orange-500"/> Streak</div>
                    <div className="text-2xl font-black text-gray-900 dark:text-white">{stats.streak} <span className="text-sm font-medium text-gray-500">Days</span></div>
                  </div>
                  <div className="bg-gradient-to-br from-white/40 to-white/10 dark:from-gray-800/40 dark:to-gray-800/10 p-4 rounded-xl border border-emerald-200/30 dark:border-emerald-700/30 shadow-sm backdrop-blur-md hover:scale-105 transition-transform duration-300">
                    <div className="text-gray-500 dark:text-gray-400 text-xs font-semibold mb-1 flex items-center gap-1.5"><Award size={14} className="text-emerald-500"/> Level</div>
                    <div className="text-2xl font-black text-gray-900 dark:text-white">{stats.level}</div>
                  </div>
                </div>

                <div className="space-y-3 pt-4 border-t border-primary-200/50 dark:border-primary-500/20 mt-6">
                  <button className="w-full py-2.5 rounded-xl bg-red-500/10 border border-red-200/50 text-red-600 dark:text-red-400 font-medium hover:bg-red-500/20 transition-all duration-300 flex items-center justify-center gap-2">
                    <LogOut size={16} />
                    Sign Out
                  </button>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Right Column - Forms */}
          <div className="lg:w-3/5 space-y-6">
            {/* Personal Information */}
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={fadeUp}
              className="relative group"
            >
              <div className="relative p-6 rounded-2xl bg-white/70 dark:bg-white/5 backdrop-blur-xl border border-primary-200/50 dark:border-primary-500/20 shadow-xl hover:shadow-2xl transition-all duration-300">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                  <User className="text-primary-600" size={20} />
                  Personal Information
                </h3>
                
                <div className="grid md:grid-cols-2 gap-5">
                  {/* Name Field */}
                  <div className="space-y-2">
                    <label className="font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-2 text-sm">
                      <User size={14} className="text-primary-600" />
                      Full Name
                    </label>
                    <input
                      type="text"
                      name="name"
                      value={user.name}
                      onChange={handleInputChange}
                      className="w-full p-3 rounded-xl bg-white/50 dark:bg-gray-900/50 border border-gray-300/50 dark:border-gray-700/50 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500/50 hover:bg-white/70 dark:hover:bg-gray-800/60 transition-all duration-300 shadow-sm"
                    />
                  </div>

                  {/* Email Field */}
                  <div className="space-y-2">
                    <label className="font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-2 text-sm">
                      <Mail size={14} className="text-primary-600" />
                      Email Address
                    </label>
                    <input
                      type="email"
                      name="email"
                      value={user.email}
                      onChange={handleInputChange}
                      className="w-full p-3 rounded-xl bg-white/50 dark:bg-gray-900/50 border border-gray-300/50 dark:border-gray-700/50 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500/50 hover:bg-white/70 dark:hover:bg-gray-800/60 transition-all duration-300 shadow-sm"
                    />
                  </div>

                  {/* Password Field */}
                  <div className="space-y-2">
                    <label className="font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-2 text-sm">
                      <Lock size={14} className="text-primary-600" />
                      New Password
                    </label>
                    <div className="relative">
                      <input
                        type={showPassword ? "text" : "password"}
                        name="password"
                        value={user.password}
                        onChange={handleInputChange}
                        placeholder="••••••••"
                        className="w-full p-3 rounded-xl bg-white/50 dark:bg-gray-900/50 border border-gray-300/50 dark:border-gray-700/50 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500/50 hover:bg-white/70 dark:hover:bg-gray-800/60 transition-all duration-300 shadow-sm pr-10"
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-primary-600 transition-colors"
                      >
                        {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                      </button>
                    </div>
                  </div>

                  {/* Language Field */}
                  <div className="space-y-2">
                    <label className="font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-2 text-sm">
                      <Globe size={14} className="text-primary-600" />
                      Preferred Language
                    </label>
                    <select
                      name="preferredLanguage"
                      value={user.preferredLanguage}
                      onChange={handleInputChange}
                      className="w-full p-3 rounded-xl bg-white/50 dark:bg-gray-900/50 border border-gray-300/50 dark:border-gray-700/50 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500/50 hover:bg-white/70 dark:hover:bg-gray-800/60 transition-all duration-300 shadow-sm"
                    >
                      <option value="ASL">American Sign Language (ASL)</option>
                      <option value="BSL">British Sign Language (BSL)</option>
                      <option value="LSF">French Sign Language (LSF)</option>
                      <option value="DGS">German Sign Language (DGS)</option>
                    </select>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Preferences */}
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={fadeUp}
              className="relative group"
            >
              <div className="relative p-6 rounded-2xl bg-white/70 dark:bg-white/5 backdrop-blur-xl border border-primary-200/50 dark:border-primary-500/20 shadow-xl hover:shadow-2xl transition-all duration-300">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                  <Settings className="text-primary-600" size={20} />
                  Account Preferences
                </h3>
                
                <div className="flex flex-col gap-5">
                  {/* Settings Switches */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between py-2">
                      <span className="text-gray-700 dark:text-gray-300 font-medium text-sm flex items-center gap-2">
                        <Bell size={14} className="text-primary-600" />
                        Push Notifications
                      </span>
                      <button
                        onClick={() => setUser({...user, notifications: !user.notifications})}
                        className={`relative inline-flex h-5 w-10 items-center rounded-full transition-all duration-300 ${
                          user.notifications ? 'bg-primary-600' : 'bg-gray-300 dark:bg-gray-700'
                        }`}
                      >
                        <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-all duration-300 ${
                          user.notifications ? 'translate-x-5' : 'translate-x-0.5'
                        }`} />
                      </button>
                    </div>

                    <div className="flex items-center justify-between py-2">
                      <span className="text-gray-700 dark:text-gray-300 font-medium text-sm flex items-center gap-2">
                        <Shield size={14} className="text-primary-600" />
                        Two-Factor Authentication
                      </span>
                      <button
                        onClick={() => setUser({...user, twoFactor: !user.twoFactor})}
                        className={`relative inline-flex h-5 w-10 items-center rounded-full transition-all duration-300 ${
                          user.twoFactor ? 'bg-primary-600' : 'bg-gray-300 dark:bg-gray-700'
                        }`}
                      >
                        <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-all duration-300 ${
                          user.twoFactor ? 'translate-x-5' : 'translate-x-0.5'
                        }`} />
                      </button>
                    </div>

                    <div className="flex items-center justify-between py-2">
                      <span className="text-gray-700 dark:text-gray-300 font-medium text-sm flex items-center gap-2">
                        <Save size={14} className="text-primary-600" />
                        Auto Save Progress
                      </span>
                      <button
                        onClick={() => setUser({...user, autoSave: !user.autoSave})}
                        className={`relative inline-flex h-5 w-10 items-center rounded-full transition-all duration-300 ${
                          user.autoSave ? 'bg-primary-600' : 'bg-gray-300 dark:bg-gray-700'
                        }`}
                      >
                        <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-all duration-300 ${
                          user.autoSave ? 'translate-x-5' : 'translate-x-0.5'
                        }`} />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Save Button */}
            <motion.button
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={fadeUp}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleSave}
              className="relative overflow-hidden w-full py-4 rounded-full bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 text-white font-bold text-lg shadow-lg shadow-primary-500/30 hover:shadow-primary-500/50 transition-all duration-300 flex items-center justify-center gap-3 group"
            >
              <span className="relative z-10 flex items-center gap-2">
                <Save size={20} />
                Save All Changes
                <Sparkles size={16} className="group-hover:rotate-12 transition-transform" />
              </span>
              
              {/* Animated shine */}
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                animate={{
                  x: ["-100%", "200%"],
                }}
                transition={{
                  duration: 1.5,
                  repeat: Infinity,
                  ease: "linear",
                }}
              />
            </motion.button>
          </div>
        </div>
      </div>

      {/* Scroll Indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2 }}
        className="absolute bottom-10 left-1/2 transform -translate-x-1/2"
      >
        <motion.div
          animate={{ 
            y: [0, 10, 0],
            opacity: [0.5, 1, 0.5]
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          className="text-center"
        >
        </motion.div>
      </motion.div>
    </div>
  );
}
