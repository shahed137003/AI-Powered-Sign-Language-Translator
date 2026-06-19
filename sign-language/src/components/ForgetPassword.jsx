import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useTheme } from "../context/ThemeContext";
import axios from 'axios';
import { FaEnvelope, FaKey, FaArrowLeft, FaUserPlus, FaShieldAlt, FaCheck } from 'react-icons/fa';
import { BsRobot, BsLock, BsClock, BsStars } from 'react-icons/bs';
import { motion, AnimatePresence } from 'framer-motion';
import { TbSparkles } from "react-icons/tb";

export default function ForgetPassword() {
  const { themeColor } = useTheme();
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationFrameRef = useRef(null);
  const [isDark, setIsDark] = useState(false);

  // Detect dark mode
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

  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setMessage('');

    try {
      await axios.post(`${API_URL}/password/forget`, { email });
      setMessage('Reset code has been sent to your email. Check your inbox!');
      setTimeout(() => {
        navigate('/reset-password', { state: { email } });
      }, 2000);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to send reset code. Please check your email and try again.');
    } finally {
      setLoading(false);
    }
  };

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

  const features = [
    {
      icon: <BsLock className="text-xl" />,
      title: "Encrypted Code",
      description: "6-digit verification code sent via secure channel",
      gradient: "from-primary-500 to-primary-700"
    },
    {
      icon: <BsClock className="text-xl" />,
      title: "15-Minute Window",
      description: "Verification code expires for added security",
      gradient: "from-pink-500 to-rose-500"
    },
    {
      icon: <FaShieldAlt className="text-xl" />,
      title: "Identity Protection",
      description: "Advanced algorithms verify legitimate requests",
      gradient: "from-blue-500 to-indigo-500"
    },
    {
      icon: <BsRobot className="text-xl" />,
      title: "AI Monitoring",
      description: "Real-time fraud detection and prevention",
      gradient: "from-violet-500 to-primary-600"
    }
  ];

  return (
    <div className="relative w-full min-h-screen bg-gradient-to-br from-gray-50 via-white to-primary-50/60 dark:from-primary-bg-1 dark:via-primary-bg-2 dark:to-primary-bg-3 overflow-hidden selection:bg-primary-500 selection:text-white transition-all duration-700">
      
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

      <div className="relative z-10 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
        
        {/* Header Section */}
        <motion.div
          initial="hidden"
          animate="visible"
          variants={fadeUp}
          className="text-center mb-16"
        >
          {/* Premium Badge */}
          <motion.div
            variants={fadeUp}
            whileHover={{ scale: 1.05, rotate: 1 }}
            className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-gradient-to-r from-primary-500/15 via-primary-400/10 to-primary-300/10 border border-primary-200/60 dark:border-primary-700/60 backdrop-blur-xl shadow-lg shadow-primary-500/10 relative overflow-hidden group mb-8"
          >
            <div className="relative">
              <span className="absolute animate-ping inline-flex h-3.5 w-3.5 rounded-full bg-primary-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-gradient-to-r from-primary-500 to-primary-400" />
            </div>
            <span className="text-sm font-bold bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 bg-clip-text text-transparent">
              ACCOUNT SECURITY
            </span>
            <TbSparkles className="text-primary-500 ml-1" />
            <div className="absolute -inset-1 bg-gradient-to-r from-primary-500/0 via-primary-400/10 to-primary-500/0 group-hover:via-primary-400/20 transition-all duration-500" />
          </motion.div>

          {/* Main Header */}
          <motion.h1
            variants={fadeUp}
            className="font-extrabold text-4xl sm:text-5xl lg:text-[53px] leading-tight mb-6"
          >
            <span className="block text-gray-900 dark:text-white">
              Reset Your
            </span>
            <span className="block bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 bg-clip-text text-transparent">
              Password
            </span>
          </motion.h1>

          <motion.p
            variants={fadeUp}
            className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto"
          >
            Secure your account with our AI-powered password recovery system
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

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12">
          
          {/* Left Column - Features & Info */}
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="space-y-6"
          >
            {/* Security Features Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {features.map((feature, i) => (
                <motion.div
                  key={i}
                  variants={scaleIn}
                  whileHover={{ 
                    scale: 1.03,
                    y: -4,
                    boxShadow: "0 20px 40px -15px rgba(139, 92, 246, 0.4)"
                  }}
                  className="group relative p-4 rounded-2xl backdrop-blur-xl border transition-all duration-300 overflow-hidden bg-white/70 dark:bg-white/5 border-primary-200/50 dark:border-primary-500/20 hover:border-primary-300/50 dark:hover:border-primary-500/50"
                >
                  <div className="flex items-start gap-3">
                    <div className={`p-2 rounded-xl bg-gradient-to-br ${feature.gradient} text-white shadow-lg group-hover:scale-110 transition-transform duration-300`}>
                      {feature.icon}
                    </div>
                    <div>
                      <h3 className="font-bold text-gray-900 dark:text-white text-sm mb-1">
                        {feature.title}
                      </h3>
                      <p className="text-gray-600 dark:text-gray-400 text-xs">
                        {feature.description}
                      </p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Security Information */}
            <motion.div
              variants={fadeUp}
              className="p-5 rounded-2xl backdrop-blur-xl bg-gradient-to-r from-blue-50/80 to-primary-50/50 dark:from-blue-900/20 dark:to-primary-900/20 border border-blue-200/50 dark:border-blue-500/20"
            >
              <div className="flex items-start gap-3">
                <div className="flex-shrink-0">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-r from-blue-500 to-blue-600 flex items-center justify-center">
                    <FaCheck className="text-white text-lg" />
                  </div>
                </div>
                <div>
                  <h4 className="font-bold text-gray-900 dark:text-white text-base mb-2">
                    What happens next?
                  </h4>
                  <ul className="space-y-1.5 text-sm text-gray-600 dark:text-gray-300">
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                      You'll receive a 6-digit verification code via email
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                      The code is valid for 15 minutes for your security
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                      Enter the code on the next screen to reset your password
                    </li>
                  </ul>
                </div>
              </div>
            </motion.div>

            {/* Trust Badges */}
            <motion.div
              variants={fadeUp}
              className="flex flex-wrap items-center gap-4"
            >
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                  <FaShieldAlt className="text-green-600 dark:text-green-400 text-sm" />
                </div>
                <div>
                  <div className="font-semibold text-gray-900 dark:text-white text-xs">Bank-Level Security</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">256-bit encryption</div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center">
                  <BsRobot className="text-primary-600 dark:text-primary-400 text-sm" />
                </div>
                <div>
                  <div className="font-semibold text-gray-900 dark:text-white text-xs">AI Protection</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Fraud detection</div>
                </div>
              </div>
            </motion.div>
          </motion.div>

          {/* Right Column - Reset Form */}
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeUp}
          >
            <div className="relative group h-full">
              {/* Form Glow */}
              <div className="absolute -inset-1 bg-gradient-to-r from-primary-500/20 via-pink-500/20 to-indigo-500/20 rounded-3xl blur opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              
              <div className="relative p-8 rounded-3xl bg-white/70 dark:bg-white/5 backdrop-blur-2xl border border-white/40 dark:border-white/10 shadow-2xl shadow-primary-100/30 dark:shadow-primary-900/30">
                {/* Form Header Icon */}
                <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 mx-auto mb-6">
                  <BsLock className="text-2xl text-white" />
                </div>

                {/* Success Message */}
                <AnimatePresence>
                  {message && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="mb-5 p-3 rounded-xl bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800"
                    >
                      <div className="flex items-center gap-2">
                        <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center">
                          <FaCheck className="text-white text-xs" />
                        </div>
                        <p className="text-sm text-green-700 dark:text-green-300">{message}</p>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Error Message */}
                <AnimatePresence>
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="mb-5 p-3 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800"
                    >
                      <div className="flex items-center gap-2">
                        <div className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center">
                          <span className="text-white text-xs font-bold">!</span>
                        </div>
                        <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                <form onSubmit={handleSubmit} className="space-y-5">
                  {/* Email Field */}
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center gap-2">
                      <FaEnvelope className="text-primary-500 text-sm" />
                      Email Address
                    </label>
                    <div className="relative">
                      <input
                        type="email"
                        name="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                        className="w-full px-4 py-3 pl-11 rounded-xl bg-white/40 dark:bg-gray-900/40 border border-gray-300/50 dark:border-gray-700/50 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-transparent transition-all duration-300"
                        placeholder="Enter your account email"
                      />
                      <FaEnvelope className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500 text-sm" />
                    </div>
                  </div>

                  {/* Security Note */}
                  <div className="p-3 rounded-xl bg-blue-50 dark:bg-blue-900/10 border border-blue-200 dark:border-blue-800/30">
                    <div className="flex items-start gap-2">
                      <BsClock className="text-blue-500 text-sm flex-shrink-0 mt-0.5" />
                      <p className="text-xs text-blue-700 dark:text-blue-300">
                        A 6-digit verification code will be sent to your email. The code expires in 15 minutes.
                      </p>
                    </div>
                  </div>

                  {/* Submit Button */}
                  <motion.button
                    type="submit"
                    disabled={loading}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="relative overflow-hidden w-full py-3.5 rounded-full bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 text-white font-bold text-base shadow-lg shadow-primary-500/30 hover:shadow-primary-500/50 transition-all duration-300 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed group"
                  >
                    <span className="relative z-10 flex items-center gap-2">
                      {loading ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                          Sending Code...
                        </>
                      ) : (
                        <>
                          <FaKey className="group-hover:scale-110 transition-transform" />
                          Send Verification Code
                        </>
                      )}
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

                  {/* Back to Login Link */}
                  <Link
                    to="/login"
                    className="w-full flex items-center justify-center gap-2 py-3 rounded-xl border border-gray-300/50 dark:border-gray-700/50 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-white/5 hover:border-primary-300 dark:hover:border-primary-500/50 transition-all duration-300 group text-sm font-medium"
                  >
                    <FaArrowLeft className="group-hover:-translate-x-1 transition-transform text-sm" />
                    Back to Sign In
                  </Link>

                  {/* Divider */}
                  <div className="relative">
                    <div className="absolute inset-0 flex items-center">
                      <div className="w-full border-t border-gray-300/50 dark:border-gray-700/50" />
                    </div>
                    <div className="relative flex justify-center text-xs">
                      <span className="px-3 bg-white dark:bg-transparent text-gray-500 dark:text-gray-400">
                        Don't have an account?
                      </span>
                    </div>
                  </div>

                  {/* Register Link */}
                  <Link
                    to="/register"
                    className="w-full flex items-center justify-center gap-2 py-3 rounded-xl border-2 border-primary-600/50 dark:border-primary-500/50 text-primary-600 dark:text-primary-400 hover:bg-primary-50/50 dark:hover:bg-primary-900/20 hover:border-primary-700/50 dark:hover:border-primary-400/50 transition-all duration-300 group text-sm font-medium"
                  >
                    <FaUserPlus className="group-hover:scale-110 transition-transform text-sm" />
                    Create New Account
                  </Link>
                </form>

                {/* Security Footer */}
                <div className="mt-6 pt-5 border-t border-gray-200/50 dark:border-gray-800/50">
                  <div className="flex items-center justify-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                    <FaShieldAlt className="text-green-500 text-sm" />
                    <span>Your email is secured and never shared with third parties</span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Bottom Security Note */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.5 }}
          className="mt-12 text-center"
        >
          <div className="inline-flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
            <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            <span>All password recovery requests are monitored by our AI security system</span>
          </div>
        </motion.div>
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
