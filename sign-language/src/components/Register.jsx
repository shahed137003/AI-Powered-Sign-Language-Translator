import React, { useState, useEffect, useRef, useContext, useCallback, useMemo } from "react";
import { useTheme } from "../context/ThemeContext";
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from "../context/useAuth";
import { FaUser, FaEnvelope, FaLock, FaUserPlus, FaSignInAlt, FaEye, FaEyeSlash, FaCheck, FaShieldAlt, FaCrown } from 'react-icons/fa';
import { BsRobot, BsStars, BsLightningFill } from 'react-icons/bs';
import { GiArtificialIntelligence } from 'react-icons/gi';
import { TbSparkles } from 'react-icons/tb';
import { motion } from 'framer-motion';

export default function Register() {
  const { themeColor } = useTheme();
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [success, setSuccess] = useState(false);
  
  const { register } = useAuth();
  const navigate = useNavigate();
  
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationFrameRef = useRef(null);
  
  // Detect dark mode
  const [isDark, setIsDark] = useState(
    document.documentElement.classList.contains("dark")
  );

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

  // Particle system - matching home page
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

    particlesRef.current = Array.from({ length: 120 }).map(() => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      size: Math.random() * 4 + 1,
      speedX: Math.random() * 0.5 - 0.25,
      speedY: Math.random() * 0.5 - 0.25,
      color: colors[Math.floor(Math.random() * colors.length)],
      opacity: Math.random() * 0.6 + 0.2,
      glow: Math.random() > 0.7,
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
            particle.x, particle.y, particle.size * 4
          );
          glowGradient.addColorStop(0, particle.color + '99');
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

          if (distance < 100) {
            ctx.beginPath();
            ctx.strokeStyle = particle.color + '44';
            ctx.lineWidth = 0.6 * (1 - distance / 100);
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
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      window.removeEventListener('resize', handleResize);
    };
  }, [isDark, themeColor]);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    if (!formData.username.trim()) {
      setError('Username is required');
      return;
    }
    
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    
    if (formData.password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    setLoading(true);

    try {
      const result = await register({
        username: formData.username.trim(),
        email: formData.email,
        password: formData.password
      });
      
      if (result.success) {
        setSuccess(true);
        setTimeout(() => {
          navigate('/login');
        }, 2000);
      } else {
        setError(result.error || 'Registration failed. Please try again.');
      }
    } catch (err) {
      setError('An unexpected error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Animation variants matching home page
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
    <div className="relative w-full min-h-screen bg-gradient-to-br from-gray-50 via-white to-primary-50/60 dark:from-primary-bg-1 dark:via-primary-bg-2 dark:to-primary-bg-3 overflow-hidden selection:bg-primary-500 selection:text-white transition-all duration-700">
      
      {/* Premium Canvas Particles - Matching Home Page */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none opacity-60"
      />

      {/* Premium Geometric Grid - Same as homepage */}
      <div className="absolute inset-0 opacity-40 dark:opacity-60 pointer-events-none">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(90deg, rgba(168, 85, 247, 0.1) 1px, transparent 1px),
            linear-gradient(180deg, rgba(168, 85, 247, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px'
        }} />
      </div>

      {/* Animated gradient orbs - Matching Home Page */}
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

      <div className="relative z-10 w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-24">
        
        {/* Premium Badge - Matching home page */}
        <motion.div
          initial="hidden"
          animate="show"
          variants={fadeUp}
          className="text-center mb-12"
        >
          <motion.div
            variants={fadeUp}
            whileHover={{ scale: 1.05 }}
            className="inline-flex items-center gap-3 px-6 py-3 rounded-full bg-gradient-to-r from-primary-500/20 via-primary-400/10 to-primary-300/20 border-2 border-primary-300/40 dark:border-primary-600/40 backdrop-blur-xl shadow-2xl shadow-primary-500/20 relative overflow-hidden group mb-8"
          >
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
              className="p-1 rounded-full bg-gradient-to-r from-primary-500 to-primary-400"
            >
              <FaCrown className="text-white text-sm" />
            </motion.div>
            <span className="text-sm font-extrabold bg-gradient-to-r from-primary-700 via-primary-600 to-primary-500 dark:from-primary-400 dark:via-primary-300 dark:to-primary-200 bg-clip-text text-transparent">
              JOIN OUR AI COMMUNITY
            </span>
            <TbSparkles className="text-primary-500 text-lg" />
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
          </motion.div>

          {/* Main Header */}
          <motion.h1
            variants={fadeUp}
            className="font-black text-4xl sm:text-5xl lg:text-[56px] leading-tight mb-6"
          >
            <motion.span
              variants={fadeUp}
              className="block text-gray-900 dark:text-white"
            >
              Create Your
            </motion.span>
            <motion.span
              variants={fadeUp}
              transition={{ delay: 0.1 }}
              className="block bg-gradient-to-r from-primary-700 via-primary-500 to-primary-400 bg-clip-text text-transparent"
              animate={{
                backgroundPosition: ["0%", "100%", "0%"],
              }}
              transition={{
                duration: 8,
                repeat: Infinity,
                ease: "linear",
              }}
              style={{
                backgroundSize: "200% auto",
              }}
            >
              Premium Account
            </motion.span>
          </motion.h1>

          <motion.p
            variants={fadeUp}
            transition={{ delay: 0.2 }}
            className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto"
          >
            Unlock the full potential of LinguaSign with advanced AI translation features
            and seamless communication tools
          </motion.p>

          {/* Decorative line */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
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

        {/* Registration Form Section */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 lg:gap-12">
          
          {/* Left Column - Benefits (3/5 width) */}
          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            variants={{
              hidden: { opacity: 0, x: -50 },
              show: {
                opacity: 1,
                x: 0,
                transition: {
                  duration: 0.8,
                  ease: "easeOut"
                }
              }
            }}
            className="lg:col-span-3 space-y-8"
          >
                        {/* Premium Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                {
                  icon: GiArtificialIntelligence,
                  title: "AI-Powered Translation",
                  description: "Advanced neural networks with 99% gesture recognition accuracy",
                  color: "from-primary-400 to-primary-600",
                  delay: 0.1
                },
                {
                  icon: BsLightningFill,
                  title: "Real-time Processing",
                  description: "Instant translation with sub-second latency",
                  color: "from-primary-500 to-primary-700",
                  delay: 0.2
                },
                {
                  icon: FaShieldAlt,
                  title: "Enterprise Security",
                  description: "End-to-end encryption and privacy-first design",
                  color: "from-primary-600 to-primary-800",
                  delay: 0.3
                },
                {
                  icon: BsStars,
                  title: "Premium Features",
                  description: "Access to all advanced tools and customizations",
                  color: "from-primary-700 to-primary-900",
                  delay: 0.4
                }
              ].map((feature, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: feature.delay }}
                  whileHover={{ y: -5, scale: 1.05 }}
                  className="group relative cursor-pointer h-full"
                >
                  <div className={`absolute -inset-0.5 bg-gradient-to-br ${feature.color} rounded-3xl blur opacity-0 group-hover:opacity-60 transition-opacity duration-500`} />
                  
                  <div className="relative h-full p-4 flex flex-col items-center text-center justify-center rounded-3xl bg-white/70 dark:bg-white/5 backdrop-blur-xl border border-primary-200/50 dark:border-primary-500/20 shadow-lg shadow-primary-100/20 dark:shadow-primary-900/20 group-hover:bg-white/90 dark:group-hover:bg-white/10 transition-colors duration-300 overflow-hidden">
                    <feature.icon className="absolute -bottom-4 -right-4 text-7xl text-gray-400 dark:text-gray-500 opacity-[0.06] group-hover:opacity-[0.12] group-hover:scale-110 group-hover:-rotate-12 transition-all duration-500 pointer-events-none" />
                    <div className={`p-3 rounded-2xl bg-gradient-to-br ${feature.color} shadow-lg mb-3 group-hover:-translate-y-1 transition-transform duration-300`}>
                      <feature.icon className="text-2xl text-white drop-shadow-md group-hover:scale-110 transition-transform duration-300" />
                    </div>
                    <h3 className="font-bold text-gray-900 dark:text-white mb-1 text-sm z-10">{feature.title}</h3>
                    <p className="text-gray-600 dark:text-gray-300 text-xs font-medium z-10 break-words w-full">{feature.description}</p>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Trust Badges */}
            <motion.div
              variants={fadeUp}
              className="p-6 rounded-2xl backdrop-blur-xl bg-white/80 dark:bg-white/5 border-2 border-primary-200/50 dark:border-primary-800/50"
            >
              <div className="flex flex-wrap items-center gap-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                    <FaCheck className="text-green-600 dark:text-green-400" />
                  </div>
                  <div>
                    <div className="font-bold text-gray-900 dark:text-white">100% Secure</div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">Encrypted Data</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                    <BsRobot className="text-blue-600 dark:text-blue-400" />
                  </div>
                  <div>
                    <div className="font-bold text-gray-900 dark:text-white">AI-Powered</div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">Advanced Models</div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>

          {/* Right Column - Registration Form (2/5 width) */}
          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            variants={{
              hidden: { opacity: 0, x: 50 },
              show: {
                opacity: 1,
                x: 0,
                transition: {
                  duration: 0.8,
                  ease: "easeOut",
                  delay: 0.2
                }
              }
            }}
            className="lg:col-span-2"
          >
            <div className="p-8 rounded-3xl backdrop-blur-xl bg-white/80 dark:bg-white/5 border-2 border-primary-200/50 dark:border-primary-800/50 shadow-2xl shadow-primary-500/20">
              <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-r from-primary-custom-1 via-primary-custom-2 to-primary-custom-3 mx-auto mb-8 shadow-lg">
                <FaUserPlus className="text-2xl text-white" />
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Username Field */}
                <div className="space-y-2">
                  <label className="text-sm font-bold text-gray-700 dark:text-gray-300 flex items-center gap-2">
                    <FaUser className="text-primary-500" />
                    Username
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      name="username"
                      value={formData.username}
                      onChange={handleChange}
                      required
                      className="w-full px-4 py-3 pl-12 bg-white/50 dark:bg-gray-900/50 border-2 border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 transition-all"
                      placeholder="Enter your username"
                    />
                    <FaUser className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500" />
                  </div>
                </div>

                {/* Email Field */}
                <div className="space-y-2">
                  <label className="text-sm font-bold text-gray-700 dark:text-gray-300 flex items-center gap-2">
                    <FaEnvelope className="text-primary-500" />
                    Email Address
                  </label>
                  <div className="relative">
                    <input
                      type="email"
                      name="email"
                      value={formData.email}
                      onChange={handleChange}
                      required
                      className="w-full px-4 py-3 pl-12 bg-white/50 dark:bg-gray-900/50 border-2 border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 transition-all"
                      placeholder="you@example.com"
                    />
                    <FaEnvelope className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500" />
                  </div>
                </div>

                {/* Password Field */}
                <div className="space-y-2">
                  <label className="text-sm font-bold text-gray-700 dark:text-gray-300 flex items-center gap-2">
                    <FaLock className="text-primary-500" />
                    Password
                  </label>
                  <div className="relative">
                    <input
                      type={showPassword ? "text" : "password"}
                      name="password"
                      value={formData.password}
                      onChange={handleChange}
                      required
                      minLength={6}
                      className="w-full px-4 py-3 pl-12 pr-12 bg-white/50 dark:bg-gray-900/50 border-2 border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 transition-all"
                      placeholder="Minimum 6 characters"
                    />
                    <FaLock className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500" />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-primary-500 dark:hover:text-primary-400 transition-colors"
                    >
                      {showPassword ? <FaEyeSlash /> : <FaEye />}
                    </button>
                  </div>
                </div>

                {/* Confirm Password Field */}
                <div className="space-y-2">
                  <label className="text-sm font-bold text-gray-700 dark:text-gray-300 flex items-center gap-2">
                    <FaLock className="text-primary-500" />
                    Confirm Password
                  </label>
                  <div className="relative">
                    <input
                      type={showConfirmPassword ? "text" : "password"}
                      name="confirmPassword"
                      value={formData.confirmPassword}
                      onChange={handleChange}
                      required
                      minLength={6}
                      className="w-full px-4 py-3 pl-12 pr-12 bg-white/50 dark:bg-gray-900/50 border-2 border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 transition-all"
                      placeholder="Re-enter your password"
                    />
                    <FaLock className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500" />
                    <button
                      type="button"
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-primary-500 dark:hover:text-primary-400 transition-colors"
                    >
                      {showConfirmPassword ? <FaEyeSlash /> : <FaEye />}
                    </button>
                  </div>
                </div>

                {/* Error Message */}
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 rounded-xl bg-red-50 dark:bg-red-900/20 border-2 border-red-200 dark:border-red-800"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center">
                        <span className="text-white text-xs font-bold">!</span>
                      </div>
                      <p className="text-sm text-red-700 dark:text-red-300 font-medium">{error}</p>
                    </div>
                  </motion.div>
                )}

                {/* Success Message */}
                {success && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 rounded-xl bg-green-50 dark:bg-green-900/20 border-2 border-green-200 dark:border-green-800"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center">
                        <FaCheck className="text-white text-xs" />
                      </div>
                      <p className="text-sm text-green-700 dark:text-green-300 font-medium">
                        Account created successfully! Redirecting to login...
                      </p>
                    </div>
                  </motion.div>
                )}

                {/* Submit Button */}
                <motion.button
                  type="submit"
                  disabled={loading}
                  whileHover={{ scale: 1.02, boxShadow: "0 0 25px rgba(160, 68, 255, 0.6)" }}
                  whileTap={{ scale: 0.98 }}
                  className="w-full relative overflow-hidden px-6 py-4 bg-gradient-to-r from-primary-custom-1 via-primary-custom-2 to-primary-custom-3 text-white font-bold rounded-xl shadow-lg shadow-primary-500/40 hover:shadow-primary-500/60 transition-all duration-300 flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed group"
                >
                  <span className="relative z-10 flex items-center gap-3">
                    {loading ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Creating Account...
                      </>
                    ) : (
                      <>
                        <FaUserPlus className="group-hover:scale-110 transition-transform" />
                        Create Premium Account
                      </>
                    )}
                  </span>
                  <div className="absolute top-0 left-0 w-full h-full bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 z-0 rounded-xl" />
                </motion.button>

                {/* Divider */}
                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t-2 border-gray-300/50 dark:border-gray-700/50" />
                  </div>
                  <div className="relative flex justify-center text-sm">
                    <span className="px-4 bg-white/80 dark:bg-transparent text-gray-500 dark:text-gray-400 font-medium">
                      Already have an account?
                    </span>
                  </div>
                </div>

                {/* Login Link */}
                <Link
                  to="/login"
                  className="w-full px-6 py-4 border-2 border-primary-600/50 dark:border-primary-500/50 text-primary-600 dark:text-primary-400 hover:bg-primary-50/50 dark:hover:bg-primary-900/20 hover:border-primary-700/50 dark:hover:border-primary-400/50 rounded-xl font-bold transition-all duration-300 flex items-center justify-center gap-3 group"
                >
                  <FaSignInAlt className="group-hover:scale-110 transition-transform" />
                  Sign In Now
                </Link>
              </form>

              {/* Terms */}
              <div className="mt-8 pt-6 border-t-2 border-gray-200/50 dark:border-gray-800/50">
                <p className="text-xs text-center text-gray-500 dark:text-gray-400">
                  By signing up, you agree to our{' '}
                  <a href="#" className="text-primary-600 dark:text-primary-400 hover:underline font-semibold">
                    Terms of Service
                  </a>{' '}
                  and{' '}
                  <a href="#" className="text-primary-600 dark:text-primary-400 hover:underline font-semibold">
                    Privacy Policy
                  </a>
                </p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Security Footer */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.5 }}
          className="mt-12 text-center"
        >
          <div className="inline-flex items-center gap-3 text-sm text-gray-500 dark:text-gray-400">
            <FaShieldAlt className="text-primary-500" />
            <span>All data is encrypted and secured with 256-bit SSL encryption</span>
          </div>
        </motion.div>
      </div>

      <style jsx>{`
        @keyframes gradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .animate-gradient {
          background-size: 200% auto;
          animation: gradient 3s linear infinite;
        }
      `}</style>
    </div>
  );
}
