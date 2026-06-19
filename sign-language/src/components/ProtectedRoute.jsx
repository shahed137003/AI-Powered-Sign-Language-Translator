import React, { useEffect, useRef } from 'react';
import { useLocation, Link } from 'react-router-dom';
import { useAuth } from '../context/useAuth';
import { useTheme } from '../context/ThemeContext';
import { motion } from 'framer-motion';
import { FaLock, FaSignInAlt, FaShieldAlt, FaCrown, FaGem } from 'react-icons/fa';
import { TbSparkles, TbHandLoveYou } from 'react-icons/tb';
import { BsLightningCharge } from 'react-icons/bs';

export default function ProtectedRoute({ children }) {
  const { isAuthenticated } = useAuth();
  const { themeColor } = useTheme();
  const location = useLocation();
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationFrameRef = useRef(null);
  const [isDark, setIsDark] = React.useState(false);

  const gridColor = themeColor === "midnight-blue" ? "rgba(99, 102, 241, 0.1)" : "rgba(168, 85, 247, 0.1)";

  // Detect dark mode
  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains("dark"));
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    setIsDark(document.documentElement.classList.contains("dark"));
    return () => observer.disconnect();
  }, []);

  // Particle system
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
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      window.removeEventListener('resize', handleResize);
    };
  }, [isDark, themeColor]);

  if (!isAuthenticated) {
    let message = "Please log in to access this page.";
    let subMessage = "This area requires authentication to ensure your privacy and security.";
    
    if (location.pathname === '/translate') {
      message = "Access to Translation Tool";
      subMessage = "Please log in to use our AI-powered sign language translation features.";
    } else if (location.pathname === '/profile') {
      message = "Profile Access Required";
      subMessage = "Please log in to view and manage your personal profile settings.";
    } else if (location.pathname === '/chat') {
      message = "Chatbot Access";
      subMessage = "Please log in to interact with our AI sign language chatbot.";
    } else if (location.pathname === '/dashboard') {
      message = "Dashboard Restricted";
      subMessage = "Please log in to access your personal dashboard and analytics.";
    }

    const fadeUp = {
      hidden: { opacity: 0, y: 30 },
      visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: [0.22, 1, 0.36, 1] } }
    };

    const scaleIn = {
      hidden: { opacity: 0, scale: 0.9 },
      visible: { opacity: 1, scale: 1, transition: { duration: 0.6, ease: "backOut" } }
    };

    return (
      <div className="relative w-full min-h-screen bg-gradient-to-br from-gray-50 via-white to-primary-50/60 dark:from-[#0a0518] dark:via-[#110a2e] dark:to-[#1e0f5c] overflow-hidden selection:bg-primary-500 selection:text-white transition-all duration-700">
        
        {/* Canvas Particles */}
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none opacity-60" />

        {/* Premium Geometric Grid */}
        <div className="absolute inset-0 opacity-40 dark:opacity-60 pointer-events-none">
          <div className="absolute inset-0" style={{
            backgroundImage: `
              linear-gradient(90deg, ${gridColor} 1px, transparent 1px),
              linear-gradient(180deg, ${gridColor} 1px, transparent 1px)
            `,
            backgroundSize: '40px 40px'
          }} />
        </div>

        {/* Animated gradient orbs */}
        <motion.div
          className="absolute top-20 left-20 w-[600px] h-[600px] bg-primary-600/10 rounded-full blur-[120px]"
          animate={{ x: [0, 100, 0], y: [0, -100, 0] }}
          transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
        />
        <motion.div
          className="absolute bottom-20 right-20 w-[600px] h-[600px] bg-primary-400/10 rounded-full blur-[120px]"
          animate={{ x: [0, -100, 0], y: [0, 100, 0] }}
          transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
        />

   

        <div className="relative z-10 w-full max-w-lg mx-auto px-4 min-h-screen flex flex-col items-center justify-center">
          
          <motion.div
            initial="hidden"
            animate="visible"
            variants={fadeUp}
            className="text-center"
          >
            {/* Animated Icon */}
            <motion.div
              variants={scaleIn}
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-gradient-to-br from-primary-500/20 to-primary-600/20 border-2 border-primary-300/40 dark:border-primary-600/40 shadow-2xl backdrop-blur-xl mb-4"
            >
              <FaLock className="text-5xl text-primary-500" />
            </motion.div>

            {/* Premium Badge */}
            {/* <motion.div
              variants={scaleIn}
              whileHover={{ scale: 1.05 }}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-gradient-to-r from-primary-500/20 via-primary-400/10 to-primary-300/20 border-2 border-primary-300/40 dark:border-primary-600/40 backdrop-blur-xl shadow-2xl shadow-primary-500/20 relative overflow-hidden group mb-8"
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                className="p-1 rounded-full bg-gradient-to-r from-primary-500 to-primary-400"
              >
                <FaCrown className="text-white text-sm" />
              </motion.div>
              <span className="text-sm font-extrabold bg-gradient-to-r from-primary-700 via-primary-600 to-primary-500 dark:from-primary-400 dark:via-primary-300 dark:to-primary-200 bg-clip-text text-transparent">
                ACCESS RESTRICTED
              </span>
              <TbSparkles className="text-primary-500 text-lg" />
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
            </motion.div> */}

            {/* Main Title */}
            <motion.h1 
              variants={fadeUp}
              className="font-black text-4xl sm:text-5xl leading-tight mb-4 text-gray-900 dark:text-white"
            >
              {message}
            </motion.h1>

            {/* Subtitle */}
            <motion.p 
              variants={fadeUp}
              className="text-lg text-gray-600 dark:text-gray-300 mb-6"
            >
              {subMessage}
            </motion.p>

            {/* Feature Cards */}
            <motion.div
              variants={scaleIn}
              className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-md mx-auto mb-10"
            >
              <div className="p-4 rounded-xl bg-white/60 dark:bg-white/5 backdrop-blur-sm border border-primary-200/50 dark:border-primary-500/20 text-center">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary-500/20 to-primary-400/20 flex items-center justify-center mx-auto mb-2">
                  <FaShieldAlt className="text-primary-500 text-lg" />
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Enterprise Security</p>
              </div>
              <div className="p-4 rounded-xl bg-white/60 dark:bg-white/5 backdrop-blur-sm border border-primary-200/50 dark:border-primary-500/20 text-center">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary-500/20 to-primary-400/20 flex items-center justify-center mx-auto mb-2">
                  <FaGem className="text-primary-500 text-lg" />
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Premium Features</p>
              </div>
            </motion.div>

            {/* Login Button */}
            <motion.div variants={scaleIn}>
              <Link to="/login" state={{ from: location }}>
                <motion.button
                  whileHover={{ scale: 1.02, y: -2 }}
                  whileTap={{ scale: 0.98 }}
                  className="relative group overflow-hidden rounded-xl bg-gradient-to-r from-primary-600 to-primary-500 text-white shadow-xl shadow-primary-500/25 px-10 py-4 font-bold text-lg transition-all hover:shadow-primary-500/40 inline-flex items-center justify-center gap-3"
                >
                  <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-out" />
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
                  <span className="relative z-10 flex items-center gap-2">
                    <FaSignInAlt />
                    Sign In to Continue
                  </span>
                </motion.button>
              </Link>
            </motion.div>

            {/* Additional Info */}
            <motion.p 
              variants={fadeUp}
              className="text-sm text-gray-500 dark:text-gray-400 mt-6 flex items-center justify-center gap-2"
            >
              <BsLightningCharge className="text-primary-500 text-xs" />
              Secure login powered by AI authentication
            </motion.p>
          </motion.div>

    
        </div>
      </div>
    );
  }

  return children;
}
