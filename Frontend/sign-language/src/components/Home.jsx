import React, { useState, useEffect, useRef, useContext, useCallback, useMemo } from "react";
import { useTheme } from "../context/ThemeContext";
import { FaMobileAlt, FaGlobeAmericas, FaRocket, FaCrown } from "react-icons/fa";
import { BsMicFill, BsRobot, BsStars, BsLightningFill, BsCheckCircleFill } from "react-icons/bs";
import { TbHandLoveYou, TbArrowsExchange, TbSparkles, TbHexagon } from "react-icons/tb";
import { GiArtificialIntelligence, GiRingingBell, GiBrain, GiHand } from "react-icons/gi";
import { RiTranslate2, RiShieldCheckFill } from "react-icons/ri";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { BsChatDots, BsChatFill, BsChatText } from "react-icons/bs";
const allAssets = import.meta.glob("../assets/**/*.{svg,png,jpg,jpeg}", { eager: true, import: "default" });

export default function Home() {
  const { themeColor } = useTheme();
  const gridColor = themeColor === "midnight-blue" ? "rgba(99, 102, 241, 0.1)" : "rgba(168, 85, 247, 0.1)";
  const themeFolder = themeColor === "midnight-blue" ? "blue" : "purple";
  const navigate = useNavigate();
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationFrameRef = useRef(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [hoveredFeature, setHoveredFeature] = useState(null);
  const [expandedFeature, setExpandedFeature] = useState(null);

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
    });return () => observer.disconnect();
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
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      window.removeEventListener('resize', handleResize);
    };
  }, [isDark, themeColor]);

  // Animation variants
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

  const fade = {
    hidden: { opacity: 0 },
    show: { 
      opacity: 1,
      transition: {
        duration: 1,
        ease: "easeOut"
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

  const features = [
    { 
      icon: <GiBrain />, 
      text: "AI Gesture Recognition", 
      desc: "Deep learning models with 99% accuracy", 
    },
    { 
      icon: <BsMicFill />, 
      text: "Speech-to-Text", 
      desc: "Real-time transcription in 50+ languages",
    },
    { 
      icon: <GiHand />, 
      text: "Customizable Avatars", 
      desc: "Personalize digital sign language interpreters",
    },
    { 
      icon: <FaMobileAlt />, 
      text: "Mobile & Web Ready", 
      desc: "Seamless cross-platform experience",
    },
  ];
  
  const getHeroImage = () => {
    const darkPath = `../assets/${themeFolder}/heroDark.svg`;
    const lightPath = `../assets/${themeFolder}/hero.svg`;
    const imagePath = `../assets/${themeFolder}/image.svg`;
    
    if (isDark) {
      return allAssets[darkPath] || allAssets[imagePath] || allAssets[lightPath];
    }
    return allAssets[lightPath] || allAssets[imagePath] || allAssets[darkPath];
  };

  // Get theme-specific gradient for buttons
  const getGradientColors = () => {
    if (themeColor === "midnight-blue") {
      return "from-indigo-500 via-blue-600 to-indigo-700";
    }
    return "from-[#6A3093] via-[#A044FF] to-[#BF5AE0]";
  };

  const getHoverShadow = () => {
    if (themeColor === "midnight-blue") {
      return "shadow-indigo-500/40 hover:shadow-indigo-500/60";
    }
    return "shadow-purple-400/40 hover:shadow-purple-500/60";
  };

  const getFocusRing = () => {
    if (themeColor === "midnight-blue") {
      return "focus:ring-indigo-500/50";
    }
    return "focus:ring-purple-500/50";
  };

  return (
    <div className="relative w-full min-h-screen bg-gradient-to-br from-gray-50 via-white to-primary-50/60 dark:from-primary-bg-1 dark:via-primary-bg-2 dark:to-primary-bg-3 overflow-hidden selection:bg-primary-500 selection:text-white transition-all duration-700">
      
      {/* Premium Canvas Particles */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none opacity-60"
      />

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

      {/* MAIN CONTENT */}
      <div className="relative z-10 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 lg:py-28 flex flex-col lg:flex-row items-center justify-between gap-16 lg:gap-24">

        {/* LEFT COLUMN */}
        <motion.div
          initial="hidden"
          animate="show"
          variants={{
            hidden: { opacity: 0 },
            show: {
              opacity: 1,
              transition: {
                staggerChildren: 0.1,
                delayChildren: 0.2
              }
            }
          }}
          className="w-full lg:w-1/2 space-y-5"
        >
          {/* Premium Badge */}
          <motion.div
            variants={fadeUp}
            whileHover={{ scale: 1.05 }}
            className="inline-flex items-center gap-3 px-6 py-3 rounded-full bg-gradient-to-r from-primary-500/20 via-primary-400/10 to-primary-300/20 border-2 border-primary-300/40 dark:border-primary-600/40 backdrop-blur-xl shadow-2xl shadow-primary-500/20 relative overflow-hidden group"
          >
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
              className="p-1 rounded-full bg-gradient-to-r from-primary-500 to-primary-400"
            >
              <FaCrown className="text-white text-sm" />
            </motion.div>
            <span className="text-sm font-extrabold bg-gradient-to-r from-primary-700 via-primary-600 to-primary-500 dark:from-primary-400 dark:via-primary-300 dark:to-primary-200 bg-clip-text text-transparent">
              AI INTEGRATED V3.0
            </span>
            <TbSparkles className="text-primary-500 text-lg" />
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
          </motion.div>

          {/* HEADLINE */}
          <div className="space-y-1">
            <motion.h1 variants={fadeUp} className="font-black text-5xl sm:text-6xl lg:text-[58px] leading-[1.1] tracking-tight">
              <motion.span variants={fadeUp} className="block text-gray-900 dark:text-white">
                AI Powered
              </motion.span>
              <motion.span
                variants={fadeUp}
                className="block bg-gradient-to-r from-primary-700 via-primary-500 to-primary-400 bg-clip-text text-transparent"
                animate={{
                  backgroundPosition: ["0%", "100%", "0%"],
                }}
                transition={{
                  delay: 0.1,
                  duration: 8,
                  repeat: Infinity,
                  ease: "linear",
                }}
                style={{
                  backgroundSize: "200% auto",
                }}
              >
                Sign Language
              </motion.span>
              <motion.span
                variants={fadeUp}
                transition={{ delay: 0.2 }}
                className="block text-gray-900 dark:text-white"
              >
                Translator
              </motion.span>
            </motion.h1>
          </div>

          {/* FEATURE CARDS - 2x2 GRID FIX */}
          <motion.div
            variants={{
              hidden: { opacity: 0 },
              show: {
                opacity: 1,
                transition: {
                  staggerChildren: 0.15,
                  delayChildren: 0.3
                }
              }
            }}
            initial="hidden"
            animate="show"
            className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-8"
          >
            {features.map((feature, i) => (
              <motion.div
                key={i}
                variants={scaleIn}
                whileHover={{ 
                  scale: 1.05,
                  y: -5,
                  boxShadow: "0 25px 40px -15px rgba(139, 92, 246, 0.4)",
                }}
                onMouseEnter={() => setHoveredFeature(i)}
                onMouseLeave={() => setHoveredFeature(null)}
                onClick={() => setExpandedFeature(expandedFeature === i ? null : i)}
                className={`group relative p-4 rounded-2xl backdrop-blur-xl border-2 transition-all duration-500 overflow-hidden cursor-pointer ${
                  hoveredFeature === i || expandedFeature === i
                    ? 'bg-white dark:bg-white/15 border-primary-400 dark:border-primary-400 shadow-2xl' 
                    : 'bg-white/80 dark:bg-white/5 border-primary-200/50 dark:border-primary-800/50 shadow-lg hover:border-primary-300 dark:hover:border-primary-600'
                }`}
              >
                <div className="absolute inset-0 bg-gradient-to-br from-primary-400 to-primary-600 opacity-0 group-hover:opacity-10 transition-opacity duration-700" />
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
                <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-br from-primary-400 to-primary-600 opacity-0 group-hover:opacity-20 rounded-bl-2xl transition-all duration-500" />
                <div className="absolute bottom-0 left-0 w-16 h-16 bg-gradient-to-tr from-primary-400 to-primary-600 opacity-0 group-hover:opacity-20 rounded-tr-2xl transition-all duration-500" />
                
                <div className="relative z-10">
                  <div className="flex items-start gap-3">
                    <div className="p-2.5 rounded-xl bg-gradient-to-br from-primary-400 to-primary-600 text-white shadow-xl group-hover:scale-110 transition-all duration-500">
                      <div className="text-xl">{feature.icon}</div>
                    </div>
                    <div className="flex-1">
                      <div className="font-black text-gray-900 dark:text-white text-[15px] mb-0.5">
                        {feature.text}
                      </div>
                      <div className="text-[12px] text-gray-600 dark:text-gray-400 leading-snug">
                        {feature.desc}
                      </div>
                    </div>
                  </div>
                </div>
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-primary-400 to-primary-600 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-700 origin-left" />
              </motion.div>
            ))}
          </motion.div>

          {/* CTA Buttons - Unified Styling */}
         {/* CTA Buttons - Reorganized & Enhanced */}
<div className="flex flex-col sm:flex-row flex-wrap items-center gap-4 mt-8">
  
  {/* Primary Actions - Main CTA */}
  <div className="flex flex-col sm:flex-row items-center gap-3 w-full sm:w-auto">
    {/* Start Translating - Primary Action */}
    <motion.button
      variants={fadeUp}
      whileHover={{
        scale: 1.05,
        boxShadow: "0 0 30px rgba(160, 68, 255, 0.5)",
      }}
      whileTap={{ scale: 0.95 }}
      onClick={() => navigate("/translate")}
      className={`relative overflow-hidden px-8 py-4 rounded-full bg-gradient-to-r ${getGradientColors()} text-white font-bold text-lg shadow-xl ${getHoverShadow()} transition-all duration-300 group w-full sm:w-auto focus:outline-none focus:ring-4 ${getFocusRing()}`}
    >
      <span className="relative z-10 flex items-center justify-center gap-3">
        <TbHandLoveYou className="text-2xl group-hover:rotate-12 group-hover:scale-110 transition-transform duration-300" />
        <span className="tracking-wide">Start Translating</span>
        <motion.span
          animate={{ x: [0, 5, 0] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
          className="text-lg"
        >
          →
        </motion.span>
      </span>
      <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-white/0 via-white/20 to-white/0 translate-y-full group-hover:translate-y-0 transition-transform duration-500 z-0 rounded-full" />
    </motion.button>

    {/* AI Chatbot - Secondary CTA */}
    <motion.button
      variants={fadeUp}
      whileHover={{
        scale: 1.05,
        boxShadow: "0 0 25px rgba(160, 68, 255, 0.4)",
      }}
      whileTap={{ scale: 0.95 }}
      onClick={() => navigate("/chatbot")}
      className={`relative overflow-hidden px-6 py-4 rounded-full bg-gradient-to-r ${getGradientColors()} text-white font-bold text-lg shadow-lg ${getHoverShadow()} transition-all duration-300 group w-full sm:w-auto focus:outline-none focus:ring-4 ${getFocusRing()}`}
    >
      <span className="relative z-10 flex items-center justify-center gap-2">
        <BsRobot className="text-xl group-hover:rotate-12 transition-transform duration-300" />
        <span className="tracking-wide">AI Chatbot</span>

      </span>
      <div className="absolute top-0 left-0 w-full h-full bg-white/20 -translate-x-full group-hover:translate-x-0 transition-transform duration-500 ease-out z-0 rounded-full" />
    </motion.button>
  </div>

  {/* Divider - Optional visual separator */}
  <div className="hidden sm:flex items-center">
    <div className="w-px h-8 bg-gray-300/50 dark:bg-gray-700/50" />
  </div>
  <div className="flex sm:hidden w-full max-w-xs">
    <div className="w-full h-px bg-gray-300/50 dark:bg-gray-700/50" />
  </div>

  {/* Secondary Actions */}
  <div className="flex flex-col sm:flex-row items-center gap-3 w-full sm:w-auto">
    {/* Chatting - Ghost Button */}
    <motion.button
      variants={fadeUp}
      whileHover={{ 
        scale: 1.05,
        boxShadow: "0 0 20px rgba(160, 68, 255, 0.15)",
      }}
      whileTap={{ scale: 0.95 }}
      onClick={() => navigate("/chat")}
      className="px-6 py-4 rounded-full font-bold text-lg text-gray-700 dark:text-gray-200 border-2 border-gray-300 dark:border-gray-600 hover:border-primary-400 dark:hover:border-primary-500 bg-white/60 dark:bg-white/5 hover:bg-primary-50/50 dark:hover:bg-primary-900/20 shadow-sm hover:shadow-md transition-all duration-300 flex items-center justify-center gap-2 group w-full sm:w-auto"
    >
      <BsChatDots className="text-xl group-hover:scale-110 group-hover:text-primary-500 transition-all duration-300" />
      <span className="group-hover:text-primary-500 dark:group-hover:text-primary-400 transition-colors duration-300">
        Chatting
      </span>
    </motion.button>
  </div>
</div>

{/* Trust Badges - Added for better UX */}
<motion.div
  variants={fadeUp}
  className="flex flex-wrap items-center justify-center sm:justify-start gap-4 sm:gap-6 mt-6 pt-4 border-t border-gray-200/50 dark:border-gray-700/50"
>
  <div className="flex items-center gap-2">
    <RiShieldCheckFill className="text-primary-500 text-sm" />
    <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">256-bit SSL</span>
  </div>
  <div className="flex items-center gap-2">
    <BsLightningFill className="text-primary-500 text-sm" />
    <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">Real-time Translation</span>
  </div>
  <div className="flex items-center gap-2">
    <FaGlobeAmericas className="text-primary-500 text-sm" />
    <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">Accessible Communication</span>
  </div>
  <div className="flex items-center gap-2">
    <BsCheckCircleFill className="text-primary-500 text-sm" />
    <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">92.36% Accuracy</span>
  </div>
</motion.div>
        </motion.div>


        {/* RIGHT COLUMN - Image */}
        <motion.div
          variants={fade}
          initial="hidden"
          animate="show"
          transition={{ duration: 1.5 }}
          className="w-full lg:w-[41%] relative flex justify-center"
        >
          <div className="relative w-full max-w-2xl">
            {/* Orbital Rings */}
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 40, repeat: Infinity, ease: "linear" }}
              className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[85%] h-[85%] border-2 border-primary-400/30 rounded-full"
            />
            <motion.div
              animate={{ rotate: -360 }}
              transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
              className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[65%] h-[65%] border border-primary-400/20 rounded-full "
            />

            {/* Main Image */}
            <div>
              <motion.div className="relative">
                <img
                  src={getHeroImage()}
                  alt="LinguaSign Premium AI Translator"
                  className="w-full transform transition-all duration-700 group-hover:scale-105 relative z-10"
                />
              </motion.div>

              {/* Floating Elements */}
              <motion.div
                animate={{ 
                  y: [0, -15, 0],
                  rotate: [0, 5, -5, 0]
                }}
                transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
                className="absolute top-4 right-4 z-30"
              >
                <div className="backdrop-blur-xl bg-white/95 dark:bg-gray-800/95 border-2 border-primary-300 dark:border-primary-600 p-3.5 rounded-2xl shadow-2xl flex items-center gap-3">
                  <div className="w-2.5 h-2.5 bg-gradient-to-r from-primary-500 to-primary-400 rounded-full animate-pulse" />
                  <span className="font-extrabold text-gray-900 dark:text-white text-sm tracking-wide">
                    AI Live Avatars
                  </span>
                </div>
              </motion.div>

              <motion.div
                animate={{ 
                  y: [0, 15, 0],
                  rotate: [0, -5, 5, 0]
                }}
                transition={{ duration: 6, repeat: Infinity, ease: "easeInOut", delay: 1 }}
                className="absolute bottom-4 left-4 z-30"
              >
                <div className="backdrop-blur-xl bg-white/95 dark:bg-gray-800/95 border-2 border-primary-300 dark:border-primary-600 px-5 py-2.5 rounded-full shadow-2xl">
                  <span className="font-extrabold text-transparent bg-gradient-to-r from-primary-700 to-primary-500 dark:from-primary-300 dark:to-primary-400 bg-clip-text text-sm flex items-center gap-2">
                    Real-time translation
                  </span>
                </div>
              </motion.div>
            </div>
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
          animation: gradient 8s ease infinite;
        }
      `}</style>
    </div>
  );
}