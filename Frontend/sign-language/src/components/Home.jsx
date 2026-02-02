import React, { useEffect, useState, useRef } from "react";
import Hero from "../assets/hero.svg";
import HeroDark from "../assets/heroDark.svg";
import { FaMobileAlt, FaGlobeAmericas, FaRocket, FaCrown } from "react-icons/fa";
import { BsMicFill, BsRobot, BsStars, BsLightningFill } from "react-icons/bs";
import { TbHandLoveYou, TbArrowsExchange, TbSparkles } from "react-icons/tb";
import { GiArtificialIntelligence, GiRingingBell } from "react-icons/gi";
import { RiTranslate2, RiShieldCheckFill } from "react-icons/ri";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import Logo from "./Logo";

export default function Home() {
  const navigate = useNavigate();
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationFrameRef = useRef(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [hoveredFeature, setHoveredFeature] = useState(null);

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



  // Particle system (Static Movement Only)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const colors = isDark 
      ? ['#A855F7', '#9333EA', '#7C3AED', '#6D28D9', '#8B5CF6'] 
      : ['#8B5CF6', '#7C3AED', '#6D28D9', '#9333EA', '#A855F7'];

    particlesRef.current = Array.from({ length: 100 }).map(() => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      size: Math.random() * 3 + 1, // Slightly smaller
      speedX: Math.random() * 0.5 - 0.25,
      speedY: Math.random() * 0.5 - 0.25,
      color: colors[Math.floor(Math.random() * colors.length)],
      opacity: Math.random() * 0.5 + 0.1,
      glow: Math.random() > 0.8,
    }));

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // --- MOUSE GRADIENT BACKGROUND REMOVED ---

      particlesRef.current.forEach(particle => {
        // --- MOUSE PUSH INTERACTION REMOVED ---

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
            ctx.strokeStyle = particle.color + '33'; // Static subtle opacity
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

    // Handle resize
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
  }, [isDark, mousePosition]);

  // Enhanced animation variants
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
    { icon: <BsRobot />, text: "AI Gesture Recognition", desc: "Deep learning models with 99% accuracy", glow: true },
    { icon: <BsMicFill />, text: "Speech-to-Text", desc: "Real-time transcription in 50+ languages", glow: true },
    { icon: <FaMobileAlt />, text: "Mobile & Web Ready", desc: "Seamless cross-platform experience", glow: false },
    { icon: <TbHandLoveYou />, text: "Customizable Avatars", desc: "Personalize digital sign language interpreters", glow: true },
    { icon: <RiShieldCheckFill />, text: "Secure & Private", desc: "End-to-end encryption for all conversations", glow: false },
    { icon: <GiRingingBell />, text: "Real-time Alerts", desc: "Instant notifications for translations", glow: false },
  ];

  return (
    <div className="relative w-full min-h-screen bg-gradient-to-br from-gray-50 via-white to-purple-50/60 dark:from-[#0a0518] dark:via-[#110a2e] dark:to-[#1e0f5c] overflow-hidden selection:bg-purple-500 selection:text-white transition-all duration-700">
      
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



      {/* MAIN CONTENT */}
      <div className="relative z-10 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 lg:py-28 flex flex-col lg:flex-row items-center justify-between gap-16 lg:gap-24">

        {/* LEFT COLUMN - Enhanced */}
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
            whileHover={{ scale: 1.05, rotate: 1 }}
            className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-gradient-to-r from-purple-500/15 via-purple-400/10 to-purple-300/10 border border-purple-200/60 dark:border-purple-700/60 backdrop-blur-xl shadow-lg shadow-purple-500/10 relative overflow-hidden group"
          >
            <div className="relative">
              <span className="absolute animate-ping inline-flex h-3.5 w-3.5 rounded-full bg-purple-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-gradient-to-r from-purple-500 to-purple-400" />
            </div>
            <span className="text-sm font-bold bg-gradient-to-r from-purple-600 via-purple-500 to-purple-400 bg-clip-text text-transparent">
             AI Integrated v3.0 
            </span>
            <div className="absolute -inset-1 bg-gradient-to-r from-purple-500/0 via-purple-400/10 to-purple-500/0 group-hover:via-purple-400/20 transition-all duration-500" />
          </motion.div>

          {/* HEADLINE with Enhanced Typography */}
          <div className="space-y-1">
       <motion.h1
  variants={fadeUp}
  className="font-extrabold text-4xl sm:text-5xl lg:text-[53px] leading-tight"
>
  <motion.span
    variants={fadeUp}
    className="block text-gray-900 dark:text-white"
  >
    AI Powered
  </motion.span>
  <motion.span
    variants={fadeUp}
    transition={{ delay: 0.1 }}
    className="block bg-gradient-to-r from-purple-600 to-purple-400 bg-clip-text text-transparent"
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

          {/* Premium Feature Grid with Glow Effects */}
          <motion.div
            variants={{
              hidden: { opacity: 0 },
              show: {
                opacity: 1,
                transition: {
                  staggerChildren: 0.1,
                  delayChildren: 0.5
                }
              }
            }}
            className="grid grid-cols-1 sm:grid-cols-2 gap-5"
          >
            {features.slice(0, 4).map((feature, i) => (
              <motion.div
                key={i}
                variants={scaleIn}
                whileHover={{ 
                  scale: 1.05,
                  y: -8,
                  boxShadow: "0 20px 40px -15px rgba(139, 92, 246, 0.4)"
                }}
                onMouseEnter={() => setHoveredFeature(i)}
                onMouseLeave={() => setHoveredFeature(null)}
                className={`group relative p-3 rounded-2xl backdrop-blur-xl border transition-all duration-300 overflow-hidden ${
                  hoveredFeature === i 
                    ? 'bg-white/90 dark:bg-white/10 border-purple-300/50 dark:border-purple-500/50 shadow-2xl' 
                    : 'bg-white/70 dark:bg-white/5 border-white/30 dark:border-white/10 shadow-lg'
                }`}
              >
         

                <div className="relative z-10 flex items-start gap-4">
                  <div className={`p-2 rounded-xl bg-gradient-to-br from-purple-500/20 to-purple-400/20 text-purple-600 dark:text-purple-400 group-hover:from-purple-500/30 group-hover:to-purple-400/30 transition-all duration-300 ${
                    hoveredFeature === i ? 'scale-110' : ''
                  }`}
                  
                  >
                    <div className="text-2xl">{feature.icon}</div>
                  </div>
                  <div>
                    <div className="font-bold text-gray-900 dark:text-white text-[16px] mb-1">
                      {feature.text}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {feature.desc}
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>

          {/* Premium CTA Buttons */}
       <div className="flex flex-wrap items-center gap-4 pt-2">
            <motion.button
              variants={fadeUp}
              whileHover={{
                scale: 1.05,
                boxShadow: "0 0 25px rgba(160, 68, 255, 0.6)",
              }}
              whileTap={{ scale: 0.95 }}
              onClick={() => navigate("/translate")}
              className="relative overflow-hidden px-8 py-4 rounded-full bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] text-white font-bold text-lg shadow-lg shadow-purple-400/40 hover:shadow-purple-500/60 transition-all group focus:outline-none focus:ring-4 focus:ring-purple-500/50"
            >
              <span className="relative z-10 flex items-center gap-2">
                Start Translating <TbHandLoveYou className="text-2xl" />
              </span>
              <div className="absolute top-0 left-0 w-full h-full bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 z-0 rounded-full" />
            </motion.button>

            <motion.button
              variants={fadeUp}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => navigate("/chatbot")}
              className="px-8 py-4 rounded-full font-bold text-lg text-gray-800 dark:text-white border-2 border-gray-300 dark:border-gray-700 hover:border-purple-500 bg-white/70 dark:bg-transparent hover:bg-gray-100 dark:hover:bg-white/5 shadow-sm hover:shadow-md transition-all flex items-center gap-2"
            >
              <BsRobot className="text-xl" />
              Chatbot
            </motion.button>
          </div>
        </motion.div>

        {/* RIGHT COLUMN - Premium Image Display */}
        <motion.div
          variants={fade}
          initial="hidden"
          animate="show"
          transition={{ duration: 1.5 }}
          className="w-full lg:w-[41%] relative flex justify-center"
        >
          {/* Image Container with Premium Effects */}
          <div className="relative w-full max-w-2xl">
            {/* Glow Effect Behind Image */}
            {/* <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[120%] h-[120%] bg-gradient-to-br from-purple-600/20 via-purple-500/15 to-purple-400/10 blur-3xl rounded-full" /> */}
            
            {/* Orbital Rings */}
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 35, repeat: Infinity, ease: "linear" }}
              className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[80%] h-[80%] border-2 border-purple-500/30 rounded-full"
            />
            <motion.div
              animate={{ rotate: -360 }}
              transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
              className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[60%] h-[60%] border border-purple-400/20 rounded-full"
            />

            {/* Premium Image Display */}
            <motion.div
              initial={{ opacity: 0}}
              animate={{ opacity: 1}}
              transition={{ duration: 1, ease: "backOut" }}
              className="relative z-20 group"
          
            >
              {/* Image Frame */}
              <div className="relative  transition-all duration-500">
                <img
                  src={Hero}
                  alt="LinguaSign Premium AI Translator Interface"
                  className="w-full transform transition-transform duration-700"
                />
                
                {/* Image Overlay Glow
                <div className="absolute inset-0 bg-gradient-to-t from-purple-600/5 via-transparent to-purple-400/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" /> */}
              </div>
              
              {/* Floating Elements on Image */}
          <motion.div
  animate={{ y: [0, -10, 0] }}
  transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
  className="absolute top-4 right-4 z-20"
>
  <div className="backdrop-blur-md bg-white/90 dark:bg-gray-800/90 border border-purple-200 dark:border-purple-700 p-3 rounded-xl shadow-lg flex items-center gap-2">
    <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
    <span className="font-semibold text-gray-800 dark:text-white text-sm">
      AI Live Avatars
    </span>
  </div>
</motion.div>

<motion.div
  animate={{ y: [0, 10, 0] }}
  transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
  className="absolute bottom-4 left-4 z-20"
>
  <div className="backdrop-blur-md bg-white/90 dark:bg-gray-800/90 border border-purple-200 dark:border-purple-700 px-4 py-2 rounded-full shadow-lg">
    <span className="font-semibold text-purple-600 dark:text-purple-400 text-sm">
      Real-time translation 
    </span>
  </div>
</motion.div>
            </motion.div>

         
          </div>
        </motion.div>
      </div>

      {/* Premium Scroll Indicator */}
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
          <div className="text-sm text-gray-500 dark:text-gray-400 mb-3 font-medium">
            Discover More Features ↓
          </div>
          <div className="w-8 h-12 border-2 border-purple-300/50 dark:border-purple-700/50 rounded-full mx-auto relative overflow-hidden">
            <motion.div
              animate={{ y: [0, 20, 0] }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: "easeInOut"
              }}
              className="absolute top-2 left-1/2 transform -translate-x-1/2 w-1 h-3 bg-gradient-to-b from-purple-500 to-purple-400 rounded-full"
            />
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}