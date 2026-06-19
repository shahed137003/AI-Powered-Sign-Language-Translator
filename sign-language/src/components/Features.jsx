import React, { useEffect, useRef, useState } from "react";
import { useTheme } from "../context/ThemeContext";
import { motion, AnimatePresence } from "framer-motion";
import { TbSparkles, TbMicrophone2, TbVideo, TbMessageCircle, TbDeviceMobile, TbUserEdit, TbSettingsAutomation, TbBrain, TbRocket, TbEye, TbClock } from "react-icons/tb";
import { FaCrown, FaStar, FaCheckCircle, FaUsers, FaChartLine } from "react-icons/fa";
import { BsCheckCircleFill, BsLightningFill, BsAward, BsGraphUp } from "react-icons/bs";

const features = [
  {
    imgName: "feature1.svg",
    title: "Voice-to-Text Translation",
    description: "Convert spoken language into written text in real-time. Ideal for facilitating communication for people with hearing impairments, meetings, or classrooms. Our AI ensures accurate transcription context understanding.",
    tags: ["Real-time", "99% Accuracy", "50+ Languages"],
    icon: <TbMicrophone2 />,
    statValue: "99%",
    statLabel: "Accuracy",
    users: "50K+",
    rating: 4.9,
    speed: "0.3s",
  },
  {
    imgName: "feature2.svg",
    title: "Video Upload Translation",
    description: "Upload pre-recorded videos or links and automatically generate synchronized sign language animations. Perfect for creating educational content, presentations, or social media posts accessible to the deaf community.",
    tags: ["Auto-sync", "HD Quality", "Batch Upload"],
    icon: <TbVideo />,
    statValue: "4K",
    statLabel: "Quality",
    users: "35K+",
    rating: 4.8,
    speed: "Instant",
  },
  {
    imgName: "feature4.svg",
    title: "Mobile Application",
    description: "Experience the full suite of features on your smartphone. Translate spoken language, type text, upload videos, or chat with the AI bot anytime, anywhere. Designed for portability and seamless integration.",
    tags: ["iOS & Android", "Offline Mode", "Cloud Sync"],
    icon: <TbDeviceMobile />,
    statValue: "100K+",
    statLabel: "Downloads",
    users: "80K+",
    rating: 4.9,
    speed: "Sync",
  },
  {
    imgName: "profile.svg",
    title: "Edit Profile & Personalization",
    description: "Customize your personal experience with a fully editable profile. Update your communication preferences and accessibility settings to ensure a tailored environment that manages your identity securely.",
    tags: ["Custom Avatar", "Preferences", "Secure Storage"],
    icon: <TbUserEdit />,
    statValue: "15+",
    statLabel: "Themes",
    users: "45K+",
    rating: 4.6,
    speed: "Auto",
  },
  {
    imgName: "customize.svg",
    title: "Accessibility Customization",
    description: "Enhance usability through advanced accessibility controls. Adjust avatar animation speed, customize color themes, enlarge handshape visuals, and modify gesture clarity for an inclusive experience.",
    tags: ["WCAG Compliant", "High Contrast", "Adjustable Speed"],
    icon: <TbSettingsAutomation />,
    statValue: "25+",
    statLabel: "Settings",
    users: "30K+",
    rating: 4.8,
    speed: "Flexible",
  },
  {
    imgName: "Ai2.svg",
    title: "AI-Powered Communication Engine",
    description: "Built on advanced AI models that process speech, text, and gestures with exceptional accuracy. Neural networks continuously learn from diverse signing styles, accents, and contexts—ensuring smarter, faster, and more natural communication every time you use the platform.",
    tags: ["Deep Learning", "Real-time Processing", "Continuous Learning"],
    icon: <TbBrain />,
    statValue: "99.5%",
    statLabel: "Accuracy",
    users: "All",
    rating: 5.0,
    speed: "0.2s",
  },
];

const allAssets = import.meta.glob("../assets/**/*.{svg,png,jpg,jpeg}", { eager: true, import: "default" });

export default function Features() {
  const { themeColor } = useTheme();
  const gridColor = themeColor === "midnight-blue" ? "rgba(99, 102, 241, 0.1)" : "rgba(168, 85, 247, 0.1)";
  const themeFolder = themeColor === "midnight-blue" ? "blue" : "purple";

  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationFrameRef = useRef(null);

  const [hoveredCard, setHoveredCard] = useState(null);
      const [cardAnimations, setCardAnimations] = useState([]);
  
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

  // Initialize random animation delays and durations for each card
  useEffect(() => {
    const animations = features.map(() => ({
      floatDelay: Math.random() * 2,
      floatDuration: 3 + Math.random() * 2,
      floatAmplitude: 8 + Math.random() * 7,
      rotateDelay: Math.random() * 3,
      rotateDuration: 4 + Math.random() * 3,
      rotateAmplitude: 2 + Math.random() * 3,
      scaleDelay: Math.random() * 4,
      scaleDuration: 5 + Math.random() * 3,
      pulseDelay: Math.random() * 2,
      pulseDuration: 2 + Math.random() * 1.5,
    }));
    setCardAnimations(animations);
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

  
  const cardVariants = {
    hidden: { opacity: 0, y: 50, scale: 0.9 },
    visible: (i) => ({
      opacity: 1,
      y: 0,
      scale: 1,
      transition: { 
        delay: i * 0.1,
        duration: 0.5,
        ease: [0.22, 1, 0.36, 1]
      }
    }),
    exit: (i) => ({
      opacity: 0,
      y: -30,
      scale: 0.9,
      transition: { 
        delay: i * 0.05,
        duration: 0.3 
      }
    })
  };

  // Floating animation for continuous movement
  const getFloatingAnimation = (index) => {
    const anim = cardAnimations[index % cardAnimations.length];
    if (!anim) return {};
    
    return {
      y: [0, -anim.floatAmplitude, 0, anim.floatAmplitude, 0],
      rotate: [0, anim.rotateAmplitude, -anim.rotateAmplitude, anim.rotateAmplitude, 0],
      transition: {
        y: {
          duration: anim.floatDuration,
          delay: anim.floatDelay,
          repeat: Infinity,
          ease: "easeInOut",
        },
        rotate: {
          duration: anim.rotateDuration,
          delay: anim.rotateDelay,
          repeat: Infinity,
          ease: "easeInOut",
        }
      }
    };
  };

  // Pulse animation for continuous scaling
  const getPulseAnimation = (index) => {
    const anim = cardAnimations[index % cardAnimations.length];
    if (!anim) return {};
    
    return {
      scale: [1, 1.02, 1],
      transition: {
        duration: anim.pulseDuration,
        delay: anim.pulseDelay,
        repeat: Infinity,
        ease: "easeInOut",
      }
    };
  };

  // Subtle shadow animation
  const getShadowAnimation = (index) => {
    const anim = cardAnimations[index % cardAnimations.length];
    if (!anim) return {};
    
    return {
      boxShadow: [
        "0 20px 40px -15px rgba(139, 92, 246, 0.15)",
        "0 25px 50px -15px rgba(139, 92, 246, 0.25)",
        "0 20px 40px -15px rgba(139, 92, 246, 0.15)",
      ],
      transition: {
        duration: anim.pulseDuration,
        delay: anim.pulseDelay,
        repeat: Infinity,
        ease: "easeInOut",
      }
    };
  };

  return (
    <div id="features"
      className="
        relative w-full py-24 px-6 lg:px-12 overflow-hidden
        bg-gradient-to-br 
        from-gray-50 via-white to-primary-50/60
        dark:from-primary-bg-1 dark:via-primary-bg-2 dark:to-primary-bg-3
        transition-all duration-700
      "
    >
      {/* Background Effects */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none opacity-60 z-0"
      />

      <div className="absolute inset-0 opacity-40 dark:opacity-60 pointer-events-none z-0">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(90deg, ${gridColor} 1px, transparent 1px),
            linear-gradient(180deg, ${gridColor} 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px'
        }} />
      </div>

      <motion.div
        className="absolute top-20 left-20 w-[500px] h-[500px] bg-primary-500/10 rounded-full blur-[120px] z-0 pointer-events-none"
        animate={{ x: [0, 80, 0], y: [0, -80, 0] }}
        transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
      />
      <motion.div
        className="absolute bottom-20 right-20 w-[500px] h-[500px] bg-primary-600/10 rounded-full blur-[120px] z-0 pointer-events-none"
        animate={{ x: [0, -80, 0], y: [0, 80, 0] }}
        transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
      />

      {/* Header Section */}
      <div className="relative z-10 max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
          className="text-center mb-16"
        >
          <motion.div whileHover={{ scale: 1.05 }} className="inline-flex items-center gap-3 px-6 py-3 rounded-full bg-gradient-to-r from-primary-500/20 via-primary-400/10 to-primary-300/20 border-2 border-primary-300/40 dark:border-primary-600/40 backdrop-blur-xl shadow-2xl shadow-primary-500/20 relative overflow-hidden group mb-8">
            <motion.div animate={{ rotate: 360 }} transition={{ duration: 3, repeat: Infinity, ease: "linear" }} className="p-1 rounded-full bg-gradient-to-r from-primary-500 to-primary-600">
              <FaCrown className="text-white text-sm" />
            </motion.div>
            <span className="text-sm font-extrabold bg-gradient-to-r from-primary-700 via-primary-600 to-primary-500 dark:from-primary-400 dark:via-primary-300 dark:to-primary-200 bg-clip-text text-transparent">POWERED BY LINGUASIGN AI</span>
            <TbSparkles className="text-primary-500 text-lg" />
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
          </motion.div>

          <h1 className="font-black text-4xl sm:text-5xl lg:text-[53px] leading-tight mb-6">
            <span className="block text-gray-900 dark:text-white">Empower Communication</span>
            <span className="block bg-gradient-to-r from-primary-700 via-primary-600 to-primary-500 bg-clip-text text-transparent animate-gradient">With Sign Language AI</span>
          </h1>
          
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Our platform bridges the gap between spoken and sign languages using state-of-the-art recognition technology.
          </p>

          <div className="flex items-center justify-center gap-8 mt-10">
            <div className="w-12 h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent rounded-full" />
            <motion.div animate={{ rotate: 360 }} transition={{ duration: 20, repeat: Infinity, ease: "linear" }} className="w-6 h-6 rounded-full border-2 border-primary-400/50" />
            <div className="w-12 h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent rounded-full" />
          </div>
        </motion.div>

        
        {/* Continuous Auto-Scrolling Carousel */}
        <div className="relative z-10 w-full overflow-hidden py-12 px-0 mt-8 mask-image-linear-edges">
          
          <motion.div
            className="flex gap-8 w-max px-4"
            animate={{ x: ["0%", "-50%"] }}
            transition={{ 
              duration: 120, 
              repeat: Infinity, 
              ease: "linear",
            }}
          >

            {[...features, ...features].map((feature, index) => {
              const cardAnim = cardAnimations[index % cardAnimations.length];
              
              return (
                <motion.div
                  key={`${feature.title}-${index}`}
                  custom={index}
                  variants={cardVariants}
                  initial="hidden"
                  whileInView="visible"
                  viewport={{ once: true, margin: "100px" }}
                  onMouseEnter={() => setHoveredCard(index)}
                  onMouseLeave={() => setHoveredCard(null)}
                  style={{
                    boxShadow: "0 20px 40px -15px rgba(139, 92, 246, 0.15)",
                  }}
                  whileHover={{
                    scale: 1.02,
                    transition: { type: "spring", stiffness: 400, damping: 25 }
                  }}
                  className="w-[320px] md:w-[350px] shrink-0 group relative bg-white/80 dark:bg-white/5 backdrop-blur-xl rounded-3xl border-2 border-primary-200/50 dark:border-primary-800/50 shadow-xl overflow-hidden flex flex-col transition-all duration-300 hover:border-primary-300 dark:hover:border-primary-600 cursor-pointer"
                >
                  <div className="h-full flex flex-col">
                    {/* Animated Border Glow */}
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-primary-600 to-primary-400 rounded-3xl opacity-0 group-hover:opacity-30 transition-opacity duration-500 blur-xl" />
                    
                    {/* Card Hover Gradients */}
                    <div className="absolute inset-0 bg-gradient-to-br from-primary-400 to-primary-600 opacity-0 group-hover:opacity-[0.08] transition-opacity duration-700" />
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
                    
                    {/* Corner Accents */}
                    <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-br from-primary-400 to-primary-600 opacity-0 group-hover:opacity-20 rounded-bl-3xl transition-all duration-500" />
                    <div className="absolute bottom-0 left-0 w-24 h-24 bg-gradient-to-tr from-primary-400 to-primary-600 opacity-0 group-hover:opacity-20 rounded-tr-3xl transition-all duration-500" />

                    <div className="relative p-5 pb-4 flex-grow flex flex-col z-10 h-full">
                      
                      <div className="flex justify-between items-start mb-6">
                        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary-100 to-primary-50 dark:from-primary-900/50 dark:to-primary-800/50 border border-primary-200 dark:border-primary-700 flex items-center justify-center shadow-inner group-hover:scale-110 transition-transform duration-500">
                          <span className="text-3xl font-black bg-gradient-to-br from-primary-600 to-primary-400 bg-clip-text text-transparent">
                            {(index % features.length) + 1}
                          </span>
                        </div>
                        <div className="px-3 py-1.5 rounded-full bg-primary-100/50 dark:bg-primary-900/30 border border-primary-200 dark:border-primary-700 backdrop-blur-md">
                          <span className="text-xs font-bold text-primary-700 dark:text-primary-300">AI Powered</span>
                        </div>
                      </div>

                      {/* Feature Image Container */}
                      <div className="relative pt-6 pb-2 px-6 flex items-center justify-center">
                        <motion.div 
                          className="absolute inset-0 bg-primary-500/20 blur-3xl rounded-full scale-75 opacity-0 group-hover:opacity-100 transition-opacity duration-700"
                          animate={hoveredCard === index ? { scale: [0.75, 1, 0.75] } : {}}
                          transition={{ duration: 2, repeat: Infinity }}
                        />
                        <motion.img
                          src={allAssets[`../assets/${themeFolder}/${feature.imgName}`]}
                          alt={feature.title}
                          className="relative z-10 w-28 h-28 object-contain drop-shadow-xl"
                          animate={hoveredCard === index ? { scale: 1.1, rotate: [0, -3, 3, 0] } : { scale: 1 }}
                          transition={{ duration: 0.5 }}
                        />
                      </div>

                      {/* Feature Text */}
                      <div className="relative z-10 p-4 pt-2 flex-grow flex flex-col">
                        <h2 className="text-xl font-black mb-3 bg-gradient-to-r from-primary-700 via-primary-600 to-primary-500 dark:from-primary-300 dark:via-primary-200 dark:to-primary-100 bg-clip-text text-transparent">
                          {feature.title}
                        </h2>
                        
                        <p className="text-gray-600 dark:text-gray-300 leading-relaxed text-sm mb-4">
                          {feature.description.length > 100 
                            ? `${feature.description.substring(0, 100)}...` 
                            : feature.description}
                        </p>

                        {/* Stats Grid */}
                        <div className="grid grid-cols-4 gap-2 mb-4 p-3 rounded-xl bg-primary-50/50 dark:bg-primary-900/20 border border-primary-200/30 dark:border-primary-700/30">
                          <motion.div className="text-center" whileHover={{ scale: 1.05 }}>
                            <div className="text-sm font-black bg-gradient-to-r from-primary-600 to-primary-500 bg-clip-text text-transparent">
                              {feature.statValue}
                            </div>
                            <div className="text-[9px] text-primary-600 dark:text-primary-400 font-medium">{feature.statLabel}</div>
                          </motion.div>
                          <motion.div className="text-center" whileHover={{ scale: 1.05 }}>
                            <div className="flex items-center justify-center gap-0.5">
                              <FaStar className="text-primary-500 text-[10px]" />
                              <span className="text-sm font-black text-gray-800 dark:text-white">{feature.rating}</span>
                            </div>
                            <div className="text-[9px] text-primary-600 dark:text-primary-400 font-medium">Rating</div>
                          </motion.div>
                          <motion.div className="text-center" whileHover={{ scale: 1.05 }}>
                            <div className="text-sm font-black text-gray-800 dark:text-white">
                              {feature.users}
                            </div>
                            <div className="text-[9px] text-primary-600 dark:text-primary-400 font-medium">Users</div>
                          </motion.div>
                          <motion.div className="text-center" whileHover={{ scale: 1.05 }}>
                            <div className="flex items-center justify-center gap-0.5">
                              <TbClock className="text-primary-500 text-[10px]" />
                              <span className="text-sm font-black text-gray-800 dark:text-white">{feature.speed}</span>
                            </div>
                            <div className="text-[9px] text-primary-600 dark:text-primary-400 font-medium">Response</div>
                          </motion.div>
                        </div>

                        {/* Tags */}
                        <div className="flex flex-wrap gap-2 mt-auto">
                          {feature.tags.map((tag, i) => (
                            <motion.span
                              key={i}
                              initial={{ opacity: 0, scale: 0.8 }}
                              whileInView={{ opacity: 1, scale: 1 }}
                              transition={{ delay: i * 0.05 }}
                              whileHover={{ scale: 1.05 }}
                              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-primary-100/80 dark:bg-primary-800/30 text-xs font-semibold text-primary-700 dark:text-primary-300 border border-primary-200/50 dark:border-primary-700/50"
                            >
                              <BsCheckCircleFill className="text-[8px] text-primary-500" />
                              {tag}
                            </motion.span>
                          ))}
                        </div>
                      </div>

                      {/* Animated Bottom Line */}
                      <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-primary-400 via-primary-500 to-primary-600 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-700 origin-left rounded-b-3xl" />
                      
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </motion.div>
        </div>
      </div>

      <style>{`
        .mask-image-linear-edges {
          -webkit-mask-image: linear-gradient(to right, transparent, black 5%, black 95%, transparent);
          mask-image: linear-gradient(to right, transparent, black 5%, black 95%, transparent);
        }
        @keyframes gradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .animate-gradient {
          background-size: 200% auto;
          animation: gradient 4s linear infinite;
        }
      `}</style>
    </div>
  );
}


