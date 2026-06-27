import React, { useEffect, useState, useRef } from "react";
import { motion } from "framer-motion";
import { useTheme } from "../context/ThemeContext";
import { 
  FaMicrophone, FaVideo, FaRobot, FaMobileAlt, FaHands, FaEye, 
  FaUsers, FaChartLine, FaGlobe, FaLightbulb, FaHeart, FaShieldAlt, 
  FaSyncAlt, FaBrain, FaRocket, FaAward, FaCrown, FaGem
} from "react-icons/fa";
import { FaLaptopCode, FaCode, FaUserAlt, FaCloud } from "react-icons/fa";
import { BsStars, BsLightningCharge, BsCheckCircleFill } from "react-icons/bs";
import { TbSparkles, TbHandLoveYou, TbHexagon } from "react-icons/tb";
import { GiArtificialIntelligence, GiHand } from "react-icons/gi";

export default function About() {
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationFrameRef = useRef(null);
  const [isDark, setIsDark] = useState(false);
  const [hoveredCard, setHoveredCard] = useState(null);
  const [expandedItem, setExpandedItem] = useState(null);
  const { themeColor } = useTheme();

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

  const fadeUp = {
    hidden: { opacity: 0, y: 40 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.8, ease: [0.22, 1, 0.36, 1] }
    }
  };

  const scaleIn = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: { 
      opacity: 1, 
      scale: 1,
      transition: { duration: 0.6, ease: "backOut" }
    }
  };

  const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.2 }
    }
  };

  const stats = [
    { icon: <FaUsers />, value: "500K+", label: "Active Users", gradient: "from-primary-500 to-primary-700" },
    { icon: <FaGlobe />, value: "50+", label: "Languages", gradient: "from-primary-600 to-primary-800" },
    { icon: <FaRobot />, value: "99%", label: "Accuracy", gradient: "from-primary-500 to-primary-700" },
    { icon: <BsLightningCharge />, value: "<100ms", label: "Response Time", gradient: "from-primary-600 to-primary-800" },
  ];

  const features = [
    { icon: <FaVideo />, text: "Video Translation", desc: "Upload any video and get instant sign language translation", gradient: "from-primary-500 to-primary-700" },
    { icon: <FaRobot />, text: "AI Chatbot", desc: "Practice sign language with our intelligent chatbot", gradient: "from-primary-600 to-primary-800" },
    { icon: <FaMobileAlt />, text: "Cross-Platform", desc: "Available on web, mobile, and desktop", gradient: "from-primary-500 to-primary-700" },
    { icon: <FaUsers />, text: "Multi-User", desc: "Real-time translation for group conversations", gradient: "from-primary-600 to-primary-800" },
    { icon: <FaGlobe />, text: "Global Support", desc: "100+ spoken languages supported", gradient: "from-primary-500 to-primary-700" },
    { icon: <FaShieldAlt />, text: "Enterprise Security", desc: "Bank-level encryption for all data", gradient: "from-primary-600 to-primary-800" },
  ];

  const team = [
    { name: "Shahd Mohamed", role: "AI & Frontend Engineer", icon: <FaCode />, expertise: ["React", "TensorFlow.js", "Framer Motion"], gradient: "from-primary-500 to-primary-700" },
    { name: "Demiana Ayman", role: "AI & Backend Engineer", icon: <FaLaptopCode />, expertise: ["Node.js", "Python", "AWS"], gradient: "from-primary-600 to-primary-800" },
    { name: "Kareem Reda", role: "AI Engineer", icon: <FaBrain />, expertise: ["PyTorch", "Computer Vision", "MLOps"], gradient: "from-primary-500 to-primary-700" },
    { name: "Yahya Aboamer", role: "AI Engineer", icon: <FaCode />, expertise: ["Deep Learning", "TensorFlow", "Kubernetes"], gradient: "from-primary-600 to-primary-800" },
    { name: "Mariam Hany", role: "AI Engineer", icon: <FaBrain />, expertise: ["Deep Learning", "Computer Vision", "TensorFlow"], gradient: "from-primary-500 to-primary-700" },
    { name: "Hussam Elsayed", role: "AI Engineer", icon: <FaUserAlt />, expertise: ["TensorFlow", "Data Science", "Docker"], gradient: "from-primary-600 to-primary-800" },
  ];

  return (
    <div className="relative w-full min-h-screen bg-gradient-to-br from-gray-50 via-white to-primary-50/60 dark:from-primary-bg-1 dark:via-primary-bg-2 dark:to-primary-bg-3 overflow-hidden selection:bg-primary-500 selection:text-white transition-all duration-700 py-24 px-4 sm:px-6 lg:px-8">
      
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

      {/* Header */}
      <div className="relative z-10 max-w-7xl mx-auto">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          className="text-center mb-16"
        >
          {/* Premium Badge - matching home page */}
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="inline-flex items-center gap-3 px-6 py-3 rounded-full bg-white/50 dark:bg-white/5 border border-primary-200/50 dark:border-primary-500/20 backdrop-blur-xl shadow-xl shadow-primary-500/5 relative overflow-hidden group mb-8"
          >
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
              className="p-1 rounded-full bg-gradient-to-r from-primary-500 to-primary-400"
            >
              <FaCrown className="text-white text-sm" />
            </motion.div>
            <span className="text-sm font-extrabold bg-gradient-to-r from-primary-700 via-primary-600 to-primary-500 dark:from-primary-400 dark:via-primary-300 dark:to-primary-200 bg-clip-text text-transparent uppercase tracking-wider">
              About LinguaSign
            </span>
            <TbSparkles className="text-primary-500 text-lg" />
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
          </motion.div>

          <div className="space-y-1 mb-6">
            <motion.h1 className="font-black text-5xl sm:text-6xl lg:text-[58px] leading-[1.1] tracking-tight">
              <span className="block text-gray-900 dark:text-white">
                Our Story
              </span>
              <motion.span
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
                Revolutionizing Communication
              </motion.span>
              <span className="block text-gray-900 dark:text-white">
                Through AI Innovation
              </span>
            </motion.h1>
          </div>
          
          <motion.p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Discover the vision, mission, and team behind LinguaSign — transforming communication through AI-powered sign language translation.
          </motion.p>

          {/* Decorative Elements */}
          <motion.div className="flex items-center justify-center gap-8 mt-10">
            <div className="w-12 h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent rounded-full" />
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="w-6 h-6 rounded-full border-2 border-primary-400/50"
            />
            <div className="w-12 h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent rounded-full" />
          </motion.div>
        </motion.div>

        {/* Stats Section */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={staggerContainer}
          className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-20"
        >
          {stats.map((stat, i) => (
            <motion.div
              key={i}
              variants={scaleIn}
              whileHover={{ scale: 1.05, y: -5 }}
              className="relative group"
            >
              <div className="absolute -inset-0.5 bg-gradient-to-r from-primary-500/30 to-primary-600/30 rounded-2xl blur opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              <div className="relative p-6 rounded-2xl bg-white/80 dark:bg-white/10 backdrop-blur-xl border-2 border-primary-200/50 dark:border-primary-500/20 shadow-xl hover:shadow-2xl transition-all duration-300 text-center">
                <div className={`p-3 rounded-xl bg-gradient-to-br ${stat.gradient} inline-flex mb-4 group-hover:scale-110 transition-transform`}>
                  <div className="text-white text-2xl">{stat.icon}</div>
                </div>
                <div className="text-3xl font-bold text-gray-900 dark:text-white mb-1">{stat.value}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">{stat.label}</div>
              </div></motion.div>
          ))}
        </motion.div>

        {/* Mission & Vision Section */}
        <div className="grid lg:grid-cols-2 gap-8 mb-20">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeUp}
            className="relative group"
          >
            <div className="absolute -inset-0.5 bg-gradient-to-r from-primary-500/30 to-primary-600/30 rounded-2xl blur opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <div className="relative p-8 rounded-2xl bg-white/80 dark:bg-white/10 backdrop-blur-xl border-2 border-primary-200/50 dark:border-primary-500/20 shadow-xl hover:shadow-2xl transition-all duration-300">
              <div className="absolute inset-0 bg-gradient-to-br from-primary-400 to-primary-600 opacity-0 group-hover:opacity-5 transition-opacity duration-700" />
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
              <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-br from-primary-400 to-primary-600 opacity-0 group-hover:opacity-10 rounded-bl-3xl transition-all duration-500" />
              
              <div className="flex items-center gap-4 mb-6">
                <div className="p-3 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 text-white group-hover:scale-110 transition-transform">
                  <FaHeart className="text-2xl" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white">Our Mission</h3>
              </div>
              <p className="text-lg leading-relaxed text-gray-700 dark:text-gray-300">
                To create a world where communication barriers cease to exist, empowering every individual to connect, learn, and thrive through accessible AI technology.
              </p>
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-primary-400 to-primary-600 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-700 origin-left" />
            </div></motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeUp}
            transition={{ delay: 0.2 }}
            className="relative group"
          >
            <div className="absolute -inset-0.5 bg-gradient-to-r from-primary-500/30 to-primary-600/30 rounded-2xl blur opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <div className="relative p-8 rounded-2xl bg-white/80 dark:bg-white/10 backdrop-blur-xl border-2 border-primary-200/50 dark:border-primary-500/20 shadow-xl hover:shadow-2xl transition-all duration-300">
              <div className="absolute inset-0 bg-gradient-to-br from-primary-500 to-primary-700 opacity-0 group-hover:opacity-5 transition-opacity duration-700" />
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
              <div className="absolute top-0 left-0 w-24 h-24 bg-gradient-to-tr from-primary-500 to-primary-700 opacity-0 group-hover:opacity-10 rounded-br-3xl transition-all duration-500" />
              
              <div className="flex items-center gap-4 mb-6">
                <div className="p-3 rounded-xl bg-gradient-to-br from-primary-600 to-primary-800 text-white group-hover:scale-110 transition-transform">
                  <FaEye className="text-2xl" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white">Our Vision</h3>
              </div>
              <p className="text-lg leading-relaxed text-gray-700 dark:text-gray-300">
                To become the global standard for AI-powered sign language translation, transforming how humanity connects across languages, cultures, and abilities.
              </p>
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-primary-500 to-primary-700 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-700 origin-right" />
            </div></motion.div>
        </div>

        {/* Features Section */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={staggerContainer}
          className="mb-20"
        >
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4 flex items-center justify-center gap-3">
              <BsStars className="text-primary-500" />
              Powerful Features
              <BsLightningCharge className="text-primary-500" />
            </h2>
            <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              Discover what makes LinguaSign the premier sign language translation platform
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, i) => (
              <motion.div
                key={i}
                variants={scaleIn}
                whileHover={{ scale: 1.03, y: -5 }}
                onMouseEnter={() => setHoveredCard(i)}
                onMouseLeave={() => setHoveredCard(null)}
                onClick={() => setExpandedItem(expandedItem === i ? null : i)}
                className={`relative group cursor-pointer transition-all duration-300 ${
                  hoveredCard === i ? 'z-10' : ''
                }`}
              >
                <div className="absolute -inset-0.5 bg-gradient-to-r from-primary-500/30 to-primary-600/30 rounded-2xl blur opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                <div className={`relative p-6 rounded-2xl bg-white/80 dark:bg-white/10 backdrop-blur-xl border-2 transition-all duration-300 ${
                  hoveredCard === i || expandedItem === i
                    ? 'border-primary-400 shadow-2xl shadow-primary-500/20 bg-white/95 dark:bg-white/15' 
                    : 'border-primary-200/50 dark:border-primary-500/20 shadow-lg'
                }`}>
                  <div className="absolute inset-0 bg-gradient-to-br from-primary-400 to-primary-600 opacity-0 group-hover:opacity-5 transition-opacity duration-700" />
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
                  <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-br from-primary-400 to-primary-600 opacity-0 group-hover:opacity-10 rounded-bl-2xl transition-all duration-500" />
                  
                  <div className="flex items-start gap-4">
                    <div className={`p-3 rounded-xl bg-gradient-to-br ${feature.gradient} text-white group-hover:scale-110 transition-transform`}>
                      {feature.icon}
                    </div>
                    <div>
                      <h3 className="font-bold text-gray-900 dark:text-white mb-2">{feature.text}</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">{feature.desc}</p>
                    </div>
                  </div>
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-primary-400 to-primary-600 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-700 origin-left" />
                </div></motion.div>
            ))}
          </div></motion.div>

        {/* Team Section */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={staggerContainer}
          className="mb-16"
        >
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4 flex items-center justify-center gap-3">
              <FaBrain className="text-primary-500" />
              Meet Our Team
              <FaRocket className="text-primary-500" />
            </h2>
            <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              The brilliant minds behind LinguaSign's groundbreaking technology
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {team.map((person, i) => (
              <motion.div
                key={i}
                variants={scaleIn}
                whileHover={{ scale: 1.03, y: -5 }}
                className="relative group"
              >
                <div className="absolute -inset-0.5 bg-gradient-to-r from-primary-500/30 to-primary-600/30 rounded-2xl blur opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                <div className="relative p-6 rounded-2xl bg-white/80 dark:bg-white/10 backdrop-blur-xl border-2 border-primary-200/50 dark:border-primary-500/20 shadow-xl hover:shadow-2xl transition-all duration-300">
                  <div className="absolute inset-0 bg-gradient-to-br from-primary-400 to-primary-600 opacity-0 group-hover:opacity-5 transition-opacity duration-700" />
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
                  
                  <div className="flex flex-col items-center text-center">
                    <div className={`w-20 h-20 rounded-xl bg-gradient-to-br ${person.gradient} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform shadow-lg`}>
                      <div className="text-white text-3xl">{person.icon}</div>
                    </div>
                    <h3 className="font-bold text-xl text-gray-900 dark:text-white mb-1">{person.name}</h3>
                    <p className="text-sm text-primary-600 dark:text-primary-400 mb-3">{person.role}</p>
                    <div className="flex flex-wrap justify-center gap-2">
                      {person.expertise.map((skill, j) => (
                        <span key={j} className="px-2 py-1 rounded-full text-xs font-medium bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300">
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-primary-400 to-primary-600 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-700" />
                </div></motion.div>
            ))}
          </div></motion.div>
      </div>

      <style jsx>{`
        @keyframes gradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .animate-gradient {
          background-size: 200% auto;
          animation: gradient 8s ease infinite;
        }
      `}</style>
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
