import React, { useState, useEffect, useRef } from "react";
import { useTheme } from "../context/ThemeContext";
import { 
  FaFacebookF, 
  FaTwitter, 
  FaLinkedinIn, 
  FaInstagram, 
  FaEnvelope, 
  FaUser, 
  FaPhone,
  FaMapMarkerAlt,
  FaPaperPlane,
  FaRegClock
} from "react-icons/fa";
import { motion, AnimatePresence } from "framer-motion";
import { TbSparkles, TbMailForward, TbMessage2, TbHandLoveYou } from "react-icons/tb";
import { FiSend } from "react-icons/fi";
import { BsArrowRight } from "react-icons/bs";

export default function Contact() {
  const { themeColor } = useTheme();
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: "",
  });
  const [focusedInput, setFocusedInput] = useState(null);
  const [success, setSuccess] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isDark, setIsDark] = useState(false);

  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationFrameRef = useRef(null);

  // Detect dark mode
  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains("dark"));
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });

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
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      window.removeEventListener('resize', handleResize);
    };
  }, [isDark, themeColor]);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsLoading(true);
    
    setTimeout(() => {
      console.log("Message sent:", formData); 
      setSuccess(true);
      setFormData({ name: "", email: "", message: "" });
      setIsLoading(false);
      setTimeout(() => setSuccess(false), 4000);
    }, 1500);
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

  const socialLinks = [
    { icon: FaFacebookF, href: "https://facebook.com", color: "#3b5998", label: "Facebook" },
    { icon: FaTwitter, href: "https://twitter.com", color: "#1da1f2", label: "Twitter" },
    { icon: FaLinkedinIn, href: "https://linkedin.com", color: "#0077b5", label: "LinkedIn" },
    { icon: FaInstagram, href: "https://instagram.com", color: "#e4405f", label: "Instagram" },
  ];

  const contactInfo = [
    { icon: FaEnvelope, title: "Email Support", value: "support@linguasign.io", color: "from-primary-400 to-primary-600" },
    { icon: FaPhone, title: "Phone Support", value: "+1 (555) 123-4567", color: "from-primary-500 to-primary-700" },
    { icon: FaMapMarkerAlt, title: "Headquarters", value: "San Francisco, CA", color: "from-primary-600 to-primary-800" },
    { icon: FaRegClock, title: "Business Hours", value: "Mon-Fri, 9AM-6PM PST", color: "from-primary-700 to-primary-900" },
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

      {/* Header */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          className="text-center mb-20"
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
              Get in Touch
            </span>
            <TbSparkles className="text-primary-500 ml-1" />
            <div className="absolute -inset-1 bg-gradient-to-r from-primary-500/0 via-primary-400/10 to-primary-500/0 group-hover:via-primary-400/20 transition-all duration-500" />
          </motion.div>

          <motion.h1
            variants={fadeUp}
            className="font-extrabold text-4xl sm:text-5xl lg:text-[53px] leading-tight mb-6"
          >
            <span className="block text-gray-900 dark:text-white">
              Let's Start a
            </span>
            <span className="block bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 bg-clip-text text-transparent">
              Conversation
            </span>
          </motion.h1>
          
          <motion.p
            variants={fadeUp}
            className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto"
          >
            Have questions, ideas, or want to collaborate? We're here to help you bridge the communication gap.
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

        {/* Contact Grid */}
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Contact Information Cards */}
          <div className="lg:col-span-1 space-y-6">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={staggerContainer}
              className="grid grid-cols-2 gap-4"
            >
              {contactInfo.map((info, index) => (
                  <motion.div
                    key={index}
                    variants={scaleIn}
                    whileHover={{ y: -5, scale: 1.05 }}
                    className="group relative cursor-pointer h-full"
                  >
                    <div className={`absolute -inset-0.5 bg-gradient-to-br ${info.color} rounded-3xl blur opacity-0 group-hover:opacity-60 transition-opacity duration-500`} />
                    <div className="relative h-full p-4 sm:p-5 flex flex-col items-center text-center justify-center rounded-3xl bg-white/70 dark:bg-white/5 backdrop-blur-xl border border-primary-200/50 dark:border-primary-500/20 shadow-lg shadow-primary-100/20 dark:shadow-primary-900/20 group-hover:bg-white/90 dark:group-hover:bg-white/10 transition-colors duration-300 overflow-hidden">
                      <info.icon className="absolute -bottom-4 -right-4 text-7xl sm:text-8xl text-gray-400 dark:text-gray-500 opacity-[0.06] group-hover:opacity-[0.12] group-hover:scale-110 group-hover:-rotate-12 transition-all duration-500 pointer-events-none" />
                      
                      <div className={`p-3 sm:p-4 rounded-2xl bg-gradient-to-br ${info.color} shadow-lg mb-3 group-hover:-translate-y-1 transition-transform duration-300`}>
                        <info.icon className="text-2xl sm:text-3xl text-white drop-shadow-md group-hover:scale-110 transition-transform duration-300" />
                      </div>
                      <h3 className="font-bold text-gray-900 dark:text-white text-sm mb-1 z-10">{info.title}</h3>
                      <p className="text-gray-600 dark:text-gray-300 text-xs font-medium z-10">{info.value}</p>
                    </div>
                  </motion.div>
                ))}
            </motion.div>

            {/* Social Links */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: 0.4 }}
            >
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <TbMessage2 className="text-primary-600" />
                Follow Our Journey
              </h3>
              <div className="flex gap-3">
                {socialLinks.map((link, index) => (
                  <motion.a
                    key={index}
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    whileHover={{ scale: 1.1, y: -3 }}
                    whileTap={{ scale: 0.95 }}
                    className="relative group"
                  >
                    <div className="relative p-3 rounded-full bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border border-gray-300/50 dark:border-gray-700/50 shadow-lg hover:shadow-xl transition-all duration-300"
                      style={{ color: link.color }}
                    >
                      <link.icon className="text-lg" />
                    </div>
                  </motion.a>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Contact Form */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="lg:col-span-2"
          >
            <div className="relative group">
              <div className="relative p-8 rounded-2xl bg-white/70 dark:bg-white/5 backdrop-blur-xl border border-primary-200/50 dark:border-primary-500/20 shadow-xl hover:shadow-2xl transition-all duration-300">
                <div className="flex items-center gap-3 mb-8">
                  <div className="p-2 rounded-xl bg-gradient-to-br from-primary-500/20 to-primary-400/20">
                    <TbMailForward className="text-2xl text-primary-600 dark:text-primary-400" />
                  </div>
                  <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
                    Send Us a Message
                  </h3>
                </div>

                <form onSubmit={handleSubmit} className="space-y-6">
                  {/* Name Input */}
                  <div className="relative">
                    <input
                      type="text"
                      name="name"
                      id="name"
                      value={formData.name}
                      onChange={handleChange}
                      onFocus={() => setFocusedInput('name')}
                      onBlur={() => setFocusedInput(null)}
                      required
                      className="peer w-full px-4 pt-6 pb-2 rounded-xl bg-white/40 dark:bg-gray-900/40 border border-gray-300/50 dark:border-gray-700/50 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-transparent transition-all duration-300"
                      placeholder=" "
                    />
                    <label 
                      htmlFor="name" 
                      className={`absolute left-4 top-4 text-gray-500 dark:text-gray-400 transition-all duration-300 pointer-events-none flex items-center gap-2
                        ${(focusedInput === 'name' || formData.name) ? 'text-xs -translate-y-3 text-primary-600 dark:text-primary-400 font-semibold' : 'text-base translate-y-0'}
                      `}
                    >
                      <FaUser size={14} />
                      Your Name
                    </label>
                  </div>

                  {/* Email Input */}
                  <div className="relative">
                    <input
                      type="email"
                      name="email"
                      id="email"
                      value={formData.email}
                      onChange={handleChange}
                      onFocus={() => setFocusedInput('email')}
                      onBlur={() => setFocusedInput(null)}
                      required
                      className="peer w-full px-4 pt-6 pb-2 rounded-xl bg-white/40 dark:bg-gray-900/40 border border-gray-300/50 dark:border-gray-700/50 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-transparent transition-all duration-300"
                      placeholder=" "
                    />
                    <label 
                      htmlFor="email" 
                      className={`absolute left-4 top-4 text-gray-500 dark:text-gray-400 transition-all duration-300 pointer-events-none flex items-center gap-2
                        ${(focusedInput === 'email' || formData.email) ? 'text-xs -translate-y-3 text-primary-600 dark:text-primary-400 font-semibold' : 'text-base translate-y-0'}
                      `}
                    >
                      <FaEnvelope size={14} />
                      Your Email
                    </label>
                  </div>

                  {/* Message Input */}
                  <div className="relative">
                    <textarea
                      name="message"
                      id="message"
                      value={formData.message}
                      onChange={handleChange}
                      onFocus={() => setFocusedInput('message')}
                      onBlur={() => setFocusedInput(null)}
                      required
                      rows="4"
                      className="peer w-full px-4 pt-6 pb-2 rounded-xl bg-white/40 dark:bg-gray-900/40 border border-gray-300/50 dark:border-gray-700/50 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-transparent resize-none transition-all duration-300"
                      placeholder=" "
                    />
                    <label 
                      htmlFor="message" 
                      className={`absolute left-4 top-4 text-gray-500 dark:text-gray-400 transition-all duration-300 pointer-events-none flex items-center gap-2
                        ${(focusedInput === 'message' || formData.message) ? 'text-xs -translate-y-3 text-primary-600 dark:text-primary-400 font-semibold' : 'text-base translate-y-0'}
                      `}
                    >
                      Your Message
                    </label>
                  </div>

                  {/* Submit Button */}
                  <motion.button
                    type="submit"
                    disabled={isLoading}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="relative overflow-hidden w-full py-3.5 rounded-full bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 text-white font-bold text-base shadow-lg shadow-primary-500/30 hover:shadow-primary-500/50 transition-all duration-300 disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2 group"
                  >
                    <span className="relative z-10 flex items-center gap-2">
                      {isLoading ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                          Sending...
                        </>
                      ) : (
                        <>
                          Send Message
                          <FiSend size={16} className="group-hover:translate-x-1 transition-transform" />
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

                  {/* Success Message */}
                  <AnimatePresence>
                    {success && (
                      <motion.div
                        initial={{ opacity: 0, y: 10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 10, scale: 0.95 }}
                        className="p-4 rounded-xl bg-gradient-to-r from-green-500/90 to-emerald-500/90 backdrop-blur-md border border-green-400/50 shadow-xl"
                      >
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
                            <FaPaperPlane className="text-white" size={16} />
                          </div>
                          <div>
                            <p className="font-bold text-white">Message Sent!</p>
                            <p className="text-sm text-green-50">
                              Thank you for reaching out. We'll respond shortly.
                            </p>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </form>
              </div>
            </div>
          </motion.div>
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
