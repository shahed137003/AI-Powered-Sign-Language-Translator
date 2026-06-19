import React, { useState, useEffect, useRef, useContext, useCallback, useMemo } from "react";
import { useTheme } from "../context/ThemeContext";
import { FaFacebookF, FaTwitter, FaLinkedinIn, FaInstagram, FaChevronRight, FaEnvelope, FaHeart, FaHands, FaUserFriends, FaGlobe, FaCrown } from "react-icons/fa";
import { BsLightningFill, BsStars, BsRobot } from "react-icons/bs";
import { TbSparkles, TbMessageChatbot } from "react-icons/tb";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";

export default function Footer() {
  const { themeColor } = useTheme();
  const [email, setEmail] = useState("");
  const [isSubscribed, setIsSubscribed] = useState(false);
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

  // Particle system for footer
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

    particlesRef.current = Array.from({ length: 60 }).map(() => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      size: Math.random() * 2 + 0.5,
      speedX: Math.random() * 0.3 - 0.15,
      speedY: Math.random() * 0.3 - 0.15,
      color: colors[Math.floor(Math.random() * colors.length)],
      opacity: Math.random() * 0.4 + 0.1,
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
        ctx.fillStyle = particle.color + Math.floor(particle.opacity * 255).toString(16).padStart(2, '0');
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
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      window.removeEventListener('resize', handleResize);
    };
  }, [isDark, themeColor]);

  const handleSubscribe = (e) => {
    e.preventDefault();
    if (email) {
      setIsSubscribed(true);
      setEmail("");
      setTimeout(() => setIsSubscribed(false), 3000);
    }
  };

  const socialLinks = [
    { icon: FaFacebookF, href: "https://facebook.com", color: "#1877F2" },
    { icon: FaTwitter, href: "https://twitter.com", color: "#1DA1F2" },
    { icon: FaLinkedinIn, href: "https://linkedin.com", color: "#0077B5" },
    { icon: FaInstagram, href: "https://instagram.com", color: "#E4405F" },
  ];

  const quickLinks = [
    { name: "Home", href: "/", icon: <BsStars className="text-sm" /> },
    { name: "Features", href: "/#features", icon: <BsLightningFill className="text-sm" /> },
      { name: "Help Guide", href: "/#guide", icon: <TbMessageChatbot className="text-sm" /> },
    { name: "Translate", href: "/translate", icon: <FaHands className="text-sm" /> },
    { name: "Chatting", href: "/chat", icon: <TbMessageChatbot className="text-sm" /> },
    { name: "Contact", href: "/contactus", icon: <FaEnvelope className="text-sm" /> },
  ];

  const legalLinks = [
    { name: "Terms of Service", href: "/terms" },
    { name: "Privacy Policy", href: "/privacy" },
    { name: "Cookie Policy", href: "/cookies" },
    { name: "Accessibility", href: "/accessibility" },
    { name: "Support Center", href: "/support" },
    { name: "Documentation", href: "/docs" },
  ];

  return (
    <footer className="relative w-full bg-gradient-to-br from-gray-50 via-white to-primary-50/60 dark:from-primary-bg-1 dark:via-primary-bg-2 dark:to-primary-bg-3 text-gray-800 dark:text-gray-300 pt-20 pb-8 px-4 sm:px-6 lg:px-8 border-t-2 border-primary-200/30 dark:border-primary-900/30 overflow-hidden transition-all duration-700">
      
      {/* Premium Canvas Particles */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none opacity-40"
      />

      {/* Premium Geometric Grid */}
      <div className="absolute inset-0 opacity-30 dark:opacity-40 pointer-events-none">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(90deg, rgba(168, 85, 247, 0.08) 1px, transparent 1px),
            linear-gradient(180deg, rgba(168, 85, 247, 0.08) 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px'
        }} />
      </div>

      {/* Animated gradient orbs - matching home page */}
      <motion.div
        className="absolute bottom-20 left-20 w-[400px] h-[400px] bg-primary-600/10 rounded-full blur-[120px]"
        animate={{
          x: [0, 50, 0],
          y: [0, -50, 0],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "linear",
        }}
      />
      <motion.div
        className="absolute top-20 right-20 w-[400px] h-[400px] bg-primary-400/10 rounded-full blur-[120px]"
        animate={{
          x: [0, -50, 0],
          y: [0, 50, 0],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "linear",
        }}
      />

      <div className="relative z-10 max-w-7xl mx-auto">
        
        {/* Premium Badge - matching home page */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="flex justify-center mb-12"
        >
          <div className="inline-flex items-center gap-3 px-6 py-3 rounded-full bg-gradient-to-r from-primary-500/20 via-primary-400/10 to-primary-300/20 border-2 border-primary-300/40 dark:border-primary-600/40 backdrop-blur-xl shadow-2xl shadow-primary-500/20 relative overflow-hidden group">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
              className="p-1 rounded-full bg-gradient-to-r from-primary-500 to-primary-400"
            >
              <FaCrown className="text-white text-sm" />
            </motion.div>
            <span className="text-sm font-extrabold bg-gradient-to-r from-primary-700 via-primary-600 to-primary-500 dark:from-primary-400 dark:via-primary-300 dark:to-primary-200 bg-clip-text text-transparent">
              BRIDGING COMMUNICATION GAPS
            </span>
            <TbSparkles className="text-primary-500 text-lg" />
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
          </div>
        </motion.div>

        {/* Main Footer Content */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12 mb-12">
          
          {/* 1. Brand Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="space-y-6"
          >
            {/* Logo */}
            <div className="flex items-center gap-3">
              <motion.div
                whileHover={{ scale: 1.05, rotate: 5 }}
                className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary-custom-1 to-primary-custom-2 flex items-center justify-center shadow-lg shadow-primary-500/30"
              >
                <FaHands className="text-white text-xl" />
              </motion.div>
              <h3 className="text-2xl font-black">
                <span className="bg-gradient-to-r from-primary-custom-1 via-primary-custom-2 to-primary-custom-3 dark:from-primary-custom-1 dark:to-primary-custom-2 bg-clip-text text-transparent">
                  LinguaSign
                </span>
              </h3>
            </div>
            
            <p className="text-gray-600 dark:text-gray-400 leading-relaxed text-sm">
              Breaking communication barriers with AI-powered sign language translation. 
              Making the world accessible for everyone through cutting-edge technology.
            </p>
            
            {/* Tagline with gradient */}
            <div className="flex items-center gap-2 text-primary-600 dark:text-primary-400 text-sm font-semibold">
              <TbSparkles className="text-lg animate-pulse" />
              <span>Inclusive Technology for All</span>
            </div>
          </motion.div>

          {/* 2. Quick Links */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1, duration: 0.6 }}
          >
            <h4 className="text-xl font-black mb-6 text-gray-900 dark:text-white flex items-center gap-2">
              <div className="w-1 h-6 bg-gradient-to-b from-primary-500 to-primary-400 rounded-full" />
              Navigation
            </h4>
            <ul className="space-y-3">
              {quickLinks.map((link, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.05 }}
                >
                  <Link to={link.href}
                    className="flex items-center group text-gray-700 dark:text-gray-400 hover:text-primary-500 dark:hover:text-primary-400 transition-all duration-300 p-2 rounded-lg hover:bg-primary-50/50 dark:hover:bg-primary-900/20"
                  >
                    <span className="mr-3 opacity-80 group-hover:scale-110 group-hover:rotate-12 transition-all duration-300">
                      {link.icon}
                    </span>
                    <span className="flex-1 text-sm font-medium">{link.name}</span>
                    <FaChevronRight className="w-3 h-3 opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-300" />
                  </Link>
                </motion.li>
              ))}
            </ul>
          </motion.div>

          {/* 3. Resources */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2, duration: 0.6 }}
          >
            <h4 className="text-xl font-black mb-6 text-gray-900 dark:text-white flex items-center gap-2">
              <div className="w-1 h-6 bg-gradient-to-b from-primary-500 to-primary-400 rounded-full" />
              Resources
            </h4>
            <ul className="space-y-3">
              {legalLinks.map((link, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.05 }}
                >
                  <Link to={link.href}
                    className="flex items-center group text-gray-700 dark:text-gray-400 hover:text-primary-500 dark:hover:text-primary-400 transition-all duration-300 p-2 rounded-lg hover:bg-primary-50/50 dark:hover:bg-primary-900/20"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-primary-400 mr-3 opacity-60 group-hover:opacity-100 group-hover:scale-150 transition-all duration-300" />
                    <span className="flex-1 text-sm font-medium">{link.name}</span>
                    <FaChevronRight className="w-3 h-3 opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-300" />
                  </Link>
                </motion.li>
              ))}
            </ul>
          </motion.div>

          {/* 4. Newsletter & Social */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.3, duration: 0.6 }}
            className="space-y-6"
          >
            <div>
              <h4 className="text-xl font-black mb-4 text-gray-900 dark:text-white flex items-center gap-2">
                <div className="w-1 h-6 bg-gradient-to-b from-primary-500 to-primary-400 rounded-full" />
                Stay Updated
              </h4>
              <p className="text-gray-600 dark:text-gray-400 mb-6 text-sm">
                Get the latest updates, tips, and news about accessibility technology.
              </p>
              
              {/* Subscription Form - matching home page button style */}
              <form onSubmit={handleSubscribe} className="space-y-3">
                <div className="relative">
                  <input
                    type="email"
                    placeholder="Your email address"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className="w-full p-4 pl-12 rounded-xl border-2 border-primary-300/50 dark:border-primary-700/50 bg-white/80 dark:bg-white/5 text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 placeholder:text-gray-500 dark:placeholder:text-gray-500 transition-all duration-300 backdrop-blur-sm"
                  />
                  <FaEnvelope className="absolute left-4 top-1/2 transform -translate-y-1/2 text-primary-500" />
                </div>
                
                <motion.button
                  type="submit"
                  whileHover={{ scale: 1.02, boxShadow: "0 0 20px rgba(160, 68, 255, 0.6)" }}
                  whileTap={{ scale: 0.98 }}
                  className="w-full relative overflow-hidden px-6 py-4 rounded-xl bg-gradient-to-r from-primary-custom-1 via-primary-custom-2 to-primary-custom-3 text-white font-bold shadow-lg shadow-primary-400/40 hover:shadow-primary-500/60 transition-all duration-300 group"
                >
                  <span className="relative z-10 flex items-center justify-center gap-2">
                    Subscribe Now
                    <FaChevronRight className="group-hover:translate-x-1 transition-transform duration-300" />
                  </span>
                  <div className="absolute top-0 left-0 w-full h-full bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 z-0 rounded-xl" />
                </motion.button>
                
                {isSubscribed && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="p-3 rounded-lg bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/30 text-green-600 dark:text-green-400 text-sm text-center font-semibold"
                  >
                    ✓ Successfully subscribed! Thank you 🎉
                  </motion.div>
                )}
              </form>
            </div>

            {/* Social Media - matching home page style */}
            <div>
              <h5 className="text-lg font-black mb-4 text-gray-900 dark:text-white">Follow Us</h5>
              <div className="flex gap-3">
                {socialLinks.map((link, index) => (
                  <motion.a
                    key={index}
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    initial={{ scale: 0 }}
                    whileInView={{ scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1, type: "spring", stiffness: 300 }}
                    whileHover={{ 
                      scale: 1.15, 
                      y: -3,
                      transition: { type: "spring", stiffness: 400 }
                    }}
                    whileTap={{ scale: 0.9 }}
                    className="p-3 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 text-white shadow-lg hover:shadow-[0_0_20px_rgba(168,85,247,0.5)] transition-all duration-300"
                  >
                    <link.icon className="text-lg" />
                  </motion.a>
                ))}
              </div>
            </div>
          </motion.div>
        </div>

        {/* Divider with gradient */}
        <motion.div
          initial={{ scaleX: 0 }}
          whileInView={{ scaleX: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="h-px bg-gradient-to-r from-transparent via-primary-500/50 to-transparent my-12"
        />

        {/* Footer Bottom */}
        <div className="flex flex-col md:flex-row justify-between items-center gap-6 pt-6">
          {/* Copyright */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-gray-500 dark:text-gray-400 text-sm text-center md:text-left"
          >
            &copy; {new Date().getFullYear()} LinguaSign. All rights reserved.
            <span className="mx-2">•</span>
            Making communication accessible for everyone
          </motion.div>

          {/* Made with Love - animated heart */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            whileHover={{ scale: 1.05 }}
            className="flex items-center gap-2 text-gray-600 dark:text-gray-400 text-sm font-medium"
          >
            <span>Made with</span>
            <motion.div
              animate={{ scale: [1, 1.3, 1] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
            >
              <FaHeart className="text-red-500" />
            </motion.div>
            <span>for the community</span>
          </motion.div>

          {/* Additional Links */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="flex items-center gap-6 text-sm"
          >
            {["Accessibility", "Sitemap", "Status"].map((item, i) => (
              <Link 
                key={i}
                to={`/${item.toLowerCase()}`}
                className="text-gray-600 dark:text-gray-400 hover:text-primary-500 dark:hover:text-primary-400 transition-colors duration-300 font-medium"
              >
                {item}
              </Link>
            ))}
          </motion.div>
        </div>

      </div>
    </footer>
  );
}
