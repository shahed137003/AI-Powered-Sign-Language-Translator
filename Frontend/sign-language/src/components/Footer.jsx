import React, { useState } from "react";
import { FaFacebookF, FaTwitter, FaLinkedinIn, FaInstagram, FaChevronRight, FaEnvelope, FaHeart, FaHands, FaUserFriends, FaGlobe } from "react-icons/fa";
import { BsLightningFill, BsStars } from "react-icons/bs";
import { TbSparkles } from "react-icons/tb";
import { motion } from "framer-motion";

export default function Footer() {
  const [email, setEmail] = useState("");
  const [isSubscribed, setIsSubscribed] = useState(false);

  const handleSubscribe = (e) => {
    e.preventDefault();
    if (email) {
      console.log("Subscribing email:", email);
      setIsSubscribed(true);
      setEmail("");
      
      // Reset success message after 3 seconds
      setTimeout(() => setIsSubscribed(false), 3000);
    }
  };

  const socialLinks = [
    { 
      icon: FaFacebookF, 
      href: "https://facebook.com", 
      gradient: "from-[#1877F2] to-[#0D5DBD]",
      hover: "hover:shadow-[0_0_20px_#1877F2]"
    },
    { 
      icon: FaTwitter, 
      href: "https://twitter.com", 
      gradient: "from-[#1DA1F2] to-[#0C8BD9]",
      hover: "hover:shadow-[0_0_20px_#1DA1F2]"
    },
    { 
      icon: FaLinkedinIn, 
      href: "https://linkedin.com", 
      gradient: "from-[#0077B5] to-[#00669C]",
      hover: "hover:shadow-[0_0_20px_#0077B5]"
    },
    { 
      icon: FaInstagram, 
      href: "https://instagram.com", 
      gradient: "from-[#E4405F] via-[#D92D4F] to-[#C22945]",
      hover: "hover:shadow-[0_0_20px_#E4405F]"
    },
  ];

  const quickLinks = [
    { name: "Home", href: "/", icon: <BsStars className="text-sm" /> },
    { name: "About Us", href: "/about", icon: <FaUserFriends className="text-sm" /> },
    { name: "Features", href: "/#features", icon: <BsLightningFill className="text-sm" /> },
    { name: "Translate", href: "/translate", icon: <FaHands className="text-sm" /> },
    { name: "Chatbot", href: "/chatbot", icon: <FaGlobe className="text-sm" /> },
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

  const stats = [
    { value: "10M+", label: "Users Worldwide" },
    { value: "99%", label: "Accuracy Rate" },
    { value: "24/7", label: "Support Available" },
    { value: "50+", label: "Languages" },
  ];

  return (
    <footer className="relative w-full bg-gradient-to-br from-gray-50 via-white to-purple-50/60 dark:from-[#0a0518] dark:via-[#110a2e] dark:to-[#1e0f5c] text-gray-800 dark:text-gray-300 pt-20 pb-8 px-4 sm:px-6 lg:px-8 border-t border-purple-200/30 dark:border-purple-900/30 overflow-hidden transition-colors duration-500">
      
      {/* Background Effects */}
      <div className="absolute inset-0 pointer-events-none">
        {/* Geometric Grid */}
        <div className="absolute inset-0 opacity-5 dark:opacity-5">
          <div className="absolute inset-0" style={{
            backgroundImage: `
              linear-gradient(90deg, rgba(168, 85, 247, 0.1) 1px, transparent 1px),
              linear-gradient(180deg, rgba(168, 85, 247, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: '40px 40px'
          }} />
        </div>
        
        {/* Gradient Orbs */}
        <motion.div 
          animate={{ opacity: [0.1, 0.2, 0.1], scale: [1, 1.05, 1] }}
          transition={{ duration: 8, repeat: Infinity }}
          className="absolute bottom-0 right-0 w-[300px] h-[300px] bg-gradient-to-tr from-purple-600/10 to-pink-500/10 rounded-full blur-3xl"
        />
        <motion.div 
          animate={{ opacity: [0.05, 0.15, 0.05], scale: [1, 1.03, 1] }}
          transition={{ duration: 10, repeat: Infinity, delay: 1 }}
          className="absolute top-0 left-0 w-[200px] h-[200px] bg-gradient-to-br from-blue-500/10 to-cyan-500/10 rounded-full blur-3xl"
        />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto">
        
  

        {/* Main Footer Content */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12 mb-12">
          
          {/* 1. Brand Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            {/* Logo */}
            <div className="flex items-center gap-3">
              <motion.div
                animate={{ rotate: [0, 10, 0] }}
                transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                className="w-12 h-12 rounded-xl bg-gradient-to-br from-[#6A3093] to-[#A044FF] flex items-center justify-center"
              >
                <FaHands className="text-white text-xl" />
              </motion.div>
              <h3 className="text-2xl font-bold">
                <span className="bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
                  LinguaSign
                </span>
              </h3>
            </div>
            
            <p className="text-gray-600 dark:text-gray-400 leading-relaxed text-sm">
              Breaking communication barriers with AI-powered sign language translation. 
              Making the world accessible for everyone through cutting-edge technology.
            </p>
            
            {/* Tagline */}
            <div className="flex items-center gap-2 text-purple-600 dark:text-purple-400 text-sm font-medium">
              <TbSparkles className="text-lg" />
              <span>Inclusive Technology for All</span>
            </div>
          </motion.div>

          {/* 2. Quick Links */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
          >
            <h4 className="text-xl font-semibold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
              <FaChevronRight className="text-purple-500" />
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
                  <a 
                    href={link.href}
                    className="flex items-center group text-gray-700 dark:text-gray-400 hover:text-purple-500 dark:hover:text-purple-400 transition-all duration-300 p-2 rounded-lg hover:bg-purple-50 dark:hover:bg-purple-900/20"
                  >
                    <span className="mr-3 opacity-80 group-hover:scale-110 transition-transform">
                      {link.icon}
                    </span>
                    <span className="flex-1">{link.name}</span>
                    <FaChevronRight className="w-3 h-3 opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all" />
                  </a>
                </motion.li>
              ))}
            </ul>
          </motion.div>

          {/* 3. Resources */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
          >
            <h4 className="text-xl font-semibold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
              <FaChevronRight className="text-purple-500" />
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
                  <a 
                    href={link.href}
                    className="flex items-center group text-gray-700 dark:text-gray-400 hover:text-purple-500 dark:hover:text-purple-400 transition-all duration-300 p-2 rounded-lg hover:bg-purple-50 dark:hover:bg-purple-900/20"
                  >
                    <span className="w-2 h-2 rounded-full bg-purple-400 mr-3 opacity-60 group-hover:opacity-100 group-hover:scale-125 transition-all" />
                    <span className="flex-1">{link.name}</span>
                    <FaChevronRight className="w-3 h-3 opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all" />
                  </a>
                </motion.li>
              ))}
            </ul>
          </motion.div>

          {/* 4. Newsletter & Social */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.3 }}
            className="space-y-6"
          >
            <div>
              <h4 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
                <FaEnvelope className="text-purple-500" />
                Stay Updated
              </h4>
              <p className="text-gray-600 dark:text-gray-400 mb-6 text-sm">
                Get the latest updates, tips, and news about accessibility technology.
              </p>
              
              {/* Subscription Form */}
              <form onSubmit={handleSubscribe} className="space-y-3">
                <div className="relative">
                  <input
                    type="email"
                    placeholder="Your email address"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className="w-full p-4 pl-12 rounded-xl border border-purple-300/50 dark:border-purple-700/50 bg-white/90 dark:bg-white/10 text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-purple-500 placeholder:text-gray-500 dark:placeholder:text-gray-500 transition-all duration-300 backdrop-blur-sm"
                  />
                  <FaEnvelope className="absolute left-4 top-1/2 transform -translate-y-1/2 text-purple-500" />
                </div>
                
                <motion.button
                  type="submit"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="w-full relative overflow-hidden px-6 py-4 rounded-xl bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] text-white font-semibold shadow-lg shadow-purple-500/40 hover:shadow-purple-500/60 transition-all duration-300 group"
                >
                  <span className="relative z-10 flex items-center justify-center gap-2">
                    Subscribe Now
                    <FaChevronRight className="group-hover:translate-x-1 transition-transform" />
                  </span>
                  <div className="absolute top-0 left-0 w-full h-full bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 z-0 rounded-xl" />
                </motion.button>
                
                {isSubscribed && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="p-3 rounded-lg bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/30 text-green-600 dark:text-green-400 text-sm text-center"
                  >
                    Successfully subscribed! Thank you 
                  </motion.div>
                )}
              </form>
            </div>

            {/* Social Media */}
            <div>
              <h5 className="text-lg font-medium mb-4 text-gray-900 dark:text-white">Follow Us</h5>
              <div className="flex gap-3">
                {socialLinks.map((link, index) => (
                  <motion.a
                    key={index}
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    initial={{ scale: 0, rotate: -180 }}
                    whileInView={{ scale: 1, rotate: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1, type: "spring" }}
                    whileHover={{ 
                      scale: 1.15, 
                      rotate: 5,
                      transition: { type: "spring", stiffness: 400 }
                    }}
                    whileTap={{ scale: 0.9 }}
                    className={`p-3 rounded-xl bg-gradient-to-br ${link.gradient} text-white shadow-lg ${link.hover} transition-all duration-300 backdrop-blur-sm border border-white/20`}
                  >
                    <link.icon className="text-lg" />
                  </motion.a>
                ))}
              </div>
            </div>
          </motion.div>
        </div>

        {/* Divider */}
        <motion.div
          initial={{ scaleX: 0 }}
          whileInView={{ scaleX: 1 }}
          viewport={{ once: true }}
          className="h-px bg-gradient-to-r from-transparent via-purple-500/30 to-transparent my-12"
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

          {/* Made with Love */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            whileHover={{ scale: 1.05 }}
            className="flex items-center gap-2 text-gray-600 dark:text-gray-400 text-sm"
          >
            <span>Made with</span>
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
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
            <a href="/accessibility" className="text-gray-600 dark:text-gray-400 hover:text-purple-500 dark:hover:text-purple-400 transition-colors">
              Accessibility
            </a>
            <a href="/sitemap" className="text-gray-600 dark:text-gray-400 hover:text-purple-500 dark:hover:text-purple-400 transition-colors">
              Sitemap
            </a>
            <a href="/status" className="text-gray-600 dark:text-gray-400 hover:text-purple-500 dark:hover:text-purple-400 transition-colors">
              Status
            </a>
          </motion.div>
        </div>

      </div>
    </footer>
  );
}