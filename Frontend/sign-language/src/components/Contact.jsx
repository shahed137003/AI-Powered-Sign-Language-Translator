import React, { useState } from "react";
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
import { motion } from "framer-motion";
import { TbSparkles, TbMailForward, TbMessage2, TbPhoneCall } from "react-icons/tb";
import { FiSend } from "react-icons/fi";

export default function Contact() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: "",
  });

  const [success, setSuccess] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      console.log("Message sent:", formData); 
      setSuccess(true);
      setFormData({ name: "", email: "", message: "" });
      setIsLoading(false);
      setTimeout(() => setSuccess(false), 4000);
    }, 1500);
  };

  // Motion variants
  const fadeUp = { 
    hidden: { opacity: 0, y: 30 }, 
    visible: { opacity: 1, y: 0 } 
  };

  const socialLinks = [
    { icon: FaFacebookF, href: "https://facebook.com", color: "#3b5998", label: "Facebook" },
    { icon: FaTwitter, href: "https://twitter.com", color: "#1da1f2", label: "Twitter" },
    { icon: FaLinkedinIn, href: "https://linkedin.com", color: "#0077b5", label: "LinkedIn" },
    { icon: FaInstagram, href: "https://instagram.com", color: "#e4405f", label: "Instagram" },
  ];

  const contactInfo = [
    { icon: FaEnvelope, title: "Email Support", value: "support@linguasign.io", color: "from-purple-500 to-pink-500" },
    { icon: FaPhone, title: "Phone Support", value: "+1 (555) 123-4567", color: "from-blue-500 to-cyan-500" },
    { icon: FaMapMarkerAlt, title: "Headquarters", value: "San Francisco, CA", color: "from-green-500 to-emerald-500" },
    { icon: FaRegClock, title: "Business Hours", value: "Mon-Fri, 9AM-6PM PST", color: "from-orange-500 to-yellow-500" },
  ];

  return (
    <div className="w-full min-h-screen py-24 px-4 sm:px-6 lg:px-8 relative overflow-hidden bg-gradient-to-br from-gray-50 via-white to-purple-50/60 dark:from-[#0a0518] dark:via-[#110a2e] dark:to-[#1e0f5c]">
      
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

      {/* Animated glows */}
      <motion.div 
        animate={{ 
          x: [0, 50, 0],
          y: [0, 30, 0]
        }}
        transition={{ 
          duration: 20, 
          repeat: Infinity, 
          ease: "linear" 
        }}
        className="absolute top-1/4 left-1/4 w-[600px] h-[600px] bg-gradient-to-r from-purple-600/10 via-pink-600/10 to-indigo-600/10 rounded-full blur-[120px]"
      />
      <motion.div 
        animate={{ 
          x: [0, -40, 0],
          y: [0, -20, 0]
        }}
        transition={{ 
          duration: 25, 
          repeat: Infinity, 
          ease: "linear" 
        }}
        className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-gradient-to-r from-indigo-600/10 via-purple-600/10 to-pink-600/10 rounded-full blur-[100px]"
      />

      {/* Floating particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(15)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-purple-500/20 rounded-full"
            initial={{ 
              x: Math.random() * 100 + 'vw', 
              y: Math.random() * 100 + 'vh',
              scale: 0 
            }}
            animate={{ 
              y: [null, -20, 20, -15],
              x: [null, 15, -15, 10],
              scale: [0, 1, 1, 0],
              opacity: [0, 0.5, 0.5, 0]
            }}
            transition={{ 
              duration: Math.random() * 8 + 15,
              repeat: Infinity,
              ease: "linear",
              delay: Math.random() * 3
            }}
          />
        ))}
      </div>

      {/* Header */}
      <div className="relative z-10 max-w-7xl mx-auto">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
          className="text-center mb-20"
        >
          {/* Premium Badge */}
          <motion.div
            whileHover={{ scale: 1.05, rotate: 1 }}
            className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-gradient-to-r from-purple-500/15 via-purple-400/10 to-purple-300/10 border border-purple-200/60 dark:border-purple-700/60 backdrop-blur-xl shadow-lg shadow-purple-500/10 relative overflow-hidden group mb-8"
          >
            <div className="relative">
              <span className="absolute animate-ping inline-flex h-3.5 w-3.5 rounded-full bg-purple-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-gradient-to-r from-purple-500 to-purple-400" />
            </div>
            <span className="text-sm font-bold bg-gradient-to-r from-purple-600 via-purple-500 to-purple-400 bg-clip-text text-transparent">
              Get in Touch
            </span>
            <TbSparkles className="text-purple-500 ml-1" />
            <div className="absolute -inset-1 bg-gradient-to-r from-purple-500/0 via-purple-400/10 to-purple-500/0 group-hover:via-purple-400/20 transition-all duration-500" />
          </motion.div>

          <motion.h1
            variants={fadeUp}
            className="font-extrabold text-4xl sm:text-5xl lg:text-[43px] leading-tight mb-6"
          >
            <span className="block text-gray-900 dark:text-white">
              Let's Start a Conversation
            </span>
            <span className="block bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
              Contact Our Team
            </span>
          </motion.h1>
          
          <motion.p
            variants={fadeUp}
            className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed"
          >
            Have questions, ideas, or want to collaborate? We're here to help you bridge the communication gap.
          </motion.p>

          {/* Decorative Elements */}
          <motion.div
            variants={fadeUp}
            className="flex items-center justify-center gap-8 mt-10"
          >
            <div className="w-12 h-1 bg-gradient-to-r from-transparent via-purple-500 to-transparent rounded-full" />
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="w-6 h-6 rounded-full border-2 border-purple-400/50"
            />
            <div className="w-12 h-1 bg-gradient-to-r from-transparent via-purple-500 to-transparent rounded-full" />
          </motion.div>
        </motion.div>

        {/* Contact Grid */}
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8 relative z-10">
          
          {/* Contact Information Cards */}
          <div className="lg:col-span-1 space-y-6">
            {contactInfo.map((info, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ y: -5, scale: 1.02 }}
                className="group relative"
              >
                <div className="absolute -inset-0.5 bg-gradient-to-r rounded-2xl blur opacity-0 group-hover:opacity-30 transition-opacity duration-500"
                  style={{ background: `linear-gradient(to right, ${info.color})` }}
                />
                <div className="relative p-6 rounded-2xl bg-gradient-to-br from-white/80 to-white/60 dark:from-white/10 dark:to-white/5 backdrop-blur-xl border border-purple-200/50 dark:border-purple-500/20 shadow-lg shadow-purple-100/20 dark:shadow-purple-900/20">
                  <div className="flex items-start gap-4">
                    <div className={`p-3 rounded-xl bg-gradient-to-br ${info.color}/10`}>
                      <info.icon className="text-2xl text-purple-600 dark:text-purple-400" />
                    </div>
                    <div>
                      <h3 className="font-bold text-gray-900 dark:text-white mb-1">{info.title}</h3>
                      <p className="text-gray-600 dark:text-gray-300">{info.value}</p>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}

            {/* Social Links */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="pt-6"
            >
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <TbMessage2 className="text-purple-600" />
                Follow Our Journey
              </h3>
              <div className="flex gap-3 flex-wrap">
                {socialLinks.map((link, index) => (
                  <motion.a
                    key={index}
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    whileHover={{ scale: 1.1, y: -3 }}
                    whileTap={{ scale: 0.95 }}
                    className="group relative"
                  >
                    <div className="absolute -inset-1 rounded-full blur opacity-0 group-hover:opacity-30 transition-opacity duration-300"
                      style={{ backgroundColor: link.color }}
                    />
                    <div className="relative p-3 rounded-full bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border border-gray-300/50 dark:border-gray-700/50 shadow-lg group-hover:shadow-xl transition-all duration-300"
                      style={{ 
                        color: link.color,
                        boxShadow: `0 4px 14px 0 ${link.color}20`
                      }}
                    >
                      <link.icon className="text-xl" />
                      <span className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 text-xs font-medium opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap">
                        {link.label}
                      </span>
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
              {/* Form Glow */}
              <div className="absolute -inset-1 bg-gradient-to-r from-purple-500/20 via-pink-500/20 to-indigo-500/20 rounded-3xl blur opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              
              {/* Main Form Card */}
              <div className="relative p-8 lg:p-10 rounded-3xl bg-gradient-to-br from-white/80 to-white/60 dark:from-white/10 dark:to-white/5 backdrop-blur-xl border border-purple-200/50 dark:border-purple-500/20 shadow-2xl shadow-purple-100/30 dark:shadow-purple-900/30">
                <div className="flex items-center gap-3 mb-8">
                  <div className="p-2 rounded-lg bg-gradient-to-r from-purple-500/10 to-pink-500/10">
                    <TbMailForward className="text-2xl text-purple-600 dark:text-purple-400" />
                  </div>
                  <h3 className="text-3xl font-bold text-gray-900 dark:text-white">
                    Send Us a Message
                  </h3>
                </div>

                <form onSubmit={handleSubmit} className="space-y-6">
                  {/* Name Input */}
                  <div className="relative group/input">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500/0 via-purple-400/0 to-purple-300/0 group-hover/input:via-purple-400/10 group-hover/input:opacity-100 opacity-0 rounded-xl transition-all duration-300" />
                    <div className="relative">
                      <span className="absolute left-4 top-1/2 -translate-y-1/2 text-purple-400 dark:text-purple-500">
                        <FaUser size={18} />
                      </span>
                      <input
                        type="text"
                        name="name"
                        value={formData.name}
                        onChange={handleChange}
                        required
                        className="w-full pl-12 pr-4 py-4 rounded-xl bg-white/70 dark:bg-gray-800/50 border border-gray-300 dark:border-gray-700 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-purple-500/50 transition-all duration-300"
                        placeholder="Your Name"
                      />
                    </div>
                  </div>

                  {/* Email Input */}
                  <div className="relative group/input">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500/0 via-purple-400/0 to-purple-300/0 group-hover/input:via-purple-400/10 group-hover/input:opacity-100 opacity-0 rounded-xl transition-all duration-300" />
                    <div className="relative">
                      <span className="absolute left-4 top-1/2 -translate-y-1/2 text-purple-400 dark:text-purple-500">
                        <FaEnvelope size={18} />
                      </span>
                      <input
                        type="email"
                        name="email"
                        value={formData.email}
                        onChange={handleChange}
                        required
                        className="w-full pl-12 pr-4 py-4 rounded-xl bg-white/70 dark:bg-gray-800/50 border border-gray-300 dark:border-gray-700 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-purple-500/50 transition-all duration-300"
                        placeholder="Your Email"
                      />
                    </div>
                  </div>

                  {/* Message Input */}
                  <div className="relative group/input">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500/0 via-purple-400/0 to-purple-300/0 group-hover/input:via-purple-400/10 group-hover/input:opacity-100 opacity-0 rounded-xl transition-all duration-300" />
                    <div className="relative">
                      <textarea
                        name="message"
                        rows="6"
                        value={formData.message}
                        onChange={handleChange}
                        required
                        className="w-full p-4 rounded-xl bg-white/70 dark:bg-gray-800/50 border border-gray-300 dark:border-gray-700 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-purple-500/50 resize-none transition-all duration-300"
                        placeholder="Your Message"
                      />
                    </div>
                  </div>

                  {/* Submit Button */}
                  <motion.button
                    type="submit"
                    disabled={isLoading}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="w-full py-4 rounded-xl font-bold text-lg text-white bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] shadow-2xl shadow-purple-500/30 hover:shadow-purple-500/50 transition-all duration-300 relative overflow-hidden group disabled:opacity-70 disabled:cursor-not-allowed"
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-white/5 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
                    <div className="relative z-10 flex items-center justify-center gap-3">
                      {isLoading ? (
                        <>
                          <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                          <span>Sending...</span>
                        </>
                      ) : (
                        <>
                          <FiSend size={20} />
                          <span>Send Message</span>
                        
                        </>
                      )}
                    </div>
                  </motion.button>

                  {/* Success Message */}
                  {success && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="p-4 rounded-xl bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/20 dark:border-emerald-500/20"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-r from-green-500 to-emerald-500 flex items-center justify-center">
                          <FaPaperPlane className="text-white" size={18} />
                        </div>
                        <div>
                          <p className="font-semibold text-green-700 dark:text-green-400">Message Sent!</p>
                          <p className="text-sm text-green-600 dark:text-green-300">
                            Thank you for reaching out. We'll respond within 24 hours.
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </form>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}