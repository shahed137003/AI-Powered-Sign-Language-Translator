import React, { useState } from "react";
<<<<<<< HEAD
import { FaFacebookF, FaTwitter, FaLinkedinIn, FaInstagram } from "react-icons/fa";
=======
import { FaFacebookF, FaTwitter, FaLinkedinIn, FaInstagram, FaEnvelope, FaUser } from "react-icons/fa";
>>>>>>> e251330 (Add frontend, backend, and ai_service)
import { motion } from "framer-motion";

export default function Contact() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: "",
  });

  const [success, setSuccess] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
<<<<<<< HEAD
    console.log("Message sent:", formData);
=======
    // In a real application, you would send this data to a server here.
    console.log("Message sent:", formData); 
>>>>>>> e251330 (Add frontend, backend, and ai_service)
    setSuccess(true);
    setFormData({ name: "", email: "", message: "" });
    setTimeout(() => setSuccess(false), 4000);
  };

  // Motion variants
<<<<<<< HEAD
  const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

  return (
    <div className="w-full bg-gray-50 dark:bg-gray-900 py-24 px-4 sm:px-6 lg:px-20">
=======
  const fadeUp = { hidden: { opacity: 0, y: 30 }, visible: { opacity: 1, y: 0 } };

  const socialLinks = [
    { icon: FaFacebookF, href: "https://facebook.com", color: "#3b5998" },
    { icon: FaTwitter, href: "https://twitter.com", color: "#1da1f2" },
    { icon: FaLinkedinIn, href: "https://linkedin.com", color: "#0077b5" },
    { icon: FaInstagram, href: "https://instagram.com", color: "#e4405f" },
  ];

  return (
    <div className="w-full bg-gray-50 dark:bg-[#0f0c29] py-24 px-6 lg:px-20 relative overflow-hidden transition-colors duration-500">
      
      {/* Background Glows */}
      
      <div className="absolute top-1/4 left-1/4 w-[300px] h-[300px] bg-indigo-600/10 rounded-full blur-[100px] pointer-events-none" />

      {/* --- HEADER --- */}
>>>>>>> e251330 (Add frontend, backend, and ai_service)
      <motion.div
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        variants={fadeUp}
        transition={{ duration: 0.8 }}
<<<<<<< HEAD
        className="max-w-7xl mx-auto text-center mb-12"
      >
        <h2 className="text-4xl sm:text-5xl font-extrabold mb-4 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
          Contact Us
        </h2>
        <div className="w-24 h-1 mx-auto mb-10 rounded-full bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF]"></div>
        <p className="text-gray-700 dark:text-gray-200 text-lg sm:text-xl">
          Have a question or want to get in touch? Send us a message or reach out via social media. We’d love to hear from you!
        </p>
      </motion.div>

      <div className="max-w-7xl mx-auto flex flex-col lg:flex-row gap-12">
=======
        className="max-w-7xl mx-auto text-center mb-16 relative z-10"
      >
        <span className="text-purple-600 dark:text-purple-400 font-bold tracking-widest uppercase text-sm mb-2 block">
            Get in Touch
        </span>
        <h2 className="text-4xl sm:text-5xl font-extrabold mb-4 text-gray-900 dark:text-white">
          <span className="bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
            Contact Us
          </span>
        </h2>
        <p className="text-gray-600 dark:text-gray-400 text-lg sm:text-xl max-w-2xl mx-auto">
          Have a question or want to get in touch? Send us a secure message or reach out via social media.
        </p>
      </motion.div>

      {/* --- CONTACT GRID --- */}
      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-12 relative z-10">
        
>>>>>>> e251330 (Add frontend, backend, and ai_service)
        {/* Contact Form */}
        <motion.form
          initial={{ opacity: 0, x: -50 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          onSubmit={handleSubmit}
<<<<<<< HEAD
          className="flex-1 bg-purple-50 dark:bg-gray-800/40 backdrop-blur-lg border border-white/30 dark:border-gray-600 p-8 rounded-3xl shadow-2xl flex flex-col gap-6 hover:shadow-3xl transition duration-500 transform hover:-translate-y-1"
        >
=======
          className="p-8 lg:p-10 dark:bg-[#1a163a]/60 backdrop-blur-xl border border-gray-200 dark:border-purple-500/20 rounded-3xl shadow-xl flex flex-col gap-6 
            transition duration-500 hover:shadow-purple-900/40"
        >
          <h3 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">Send a Message</h3>
          
>>>>>>> e251330 (Add frontend, backend, and ai_service)
          {/* Name Input */}
          <div className="relative">
            <input
              type="text"
              name="name"
              value={formData.name}
              onChange={handleChange}
              required
              placeholder="Your Name"
<<<<<<< HEAD
              className="w-full p-4 pl-12 border border-gray-300 dark:border-gray-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#A044FF] placeholder-transparent peer bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-200"
            />
            <label className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 text-sm peer-placeholder-shown:top-1/2 peer-placeholder-shown:text-gray-400 peer-placeholder-shown:text-base peer-focus:top-2 peer-focus:text-[#6A3093] peer-focus:text-sm transition-all">
=======
              className="w-full p-4 pl-12 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 placeholder-transparent peer bg-white/70 dark:bg-gray-700/50 text-gray-900 dark:text-gray-200 transition-colors"
            />
            <FaUser className="absolute left-4 top-1/2 transform -translate-y-1/2 text-purple-400 peer-focus:text-purple-600 transition-colors" />
            <label className="absolute left-12 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 text-base peer-placeholder-shown:top-1/2 peer-focus:top-3 peer-focus:text-purple-600 peer-focus:dark:text-purple-400 peer-focus:text-sm transition-all pointer-events-none">
>>>>>>> e251330 (Add frontend, backend, and ai_service)
              Your Name
            </label>
          </div>

          {/* Email Input */}
          <div className="relative">
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
              placeholder="Your Email"
<<<<<<< HEAD
              className="w-full p-4 pl-12 border border-gray-300 dark:border-gray-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#A044FF] placeholder-transparent peer bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-200"
            />
            <label className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 text-sm peer-placeholder-shown:top-1/2 peer-placeholder-shown:text-gray-400 peer-placeholder-shown:text-base peer-focus:top-2 peer-focus:text-[#6A3093] peer-focus:text-sm transition-all">
=======
              className="w-full p-4 pl-12 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 placeholder-transparent peer bg-white/70 dark:bg-gray-700/50 text-gray-900 dark:text-gray-200 transition-colors"
            />
            <FaEnvelope className="absolute left-4 top-1/2 transform -translate-y-1/2 text-purple-400 peer-focus:text-purple-600 transition-colors" />
             <label className="absolute left-12 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 text-base peer-placeholder-shown:top-1/2 peer-focus:top-3 peer-focus:text-purple-600 peer-focus:dark:text-purple-400 peer-focus:text-sm transition-all pointer-events-none">
>>>>>>> e251330 (Add frontend, backend, and ai_service)
              Your Email
            </label>
          </div>

          {/* Message Input */}
          <div className="relative">
            <textarea
              name="message"
              rows="6"
              value={formData.message}
              onChange={handleChange}
              required
              placeholder="Your Message"
<<<<<<< HEAD
              className="w-full p-4 pt-6 border border-gray-300 dark:border-gray-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#A044FF] placeholder-transparent peer resize-none bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-200"
            ></textarea>
            <label className="absolute left-4 top-3 text-gray-500 dark:text-gray-400 text-sm peer-placeholder-shown:top-6 peer-placeholder-shown:text-gray-400 peer-placeholder-shown:text-base peer-focus:top-3 peer-focus:text-[#6A3093] peer-focus:text-sm transition-all">
=======
              className="w-full p-4 pt-10 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 placeholder-transparent peer resize-none bg-white/70 dark:bg-gray-700/50 text-gray-900 dark:text-gray-200 transition-colors"
            ></textarea>
            <label className="absolute left-4 top-3 text-gray-500 dark:text-gray-400 text-base peer-placeholder-shown:top-6 peer-focus:top-3 peer-focus:text-purple-600 peer-focus:dark:text-purple-400 peer-focus:text-sm transition-all pointer-events-none">
>>>>>>> e251330 (Add frontend, backend, and ai_service)
              Your Message
            </label>
          </div>

          {/* Submit Button */}
          <motion.button
            type="submit"
<<<<<<< HEAD
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.98 }}
            transition={{ type: "spring", stiffness: 300 }}
            className="px-6 py-3 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] text-white font-semibold rounded-full shadow-xl transform transition duration-300"
=======
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            transition={{ type: "spring", stiffness: 300 }}
            className="px-6 py-4 bg-gradient-to-r from-[#6A3093] to-[#A044FF]  text-white font-bold rounded-full shadow-lg shadow-purple-500/40 transform transition duration-300"
>>>>>>> e251330 (Add frontend, backend, and ai_service)
          >
            Send Message
          </motion.button>

          {/* Success Message */}
          {success && (
            <motion.p
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
<<<<<<< HEAD
              className="text-green-500 dark:text-green-400 text-center mt-4 animate-pulse"
            >
              Your message has been sent successfully!
=======
              className="text-green-500 dark:text-green-400 text-center mt-2 font-medium"
            >
              Your message has been sent successfully! We'll be in touch soon.
>>>>>>> e251330 (Add frontend, backend, and ai_service)
            </motion.p>
          )}
        </motion.form>

<<<<<<< HEAD
        {/* Social Links */}
=======
        {/* Social Links & Info */}
>>>>>>> e251330 (Add frontend, backend, and ai_service)
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
<<<<<<< HEAD
          className="flex-1 flex flex-col items-center lg:items-start justify-center gap-6"
        >
          <h3 className="text-2xl font-semibold text-[#6A3093] mb-4">Follow Us</h3>
          <p className="text-gray-700 dark:text-gray-200 mb-4 text-center lg:text-left">
            Connect with us on social media for updates, tips, and news.
          </p>
          <div className="flex gap-4">
            <a href="https://facebook.com" target="_blank" rel="noopener noreferrer"
               className="p-4 bg-purple-50 dark:bg-gray-700 rounded-full shadow-lg hover:bg-[#3b5998] hover:text-white transition duration-300">
              <FaFacebookF />
            </a>
            <a href="https://twitter.com" target="_blank" rel="noopener noreferrer"
               className="p-4 bg-purple-50 dark:bg-gray-700 rounded-full shadow-lg hover:bg-[#1da1f2] hover:text-white transition duration-300">
              <FaTwitter />
            </a>
            <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer"
               className="p-4 bg-purple-50 dark:bg-gray-700 rounded-full shadow-lg hover:bg-[#0077b5] hover:text-white transition duration-300">
              <FaLinkedinIn />
            </a>
            <a href="https://instagram.com" target="_blank" rel="noopener noreferrer"
               className="p-4 bg-purple-50 dark:bg-gray-700 rounded-full shadow-lg hover:bg-[#e4405f] hover:text-white transition duration-300">
              <FaInstagram />
            </a>
=======
          className="flex flex-col gap-6 lg:justify-start lg:pt-10"
        >
          {/* General Info */}
          <div className="space-y-4">
              <h3 className="text-3xl font-bold text-gray-900 dark:text-white">Other Ways to Connect</h3>
              
              <div className="text-lg text-gray-700 dark:text-gray-300">
                <p className="font-semibold text-purple-500 dark:text-purple-400">Email Support</p>
                <p>support@linguasign.io</p>
              </div>

              <div className="text-lg text-gray-700 dark:text-gray-300">
                <p className="font-semibold text-purple-500 dark:text-purple-400">Press & Media</p>
                <p>press@linguasign.io</p>
              </div>
          </div>
          
          {/* Social Icons */}
          <div className="mt-6">
            <h3 className="text-2xl font-semibold text-purple-600 dark:text-purple-400 mb-4">Follow Our Journey</h3>
            <div className="flex gap-4">
              {socialLinks.map((link, index) => (
                <motion.a
                  key={index}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.15, rotate: 5 }}
                  whileTap={{ scale: 0.9 }}
                  className="p-4 rounded-full shadow-lg transition duration-300 
                             bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300"
                  style={{ 
                    // Dynamic hover effect using inline style to simulate neon glow
                    boxShadow: `0 0 10px ${link.color}30`,
                    "--hover-color": link.color 
                  }}
                  onMouseOver={(e) => {
                     e.currentTarget.style.color = link.color;
                     e.currentTarget.style.boxShadow = `0 0 15px ${link.color}90`;
                  }}
                  onMouseOut={(e) => {
                     e.currentTarget.style.color = '';
                     e.currentTarget.style.boxShadow = `0 0 10px ${link.color}30`;
                  }}
                >
                  <link.icon className="text-xl" />
                </motion.a>
              ))}
            </div>
>>>>>>> e251330 (Add frontend, backend, and ai_service)
          </div>
        </motion.div>
      </div>
    </div>
  );
<<<<<<< HEAD
}
=======
}
>>>>>>> e251330 (Add frontend, backend, and ai_service)
