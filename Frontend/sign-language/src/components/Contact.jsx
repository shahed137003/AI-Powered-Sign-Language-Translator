import React, { useState } from "react";
import { FaFacebookF, FaTwitter, FaLinkedinIn, FaInstagram, FaEnvelope, FaUser } from "react-icons/fa";
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
    // In a real application, you would send this data to a server here.
    console.log("Message sent:", formData); 
    setSuccess(true);
    setFormData({ name: "", email: "", message: "" });
    setTimeout(() => setSuccess(false), 4000);
  };

  // Motion variants
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
      <motion.div
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        variants={fadeUp}
        transition={{ duration: 0.8 }}
        className="max-w-7xl mx-auto text-center mb-16 relative z-10"
      >
        <span className="text-purple-600 dark:text-purple-400 font-bold tracking-widest uppercase text-sm mb-2 block">
            Get in Touch
        </span>
        <h2 className="text-4xl sm:text-5xl font-extrabold mb-4 text-gray-900 dark:text-white">
          <span className="bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] bg-clip-text text-transparent">
            Contact Us
          </span>
        </h2>
        <p className="text-gray-600 dark:text-gray-400 text-lg sm:text-xl max-w-2xl mx-auto">
          Have a question or want to get in touch? Send us a secure message or reach out via social media.
        </p>
      </motion.div>

      {/* --- CONTACT GRID --- */}
      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-12 relative z-10">
        
        {/* Contact Form */}
        <motion.form
          initial={{ opacity: 0, x: -50 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          onSubmit={handleSubmit}
          className="p-8 lg:p-10 dark:bg-[#1a163a]/60 backdrop-blur-xl border border-gray-200 dark:border-purple-500/20 rounded-3xl shadow-xl flex flex-col gap-6 
            transition duration-500 hover:shadow-purple-900/40"
        >
          <h3 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">Send a Message</h3>
          
          {/* Name Input */}
          <div className="relative">
            <input
              type="text"
              name="name"
              value={formData.name}
              onChange={handleChange}
              required
              placeholder="Your Name"
              className="w-full p-4 pl-12 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 placeholder-transparent peer bg-white/70 dark:bg-gray-700/50 text-gray-900 dark:text-gray-200 transition-colors"
            />
            <FaUser className="absolute left-4 top-1/2 transform -translate-y-1/2 text-purple-400 peer-focus:text-purple-600 transition-colors" />
            <label className="absolute left-12 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 text-base peer-placeholder-shown:top-1/2 peer-focus:top-3 peer-focus:text-purple-600 peer-focus:dark:text-purple-400 peer-focus:text-sm transition-all pointer-events-none">
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
              className="w-full p-4 pl-12 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 placeholder-transparent peer bg-white/70 dark:bg-gray-700/50 text-gray-900 dark:text-gray-200 transition-colors"
            />
            <FaEnvelope className="absolute left-4 top-1/2 transform -translate-y-1/2 text-purple-400 peer-focus:text-purple-600 transition-colors" />
             <label className="absolute left-12 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 text-base peer-placeholder-shown:top-1/2 peer-focus:top-3 peer-focus:text-purple-600 peer-focus:dark:text-purple-400 peer-focus:text-sm transition-all pointer-events-none">
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
              className="w-full p-4 pt-10 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 placeholder-transparent peer resize-none bg-white/70 dark:bg-gray-700/50 text-gray-900 dark:text-gray-200 transition-colors"
            ></textarea>
            <label className="absolute left-4 top-3 text-gray-500 dark:text-gray-400 text-base peer-placeholder-shown:top-6 peer-focus:top-3 peer-focus:text-purple-600 peer-focus:dark:text-purple-400 peer-focus:text-sm transition-all pointer-events-none">
              Your Message
            </label>
          </div>

          {/* Submit Button */}
          <motion.button
            type="submit"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            transition={{ type: "spring", stiffness: 300 }}
            className="px-6 py-4 bg-gradient-to-r from-[#6A3093] to-[#A044FF]  text-white font-bold rounded-full shadow-lg shadow-purple-500/40 transform transition duration-300"
          >
            Send Message
          </motion.button>

          {/* Success Message */}
          {success && (
            <motion.p
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-green-500 dark:text-green-400 text-center mt-2 font-medium"
            >
              Your message has been sent successfully! We'll be in touch soon.
            </motion.p>
          )}
        </motion.form>

        {/* Social Links & Info */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
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
          </div>
        </motion.div>
      </div>
    </div>
  );
}