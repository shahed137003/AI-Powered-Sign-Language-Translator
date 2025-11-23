import React, { useState } from "react";
import { FaFacebookF, FaTwitter, FaLinkedinIn, FaInstagram } from "react-icons/fa";
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
    console.log("Message sent:", formData);
    setSuccess(true);
    setFormData({ name: "", email: "", message: "" });
    setTimeout(() => setSuccess(false), 4000);
  };

  // Motion variants
  const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

  return (
    <div className="w-full bg-gray-50 dark:bg-gray-900 py-24 px-4 sm:px-6 lg:px-20">
      <motion.div
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        variants={fadeUp}
        transition={{ duration: 0.8 }}
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
        {/* Contact Form */}
        <motion.form
          initial={{ opacity: 0, x: -50 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          onSubmit={handleSubmit}
          className="flex-1 bg-purple-50 dark:bg-gray-800/40 backdrop-blur-lg border border-white/30 dark:border-gray-600 p-8 rounded-3xl shadow-2xl flex flex-col gap-6 hover:shadow-3xl transition duration-500 transform hover:-translate-y-1"
        >
          {/* Name Input */}
          <div className="relative">
            <input
              type="text"
              name="name"
              value={formData.name}
              onChange={handleChange}
              required
              placeholder="Your Name"
              className="w-full p-4 pl-12 border border-gray-300 dark:border-gray-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#A044FF] placeholder-transparent peer bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-200"
            />
            <label className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 text-sm peer-placeholder-shown:top-1/2 peer-placeholder-shown:text-gray-400 peer-placeholder-shown:text-base peer-focus:top-2 peer-focus:text-[#6A3093] peer-focus:text-sm transition-all">
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
              className="w-full p-4 pl-12 border border-gray-300 dark:border-gray-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#A044FF] placeholder-transparent peer bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-200"
            />
            <label className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 text-sm peer-placeholder-shown:top-1/2 peer-placeholder-shown:text-gray-400 peer-placeholder-shown:text-base peer-focus:top-2 peer-focus:text-[#6A3093] peer-focus:text-sm transition-all">
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
              className="w-full p-4 pt-6 border border-gray-300 dark:border-gray-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#A044FF] placeholder-transparent peer resize-none bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-200"
            ></textarea>
            <label className="absolute left-4 top-3 text-gray-500 dark:text-gray-400 text-sm peer-placeholder-shown:top-6 peer-placeholder-shown:text-gray-400 peer-placeholder-shown:text-base peer-focus:top-3 peer-focus:text-[#6A3093] peer-focus:text-sm transition-all">
              Your Message
            </label>
          </div>

          {/* Submit Button */}
          <motion.button
            type="submit"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.98 }}
            transition={{ type: "spring", stiffness: 300 }}
            className="px-6 py-3 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] text-white font-semibold rounded-full shadow-xl transform transition duration-300"
          >
            Send Message
          </motion.button>

          {/* Success Message */}
          {success && (
            <motion.p
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-green-500 dark:text-green-400 text-center mt-4 animate-pulse"
            >
              Your message has been sent successfully!
            </motion.p>
          )}
        </motion.form>

        {/* Social Links */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
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
          </div>
        </motion.div>
      </div>
    </div>
  );
}
