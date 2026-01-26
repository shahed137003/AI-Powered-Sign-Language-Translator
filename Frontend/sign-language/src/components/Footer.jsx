<<<<<<< HEAD
import React from "react";
import { FaFacebookF, FaTwitter, FaLinkedinIn, FaInstagram } from "react-icons/fa";

export default function Footer() {
  return (
    <footer className="w-full bg-purple-50 dark:bg-gray-900 text-gray-800 dark:text-gray-200 py-16 px-4 sm:px-6 lg:px-20 dark:border-t dark:border-[#6A3093]/60">
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-12">

        {/* About Section */}
        <div>
          <h3 className="text-2xl font-bold mb-4 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] bg-clip-text text-transparent">
            LinguaSign
          </h3>
          <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
=======
import React, { useState } from "react";
import { FaFacebookF, FaTwitter, FaLinkedinIn, FaInstagram, FaChevronRight } from "react-icons/fa";
import { motion } from "framer-motion";

export default function Footer() {
  const [email, setEmail] = useState("");

  const handleSubscribe = (e) => {
    e.preventDefault();
    // In a real app, this would send data to a mailing list service
    console.log("Subscribing email:", email);
    setEmail("");
    // Add a temporary success notification here if needed
  };

  const socialLinks = [
    { icon: FaFacebookF, href: "https://facebook.com", hoverBg: "hover:bg-[#3b5998]" },
    { icon: FaTwitter, href: "https://twitter.com", hoverBg: "hover:bg-[#1da1f2]" },
    { icon: FaLinkedinIn, href: "https://linkedin.com", hoverBg: "hover:bg-[#0077b5]" },
    { icon: FaInstagram, href: "https://instagram.com", hoverBg: "hover:bg-[#e4405f]" },
  ];

  const quickLinks = [
    { name: "About Us", href: "#about" },
    { name: "Features", href: "#features" },
    { name: "How to Use", href: "#how-to-use" },
    { name: "Contact", href: "#contact" },
    { name: "Terms of Service", href: "#terms" },
    { name: "Privacy Policy", href: "#privacy" },
  ];

  return (
    // Use the deep dark background color for consistency
    <footer className="w-full bg-gray-50 dark:bg-[#0f0c29] text-gray-800 dark:text-gray-300 py-16 px-6 lg:px-20 border-t dark:border-purple-500/30 transition-colors duration-500">
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12">

        {/* 1. About Section */}
        <div>
          <h3 className="text-3xl font-extrabold mb-4">
            <span className="bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
              LinguaSign
            </span>
          </h3>
          <p className="text-gray-600 dark:text-gray-400 leading-relaxed text-sm">
>>>>>>> e251330 (Add frontend, backend, and ai_service)
            LinguaSign is an AI-powered platform bridging communication between Deaf, hard-of-hearing, and hearing individuals. Accessible, inclusive, and interactive.
          </p>
        </div>

<<<<<<< HEAD
        {/* Quick Links */}
        <div>
          <h4 className="text-xl font-semibold mb-4">Quick Links</h4>
          <ul className="space-y-2">
            <li><a href="#about" className="hover:text-[#A044FF] transition">About Us</a></li>
            <li><a href="#features" className="hover:text-[#A044FF] transition">Features</a></li>
            <li><a href="#how-to-use" className="hover:text-[#A044FF] transition">How to Use</a></li>
            <li><a href="#contact" className="hover:text-[#A044FF] transition">Contact</a></li>
          </ul>
        </div>

        {/* Social Media */}
        <div>
          <h4 className="text-xl font-semibold mb-4">Follow Us</h4>
          <div className="flex gap-4">
            <a href="https://facebook.com" target="_blank" rel="noopener noreferrer" className="p-3 bg-purple-200 dark:bg-gray-700 rounded-full hover:bg-[#3b5998] hover:text-white transition duration-300">
              <FaFacebookF />
            </a>
            <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className="p-3 bg-purple-200 dark:bg-gray-700 rounded-full hover:bg-[#1da1f2] hover:text-white transition duration-300">
              <FaTwitter />
            </a>
            <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" className="p-3 bg-purple-200 dark:bg-gray-700 rounded-full hover:bg-[#0077b5] hover:text-white transition duration-300">
              <FaLinkedinIn />
            </a>
            <a href="https://instagram.com" target="_blank" rel="noopener noreferrer" className="p-3 bg-purple-200 dark:bg-gray-700 rounded-full hover:bg-[#e4405f] hover:text-white transition duration-300">
              <FaInstagram />
            </a>
          </div>
        </div>

        {/* Newsletter Subscription */}
        <div>
          <h4 className="text-xl font-semibold mb-4">Subscribe</h4>
          <p className="text-gray-600 dark:text-gray-300 mb-4">
            Get updates, tips, and news about LinguaSign directly in your inbox.
          </p>
          <form className="flex flex-col sm:flex-row gap-2">
            <input
              type="email"
              placeholder="Your email"
              className="p-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-[#A044FF]"
            />
            <button
              type="submit"
              className="px-6 py-3 rounded-xl bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] text-white font-semibold shadow hover:scale-105 transform transition duration-300"
=======
        {/* 2. Quick Links */}
        <div>
          <h4 className="text-xl font-semibold mb-6 text-gray-900 dark:text-white border-b border-purple-500/50 pb-2">Navigation</h4>
          <ul className="space-y-3">
            {quickLinks.slice(0, 4).map((link, index) => (
              <li key={index}>
                <a 
                  href={link.href} 
                  className="flex items-center group text-gray-700 dark:text-gray-400 hover:text-purple-500 transition duration-300 text-base"
                >
                  <FaChevronRight className="w-3 h-3 mr-2 text-purple-400 group-hover:text-purple-600 transition" />
                  {link.name}
                </a>
              </li>
            ))}
          </ul>
        </div>

        {/* 3. Legal Links & Resources */}
        <div>
          <h4 className="text-xl font-semibold mb-6 text-gray-900 dark:text-white border-b border-purple-500/50 pb-2">Resources</h4>
          <ul className="space-y-3">
             {quickLinks.slice(4).map((link, index) => (
              <li key={index}>
                <a 
                  href={link.href} 
                  className="flex items-center group text-gray-700 dark:text-gray-400 hover:text-purple-500 transition duration-300 text-base"
                >
                  <FaChevronRight className="w-3 h-3 mr-2 text-purple-400 group-hover:text-purple-600 transition" />
                  {link.name}
                </a>
              </li>
            ))}
            <li>
                <a 
                  href="#support" 
                  className="flex items-center group text-gray-700 dark:text-gray-400 hover:text-purple-500 transition duration-300 text-base"
                >
                  <FaChevronRight className="w-3 h-3 mr-2 text-purple-400 group-hover:text-purple-600 transition" />
                  Support
                </a>
              </li>
          </ul>
        </div>

        {/* 4. Newsletter Subscription & Social Media */}
        <div>
          <h4 className="text-xl font-semibold mb-6 text-gray-900 dark:text-white border-b border-purple-500/50 pb-2">Stay Connected</h4>
          <p className="text-gray-600 dark:text-gray-400 mb-4 text-sm">
            Join our community for platform updates, sign language tips, and news.
          </p>
          
          {/* Subscription Form */}
          <form onSubmit={handleSubscribe} className="flex flex-col gap-3 mb-8">
            <input
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="p-3 rounded-xl border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800/50 text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-purple-500 placeholder:text-gray-500 dark:placeholder:text-gray-500 transition-colors"
            />
            <button
              type="submit"
              className="w-full px-6 py-3 rounded-xl bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] text-white font-semibold shadow-lg shadow-purple-500/30 hover:scale-[1.01] transform transition duration-300"
>>>>>>> e251330 (Add frontend, backend, and ai_service)
            >
              Subscribe
            </button>
          </form>
<<<<<<< HEAD
        </div>
      </div>

      {/* Footer Bottom */}
      <div className="mt-12 border-t border-gray-200 dark:border-gray-700 pt-6 text-center text-[#A044FF] text-sm">
        &copy; {new Date().getFullYear()} LinguaSign. All rights reserved.
      </div>
    </footer>
  );
}
=======

          {/* Social Icons */}
          <div className="flex gap-4">
            {socialLinks.map((link, index) => (
              <motion.a
                key={index}
                href={link.href}
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.15, rotate: 5 }}
                whileTap={{ scale: 0.9 }}
                className={`p-3 rounded-full shadow-lg transition duration-300 text-gray-700 dark:text-gray-300 dark:bg-gray-800/70 hover:text-white ${link.hoverBg}`}
              >
                <link.icon className="text-xl" />
              </motion.a>
            ))}
          </div>
        </div>
      </div>

      {/* Footer Bottom (Copyright) */}
      <div className="mt-12 border-t border-gray-200 dark:border-gray-700 pt-6 text-center text-gray-500 dark:text-gray-400 text-sm">
        &copy; {new Date().getFullYear()} LinguaSign. All rights reserved. Built with purpose.
      </div>
    </footer>
  );
}
>>>>>>> e251330 (Add frontend, backend, and ai_service)
