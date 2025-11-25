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
            LinguaSign is an AI-powered platform bridging communication between Deaf, hard-of-hearing, and hearing individuals. Accessible, inclusive, and interactive.
          </p>
        </div>

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
            >
              Subscribe
            </button>
          </form>

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