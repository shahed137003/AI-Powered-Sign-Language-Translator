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
            LinguaSign is an AI-powered platform bridging communication between Deaf, hard-of-hearing, and hearing individuals. Accessible, inclusive, and interactive.
          </p>
        </div>

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
            >
              Subscribe
            </button>
          </form>
        </div>
      </div>

      {/* Footer Bottom */}
      <div className="mt-12 border-t border-gray-200 dark:border-gray-700 pt-6 text-center text-[#A044FF] text-sm">
        &copy; {new Date().getFullYear()} LinguaSign. All rights reserved.
      </div>
    </footer>
  );
}
