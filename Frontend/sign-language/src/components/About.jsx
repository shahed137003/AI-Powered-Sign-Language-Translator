import React from 'react';
import Mobile from '../assets/about.svg';
import { FaMicrophone, FaVideo, FaRobot, FaShareAlt, FaMobileAlt,FaRunning ,FaHands } from "react-icons/fa";
import { motion } from "framer-motion";

export default function About() {
  
  // --- UPDATED DATA STRUCTURE (Kept) ---
  // 1. Core Translation Journey (Sequential)
  const journeySteps = [
    { 
      icon: FaMicrophone, 
      text: "1. Speak or Type Your Message", 
      details: "Our intelligent AI processes spoken or typed English, capturing language nuances and accents for the most accurate translation." 
    },
    { 
      icon: FaHands, 
      text: "2. Real-Time AI Sign Translation", 
      details: "The AI instantly generates a synchronized, realistic sign language animation using our 3D avatar." 
    },
    { 
      icon: FaShareAlt, 
      text: "3. Communicate & Share Seamlessly", 
      details: "The sign translation appears instantly on-screen, ready to be understood, saved, or effortlessly shared across platforms." 
    },
  ];

  // 2. Key Capabilities (Non-Sequential)
  const capabilityFeatures = [
    { 
      icon: FaVideo, 
      text: "Upload Videos for Automatic Signing", 
      details: "Upload files or links to instantly generate synchronized sign animations — perfect for educational and accessible content." 
    },
    { 
      icon: FaRobot, // Reusing FaRobot for the Chatbot, as it's a key independent feature
      text: "Interactive AI Chatbot Practice", 
      details: "Practice signing with our AI-powered chatbot designed for real dialogue, instant feedback, and skill improvement." 
    },
    { 
      icon: FaMobileAlt, 
      text: "Access LinguaSign Anywhere", 
      details: "Use the mobile app to translate, learn, or chat anywhere — including offline modes for remote areas." 
    },
  ];

  // --- ANIMATION VARIANTS (Kept) ---
  const fadeUp = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0 }
  };

  const shine = {
    hidden: { opacity: 0, scale: 0.9 },
    visible: { 
      opacity: 1, 
      scale: 1,
      transition: { duration: 0.8, delay: 0.4, ease: "easeOut" } 
    }
  };

  return (
    <div className="w-full bg-gray-50 dark:bg-[#0f0c29] py-24 px-6 lg:px-20 relative overflow-hidden"> 

      {/* 🌌 Decorative Floating Lights (Crucial for the blur effect to work) */}
      <motion.div 
        animate={{ opacity: [0.4, 0.7, 0.4], scale: [1, 1.1, 1] }}
        transition={{ duration: 6, repeat: Infinity }}
        className="absolute top-0 right-0 w-[450px] h-[450px] bg-indigo-600/20 rounded-full blur-[140px]"
      />
      <motion.div 
        animate={{ opacity: [0.3, 0.6, 0.3], scale: [1, 1.05, 1] }}
        transition={{ duration: 7, repeat: Infinity }}
        className="absolute bottom-0 left-0 w-[450px] h-[450px] bg-purple-600/20 rounded-full blur-[140px]"
      />

      {/* INTRO SECTION (Kept) */}
      <div className="max-w-7xl mx-auto flex flex-col-reverse lg:flex-row items-center gap-12 mb-20 relative z-10">
        
        {/* Intro Text (Kept) */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          transition={{ duration: 0.9 }}
          className="flex-1 text-center lg:text-left space-y-4"
        >
          <motion.h1
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
            className="text-4xl sm:text-5xl font-extrabold mb-6 text-gray-900 dark:text-white"
          >
            <span className="bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
              AI-powered Communication
            </span>{" "} 
            for Everyone
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.2 }}
            className="text-gray-700 dark:text-gray-300 text-lg sm:text-xl leading-relaxed"
          >
            LinguaSign bridges communication through real-time gesture recognition, intelligent animations, and multilingual speech processing — making every conversation accessible.
          </motion.p>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.35 }}
            className="text-gray-700 dark:text-gray-300 text-lg sm:text-xl leading-relaxed"
          >
            Whether you're learning sign language, teaching, or enabling accessibility, LinguaSign elevates communication into an inclusive experience.
          </motion.p>
        </motion.div>

        {/* INTRO IMAGE with ENHANCED GLASSMOPRHISM */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          variants={shine}
          viewport={{ once: true }}
          className="flex-1 flex justify-center lg:justify-end relative"
        >
          <motion.div
            whileHover={{ scale: 1.03, rotate: 1 }}
            transition={{ duration: 0.4 }}
            // Enhanced Glassmorphism Classes:
            className="relative w-full max-w-md p-4 
                       bg-white/5 dark:bg-white/5  /* Increased Transparency */
                       rounded-2xl shadow-2xl backdrop-blur-md /* Increased Blur */
                       border border-white/20 border-opacity-30  /* Light Border for Glass Edge */
                       "
          >
            {/* Inner Neon Glow (Kept) */}
            <div className="absolute inset-0 rounded-2xl border-4 border-transparent 
            shadow-[0_0_25px_rgba(160,68,255,0.5)] transition-all duration-700"></div>

            <motion.img 
              src={Mobile} 
              alt="LinguaSign App" 
              className="w-full rounded-xl"
              animate={{ y: [0, -8, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            />
          </motion.div>
        </motion.div>
      </div>


      {/* --- CORE JOURNEY SECTION TITLE (Kept) --- */}
      <motion.h3
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        viewport={{ once: true }}
        className="relative text-center text-3xl sm:text-4xl font-extrabold mb-20 text-gray-900 dark:text-white"
      >
        {/* Left bar */}
        <motion.span
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
          className="absolute left-[30%] top-1/2 -translate-y-1/2 w-10 h-[3px] bg-gradient-to-r from-transparent to-[#A044FF] rounded-full"
        />

        {/* Text - New Title for clarity */}
        <span className="bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
          LinguaSign Journey
        </span>

        {/* Right bar */}
        <motion.span
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
          className="absolute right-[30%] top-1/2 -translate-y-1/2 w-10 h-[3px] bg-gradient-to-l from-transparent to-[#A044FF] rounded-full"
        />
      </motion.h3>


      {/* CORE JOURNEY STEPS (Vertical Timeline using journeySteps) */}
      <div className="relative max-w-5xl mx-auto mt-20">

        {/* Animated Vertical Line (Kept) */}
        <motion.div
          initial={{ height: 0 }}
          whileInView={{ height: "100%" }}
          transition={{ duration: 1.8, ease: "easeInOut" }}
          viewport={{ once: true }}
          className="absolute left-1/2 transform -translate-x-1/2 w-1 bg-gradient-to-b from-purple-300 to-purple-700 rounded-full"
        />

        {/* Timeline Steps */}
        <div className="space-y-20 relative z-10">
          {journeySteps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: index % 2 === 0 ? -70 : 70 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.9, delay: index * 0.15, ease: "easeOut" }}
              viewport={{ once: true }}
              className={`flex items-center gap-10 w-full 
                ${index % 2 === 0 ? "flex-row" : "flex-row-reverse"}`}
            >

              {/* Card - Enhanced Glassmorphism Classes */}
              <motion.div
                whileHover={{ scale: 1.03 }}
                transition={{ duration: 0.3 }}
                className="w-1/2 p-6 rounded-2xl shadow-xl 
                            bg-white/5 dark:bg-white/5 /* Increased Transparency */
                            backdrop-blur-xl border border-white/20 border-opacity-30 /* Light Border */
                            hover:shadow-purple-500/40 transition-all duration-500"
              >
                <p className="text-xl font-bold text-gray-900 dark:text-gray-200 mb-2">
                  {step.text}
                </p>
                <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed">
                  {step.details}
                </p>
              </motion.div>

              {/* Icon Orb (Kept) */}
              <motion.div
                animate={{ y: [0, -10, 0] }}
                transition={{ repeat: Infinity, duration: 3.2, ease: "easeInOut" }}
                className="w-20 h-20 rounded-full bg-purple-200 dark:bg-purple-900/50 
                            flex items-center justify-center shadow-lg border border-purple-400/30"
              >
                <step.icon className="text-3xl text-purple-700 dark:text-purple-300" />
              </motion.div>

            </motion.div>
          ))}
        </div>
      </div>
      
      {/* --- KEY CAPABILITIES SECTION (Grid) --- */}
      <div className="max-w-7xl mx-auto mt-32">
        <motion.h3
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="relative text-center text-3xl sm:text-4xl font-extrabold mb-12 text-gray-900 dark:text-white"
        >
          <span className="bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
            More Key Capabilities
          </span>
        </motion.h3>

        {/* Key Capabilities Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 lg:gap-12">
          {capabilityFeatures.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6, delay: index * 0.2 }}
              viewport={{ once: true }}
              // Enhanced Glassmorphism Classes:
              className="p-8 rounded-2xl shadow-xl 
                         bg-white/5 dark:bg-white/5 /* Increased Transparency */
                         backdrop-blur-xl border border-white/20 border-opacity-30 /* Light Border */
                         hover:shadow-purple-500/40 transition-all duration-500 flex flex-col items-center text-center space-y-4"
            >
              <div className="w-16 h-16 rounded-full bg-purple-200 dark:bg-purple-900/50 flex items-center justify-center shadow-lg border border-purple-400/30">
                <feature.icon className="text-3xl text-purple-700 dark:text-purple-300" />
              </div>
              <h4 className="text-xl font-bold text-gray-900 dark:text-gray-200">{feature.text}</h4>
              <p className="text-gray-600 dark:text-gray-400 leading-relaxed text-sm">{feature.details}</p>
            </motion.div>
          ))}
        </div>
      </div>

    </div>
  );
}