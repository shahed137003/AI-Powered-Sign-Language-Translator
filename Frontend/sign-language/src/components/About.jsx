import React from 'react';
import Mobile from '../assets/about.svg';
import { FaMicrophone, FaVideo, FaRobot, FaShareAlt, FaMobileAlt } from "react-icons/fa";
import { motion } from "framer-motion";

export default function About() {
  const steps = [
    { 
      icon: FaMicrophone, 
      text: "Voice-to-Avatar Sign Translation", 
      details: "Our AI converts spoken or typed English into real-time sign language animations, supporting multiple accents for clear, immediate communication." 
    },
    { 
      icon: FaVideo, 
      text: "Upload Videos for Automatic Signing", 
      details: "Upload videos or links to automatically generate synchronized sign language animations, perfect for accessible content creation and sharing." 
    },
    { 
      icon: FaRobot, 
      text: "Interactive AI Chatbot Practice", 
      details: "Practice sign language with our intelligent chatbot, providing interactive exercises, instant feedback, and conversational skills improvement." 
    },
    { 
      icon: FaShareAlt, 
      text: "Seamless Translation Sharing", 
      details: "Easily share animations via social media, email, or within classrooms, fostering inclusive communication across all digital platforms." 
    },
    { 
      icon: FaMobileAlt, 
      text: "Access LinguaSign Anywhere", 
      details: "The mobile app brings all platform features to your smartphone. Translate, chat, or share content on-the-go with essential offline support." 
    },
  ];

  // Motion variants for fade-in + upward movement
  const fadeUp = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0 }
  };
  
  // Motion variant for the main image effect
  const shine = {
    hidden: { opacity: 0, x: 50 },
    visible: { 
      opacity: 1, 
      x: 0, 
      transition: { duration: 0.8, delay: 0.4 } 
    }
  };

  return (
    <div className="w-full bg-gray-50 dark:bg-[#0f0c29] py-24 px-6 lg:px-20 relative overflow-hidden transition-colors duration-500">
      
      {/* Background Orbs/Glows */}
      <div className="absolute top-0 right-0 w-[400px] h-[400px] bg-indigo-600/10 rounded-full blur-[100px] pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-purple-600/10 rounded-full blur-[100px] pointer-events-none" />

      {/* --- INTRO SECTION --- */}
      <div className="max-w-7xl mx-auto flex flex-col-reverse lg:flex-row items-center gap-12 mb-20 relative z-10">
        
        {/* Text Section (Left) */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="flex-1 text-center lg:text-left space-y-4"
        >
          <h1 className="text-4xl sm:text-5xl font-extrabold mb-6 text-gray-900 dark:text-white">
            <span className="bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] bg-clip-text text-transparent">
              AI-powered Communication
            </span>{" "}
            for Everyone
          </h1>
          <p className="text-gray-700 dark:text-gray-300 text-lg sm:text-xl leading-relaxed">
            LinguaSign is an AI-powered platform that bridges communication between Deaf, hard-of-hearing, and hearing individuals. Our system uses real-time gesture recognition, speech processing, and multilingual translation to create an accessible and inclusive experience.
          </p>
          <p className="text-gray-700 dark:text-gray-300 text-lg sm:text-xl leading-relaxed">
            Whether you’re learning sign language, teaching, or simply trying to communicate inclusively, LinguaSign makes conversations natural, interactive, and efficient. 
          </p>
        </motion.div>

        {/* Image Section (Right) - Enhanced Frame */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={shine}
          className="flex-1 flex justify-center lg:justify-end relative"
        >
          <div className="relative w-full max-w-md p-4 bg-white/10 dark:bg-gray-800/50 rounded-2xl shadow-2xl backdrop-blur-sm border border-purple-500/30">
             {/* Inner Glowing Border/Reflection Effect */}
             <div className="absolute inset-0 rounded-2xl border-4 border-transparent pointer-events-none transition-all duration-700 
               shadow-[0_0_20px_rgba(160,68,255,0.4)] hover:shadow-[0_0_35px_rgba(160,68,255,0.7)]"></div>
              
             <img
              src={Mobile}
              alt="About LinguaSign Mobile Interface"
              className="w-full rounded-xl object-cover"
            />
          </div>
        </motion.div>
      </div>

      <hr className="my-16 dark:border-gray-700 border-t-2 border-dashed" />

      {/* --- HOW TO USE SECTION --- */}
      <motion.h3
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        viewport={{ once: true }}
        className="text-3xl sm:text-4xl font-extrabold mb-16 text-center text-gray-900 dark:text-white"
      >
        <span className="bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] bg-clip-text text-transparent">
          How to Use
        </span>{" "}
        LinguaSign
      </motion.h3>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
        {steps.map((step, index) => (
          <motion.div
            key={index}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeUp}
            transition={{ duration: 0.8, delay: index * 0.15 }}
            className="flex flex-col p-6 dark:bg-[#1a163a]/70 backdrop-blur-xl border border-gray-200 dark:border-white/10 rounded-2xl shadow-lg 
              hover:shadow-[0_0_25px_rgba(160,68,255,0.4)] hover:-translate-y-1 transform transition duration-500"
          >
            <div className="flex items-center justify-center w-16 h-16 mb-4 rounded-full bg-purple-100 dark:bg-purple-900/50 shadow-md">
              <step.icon className="text-3xl text-purple-600 dark:text-purple-400" />
            </div>
            
            <p className="text-gray-900 dark:text-gray-200 font-bold mb-3 text-xl">{step.text}</p>
            <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed">{step.details}</p>
          </motion.div>
        ))}
      </div>
      
    </div>
  );
}