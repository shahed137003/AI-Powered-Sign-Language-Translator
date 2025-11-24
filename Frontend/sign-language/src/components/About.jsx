import React from 'react';
import Mobile from '../assets/about.svg';
import { FaMicrophone, FaVideo, FaRobot, FaShareAlt, FaMobileAlt } from "react-icons/fa";
import { motion } from "framer-motion";

export default function About() {
  const steps = [
    { 
      icon: FaMicrophone, 
      text: "Speak or type your message using our voice-to-avatar signs translation tool.", 
      details: "Our AI accurately converts spoken or typed English into real-time sign language animations, supporting multiple accents and ensuring clear communication." 
    },
    { 
      icon: FaVideo, 
      text: "Upload a video or link to automatically generate sign language animations.", 
      details: "Perfect for educational content, presentations, or social media. The system analyzes speech, tone, and context to create precise gestures that match the original content." 
    },
    { 
      icon: FaRobot, 
      text: "Practice with our AI Sign Language Chatbot to improve your skills.", 
      details: "The chatbot provides interactive exercises, instant feedback, and conversational practice to help you learn sign language faster and more confidently." 
    },
    { 
      icon: FaShareAlt, 
      text: "Share translations with friends, students, or colleagues to make communication inclusive.", 
      details: "Easily export or share animations via social media, email, or within classrooms, fostering inclusive communication across all platforms." 
    },
    { 
      icon: FaMobileAlt, 
      text: "Access LinguaSign from anywhere with our mobile app.", 
      details: "The app brings all platform features to your smartphone. Translate, upload videos, chat with the AI, or share content on-the-go with offline support for basic translations." 
    },
  ];

  // Motion variants for fade-in + upward movement
  const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  return (
    <div className="w-full bg-gray-50 dark:bg-gray-900 py-24 px-4 sm:px-6 lg:px-20">
      {/* Intro Section */}
      <motion.h1
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        variants={fadeUp}
        transition={{ duration: 0.8 }}
        className="text-5xl font-extrabold text-center mb-4 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent"
      >
        About Us
      </motion.h1>

      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        whileInView={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.8 }}
        className="w-24 h-1 mx-auto mb-10 rounded-full bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF]"
      />

      <div className="max-w-7xl mx-auto flex flex-col-reverse lg:flex-row items-center gap-12 mb-16">
        {/* Text Section */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="flex-1 text-center lg:text-left"
        >
          <h2 className="text-4xl sm:text-5xl font-extrabold mb-6 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
            AI-powered Communication for Everyone
          </h2>
          <p className="text-gray-700 dark:text-gray-200 text-lg sm:text-xl leading-relaxed mb-4">
            LinguaSign is an AI-powered platform that bridges communication between Deaf, hard-of-hearing, and hearing individuals.
            Our system uses real-time gesture recognition, speech processing, and multilingual translation to create an accessible and inclusive experience.
          </p>
          <p className="text-gray-700 dark:text-gray-200 text-lg sm:text-xl leading-relaxed">
            Whether you’re learning sign language, teaching, or simply trying to communicate inclusively, LinguaSign makes conversations natural, interactive, and efficient.
          </p>
        </motion.div>

        {/* Image Section */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          whileInView={{ opacity: 1, x: 0 }}
          
          transition={{ duration: 0.8, delay: 0.4 }}
          className="flex-1 flex justify-center lg:justify-end"
        >
          <img
            src={Mobile}
            alt="About LinguaSign"
            className="w-full max-w-md rounded-xl shadow-xl dark:shadow-gray-700"
          />
        </motion.div>
      </div>

      <motion.h3
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        viewport={{ once: true }}
        className="text-3xl sm:text-4xl font-bold mb-12 text-center bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent"
      >
        How to Use LinguaSign
      </motion.h3>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-10">
        {steps.map((step, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: index * 0.2 }}
            viewport={{ once: true }}
            className="flex flex-col items-center p-8 bg-white/10 dark:bg-[#10141F]/70 backdrop-blur-xl border border-white/10
                       dark:border-[#6A3093]/30 rounded-2xl shadow-xl hover:shadow-[0_0_20px_rgba(176,68,255,0.5)]
                       hover:-translate-y-2 transform transition duration-500"
          >
            <step.icon className="text-6xl text-[#A044FF] mb-5" />
            <p className="text-gray-900 dark:text-gray-200 text-center font-semibold mb-3 text-lg">{step.text}</p>
            <p className="text-gray-600 dark:text-gray-400 text-center text-sm leading-relaxed">{step.details}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
