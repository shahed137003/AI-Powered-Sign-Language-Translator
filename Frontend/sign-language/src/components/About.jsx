<<<<<<< HEAD
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
=======
import React, { useEffect, useRef, useState } from 'react';
import Mobile from '../assets/about.svg';
import { FaMicrophone, FaVideo, FaRobot, FaShareAlt, FaMobileAlt, FaHands, FaEye, FaUsers, FaRocket, FaChartLine, FaGlobe, FaLightbulb, FaHeart, FaAward, FaShieldAlt, FaSyncAlt } from "react-icons/fa";
import { motion, useInView, useAnimation } from "framer-motion";
import { FaUserTie, FaLaptopCode, FaBrain, FaCode, FaUserAlt, FaHandshake, FaChartBar, FaCloud, FaDatabase } from "react-icons/fa";
import { BsStars, BsLightningFill, BsGearFill } from "react-icons/bs";
import { TbArrowWaveRightDown, TbArrowWaveLeftDown, TbSparkles, TbTargetArrow } from "react-icons/tb";

export default function About() {
  const containerRef = useRef(null);
  const headerRef = useRef(null);
  
  const isHeaderInView = useInView(headerRef, { once: true, amount: 0.5 });
  const isInView = useInView(containerRef, { once: true, amount: 0.3 });
  
  const headerControls = useAnimation();
  const controls = useAnimation();

  useEffect(() => {
    if (isHeaderInView) {
      headerControls.start("visible");
    }
  }, [headerControls, isHeaderInView]);

  useEffect(() => {
    if (isInView) {
      controls.start("visible");
    }
  }, [controls, isInView]);

  const fadeUp = {
    hidden: { opacity: 0, y: 40 },
    visible: {
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.8,
        ease: [0.22, 1, 0.36, 1]
      }
    }
  };

  const fade = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        duration: 1,
        ease: "easeOut"
      }
    }
  };

  const scaleIn = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: {
      opacity: 1, 
      scale: 1,
      transition: {
        duration: 0.6,
        ease: "backOut"
      }
    }
  };

  const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.1
      }
    }
  };

  const journeySteps = [
    { 
      icon: FaMicrophone, 
      text: "1. Speak or Type Your Message", 
      details: "Our advanced AI captures spoken or typed English with 99% accuracy, understanding subtle nuances and context.",
      gradient: "from-[#6A3093] to-[#A044FF]",
      delay: 0
    },
    { 
      icon: FaBrain, 
      text: "2. AI Processing & Analysis", 
      details: "Neural networks analyze language patterns, context, and intent for precise translation.",
      gradient: "from-[#A044FF] to-[#BF5AE0]",
      delay: 0.1
    },
    { 
      icon: FaHands, 
      text: "3. Real-Time Sign Translation", 
      details: "Generates fluid, natural sign language animations with expressive 3D avatars.",
      gradient: "from-[#BF5AE0] to-[#D946EF]",
      delay: 0.2
    },
    { 
      icon: FaShareAlt, 
      text: "4. Instant Communication", 
      details: "View, save, or share translations instantly across any platform or device.",
      gradient: "from-[#D946EF] to-[#EC4899]",
      delay: 0.3
    },
  ];

  const capabilityFeatures = [
    { 
      icon: FaVideo, 
      text: "Video-to-Sign Translation", 
      details: "Upload any video content and automatically generate synchronized sign language animations.",
      stats: "95% accuracy rate",
      gradient: "from-[#6A3093] to-[#A044FF]"
    },
    { 
      icon: FaRobot,
      text: "AI Chatbot Practice", 
      details: "Practice sign language with intelligent chatbot providing real-time feedback.",
      stats: "24/7 availability",
      gradient: "from-[#A044FF] to-[#BF5AE0]"
    },
    { 
      icon: FaMobileAlt, 
      text: "Cross-Platform Access", 
      details: "Mobile app with offline mode, camera recognition, and instant translation.",
      stats: "10M+ downloads",
      gradient: "from-[#BF5AE0] to-[#D946EF]"
    },
    { 
      icon: FaUsers, 
      text: "Multi-User Collaboration", 
      details: "Real-time translation for group conversations in meetings and classrooms.",
      stats: "50+ simultaneous users",
      gradient: "from-[#D946EF] to-[#EC4899]"
    },
    { 
      icon: FaGlobe, 
      text: "100+ Language Support", 
      details: "Translate between sign language and over 100 spoken languages worldwide.",
      stats: "Global coverage",
      gradient: "from-[#EC4899] to-[#F97316]"
    },
    { 
      icon: FaChartLine, 
      text: "Analytics Dashboard", 
      details: "Track progress, usage statistics, and performance metrics with insights.",
      stats: "Real-time analytics",
      gradient: "from-[#F97316] to-[#F59E0B]"
    },
    { 
      icon: FaShieldAlt, 
      text: "Enterprise Security", 
      details: "End-to-end encryption, compliance, and advanced security features.",
      stats: "Bank-level security",
      gradient: "from-[#F59E0B] to-[#10B981]"
    },
    { 
      icon: FaSyncAlt, 
      text: "Real-time Sync", 
      details: "Instant synchronization across all devices with cloud backup.",
      stats: "<100ms latency",
      gradient: "from-[#10B981] to-[#0EA5E9]"
    },
  ];

  const team = [
    {
      name: "Shahd Mohamed",
      role: "AI & Frontend Engineer",
      icon: FaCode,
      bio: "Expert in building intelligent, user-friendly interfaces and real-time AI-powered experiences.",
      expertise: ["React", "TensorFlow.js", "UI/UX Design", "Framer Motion"],
      gradient: "from-[#6A3093] to-[#A044FF]"
    },
    {
      name: "Demiana Ayman",
      role: "AI & Backend Engineer",
      icon: FaLaptopCode,
      bio: "Specializes in backend systems, APIs, databases, and AI integration for scalable architectures.",
      expertise: ["Node.js", "Python", "AWS", "Microservices"],
      gradient: "from-[#A044FF] to-[#BF5AE0]"
    },
    {
      name: "Kareem Reda",
      role: "AI Engineer",
      icon: FaBrain,
      bio: "Focused on machine learning, model optimization, and advanced data-driven solutions.",
      expertise: ["PyTorch", "Computer Vision", "NLP", "MLOps"],
      gradient: "from-[#BF5AE0] to-[#D946EF]"
    },
    {
      name: "Yahya Aboamer",
      role: "AI Engineer",
      icon: FaCode,
      bio: "Passionate about neural networks, deep learning applications, and system intelligence.",
      expertise: ["Deep Learning", "TensorFlow", "System Design", "Kubernetes"],
      gradient: "from-[#D946EF] to-[#EC4899]"
    },
    {
      name: "Mariam Hany",
      role: "AI & Frontend Engineer",
      icon: FaBrain,
      bio: "Combines design and AI to build seamless, accessible user experiences powered by smart technology.",
      expertise: ["Next.js", "TypeScript", "AI/ML", "WebGL"],
      gradient: "from-[#6A3093] to-[#A044FF]"
    },
    {
      name: "Hussam Elsayed",
      role: "AI Engineer",
      icon: FaUserAlt,
      bio: "Expert in neural networks, deep learning applications, and scalable AI systems.",
      expertise: ["TensorFlow", "Data Science", "API Development", "Docker"],
      gradient: "from-[#6A3093] to-[#A044FF]"
    }
  ];

  const techStack = [
    { name: "React & Next.js", icon: FaCode, color: "text-blue-500", gradient: "from-blue-500 to-cyan-500" },
    { name: "TensorFlow", icon: FaBrain, color: "text-orange-500", gradient: "from-orange-500 to-red-500" },
    { name: "Node.js", icon: FaLaptopCode, color: "text-green-500", gradient: "from-green-500 to-emerald-500" },
    { name: "AWS Cloud", icon: FaCloud, color: "text-yellow-500", gradient: "from-yellow-500 to-amber-500" },
    { name: "WebRTC", icon: FaVideo, color: "text-red-500", gradient: "from-red-500 to-pink-500" },
    { name: "WebGL", icon: BsStars, color: "text-purple-500", gradient: "from-purple-500 to-pink-500" },
  ];

  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-gray-50 via-white to-purple-50/60 dark:from-[#0a0518] dark:via-[#110a2e] dark:to-[#1e0f5c] py-24 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
      
      {/* Premium Geometric Grid matching Home page */}
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
        {[...Array(25)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-purple-500/20 rounded-full"
            initial={{ 
              x: Math.random() * 100 + 'vw', 
              y: Math.random() * 100 + 'vh',
              scale: 0 
            }}
            animate={{ 
              y: [null, -30, 30, -20],
              x: [null, 20, -20, 10],
              scale: [0, 1, 1, 0],
              opacity: [0, 0.5, 0.5, 0]
            }}
            transition={{ 
              duration: Math.random() * 10 + 20,
              repeat: Infinity,
              ease: "linear",
              delay: Math.random() * 5
            }}
          />
        ))}
      </div>

      {/* Main Container */}
      <div className="relative z-10 max-w-7xl mx-auto">
        {/* Header Section with its own ref */}
        <div ref={headerRef}>
          <motion.div
            initial="hidden"
            animate={headerControls}
            variants={fadeUp}
            className="text-center mb-20"
          >
            {/* Premium Badge matching Home page */}
            <motion.div
              variants={fadeUp}
              whileHover={{ scale: 1.05, rotate: 1 }}
              className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-gradient-to-r from-purple-500/15 via-purple-400/10 to-purple-300/10 border border-purple-200/60 dark:border-purple-700/60 backdrop-blur-xl shadow-lg shadow-purple-500/10 relative overflow-hidden group mb-8"
            >
              <div className="relative">
                <span className="absolute animate-ping inline-flex h-3.5 w-3.5 rounded-full bg-purple-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-gradient-to-r from-purple-500 to-purple-400" />
              </div>
              <span className="text-sm font-bold bg-gradient-to-r from-purple-600 via-purple-500 to-purple-400 bg-clip-text text-transparent">
                About LinguaSign
              </span>
              <TbSparkles className="text-purple-500 ml-1" />
              <div className="absolute -inset-1 bg-gradient-to-r from-purple-500/0 via-purple-400/10 to-purple-500/0 group-hover:via-purple-400/20 transition-all duration-500" />
            </motion.div>

            <motion.h1
              variants={fadeUp}
              className="font-extrabold text-4xl sm:text-5xl lg:text-[53px] leading-tight mb-6"
            >
              <motion.span
                variants={fadeUp}
                className="block text-gray-900 dark:text-white"
              >
                Our Story
              </motion.span>
              <motion.span
                variants={fadeUp}
                transition={{ delay: 0.1 }}
                className="block bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent"
              >
                Revolutionizing Communication
              </motion.span>
              <motion.span
                variants={fadeUp}
                transition={{ delay: 0.2 }}
                className="block text-gray-900 dark:text-white"
              >
                Through AI Innovation
              </motion.span>
            </motion.h1>

            <motion.p
              variants={fadeUp}
              transition={{ delay: 0.3 }}
              className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed"
            >
              Discover the vision, mission, and team behind LinguaSign — the platform 
              transforming how the world communicates through cutting-edge AI-powered 
              sign language translation.
            </motion.p>

            {/* Decorative Elements */}
            <motion.div
              variants={fadeUp}
              transition={{ delay: 0.4 }}
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
        </div>

        {/* Rest of the content with containerRef */}
        <div ref={containerRef}>
          {/* Intro Section with Enhanced Design */}
          <div className="grid lg:grid-cols-2 gap-50 items-center mb-32">
            {/* Text Content */}
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, amount: 0.3 }}
              variants={fade}
              className="space-y-6"
            >
              <div className="inline-flex items-center gap-3 text-purple-600 dark:text-purple-400 font-semibold uppercase tracking-wider text-sm mb-4">
                <FaLightbulb className="text-lg" />
                Our Innovation Journey
              </div>
              
              <h2 className="text-4xl font-bold text-gray-900 dark:text-white">
                Where <span className="bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
                  AI Meets
                </span> Human Connection
              </h2>
              
              <p className="text-lg text-gray-600 dark:text-gray-300 leading-relaxed">
                LinguaSign was born from a vision to eliminate communication barriers using 
                artificial intelligence. Our journey began with a dedicated team of AI engineers 
                passionate about creating meaningful impact.
              </p>
              
              <p className="text-lg text-gray-600 dark:text-gray-300 leading-relaxed">
                Today, we combine state-of-the-art gesture recognition, natural language processing, 
                and expressive 3D animation to create a seamless bridge between sign language 
                and spoken communication for millions worldwide.
              </p>
              
              {/* Tech Stack - Enhanced with Individual Gradients */}
              <div className="pt-6">
                <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Powered By</h4>
                <div className="flex flex-wrap gap-3">
                  {techStack.map((tech, index) => {
                    const Icon = tech.icon;
                    return (
                      <motion.div
                        key={index}
                        whileHover={{ 
                          scale: 1.05,
                          y: -2,
                          boxShadow: `0 10px 25px -5px ${tech.color}40`
                        }}
                        className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-white/80 to-white/60 dark:from-white/10 dark:to-white/5 backdrop-blur-xl border border-purple-200/50 dark:border-purple-500/20 shadow-lg shadow-purple-100/20 dark:shadow-purple-900/20 group transition-all duration-300 cursor-pointer hover:shadow-xl hover:shadow-purple-200/30 dark:hover:shadow-purple-900/40 relative overflow-hidden"
                      >
                        {/* Gradient overlay */}
                        <div className={`absolute inset-0 bg-gradient-to-r ${tech.gradient}/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />
                        
                        {/* Shimmer effect */}
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/0 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
                        
                        <div className="relative z-10 flex items-center gap-2">
                          <div className={`p-2 rounded-lg bg-gradient-to-br ${tech.gradient}/10`}>
                            <Icon className={`text-lg ${tech.color}`} />
                          </div>
                          <span className="text-sm font-medium text-gray-700 group-hover:text-gray-900 dark:text-gray-300 dark:group-hover:text-gray-100 transition-colors duration-300">
                            {tech.name}
                          </span>
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              </div>
            </motion.div>

            {/* Image with Enhanced Orbital Effects */}
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, amount: 0.3 }}
              variants={scaleIn}
              className="relative"
            >
              {/* Orbital rings */}
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 40, repeat: Infinity, ease: "linear" }}
                className="absolute -inset-8 border-2 border-purple-500/10 rounded-3xl"
              />
              <motion.div
                animate={{ rotate: -360 }}
                transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
                className="absolute -ins-12 border border-purple-400/5 rounded-3xl"
              />
              
              <motion.div
                whileHover={{ 
                  scale: 1.02,
                  boxShadow: "0 20px 40px -15px rgba(168, 85, 247, 0.15) dark:shadow-purple-900/40"
                }}
                transition={{ duration: 0.3 }}
                className="relative bg-gradient-to-br from-white/10 to-transparent backdrop-blur-xl border border-white/20 dark:border-purple-900/30 rounded-2xl p-4 shadow-2xl shadow-purple-500/10 dark:shadow-purple-900/20 overflow-hidden"
              >
                {/* Image glow */}
                <div className="absolute inset-0 bg-gradient-to-r from-purple-500/0 via-purple-400/0 to-purple-300/0 group-hover:via-purple-400/5 group-hover:opacity-100 opacity-0 transition-all duration-300" />
                
                <motion.img
                  src={Mobile}
                  alt="LinguaSign Platform"
                  className="w-full rounded-xl relative z-10"
                  whileHover={{ scale: 1.05 }}
                  transition={{ duration: 0.5 }}
                />
              </motion.div>
            </motion.div>
          </div>

          {/* Mission & Vision Section with Enhanced Design */}
          <div className="grid lg:grid-cols-2 gap-8 mb-32">
            {/* Mission Card */}
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, amount: 0.3 }}
              variants={fadeUp}
              className="relative group"
            >
              <div className="absolute -inset-0.5 bg-gradient-to-r from-[#6A3093] to-[#A044FF] rounded-3xl blur opacity-20 dark:opacity-30 group-hover:opacity-40 transition duration-500" />
              <div className="relative bg-gradient-to-br from-white/80 to-white/60 dark:from-white/10 dark:to-white/5 backdrop-blur-xl border border-purple-200/50 dark:border-purple-500/20 rounded-3xl p-8 h-full shadow-xl shadow-purple-100/30 dark:shadow-purple-900/30 group-hover:shadow-2xl group-hover:shadow-purple-200/40 dark:group-hover:shadow-purple-900/50 group-hover:border-purple-300/70 dark:group-hover:border-purple-400/40 transition-all duration-500 overflow-hidden">
                {/* Shimmer effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/0 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
                
                <div className="relative z-10">
                  <div className="inline-flex items-center gap-4 mb-6">
                    <div className="w-14 h-14 rounded-xl flex items-center justify-center bg-gradient-to-br from-purple-100 to-purple-50 shadow-inner dark:bg-gradient-to-br dark:from-[#6A3093]/20 dark:to-[#A044FF]/20 group-hover:scale-105 transition-transform duration-300">
                      <FaHeart className="text-2xl text-purple-600 group-hover:text-purple-700 dark:text-purple-400 dark:group-hover:text-purple-300 transition-colors duration-300" />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-gray-900 group-hover:text-purple-900 dark:text-white dark:group-hover:text-purple-200 transition-colors duration-300">
                        Our Mission
                      </h3>
                      <p className="text-purple-500 group-hover:text-purple-600 dark:text-purple-400 dark:group-hover:text-purple-300 text-sm transition-colors duration-300">
                        Driving Purpose
                      </p>
                    </div>
                  </div>
                  
                  <p className="text-lg leading-relaxed mb-6 text-gray-700 group-hover:text-gray-800 dark:text-gray-300 dark:group-hover:text-gray-200 transition-colors duration-300">
                    To create a world where communication barriers cease to exist, empowering 
                    every individual to connect, learn, and thrive through accessible AI technology.
                  </p>
                  
                  <div className="flex items-center gap-2 text-purple-600 group-hover:text-purple-700 dark:text-purple-400 dark:group-hover:text-purple-300 transition-colors duration-300">
                    <TbArrowWaveRightDown className="text-xl" />
                    <span className="font-medium">Building Inclusive Connections</span>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Vision Card */}
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, amount: 0.3 }}
              variants={fadeUp}
              transition={{ delay: 0.2 }}
              className="relative group"
            >
              <div className="absolute -inset-0.5 bg-gradient-to-r from-[#A044FF] to-[#BF5AE0] rounded-3xl blur opacity-20 dark:opacity-30 group-hover:opacity-40 transition duration-500" />
              <div className="relative bg-gradient-to-br from-white/80 to-white/60 dark:from-white/10 dark:to-white/5 backdrop-blur-xl border border-purple-200/50 dark:border-purple-500/20 rounded-3xl p-8 h-full shadow-xl shadow-purple-100/30 dark:shadow-purple-900/30 group-hover:shadow-2xl group-hover:shadow-purple-200/40 dark:group-hover:shadow-purple-900/50 group-hover:border-purple-300/70 dark:group-hover:border-purple-400/40 transition-all duration-500 overflow-hidden">
                {/* Shimmer effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/0 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
                
                <div className="relative z-10">
                  <div className="inline-flex items-center gap-4 mb-6">
                    <div className="w-14 h-14 rounded-xl flex items-center justify-center bg-gradient-to-br from-purple-100 to-purple-50 shadow-inner dark:bg-gradient-to-br dark:from-[#A044FF]/20 dark:to-[#BF5AE0]/20 group-hover:scale-105 transition-transform duration-300">
                      <FaEye className="text-2xl text-purple-600 group-hover:text-purple-700 dark:text-purple-400 dark:group-hover:text-purple-300 transition-colors duration-300" />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-gray-900 group-hover:text-purple-900 dark:text-white dark:group-hover:text-purple-200 transition-colors duration-300">
                        Our Vision
                      </h3>
                      <p className="text-purple-500 group-hover:text-purple-600 dark:text-purple-400 dark:group-hover:text-purple-300 text-sm transition-colors duration-300">
                        Future Focus
                      </p>
                    </div>
                  </div>
                  
                  <p className="text-lg leading-relaxed mb-6 text-gray-700 group-hover:text-gray-800 dark:text-gray-300 dark:group-hover:text-gray-200 transition-colors duration-300">
                    To become the global standard for AI-powered sign language translation, 
                    transforming how humanity connects across languages, cultures, and abilities.
                  </p>
                  
                  <div className="flex items-center gap-2 text-purple-600 group-hover:text-purple-700 dark:text-purple-400 dark:group-hover:text-purple-300 transition-colors duration-300">
                    <TbArrowWaveLeftDown className="text-xl" />
                    <span className="font-medium">Shaping Communication's Future</span>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>

          {/* Enhanced Journey Section */}
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.3 }}
            variants={staggerContainer}
            className="mb-32"
          >
            {/* Section Header */}
            <div className="text-center mb-16">
              <motion.div
                variants={fadeUp}
                className="inline-flex items-center gap-4 mb-6"
              >
                <div className="w-16 h-1 bg-gradient-to-r from-transparent via-[#A044FF] to-transparent rounded-full" />
                <span className="text-sm font-semibold uppercase tracking-wider text-purple-600 dark:text-purple-400 bg-gradient-to-r from-purple-600 to-purple-400 bg-clip-text text-transparent">
                  Our Journey
                </span>
                <div className="w-16 h-1 bg-gradient-to-r from-transparent via-[#A044FF] to-transparent rounded-full" />
              </motion.div>
              
              <motion.h2
                variants={fadeUp}
                className="font-extrabold text-4xl sm:text-5xl leading-tight mb-4"
              >
                <span className="block text-gray-900 dark:text-white">How We</span>
                <span className="block bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
                  Built LinguaSign
                </span>
              </motion.h2>
              
              <motion.p
                variants={fadeUp}
                className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto"
              >
                From concept to reality - our journey of innovation and impact
              </motion.p>
            </div>

            {/* Enhanced Journey Steps with Animated Connector */}
            <div className="relative">
              {/* Animated connecting line */}
              <div className="hidden lg:block absolute left-1/2 top-0 bottom-0 w-0.5 bg-gradient-to-b from-transparent via-purple-500/30 to-transparent -translate-x-1/2">
                <motion.div 
                  className="absolute top-0 w-full h-1/4 bg-gradient-to-b from-purple-500 to-transparent"
                  animate={{ 
                    y: ['0%', '100%', '0%'] 
                  }}
                  transition={{ 
                    duration: 3, 
                    repeat: Infinity, 
                    ease: "linear" 
                  }}
                />
              </div>
              
              <div className="space-y-12">
                {journeySteps.map((step, index) => (
                  <motion.div
                    key={index}
                    variants={fadeUp}
                    transition={{ delay: step.delay }}
                    className={`relative flex flex-col lg:flex-row items-center gap-8 ${
                      index % 2 === 0 ? 'lg:flex-row-reverse' : ''
                    }`}
                  >
                    {/* Step Card */}
                    <div className={`w-full lg:w-1/2 ${index % 2 === 0 ? 'lg:text-right' : ''}`}>
                      <motion.div
                        whileHover={{ 
                          scale: 1.03,
                          y: -5,
                          boxShadow: "0 25px 50px -20px rgba(168, 85, 247, 0.25) dark:shadow-purple-900/40"
                        }}
                        className={`
                          group p-8 rounded-2xl backdrop-blur-xl border bg-white/80 dark:bg-white/10 border-purple-300/50 dark:border-purple-500/50 shadow-xl shadow-purple-100/20 dark:shadow-2xl
                          hover:bg-white/90 dark:hover:bg-white/15 hover:shadow-2xl hover:shadow-purple-200/30 dark:hover:shadow-purple-900/40
                          transition-all duration-300 overflow-hidden
                          ${index % 2 === 0 ? 'lg:mr-12' : 'lg:ml-12'}
                        `}
                      >
                        {/* Gradient overlay */}
                        <div className={`absolute inset-0 bg-gradient-to-r ${step.gradient}/5 opacity-0 group-hover:opacity-10 transition-opacity duration-300`} />
                        
                        {/* Shimmer effect */}
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/0 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
                        
                        <div className={`flex items-start gap-4 ${index % 2 === 0 ? 'lg:flex-row-reverse' : ''}`}>
                          <div className="p-3 rounded-xl bg-gradient-to-br from-purple-100 to-purple-50 text-purple-600 border border-purple-200/50 dark:bg-gradient-to-br dark:from-[#6A3093]/20 dark:to-[#A044FF]/20 dark:text-purple-400 group-hover:scale-110 group-hover:shadow-md group-hover:shadow-purple-200/50 dark:group-hover:shadow-purple-900/30 transition-all duration-300 relative z-10">
                            <step.icon className="text-2xl" />
                          </div>
                          <div className={`${index % 2 === 0 ? 'lg:text-right' : ''} relative z-10`}>
                            <h3 className="font-bold text-xl mb-2 text-gray-900 group-hover:text-purple-900 dark:text-white dark:group-hover:text-purple-200 transition-colors duration-300">
                              {step.text}
                            </h3>
                            <p className="text-gray-600 group-hover:text-gray-700 dark:text-gray-400 dark:group-hover:text-gray-300 transition-colors duration-300">
                              {step.details}
                            </p>
                          </div>
                        </div>
                      </motion.div>
                    </div>

                    {/* Step Indicator with Pulse Animation */}
                    <div className="relative z-10">
                      <motion.div
                        animate={{ 
                          y: [0, -10, 0],
                          scale: [1, 1.05, 1]
                        }}
                        transition={{ 
                          repeat: Infinity, 
                          duration: 3.2, 
                          ease: "easeInOut" 
                        }}
                        className="w-20 h-20 rounded-full flex items-center justify-center shadow-lg border bg-gradient-to-br from-purple-100 to-purple-50 border-purple-300/50 shadow-purple-200/30 dark:bg-purple-900/50 dark:border-purple-400/30 group-hover:shadow-xl group-hover:shadow-purple-300/50 dark:group-hover:shadow-purple-900/40 group-hover:scale-105 transition-all duration-300"
                      >
                        {/* Outer pulse ring */}
                        <motion.div
                          animate={{ 
                            scale: [1, 1.2, 1],
                            opacity: [0.5, 0, 0.5]
                          }}
                          transition={{ 
                            repeat: Infinity, 
                            duration: 2, 
                            ease: "easeInOut" 
                          }}
                          className="absolute inset-0 rounded-full bg-gradient-to-br from-purple-500/30 to-pink-500/30"
                        />
                        
                        <div className="w-16 h-16 rounded-full bg-gradient-to-br from-[#6A3093] to-[#A044FF] flex items-center justify-center shadow-inner dark:shadow-purple-900/30 group-hover:scale-110 transition-all duration-300 relative z-10">
                          <step.icon className="text-white text-2xl" />
                        </div>
                      </motion.div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Enhanced Capabilities Section */}
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.2 }}
            variants={staggerContainer}
            className="mb-32"
          >
            <div className="text-center mb-16">
              <motion.h2
                variants={fadeUp}
                className="font-extrabold text-4xl sm:text-5xl leading-tight mb-4"
              >
                <span className="block text-gray-900 dark:text-white">Powerful</span>
                <span className="block bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
                  Capabilities
                </span>
              </motion.h2>
              <motion.p
                variants={fadeUp}
                className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto"
              >
                Explore the advanced features that make LinguaSign the premier sign language platform
              </motion.p>
            </div>

            {/* Enhanced Capabilities Grid */}
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {capabilityFeatures.map((feature, index) => (
                <motion.div
                  key={index}
                  variants={scaleIn}
                  whileHover={{ 
                    scale: 1.05,
                    y: -8,
                    boxShadow: "0 25px 50px -20px rgba(168, 85, 247, 0.3) dark:shadow-purple-900/40"
                  }}
                  className="group relative p-6 rounded-2xl backdrop-blur-xl border bg-white/90 dark:bg-white/10 border-purple-300/50 dark:border-purple-500/50 shadow-xl shadow-purple-100/20 dark:shadow-2xl hover:bg-white/95 dark:hover:bg-white/15 hover:border-purple-400/60 dark:hover:border-purple-400/30 transition-all duration-300 overflow-hidden"
                >
                  {/* Gradient overlay */}
                  <div className={`absolute inset-0 bg-gradient-to-br ${feature.gradient}/5 rounded-2xl opacity-10 dark:opacity-0 group-hover:opacity-20 dark:group-hover:opacity-100 transition-opacity duration-300`} />
                  
                  {/* Shimmer effect */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/0 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
                  
                  <div className="relative z-10">
                    <div className="flex flex-col items-center text-center space-y-4">
                      <div className="w-16 h-16 rounded-xl flex items-center justify-center bg-gradient-to-br from-purple-100 to-purple-50 border border-purple-200/50 dark:bg-gradient-to-br dark:from-[#6A3093]/20 dark:to-[#A044FF]/20 group-hover:scale-110 group-hover:shadow-md group-hover:shadow-purple-200/50 dark:group-hover:shadow-purple-900/30 transition-transform duration-300">
                        <feature.icon className="text-2xl text-purple-600 group-hover:text-purple-700 dark:text-purple-400 dark:group-hover:text-purple-300 transition-colors duration-300" />
                      </div>
                      <h3 className="font-bold text-lg text-gray-900 group-hover:text-purple-900 dark:text-white dark:group-hover:text-purple-200 transition-colors duration-300">
                        {feature.text}
                      </h3>
                      <p className="text-sm text-gray-600 group-hover:text-gray-700 dark:text-gray-400 dark:group-hover:text-gray-300 transition-colors duration-300">
                        {feature.details}
                      </p>
                      <div className="text-xs font-semibold px-3 py-1 rounded-full text-purple-700 bg-purple-100/80 border border-purple-200 group-hover:bg-purple-200/80 group-hover:text-purple-800 dark:text-purple-400 dark:bg-purple-900/30 dark:border-purple-700/30 dark:group-hover:bg-purple-800/30 dark:group-hover:text-purple-300 transition-all duration-300">
                        {feature.stats}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Enhanced Team Section */}
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.2 }}
            variants={staggerContainer}
            className="mb-24"
          >
            <div className="text-center mb-16">
              <motion.div
                variants={fadeUp}
                className="inline-flex items-center gap-4 mb-6"
              >
                <div className="w-16 h-1 bg-gradient-to-r from-transparent via-[#A044FF] to-transparent rounded-full" />
                <span className="text-sm font-semibold uppercase tracking-wider text-purple-600 dark:text-purple-400 bg-gradient-to-r from-purple-600 to-purple-400 bg-clip-text text-transparent">
                  The Team
                </span>
                <div className="w-16 h-1 bg-gradient-to-r from-transparent via-[#A044FF] to-transparent rounded-full" />
              </motion.div>
              
              <motion.h2
                variants={fadeUp}
                className="font-extrabold text-4xl sm:text-5xl leading-tight mb-4"
              >
                <span className="block text-gray-900 dark:text-white">Meet Our</span>
                <span className="block bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
                  Expert Team
                </span>
              </motion.h2>
              
              <motion.p
                variants={fadeUp}
                className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto"
              >
                The brilliant minds behind LinguaSign's groundbreaking technology
              </motion.p>
            </div>

            {/* Enhanced Team Grid */}
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {team.map((person, index) => (
                <motion.div
                  key={index}
                  variants={scaleIn}
                  whileHover={{ 
                    scale: 1.05,
                    y: -8,
                    boxShadow: "0 25px 50px -20px rgba(168, 85, 247, 0.25) dark:shadow-purple-900/40"
                  }}
                  className="group relative p-6 rounded-2xl backdrop-blur-xl border bg-white/90 dark:bg-white/10 border-purple-300/50 dark:border-purple-500/50 shadow-xl shadow-purple-100/20 dark:shadow-2xl hover:bg-white/95 dark:hover:bg-white/15 hover:border-purple-400/60 dark:hover:border-purple-400/30 transition-all duration-300 overflow-hidden"
                >
                  {/* Gradient overlay */}
                  <div className={`absolute inset-0 bg-gradient-to-br ${person.gradient}/5 rounded-2xl opacity-10 dark:opacity-0 group-hover:opacity-20 dark:group-hover:opacity-100 transition-opacity duration-300`} />
                  
                  {/* Shimmer effect */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/0 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
                  
                  {/* Avatar */}
                  <div className="flex flex-col items-center text-center mb-4 relative z-10">
                    <div className="w-20 h-20 rounded-xl flex items-center justify-center mb-4 bg-gradient-to-br from-purple-100 to-purple-50 border border-purple-200/50 dark:bg-gradient-to-br dark:from-[#6A3093]/20 dark:to-[#A044FF]/20 group-hover:scale-110 group-hover:shadow-md group-hover:shadow-purple-200/50 dark:group-hover:shadow-purple-900/30 transition-transform duration-300">
                      <person.icon className="text-3xl text-purple-600 group-hover:text-purple-700 dark:text-purple-400 dark:group-hover:text-purple-300 transition-colors duration-300" />
                    </div>
                    <div>
                      <h3 className="font-bold text-xl text-gray-900 group-hover:text-purple-900 dark:text-white dark:group-hover:text-purple-200 transition-colors duration-300">
                        {person.name}
                      </h3>
                      <p className="text-sm font-medium mt-1 text-purple-600 group-hover:text-purple-700 dark:text-purple-400 dark:group-hover:text-purple-300 transition-colors duration-300">
                        {person.role}
                      </p>
                    </div>
                  </div>
                  
                  {/* Bio */}
                  <p className="text-sm text-center mb-4 text-gray-600 group-hover:text-gray-700 dark:text-gray-400 dark:group-hover:text-gray-300 transition-colors duration-300 relative z-10">
                    {person.bio}
                  </p>
                  
                  {/* Expertise Tags */}
                  <div className="flex flex-wrap justify-center gap-2 relative z-10">
                    {person.expertise.map((skill, skillIndex) => (
                      <span
                        key={skillIndex}
                        className="px-3 py-1 rounded-full text-xs font-medium border bg-gradient-to-r from-purple-100 to-purple-50 text-purple-700 border-purple-200 group-hover:bg-purple-200 group-hover:text-purple-800 group-hover:border-purple-300 dark:bg-gradient-to-r dark:from-purple-500/10 dark:to-purple-400/5 dark:text-purple-400 dark:border-purple-400/20 dark:group-hover:bg-purple-800/30 dark:group-hover:text-purple-300 transition-all duration-300"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
>>>>>>> e251330 (Add frontend, backend, and ai_service)
