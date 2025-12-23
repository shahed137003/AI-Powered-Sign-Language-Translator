import React, { useEffect, useState } from "react";
import Hero from "../assets/hero.svg";
import HeroDark from "../assets/heroDark.svg";
import { FaMobileAlt } from "react-icons/fa";
import { BsMicFill, BsRobot } from "react-icons/bs";
import { TbHandLoveYou } from "react-icons/tb";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

export default function Home() {
  const navigate = useNavigate();

  // Detect dark mode
  const [isDark, setIsDark] = useState(
    document.documentElement.classList.contains("dark")
  );

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains("dark"));
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });

    return () => observer.disconnect();
  }, []);

  // Animation variants
  const fadeUp = {
    hidden: { opacity: 0, y: 30 },
    show: { opacity: 1, y: 0 },
  };

  const fade = {
    hidden: { opacity: 0 },
    show: { opacity: 1 },
  };

  const scaleIn = {
    hidden: { opacity: 0, scale: 0.9 },
    show: { opacity: 1, scale: 1 },
  };

  return (
    <div className="relative w-full min-h-screen bg-gray-50 dark:bg-[#0f0c29] overflow-hidden selection:bg-purple-500 selection:text-white transition-colors duration-500">

      {/* FLOATING PARTICLES */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.45 }}
        transition={{ duration: 2 }}
        className="pointer-events-none"
      >
        {Array.from({ length: 15 }).map((_, i) => (
          <motion.div
            key={i}
            initial={{ y: 0, opacity: 0 }}
            animate={{
              y: [-10, 10, -10],
              opacity: [0.4, 0.9, 0.4],
            }}
            transition={{
              duration: 6 + i,
              repeat: Infinity,
              ease: "easeInOut",
            }}
            className="absolute w-2 h-2 bg-purple-400/40 rounded-full blur-sm"
            style={{
              top: `${Math.random() * 90}%`,
              left: `${Math.random() * 90}%`,
            }}
          />
        ))}
      </motion.div>

      {/* Background grid */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none"></div>

      {/* Background glows */}
      <div className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] bg-purple-300/20 dark:bg-purple-900/40 rounded-full blur-[100px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[500px] h-[500px] bg-purple-300/20 dark:bg-purple-900/40 rounded-full blur-[100px] pointer-events-none" />
      {/* <div className="absolute top-[20%] right-[10%] w-[300px] h-[300px] bg-pink-300/20 dark:bg-pink-900/20 rounded-full blur-[80px] pointer-events-none" /> */}

      {/* MAIN CONTENT */}
      <div className="relative z-10 w-full max-w-7xl mx-auto px-6 lg:px-12 py-12 lg:py-20 flex flex-col-reverse lg:flex-row items-center justify-between gap-12 lg:gap-20">

        {/* LEFT COLUMN */}
        <motion.div
          variants={fadeUp}
          initial="hidden"
          animate="show"
          transition={{ duration: 0.8 }}
          className="w-full lg:w-1/2 space-y-4"
        >
          {/* Badge */}
          <motion.div
            variants={fade}
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-100 dark:bg-purple-900/30 border border-purple-200 dark:border-purple-700 text-purple-700 dark:text-purple-300 text-xs font-semibold tracking-wider"
          >
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500"></span>
            </span>
            AI Integrated v2.0
          </motion.div>

          {/* HEADLINE */}
          <motion.h1
            initial="hidden"
            animate="show"
            variants={fadeUp}
            transition={{ duration: 1.2, ease: "easeOut" }}
            className="font-extrabold text-5xl sm:text-6xl leading-tight text-gray-900 dark:text-white"
          >
            <motion.span
              animate={{ opacity: 1, x: 0 }}
              initial={{ opacity: 0, x: -15 }}
              transition={{ duration: 0.7 }}
              className="block"
            >
              AI Powered
            </motion.span>

            <motion.span
              animate={{ width: "100%" }}
              initial={{ width: 0 }}
              transition={{ duration: 1.3, ease: "easeOut", delay: 0.3 }}
              className="block overflow-hidden whitespace-nowrap text-transparent bg-clip-text bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]"
            >
              Sign Language
            </motion.span>

            <motion.span
              animate={{ opacity: 1, x: 0 }}
              initial={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.8, delay: 0.6 }}
              className="block"
            >
              Translator
            </motion.span>
          </motion.h1>

          {/* Description */}
          <motion.p
            variants={fadeUp}
            transition={{ delay: 0.1 }}
            className="text-gray-600 dark:text-gray-300 text-lg sm:text-[18px] leading-relaxed max-w-lg"
          >
            Bridging communication between Deaf, hard-of-hearing, and hearing individuals with real-time gestures, speech recognition, and translations.
          </motion.p>

          {/* Feature Card */}
          <motion.div
            variants={scaleIn}
            whileHover={{ scale: 1.03 }}
            transition={{ type: "spring", stiffness: 120 }}
            className="p-4 sm:p-8 rounded-3xl backdrop-blur-3xl bg-white/70 dark:bg-[#1a163a]/60 border border-white/80 dark:border-white/10 shadow-xl shadow-purple-200/40 dark:shadow-none hover:shadow-2xl hover:shadow-purple-300/40 transition-all duration-300 relative overflow-hidden group"
          >
            <div className="absolute -right-10 -top-10 w-40 h-40 bg-purple-400/20 blur-3xl rounded-full transition-all duration-700"></div>

            <motion.h2 variants={fadeUp} className="text-2xl font-bold text-gray-800 dark:text-white mb-2">
              Real-time ASL <span className="text-purple-500">↔</span> English
            </motion.h2>

            <motion.p variants={fadeUp} className="text-sm text-gray-600 dark:text-gray-400 mb-6">
              Seamless AI-driven tools that recognize gestures and animate avatars instantly.
            </motion.p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {[
                { icon: <BsRobot />, text: "AI Gesture Recognition" },
                { icon: <BsMicFill />, text: "Speech-to-Text" },
                { icon: <FaMobileAlt />, text: "Mobile & Web Ready" },
                { icon: <TbHandLoveYou />, text: "Customizable Avatars" },
              ].map((item, i) => (

                <motion.div
                  key={i}
                  variants={fadeUp}
                  transition={{ delay: i * 0.2 }}
                  className="flex items-center gap-3 p-3 rounded-xl bg-white/90 dark:bg-white/5 border border-gray-200 dark:border-white/5 shadow-sm hover:shadow-md hover:bg-purple-50/70 dark:hover:bg-white/10 transition-all duration-300"
                >
                  <div className="p-2 rounded-lg bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300 text-lg">
                    {item.icon}
                  </div>
                  <p className="font-medium text-sm text-gray-700 dark:text-gray-200">
                    {item.text}
                  </p>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* CTA Buttons */}
          <div className="flex flex-wrap items-center gap-4 pt-2">
            <motion.button
              variants={fadeUp}
              whileHover={{
                scale: 1.05,
                boxShadow: "0 0 25px rgba(160, 68, 255, 0.6)",
              }}
              whileTap={{ scale: 0.95 }}
              onClick={() => navigate("/translate")}
              className="relative overflow-hidden px-8 py-4 rounded-full bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] text-white font-bold text-lg shadow-lg shadow-purple-400/40 hover:shadow-purple-500/60 transition-all group focus:outline-none focus:ring-4 focus:ring-purple-500/50"
            >
              <span className="relative z-10 flex items-center gap-2">
                Start Translating <TbHandLoveYou className="text-2xl" />
              </span>
              <div className="absolute top-0 left-0 w-full h-full bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 z-0 rounded-full" />
            </motion.button>

            <motion.button
              variants={fadeUp}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => navigate("/chatbot")}
              className="px-8 py-4 rounded-full font-bold text-lg text-gray-800 dark:text-white border-2 border-gray-300 dark:border-gray-700 hover:border-purple-500 bg-white/70 dark:bg-transparent hover:bg-gray-100 dark:hover:bg-white/5 shadow-sm hover:shadow-md transition-all flex items-center gap-2"
            >
              <BsRobot className="text-xl" />
              Chatbot
            </motion.button>
          </div>
        </motion.div>

        {/* RIGHT COLUMN */}
        <motion.div
          variants={fade}
          initial="hidden"
          animate="show"
          className="w-full lg:w-1/2 relative flex justify-center lg:justify-end"
        >
          {/* Orbit Lines */}
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 14, repeat: Infinity, ease: "linear" }}
            className="absolute inset-0 m-auto w-[80%] h-[80%] border border-purple-500/20 rounded-full md:w-[400px] md:h-[400px]"
          />

          <motion.div
            animate={{ rotate: -360 }}
            transition={{ duration: 18, repeat: Infinity, ease: "linear" }}
            className="absolute inset-0 m-auto w-[60%] h-[60%] border border-indigo-500/20 rounded-full md:w-[300px] md:h-[300px]"
          />

          {/* Hero Image */}
          <img
            src={isDark ? HeroDark : Hero}
            alt="Hero"
            className="relative z-10 w-full max-w-md lg:max-w-full drop-shadow-2xl"
          />

          {/* Floating AI Badge */}
          <motion.div
            animate={{ y: [0, -12, 0] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            className="absolute top-0 right-0 lg:right-10 z-20"
          >
            <div className="backdrop-blur-md bg-white/90 dark:bg-gray-800/90 border border-purple-200 dark:border-purple-700 p-4 rounded-2xl shadow-xl flex items-center gap-3">
              <div className="w-3 h-3 bg-purple-600 rounded-full animate-pulse" />
              <span className="font-semibold text-gray-800 dark:text-white text-sm">
                AI Powered
              </span>
            </div>
          </motion.div>

          {/* Floating Translation Badge */}
          <motion.div
            animate={{ y: [0, 12, 0] }}
            transition={{ duration: 5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
            className="absolute bottom-10 left-0 lg:left-[-20px] z-20"
          >
            <div className="backdrop-blur-md bg-white/90 dark:bg-gray-800/90 border border-purple-200 dark:border-purple-700 px-6 py-3 rounded-full shadow-xl">
              <span className="font-semibold text-purple-600 dark:text-purple-300 text-sm">
                Real-Time Translation
              </span>
            </div>
          </motion.div>
        </motion.div>

      </div>
    </div>
  );
}
