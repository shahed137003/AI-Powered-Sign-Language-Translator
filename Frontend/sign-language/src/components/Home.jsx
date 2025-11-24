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

  // ✨ Animation variants
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
    <div className="relative w-full bg-white dark:bg-gray-900 overflow-hidden">
      {/* BACKGROUND GLOWS */}
<div className="absolute top-[-150px] left-[-150px] w-[300px] h-[300px] bg-gradient-to-tr from-[#A044FF]/40 to-[#6A3093]/20 dark:from-[#7B1FA2]/40 dark:to-[#4A148C]/30 rounded-full blur-3xl animate-pulse" />
 <div className="absolute bottom-[-150px] right-[-150px] w-[350px] h-[350px] bg-gradient-to-br from-[#BF5AE0]/40 to-[#6A3093]/20 dark:from-[#8E24AA]/40 dark:to-[#311B92]/30 rounded-full blur-3xl animate-pulse delay-200" />
  <div className="absolute bottom-[-200px] left-[-180px] w-[320px] h-[320px] bg-[#BF5AE0]/30 dark:bg-[#8E24AA]/25 rounded-full blur-3xl" /> 
  <div className="absolute top-[-200px] right-[-180px] w-[380px] h-[380px] bg-[#BF5AE0]/30 dark:bg-[#6A1B9A]/25 rounded-full blur-3xl" /> 
  <div className="absolute inset-0 m-auto w-[420px] h-[420px] hidden dark:block bg-gradient-to-r from-[#8E24AA]/20 to-[#512DA8]/20 rounded-full blur-[120px]" />

      {/* MAIN CONTENT */}
      <div className="relative w-full flex flex-col-reverse lg:flex-row items-center justify-between px-8 lg:px-20 py-24 gap-10">

       
        <motion.div
          variants={fadeUp}
          initial="hidden"
          animate="show"
          transition={{ duration: 0.8 }}
          className="max-w-xl space-y-4"
        >
          {/* TITLE */}
          <motion.h1
            variants={fadeUp}
            transition={{ duration: 0.8 }}
            className="
              font-extrabold text-5xl sm:text-5xl lg:text-5xl leading-tight
              bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
              dark:from-[#6A3093] dark:to-[#A044FF]
              bg-clip-text text-transparent
            "
          >
            AI Powered Sign Language Translator
          </motion.h1>

          {/* DESCRIPTION */}
          <motion.p
            variants={fadeUp}
            transition={{ duration: 1 }}
            className="text-gray-600 dark:text-gray-300 text-[20px] sm:text-[18px] leading-relaxed"
          >
            Bridging communication between Deaf, hard-of-hearing, and hearing individuals with real-time gestures, speech recognition, and translations.
          </motion.p>

          {/* FEATURE BOX */}
          <motion.div
            variants={scaleIn}
            transition={{ duration: 0.8 }}
            className="p-8 bg-white/30 dark:bg-gray-800/40 backdrop-blur-xl border border-[#6A3093]/20 dark:border-[#A044FF]/20 rounded-3xl shadow-xl space-y-3"
          >
            <motion.h2
              variants={fadeUp}
              transition={{ duration: 0.8 }}
              className="text-3xl text-[#BF5AE0] dark:text-[#A044FF] font-bold drop-shadow-md"
            >
              Real-time ASL ↔ English Translation
            </motion.h2>

            <motion.p
              variants={fadeUp}
              transition={{ duration: 1 }}
              className="text-gray-700 dark:text-gray-200 text-[14px] leading-relaxed"
            >
              Communicate seamlessly using AI-driven tools that recognize gestures,
              convert speech to text, and animate avatars in real-time.
            </motion.p>

            {/* FEATURES */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {[ 
                { icon: <BsRobot />, text: "AI-driven gesture recognition" },
                { icon: <BsMicFill />, text: "Speech-to-text support" },
                { icon: <FaMobileAlt />, text: "Available on mobile & web" }
              ].map((item, i) => (
                <motion.div
                  key={i}
                  variants={fadeUp}
                  transition={{ duration: 0.6, delay: i * 0.2 }}
                  className="flex items-center gap-4 p-3 bg-white/50 dark:bg-gray-700/50 rounded-xl shadow hover:-translate-y-1 hover:shadow-lg transition"
                >
                  <div className="text-[#6A3093] dark:text-[#A044FF] text-2xl">
                    {item.icon}
                  </div>
                  <p className="font-medium text-gray-800 dark:text-gray-200">
                    {item.text}
                  </p>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* BUTTONS */}
          <div className="mt-6 flex items-center gap-4">
            <motion.button
              variants={fadeUp}
              transition={{ duration: 0.7 }}
          
              whileTap={{ scale: 0.98 }}
              className="bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] text-white px-10 py-3 rounded-full font-semibold text-lg shadow-xl flex items-center gap-3 hover:scale-105 transition-transform duration-300"
              onClick={() => navigate("/translate")}
            >
              Translate
              <TbHandLoveYou className="text-2xl" />
            </motion.button>

            <motion.button
              variants={fadeUp}
              transition={{ duration: 0.7, delay: 0.2 }}
              
              whileTap={{ scale: 0.98 }}
              className="bg-white/80 dark:bg-gray-800/70 border border-[#A044FF]/40 text-[#6A3093] dark:text-[#A044FF] px-6 py-3 rounded-full font-semibold text-lg shadow flex items-center gap-3 hover:scale-105 transition"
              onClick={() => navigate("/chatbot")}
            >
              <BsRobot className="text-2xl" />
              Chatbot
            </motion.button>
          </div>
        </motion.div>

      
        <motion.div
          variants={fade}
          initial="hidden"
          animate="show"
          transition={{ duration: 1 }}
          className="relative w-full max-w-lg lg:max-w-xl flex justify-center"
        >
          <motion.img
            src={isDark ? HeroDark : Hero}
            alt="Hero"
            initial={{ opacity: 0, scale: 1.05 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1 }}
            className="w-full drop-shadow-2xl"
          />

          {/* Floating badge 1 */}
   <motion.div
  initial={{ opacity: 0, y: -20 }}
  animate={{
    opacity: 1,
    y: [0, -12, 0],  // floating keyframes
  }}
  transition={{
    duration: 3,
    ease: "easeInOut",
    repeat: Infinity,
  }}
  className="absolute -top-5 -right-5 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] text-white font-bold px-8 py-3 rounded-full shadow-lg"
>
  AI Powered
</motion.div>

          {/* Floating badge 2 */}
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="absolute bottom-0 left-0 bg-white/70 dark:bg-gray-800/60 text-[#6A3093] dark:text-[#A044FF] font-semibold px-8 py-3 rounded-full shadow"
          >
            Real-Time Translation
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}
