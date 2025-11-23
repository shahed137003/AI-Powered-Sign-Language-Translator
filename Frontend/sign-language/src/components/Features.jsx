import React from "react";
import { motion } from "framer-motion";
import Sign from "../assets/feature1.svg";
import Avatar from "../assets/feature2.svg";
import Chatbot from "../assets/feature3.svg";
import Mobile from "../assets/feature4.svg";

const features = [
  {
    img: Sign,
    title: "Voice-to-text Translation",
    description: `Our Voice-to-text Translation feature allows users to speak in English and see words instantly converted into animated avatars performing sign language gestures.`,
  },
  {
    img: Avatar,
    title: "Video Upload Translation",
    description: `Upload videos or links to get synchronized sign language animations created with high accuracy and context understanding.`,
  },
  {
    img: Chatbot,
    title: "AI Sign Language Chatbot",
    description: `Practice sign language through interactive conversations with our AI chatbot that responds with gesture animations.`,
  },
  {
    img: Mobile,
    title: "Mobile Application",
    description: `Access all features on your smartphone, translate speech, upload videos, or chat with the AI bot anytime.`,
  },
];

// Animation variants
const parentVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.25 },
  },
};

const cardVariants = {
  hidden: { opacity: 0, y: 40, scale: 0.95 },
  show: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { duration: 0.8, ease: "easeOut" },
  },
};

export default function Features() {
  return (
    <div className="relative w-full bg-gray-50 dark:bg-gray-900 overflow-hidden py-24 px-4 sm:px-6 lg:px-20">

      {/* Intro Section */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8 }}
        className="max-w-4xl mx-auto text-center mb-20"
      >
        <h1 className="text-5xl font-extrabold mb-6 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
          Empower Communication with Sign Language
        </h1>
        <div className="w-50 h-1 mx-auto mb-10 rounded-full bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]" />
        <p className="text-gray-700 dark:text-gray-200 text-lg sm:text-xl">
          Our platform bridges the gap between spoken and sign languages.
        </p>
      </motion.div>

      {/* GRID ANIMATION WRAPPER */}
      <motion.div
        variants={parentVariants}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true }}
        className="grid grid-cols-1 sm:grid-cols-1 lg:grid-cols-2 gap-12"
      >
        {features.map((feature, index) => (
          <motion.div
            key={index}
            variants={cardVariants}
          
            className="
              relative bg-purple-100 dark:bg-gray-900/40 backdrop-blur-xl
              border border-gray-200 dark:border-[#A044FF]/20
              dark:before:absolute dark:before:inset-0 dark:before:rounded-3xl
              dark:before:bg-gradient-to-br dark:before:from-[#6A3093]/10 
              dark:before:to-[#A044FF]/10 dark:before:blur-2xl dark:before:opacity-60 before:-z-10
              rounded-3xl p-8 flex flex-col items-center text-center
              shadow-lg hover:shadow-2xl dark:shadow-[0_0_25px_rgba(160,68,255,0.15)]
              hover:dark:shadow-[0_0_40px_rgba(160,68,255,0.35)]
              transition-all duration-500 ease-in-out
              hover:scale-105
            "
          >
            <motion.img
              src={feature.img}
              alt={feature.title}
              className="w-100 h-80 mb-4"
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
            />

            <h2 className="text-3xl font-bold mb-3 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] bg-clip-text text-transparent">
              {feature.title}
            </h2>

            <p className="text-gray-700 dark:text-gray-300">
              {feature.description}
            </p>

            <div className="mt-4 w-22 h-1 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] rounded-full" />
          </motion.div>
        ))}
      </motion.div>

      {/* Call To Action */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8 }}
        className="mt-16 text-center"
      >
        <p className="text-gray-700 dark:text-gray-200 mb-6 text-lg sm:text-xl">
          Start your journey to accessible communication today.
        </p>

        <div className="flex justify-center gap-6 flex-wrap">
          <motion.button
            whileHover={{ scale: 1.1 }}
            className="px-6 py-3 rounded-full bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] text-white font-semibold shadow-lg"
          >
            Download App
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.1 }}
            className="px-6 py-3 rounded-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-500 text-gray-800 dark:text-gray-200 font-semibold shadow-lg"
          >
            Try Demo
          </motion.button>
        </div>
      </motion.div>
    </div>
  );
}
