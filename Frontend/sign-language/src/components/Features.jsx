import React from "react";
import { motion } from "framer-motion";
import Sign from "../assets/feature1.svg";
import Avatar from "../assets/feature2.svg";
import Chatbot from "../assets/feature3.svg";
import Mobile from "../assets/feature4.svg";

const features = [
  {
    img: Sign,
    title: "Voice-to-Text Translation",
    description: `Convert spoken language into written text in real-time. Ideal for facilitating communication for people with hearing impairments, meetings, or classrooms. Users can speak naturally, and our AI ensures accurate transcription and context understanding, which is then rendered into animated sign language gestures for better accessibility.`,
  },
  {
    img: Avatar,
    title: "Video Upload Translation",
    description: `Upload pre-recorded videos or links and automatically generate synchronized sign language animations. Perfect for creating educational content, presentations, or social media posts that are accessible to the deaf community. Our system analyzes gestures, context, and speech, ensuring highly accurate and context-aware avatar animations.`,
  },
  {
    img: Chatbot,
    title: "AI Sign Language Chatbot",
    description: `Interact with our intelligent chatbot capable of understanding user queries in natural language and responding with accurate sign language gestures. This feature allows users to practice conversational skills, ask questions, or learn new phrases in an engaging and interactive way, bridging the gap between text and visual communication.`,
  },
  {
    img: Mobile,
    title: "Mobile Application",
    description: `Experience the full suite of our features on your smartphone. Translate spoken language, type text, upload videos, or chat with the AI bot anytime, anywhere. The mobile app is designed for ease of use, portability, and seamless integration into daily communication, making accessibility convenient and effortless.`,
  },
];

const featureVariants = {
  hidden: { opacity: 0, x: 50 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.6 } },
};

export default function Features() {
  return (
    <div className="relative w-full bg-gray-50 dark:bg-gray-900 py-24 px-4 sm:px-6 lg:px-20">
      {/* Section Header */}
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

      {/* Features Timeline */}
      <div className="relative max-w-6xl mx-auto">
        <div className="absolute top-0 left-1/2 w-1 bg-gradient-to-b from-[#6A3093] via-[#A044FF] to-[#BF5AE0] h-full transform -translate-x-1/2"></div>

        {features.map((feature, index) => {
          const isLeft = index % 2 === 0;
          return (
            <motion.div
              key={index}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={featureVariants}
              className={`mb-16 flex flex-col lg:flex-row items-center justify-between relative ${
                isLeft ? "lg:flex-row" : "lg:flex-row-reverse"
              }`}
            >
              {/* Feature Image */}
              <div className="lg:w-1/2 flex justify-center mb-6 lg:mb-0">
                <img
                  src={feature.img}
                  alt={feature.title}
                  className="w-64 h-64 object-contain shadow-lg rounded-xl hover:scale-105 transition-transform duration-500"
                />
              </div>

              {/* Feature Text */}
              <div className="lg:w-1/2 bg-white dark:bg-gray-800/50 backdrop-blur-md p-8 rounded-2xl shadow-lg">
                <h2 className="text-3xl font-bold mb-4 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] bg-clip-text text-transparent">
                  {feature.title}
                </h2>
                <p className="text-gray-700 dark:text-gray-200 text-lg">
                  {feature.description}
                </p>
              </div>

              {/* Timeline Dot */}
              <div className="absolute left-1/2 top-8 lg:top-1/2 w-6 h-6 bg-gradient-to-tr from-[#6A3093] via-[#A044FF] to-[#BF5AE0] rounded-full transform -translate-x-1/2 -translate-y-1/2 shadow-xl"></div>
            </motion.div>
          );
        })}
      </div>

      {/* Call-to-Action */}
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
            whileHover={{ scale: 1.1, boxShadow: "0 0 20px rgba(160,68,255,0.6)" }}
            className="px-6 py-3 rounded-full bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] text-white font-semibold shadow-lg"
          >
            Download App
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.1, boxShadow: "0 0 15px rgba(160,68,255,0.3)" }}
            className="px-6 py-3 rounded-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-500 text-gray-800 dark:text-gray-200 font-semibold shadow-lg"
          >
            Try Demo
          </motion.button>
        </div>
      </motion.div>
    </div>
  );
}
