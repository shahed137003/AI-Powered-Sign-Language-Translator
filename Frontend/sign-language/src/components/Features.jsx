import React from "react";
import { motion } from "framer-motion";
import Sign from "../assets/feature1.svg";
import Avatar from "../assets/feature2.svg";
import Chatbot from "../assets/feature3.svg";
import Mobile from "../assets/feature4.svg";
import Profile from "../assets/profile.svg";
import Customize from "../assets/customize.svg";
import Ai from "../assets/Ai2.svg";
import { TbSparkles } from "react-icons/tb";

const features = [
  {
    img: Sign,
    title: "Voice-to-Text Translation",
    description:
      "Convert spoken language into written text in real-time. Ideal for facilitating communication for people with hearing impairments, meetings, or classrooms. Our AI ensures accurate transcription context understanding.",
  },
  {
    img: Avatar,
    title: "Video Upload Translation",
    description:
      "Upload pre-recorded videos or links and automatically generate synchronized sign language animations. Perfect for creating educational content, presentations, or social media posts accessible to the deaf community.",
  },
  {
    img: Chatbot,
    title: "AI Sign Language Chatbot",
    description:
      "Interact with our intelligent chatbot capable of understanding user queries in natural language and responding with accurate sign language gestures. Practice conversational skills or learn new phrases interactively.",
  },
  {
    img: Mobile,
    title: "Mobile Application",
    description:
      "Experience the full suite of features on your smartphone. Translate spoken language, type text, upload videos, or chat with the AI bot anytime, anywhere. Designed for portability and seamless integration.",
  },
  {
    img: Profile,
    title: "Edit Profile & Personalization",
    description:
      "Customize your personal experience with a fully editable profile. Update your communication preferences and accessibility settings to ensure a tailored environment that manages your identity securely.",
  },
  {
    img: Customize,
    title: "Accessibility Customization",
    description:
      "Enhance usability through advanced accessibility controls. Adjust avatar animation speed, customize color themes, enlarge handshape visuals, and modify gesture clarity for an inclusive experience.",
  },
  {
    img: Ai,
    title: "AI-Powered Communication Engine",
    description:
      "LinguaSign is built on advanced AI models that process speech, text, and gestures with exceptional accuracy. Our neural networks continuously learn from diverse signing styles, accents, and contexts—ensuring smarter, faster, and more natural communication every time you use the platform.",
  },
];

const featureVariants = {
  hidden: { opacity: 0, y: 50 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, ease: "easeOut" },
  },
};

export default function Features() {
  return (
    <div
      className="
        relative w-full py-24 px-6 lg:px-20 overflow-hidden
        bg-gradient-to-br 
        from-gray-50 via-white to-purple-50/60
        dark:from-[#0a0518] dark:via-[#110a2e] dark:to-[#1e0f5c]
        transition-all duration-700
      "
    >
      {/* Premium Geometric Grid */}
      <div className="absolute inset-0 opacity-40 dark:opacity-60 pointer-events-none">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(90deg, rgba(168, 85, 247, 0.1) 1px, transparent 1px),
            linear-gradient(180deg, rgba(168, 85, 247, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px'
        }} />
      </div>

      {/* Ambient glows */}
      <div className="absolute top-1/4 left-0 w-[500px] h-[500px] bg-purple-600/10 rounded-full blur-[120px]" />
      <div className="absolute bottom-1/4 right-0 w-[500px] h-[500px] bg-indigo-600/10 rounded-full blur-[120px]" />

      {/* HEADER - ENHANCED */}
      <div className="relative z-10 max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
          className="text-center mb-20"
        >
          {/* Premium Badge */}
          <motion.div
            whileHover={{ scale: 1.05, rotate: 1 }}
            className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-gradient-to-r from-purple-500/15 via-purple-400/10 to-purple-300/10 border border-purple-200/60 dark:border-purple-700/60 backdrop-blur-xl shadow-lg shadow-purple-500/10 relative overflow-hidden group mb-8"
          >
            <div className="relative">
              <span className="absolute animate-ping inline-flex h-3.5 w-3.5 rounded-full bg-purple-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-gradient-to-r from-purple-500 to-purple-400" />
            </div>
            <span className="text-sm font-bold bg-gradient-to-r from-purple-600 via-purple-500 to-purple-400 bg-clip-text text-transparent">
              Future of Accessibility
            </span>
            <TbSparkles className="text-purple-500 ml-1" />
            <div className="absolute -inset-1 bg-gradient-to-r from-purple-500/0 via-purple-400/10 to-purple-500/0 group-hover:via-purple-400/20 transition-all duration-500" />
          </motion.div>

          <motion.h1
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="font-extrabold text-4xl sm:text-5xl lg:text-[53px] leading-tight mb-6"
          >
            <span className="block text-gray-900 dark:text-white">
              Empower Communication
            </span>
            <span className="block bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
              With Sign Language AI
            </span>
          </motion.h1>
          
          <motion.p
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed"
          >
            Our platform bridges the gap between spoken and sign languages using
            state-of-the-art recognition technology.
          </motion.p>

          {/* Decorative Elements */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
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

      {/* TIMELINE - UNCHANGED */}
      <div className="relative max-w-7xl mx-auto z-10">
        <div className="hidden lg:block absolute left-1/2 top-0 bottom-0 w-[2px] bg-gradient-to-b from-transparent via-purple-500/40 to-transparent -translate-x-1/2" />

        {features.map((feature, index) => {
          const isLeft = index % 2 === 0;
          return (
            <motion.div
              key={index}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, margin: "-100px" }}
              variants={featureVariants}
              className={`mb-20 flex flex-col lg:flex-row items-center justify-between relative group ${
                isLeft ? "lg:flex-row" : "lg:flex-row-reverse"
              }`}
            >
              {/* IMAGE */}
              <div
                className={`w-full lg:w-5/12 flex justify-center mb-8 lg:mb-0 relative ${
                  isLeft ? "lg:justify-end lg:pr-12" : "lg:justify-start lg:pl-12"
                }`}
              >
                <div className="absolute inset-0 bg-purple-500/20 blur-3xl rounded-full scale-75 opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
                <img
                  src={feature.img}
                  alt={feature.title}
                  className="relative z-10 w-64 h-64 md:w-80 md:h-80 object-contain drop-shadow-2xl group-hover:scale-105 transition-transform duration-500"
                />
              </div>

              {/* CENTER NODE */}
              <div className="hidden lg:flex absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 items-center">
                <div className="w-8 h-8 rounded-full border border-purple-500/30 bg-[#110a2e] flex items-center justify-center shadow-[0_0_15px_rgba(160,68,255,0.4)]">
                  <div className="w-3 h-3 bg-gradient-to-r from-[#6A3093] to-[#A044FF] rounded-full" />
                </div>
                <div
                  className={`absolute top-1/2 w-12 h-[1px] bg-purple-500/30 ${
                    isLeft ? "right-full mr-4" : "left-full ml-4"
                  }`}
                />
              </div>

              {/* TEXT CARD */}
              <div
                className={`w-full lg:w-5/12 ${
                  isLeft ? "lg:pl-12" : "lg:pr-12"
                }`}
              >
                <div className="relative bg-white/60 dark:bg-white/5 backdrop-blur-xl p-8 rounded-2xl border border-gray-200 dark:border-purple-500/20 shadow-xl hover:border-purple-500/50 transition-all">
                  <h2 className="text-2xl md:text-3xl font-bold mb-4 bg-gradient-to-r from-[#6A3093] to-[#A044FF] bg-clip-text text-transparent">
                    {feature.title}
                  </h2>
                  <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}