import React from "react";
import { motion } from "framer-motion";
import Sign from "../assets/feature1.svg";
import Avatar from "../assets/feature2.svg";
import Chatbot from "../assets/feature3.svg";
import Mobile from "../assets/feature4.svg";
<<<<<<< HEAD
=======
import Profile from "../assets/profile.svg";
import Customize from "../assets/customize.svg";
import Ai from "../assets/Ai2.svg"
>>>>>>> e251330 (Add frontend, backend, and ai_service)

const features = [
  {
    img: Sign,
    title: "Voice-to-Text Translation",
<<<<<<< HEAD
    description: `Convert spoken language into written text in real-time. Ideal for facilitating communication for people with hearing impairments, meetings, or classrooms. Users can speak naturally, and our AI ensures accurate transcription and context understanding, which is then rendered into animated sign language gestures for better accessibility.`,
=======
    description: `Convert spoken language into written text in real-time. Ideal for facilitating communication for people with hearing impairments, meetings, or classrooms. Our AI ensures accurate transcription context understanding.`,
>>>>>>> e251330 (Add frontend, backend, and ai_service)
  },
  {
    img: Avatar,
    title: "Video Upload Translation",
<<<<<<< HEAD
    description: `Upload pre-recorded videos or links and automatically generate synchronized sign language animations. Perfect for creating educational content, presentations, or social media posts that are accessible to the deaf community. Our system analyzes gestures, context, and speech, ensuring highly accurate and context-aware avatar animations.`,
=======
    description: `Upload pre-recorded videos or links and automatically generate synchronized sign language animations. Perfect for creating educational content, presentations, or social media posts accessible to the deaf community.`,
>>>>>>> e251330 (Add frontend, backend, and ai_service)
  },
  {
    img: Chatbot,
    title: "AI Sign Language Chatbot",
<<<<<<< HEAD
    description: `Interact with our intelligent chatbot capable of understanding user queries in natural language and responding with accurate sign language gestures. This feature allows users to practice conversational skills, ask questions, or learn new phrases in an engaging and interactive way, bridging the gap between text and visual communication.`,
=======
    description: `Interact with our intelligent chatbot capable of understanding user queries in natural language and responding with accurate sign language gestures. Practice conversational skills or learn new phrases interactively.`,
>>>>>>> e251330 (Add frontend, backend, and ai_service)
  },
  {
    img: Mobile,
    title: "Mobile Application",
<<<<<<< HEAD
    description: `Experience the full suite of our features on your smartphone. Translate spoken language, type text, upload videos, or chat with the AI bot anytime, anywhere. The mobile app is designed for ease of use, portability, and seamless integration into daily communication, making accessibility convenient and effortless.`,
  },
];

const featureVariants = {
  hidden: { opacity: 0, x: 50 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.6 } },
=======
    description: `Experience the full suite of features on your smartphone. Translate spoken language, type text, upload videos, or chat with the AI bot anytime, anywhere. Designed for portability and seamless integration.`,
  },
  {
    img: Profile,
    title: "Edit Profile & Personalization",
    description: `Customize your personal experience with a fully editable profile. Update your communication preferences and accessibility settings to ensure a tailored environment that manages your identity securely.`,
  },
  {
    img: Customize,
    title: "Accessibility Customization",
    description: `Enhance usability through advanced accessibility controls. Adjust avatar animation speed, customize color themes, enlarge handshape visuals, and modify gesture clarity for an inclusive experience.`,
  },
  {
  img: Ai, // <-- use any AI-themed icon you prefer
  title: "AI-Powered Communication Engine",
  description: `LinguaSign is built on advanced AI models that process speech, text, and gestures with exceptional accuracy. Our neural networks continuously learn from diverse signing styles, accents, and contexts—ensuring smarter, faster, and more natural communication every time you use the platform.`,
},

];

const featureVariants = {
  hidden: { opacity: 0, y: 50 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } },
>>>>>>> e251330 (Add frontend, backend, and ai_service)
};

export default function Features() {
  return (
<<<<<<< HEAD
    <div className="relative w-full bg-gray-50 dark:bg-gray-900 py-24 px-4 sm:px-6 lg:px-20">
      {/* Section Header */}
=======
    <div className="relative w-full bg-gray-50 dark:bg-[#0f0c29] py-24 px-6 lg:px-20 overflow-hidden transition-colors duration-500">
      
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none"></div>
{/* <div
  className="
    absolute inset-0 block dark:hidden 
    bg-[radial-gradient(55%_45%_at_40%_20%,#ffb7e633,transparent_70%), 
       radial-gradient(50%_50%_at_75%_75%,#e5b3ff33,transparent_75%)] 
    backdrop-blur-[2px]
  "
></div> */}

      <div className="absolute top-1/4 left-0 w-[500px] h-[500px] bg-purple-600/10 rounded-full blur-[100px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-0 w-[500px] h-[500px] bg-indigo-600/10 rounded-full blur-[100px] pointer-events-none" />

      {/* --- HEADER --- */}
>>>>>>> e251330 (Add frontend, backend, and ai_service)
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8 }}
<<<<<<< HEAD
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
=======
        className="relative z-10 max-w-4xl mx-auto text-center mb-24"
      >
        <span className="text-purple-600 dark:text-purple-400 font-bold tracking-widest uppercase text-sm mb-2 block">
            Future of Accessibility
        </span>
        <h1 className="text-4xl md:text-5xl font-extrabold mb-6 text-gray-900 dark:text-white">
          Empower Communication with <br />
          <span className="bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
            Sign Language AI
          </span>
        </h1>
        <p className="text-gray-600 dark:text-gray-400 text-lg sm:text-xl max-w-2xl mx-auto">
          Our platform bridges the gap between spoken and sign languages using state-of-the-art recognition technology.
        </p>
      </motion.div>

      {/* --- TIMELINE CONTAINER --- */}
      <div className="relative max-w-7xl mx-auto z-10">
        
        {/* Central Line (Desktop Only) */}
        <div className="hidden lg:block absolute left-1/2 top-0 bottom-0 w-[2px] bg-gradient-to-b from-transparent via-purple-500/50 to-transparent transform -translate-x-1/2"></div>
>>>>>>> e251330 (Add frontend, backend, and ai_service)

        {features.map((feature, index) => {
          const isLeft = index % 2 === 0;
          return (
            <motion.div
              key={index}
              initial="hidden"
              whileInView="visible"
<<<<<<< HEAD
              viewport={{ once: true }}
              variants={featureVariants}
              className={`mb-16 flex flex-col lg:flex-row items-center justify-between relative ${
                isLeft ? "lg:flex-row" : "lg:flex-row-reverse"
              }`}
            >
              {/* Feature Image */}
              <div className="lg:w-1/2 flex justify-center mb-6 lg:mb-0 ">
                <img
                  src={feature.img}
                  alt={feature.title}
                  className="w-80 h-80 object-contain shadow-lg rounded-xl hover:scale-105 transition-transform duration-500 "
                />
              </div>

              {/* Feature Text */}
              <div className="lg:w-1/2 bg-purple-100 dark:bg-gray-800/50 backdrop-blur-md p-8 rounded-2xl shadow-lg">
                <h2 className="text-3xl font-bold mb-4 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] bg-clip-text text-transparent">
                  {feature.title}
                </h2>
                <p className="text-gray-700 dark:text-gray-200 text-lg">
                  {feature.description}
                </p>
              </div>

              {/* Timeline Dot */}
              <div className="absolute left-1/2 top-8 lg:top-1/2 w-6 h-6 bg-gradient-to-tr from-[#6A3093] via-[#A044FF] to-[#BF5AE0] rounded-full transform -translate-x-1/2 -translate-y-1/2 shadow-xl"></div>
=======
              viewport={{ once: true, margin: "-100px" }}
              variants={featureVariants}
              className={`mb-20 flex flex-col lg:flex-row items-center justify-between relative group ${
                isLeft ? "lg:flex-row" : "lg:flex-row-reverse"
              }`}
            >
              {/* --- IMAGE SIDE --- */}
              <div className={`w-full lg:w-5/12 flex justify-center mb-8 lg:mb-0 relative ${isLeft ? "lg:justify-end lg:pr-12" : "lg:justify-start lg:pl-12"}`}>
                 {/* Image Glow */}
                 <div className="absolute inset-0 bg-purple-500/20 blur-3xl rounded-full scale-75 opacity-0 group-hover:opacity-100 transition-opacity duration-700"></div>
                 
                 <img
                  src={feature.img}
                  alt={feature.title}
                  className="relative z-10 w-64 h-64 md:w-80 md:h-80 object-contain drop-shadow-2xl transition-transform duration-500 group-hover:scale-105"
                />
              </div>

              {/* --- CENTER NODE (Desktop) --- */}
              <div className="hidden lg:flex absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 items-center justify-center">
                 {/* Outer Ring */}
                 <div className="w-8 h-8 rounded-full border border-purple-500/30 bg-[#0f0c29] flex items-center justify-center shadow-[0_0_15px_rgba(160,68,255,0.4)]">
                    {/* Inner Dot */}
                    <div className="w-3 h-3 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] rounded-full"></div>
                 </div>
                 
                 {/* Connector Line to Card */}
                 <div className={`absolute top-1/2 w-12 h-[1px] bg-purple-500/30 ${isLeft ? "right-full mr-4" : "left-full ml-4"}`}></div>
              </div>

              {/* --- TEXT CARD SIDE --- */}
              <div className={`w-full lg:w-5/12 ${isLeft ? "lg:pl-12" : "lg:pr-12"}`}>
                <div className="
                  relative bg-white/50 dark:bg-[#1a163a]/60 backdrop-blur-xl 
                  p-8 rounded-2xl border border-gray-200 dark:border-purple-500/20 
                  shadow-xl dark:shadow-purple-900/10
                  hover:border-purple-500/50 transition-colors duration-300
                ">
                 
                  <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-bl from-purple-500/10 to-transparent rounded-tr-2xl"></div>

                  <h2 className="text-2xl md:text-3xl font-bold mb-4 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
                    {feature.title}
                  </h2>
                  <p className="text-gray-600 dark:text-gray-300 text-base leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </div>

>>>>>>> e251330 (Add frontend, backend, and ai_service)
            </motion.div>
          );
        })}
      </div>

<<<<<<< HEAD
      {/* Call-to-Action */}
=======
      {/* --- FOOTER CTA --- */}
>>>>>>> e251330 (Add frontend, backend, and ai_service)
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8 }}
<<<<<<< HEAD
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
=======
        className="mt-12 text-center relative z-10"
      >
        <p className="text-gray-600 dark:text-gray-400 mb-8 text-lg">
          Ready to break down communication barriers?
        </p>
        <div className="flex justify-center gap-6 flex-wrap">
      <motion.button
  whileHover={{ scale: 1.05 }}
  whileTap={{ scale: 0.95 }}
  className="
    relative /* Essential for absolute positioning of inner div */
    overflow-hidden /* Clips the inner div when it's outside the button */
    group /* Enables group-hover effects on children */
    px-8 py-3 rounded-full 
    bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] 
    text-white font-bold 
    shadow-lg shadow-purple-500/40 hover:shadow-purple-500/60 
    transition-all
    focus:outline-none focus:ring-4 focus:ring-purple-500/50 /* Added for accessibility */
  "
>
  {/* The main button text, ensure it's above the overlay with z-index */}
  <span className="relative z-10">
    Download App
  </span>
  
  {/* The glass overlay effect */}
  <div 
    className="
      absolute top-0 left-0 w-full h-full /* Covers the entire button */
      bg-white/20 
      translate-y-full /* Starts off-screen below the button */
      group-hover:translate-y-0 /* Slides up on hover */
      transition-transform duration-300 /* Smooth animation */
      z-0 /* Keeps it behind the text */
      rounded-full /* Matches the button's rounded shape */
    "
  />
</motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="px-8 py-3 rounded-full bg-transparent border-2 border-gray-300 dark:border-gray-600 text-gray-700 dark:text-white font-bold hover:border-purple-500 dark:hover:border-purple-500 hover:bg-purple-50 dark:hover:bg-white/5 transition-all"
          >
            View Live Demo
>>>>>>> e251330 (Add frontend, backend, and ai_service)
          </motion.button>
        </div>
      </motion.div>
    </div>
  );
<<<<<<< HEAD
}
=======
}
>>>>>>> e251330 (Add frontend, backend, and ai_service)
