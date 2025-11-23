import React, { useState, useRef } from "react";
import { motion } from "framer-motion";
import { FaMicrophone, FaCamera, FaSyncAlt } from "react-icons/fa";

export default function Translate() {
  const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };
  const options = ["Sign to text", "Text or audio to sign"];
  const [selected, setSelected] = useState(options[0]);

  // States
  const [textInput, setTextInput] = useState("");
  const [video, setVideo] = useState(null);
  const videoRef = useRef(null);

  const handleVideoRecord = () => {
    alert("Camera recording started! (simulate for now)");
    // Implement camera capture logic here
  };

  const handleAudioRecord = () => {
    alert("Audio recording started! (simulate for now)");
    // Implement audio capture logic here
  };

  return (
    <div className="w-full bg-gray-50 dark:bg-gray-900 py-24 px-4 sm:px-6 lg:px-20">
      {/* Heading */}
      <motion.div
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        variants={fadeUp}
        transition={{ duration: 0.8 }}
        className="max-w-7xl mx-auto text-center mb-12"
      >
        <h2 className="text-4xl sm:text-5xl font-extrabold mb-4 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
          Translate
        </h2>
        <div className="w-24 h-1 mx-auto mb-10 rounded-full bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF]"></div>
        <p className="text-gray-700 dark:text-gray-200 text-lg sm:text-xl">
          Record or type and translate in real time
        </p>
      </motion.div>

      {/* Options Buttons */}
      <div className="flex gap-4 justify-center mb-12 flex-wrap">
        {options.map((option) => (
          <button
            key={option}
            onClick={() => setSelected(option)}
            className={`px-6 py-3 rounded-full font-semibold transition-all duration-300 ${
              selected === option
                ? "bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] text-white shadow-lg"
                : "bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:scale-105 transform transition"
            }`}
          >
            {option}
          </button>
        ))}
      </div>

      {/* Content */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="max-w-5xl mx-auto flex flex-col items-center gap-8"
      >
        {/* Sign to Text */}
        {selected === "Sign to text" && (
          <div className="w-full flex flex-col items-center gap-4">
            <h3 className="text-2xl font-semibold mb-4 text-gray-800 dark:text-gray-200">Translate Sign Language to Text</h3>
            
            {/* Record Sign */}
            <button
              onClick={handleVideoRecord}
              className="px-6 py-3 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] text-white rounded-full shadow-lg hover:scale-105 transform transition duration-300 flex items-center gap-2"
            >
              <FaCamera /> Record Sign
            </button>

            {/* Placeholder for result */}
            <div className="mt-6 w-full max-w-md h-60 bg-gray-200 dark:bg-gray-800 rounded-xl flex items-center justify-center text-gray-400 dark:text-gray-500 text-center">
              Recognized text will appear here
            </div>
          </div>
        )}

        {/* Text or Audio to Sign */}
        {selected === "Text or audio to sign" && (
          <div className="w-full flex flex-col items-center gap-4">
            <h3 className="text-2xl font-semibold mb-4 text-gray-800 dark:text-gray-200">Translate Text or Audio to Sign Language</h3>
            
            {/* Text Input */}
            <textarea
              rows={4}
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="Type your text here..."
              className="w-full max-w-md p-4 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-200 shadow-lg focus:outline-none focus:ring-2 focus:ring-[#A044FF] resize-none"
            />

            {/* Audio Record */}
            <button
              onClick={handleAudioRecord}
              className="px-6 py-3 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] text-white rounded-full shadow-lg hover:scale-105 transform transition duration-300 flex items-center gap-2"
            >
              <FaMicrophone /> Record Audio
            </button>

            {/* Avatar Preview */}
            <div className="mt-6 w-full max-w-md h-60 bg-gray-200 dark:bg-gray-800 rounded-xl flex items-center justify-center text-gray-400 dark:text-gray-500 text-center">
              Animated Sign Language Avatar Preview
            </div>

            {/* Convert Button */}
            <button className="mt-4 px-6 py-3 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] text-white rounded-full shadow-lg hover:scale-105 transform transition duration-300 flex items-center gap-2">
              <FaSyncAlt /> Convert to Sign
            </button>
          </div>
        )}
      </motion.div>
    </div>
  );
}
