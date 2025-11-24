import React, { useState, useRef } from "react";
import { motion } from "framer-motion";
import { FaMicrophone, FaCamera, FaSyncAlt } from "react-icons/fa";

export default function Translate() {
  const fadeUp = { hidden: { opacity: 0, y: 25 }, visible: { opacity: 1, y: 0 } };
  const options = ["Sign to Text", "Text / Audio to Sign"];
  const [selected, setSelected] = useState(options[0]);

  const [textInput, setTextInput] = useState("");
  const videoRef = useRef(null);

  const handleVideoRecord = () => alert("Camera recording started (simulated)");
  const handleAudioRecord = () => alert("Audio recording started (simulated)");

  return (
    <div className="w-full bg-gray-50 dark:bg-gray-900 py-28 px-4 sm:px-6 lg:px-20 min-h-screen">

      {/* PAGE TITLE */}
      <motion.div
        initial="hidden"
        whileInView="visible"
        variants={fadeUp}
        transition={{ duration: 0.7 }}
        viewport={{ once: true }}
        className="max-w-6xl mx-auto text-center mb-16"
      >
        <h2 className="
          text-5xl sm:text-6xl font-extrabold mb-6 
          bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
          bg-clip-text text-transparent
        ">
          Translation Center
        </h2>

        <p className="text-gray-700 dark:text-gray-300 text-lg sm:text-xl max-w-2xl mx-auto">
          Convert between <span className="font-semibold">Sign Language</span>, <span className="font-semibold">Text</span>, and <span className="font-semibold">Audio</span> — all in real time.
        </p>
      </motion.div>

      {/* OPTION SWITCH */}
      <div className="flex justify-center gap-4 mb-16 flex-wrap">
        {options.map((option) => (
          <button
            key={option}
            onClick={() => setSelected(option)}
            className={`
              px-8 py-3 rounded-full font-semibold transition-all duration-300 text-lg
              ${selected === option
                ? "bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] text-white shadow-xl scale-105"
                : "bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600"}
            `}
          >
            {option}
          </button>
        ))}
      </div>

      {/* MAIN CONTENT */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
        className="max-w-5xl mx-auto"
      >
        {/* ----------------------- SIGN TO TEXT ----------------------- */}
        {selected === "Sign to Text" && (
          <div className="flex flex-col items-center gap-10">

            <div className="w-full bg-white dark:bg-[#0F1420]/50 rounded-3xl p-10 shadow-xl border border-[#A044FF]/20 backdrop-blur-xl">
              <h3 className="text-3xl font-bold text-center mb-8 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] bg-clip-text text-transparent">
                Sign Language → Text
              </h3>

              {/* Camera Button */}
              <div className="flex justify-center">
                <button
                  onClick={handleVideoRecord}
                  className="
                    px-8 py-4 rounded-full flex items-center gap-3
                    bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
                    text-white shadow-lg hover:scale-105 active:scale-95 transition-all duration-300
                  "
                >
                  <FaCamera className="text-xl" /> Start Recording
                </button>
              </div>

              {/* Result */}
              <div className="
                mt-10 w-full h-64 rounded-2xl border border-gray-300 dark:border-gray-700
                bg-gray-100 dark:bg-gray-800 flex items-center justify-center
                text-gray-500 dark:text-gray-400 text-center tracking-wide text-lg
              ">
                Recognized text will appear here
              </div>
            </div>
          </div>
        )}

        {/* ------------------- TEXT / AUDIO TO SIGN ------------------- */}
        {selected === "Text / Audio to Sign" && (
          <div className="flex flex-col items-center gap-10">

            <div className="w-full bg-white dark:bg-[#0F1420]/50 rounded-3xl p-10 shadow-xl border border-[#A044FF]/20 backdrop-blur-xl">

              <h3 className="text-3xl font-bold text-center mb-8 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] bg-clip-text text-transparent">
                Text / Audio → Sign Language
              </h3>

              {/* TEXT INPUT */}
              <textarea
                rows={4}
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                placeholder="Type something to translate..."
                className="
                  w-full p-5 rounded-xl border border-gray-300 dark:border-gray-700
                  bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-200
                  shadow-md focus:outline-none focus:ring-2 focus:ring-[#A044FF] transition resize-none
                  text-lg
                "
              />

              {/* AUDIO RECORD */}
              <div className="flex justify-center mt-6">
                <button
                  onClick={handleAudioRecord}
                  className="
                    px-8 py-4 rounded-full flex items-center gap-3
                    bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
                    text-white shadow-lg hover:scale-105 active:scale-95 transition-all duration-300
                  "
                >
                  <FaMicrophone className="text-xl" /> Record Audio
                </button>
              </div>

              {/* AVATAR PREVIEW */}
              <div className="
                mt-10 w-full h-64 rounded-2xl border border-gray-300 dark:border-gray-700
                bg-gray-100 dark:bg-gray-800 flex items-center justify-center
                text-gray-500 dark:text-gray-400 text-center text-lg tracking-wide
              ">
                Sign Language Avatar Preview
              </div>

              {/* CONVERT BUTTON */}
              <div className="flex justify-center mt-6">
                <button className="
                  px-8 py-4 rounded-full flex items-center gap-3
                  bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
                  text-white shadow-lg hover:scale-105 active:scale-95 transition-all duration-300
                ">
                  <FaSyncAlt className="text-xl" /> Convert to Sign
                </button>
              </div>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
}
