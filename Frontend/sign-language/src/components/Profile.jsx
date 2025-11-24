import React, { useState } from "react";
import ProfileImg from "../assets/profileImg.svg";
import { motion } from "framer-motion";

export default function Profile() {
  const [user, setUser] = useState({
    name: "Shahd Mohamed",
    email: "shahd@example.com",
    password: "",
    theme: "system",
  });

  const fadeUp = {
    hidden: { opacity: 0, y: 25 },
    visible: { opacity: 1, y: 0 },
  };

  return (
    <div className="w-full bg-gray-50 dark:bg-gray-900 py-20 px-4 sm:px-6 lg:px-20 min-h-screen">

      {/* ---- PAGE TITLE ---- */}
      <motion.h1
        initial="hidden"
        animate="visible"
        variants={fadeUp}
        transition={{ duration: 0.7 }}
        className="
          text-5xl sm:text-6xl font-extrabold mb-20 text-center
          bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
          bg-clip-text text-transparent tracking-tight
        "
      >
        Your Profile
      </motion.h1>

      <div className="flex flex-col lg:flex-row gap-16 items-start max-w-7xl mx-auto">

        {/* ---------------- LEFT SIDE: PROFILE OVERVIEW ---------------- */}
        <motion.div
          initial={{ opacity: 0, x: -40 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
          className="flex-1 flex flex-col items-center"
        >
          <div className="w-full max-w-sm relative">

            {/* Profile Image */}
            <div className="
              w-72 h-72 md:w-80 md:h-80 mx-auto rounded-3xl overflow-hidden
              border border-[#A044FF]/60 shadow-xl bg-white/10 dark:bg-[#0D101A]/50
              backdrop-blur-2xl transition-transform hover:scale-105 duration-500
            ">
              <img src={ProfileImg} alt="Profile" className="w-full h-full object-cover" />
            </div>

            {/* Info Card */}
            <div className="
              mt-10 p-8 bg-white dark:bg-[#0F1420]/70 backdrop-blur-xl
              border border-[#A044FF]/40 rounded-3xl shadow-lg text-center
              transition hover:-translate-y-1 hover:shadow-[0_0_25px_rgba(160,68,255,0.35)]
            ">
              <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-200 mb-2">
                {user.name}
              </h2>

              <p className="text-gray-600 dark:text-gray-400 mb-1">
                {user.email}
              </p>

              <p className="text-gray-500 dark:text-gray-400 text-sm leading-relaxed">
                Manage your profile information and personalize your experience.
              </p>
            </div>
          </div>
        </motion.div>

        {/* ---------------- RIGHT SIDE: EDIT PROFILE ---------------- */}
        <motion.div
          initial={{ opacity: 0, x: 40 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
          className="
            flex-1 w-full p-8 rounded-3xl shadow-xl
            bg-white dark:bg-[#0F1420]/60 backdrop-blur-xl
            border border-[#A044FF]/30 space-y-7
          "
        >
          <h2 className="
            text-3xl font-bold 
            bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
            bg-clip-text text-transparent mb-8
          ">
            Edit Profile
          </h2>

          {/* Name */}
          <div className="space-y-1">
            <label className="font-semibold text-gray-800 dark:text-gray-200">Full Name</label>
            <input
              type="text"
              value={user.name}
              onChange={(e) => setUser({ ...user, name: e.target.value })}
              className="
                w-full p-3 rounded-lg bg-gray-100 dark:bg-gray-800
                border border-gray-300 dark:border-gray-700
                text-gray-900 dark:text-gray-200
                focus:outline-none focus:ring-2 focus:ring-[#A044FF]/80
                transition
              "
            />
          </div>

          {/* Email */}
          <div className="space-y-1">
            <label className="font-semibold text-gray-800 dark:text-gray-200">Email Address</label>
            <input
              type="email"
              value={user.email}
              onChange={(e) => setUser({ ...user, email: e.target.value })}
              className="
                w-full p-3 rounded-lg bg-gray-100 dark:bg-gray-800
                border border-gray-300 dark:border-gray-700
                text-gray-900 dark:text-gray-200
                focus:outline-none focus:ring-2 focus:ring-[#A044FF]/80
                transition
              "
            />
          </div>

          {/* Password */}
          <div className="space-y-1">
            <label className="font-semibold text-gray-800 dark:text-gray-200">New Password</label>
            <input
              type="password"
              placeholder="••••••••"
              value={user.password}
              onChange={(e) => setUser({ ...user, password: e.target.value })}
              className="
                w-full p-3 rounded-lg bg-gray-100 dark:bg-gray-800
                border border-gray-300 dark:border-gray-700
                text-gray-900 dark:text-gray-200
                focus:outline-none focus:ring-2 focus:ring-[#A044FF]/80
                transition
              "
            />
          </div>

          {/* Save Button */}
          <button
            className="
              w-full py-3 rounded-xl font-semibold text-white 
              bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
              hover:opacity-90 transition duration-300 shadow-lg transform hover:-translate-y-1
            "
          >
            Save Changes
          </button>
        </motion.div>
      </div>
    </div>
  );
}
