import React, { useState } from "react";
import ProfileImg from "../assets/profileImg.svg";
import { motion } from "framer-motion";

export default function Profile() {
  // Simulated user data
  const [user, setUser] = useState({
    name: "Shahd Mohamed",
    email: "shahd@example.com",
    password: "",
    theme: "system",
  });

  const fadeUp = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } };

  return (
    <div className="w-full bg-gray-50 dark:bg-gray-900 py-20 px-4 sm:px-6 lg:px-20 min-h-screen">

      {/* Title */}
      <motion.h1
        initial="hidden"
        animate="visible"
        variants={fadeUp}
        transition={{ duration: 0.8 }}
        className="text-5xl sm:text-6xl font-extrabold mb-16 text-center
                   bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
                   dark:from-[#6A3093] dark:to-[#A044FF] 
                   bg-clip-text text-transparent drop-shadow-lg tracking-wide"
      >
        Welcome, {user.name}
      </motion.h1>

      <div className="flex flex-col lg:flex-row gap-16 items-start max-w-7xl mx-auto">

       
       <motion.div
  initial={{ opacity: 0, x: -40 }}
  animate={{ opacity: 1, x: 0 }}
  transition={{ duration: 0.7 }}
  className="flex-1 flex flex-col items-center space-y-6"
>
  <div className="relative w-full max-w-sm mx-auto">

    {/* Profile Picture */}
    <div className="absolute -top-10 left-1/2 -translate-x-1/2 w-80 h-80 md:w-90 md:h-90 rounded-4xl border border-[#A044FF] overflow-hidden shadow-2xl transition-transform duration-500 hover:scale-105">
      <img
        src={ProfileImg}
        alt="Profile"
        className="w-full h-full object-cover"
      />
    </div>

    {/* User Info Card */}
    <div className="mt-83 p-8 bg-white dark:bg-[#10141F]/70 backdrop-blur-xl border border-[#A044FF]/30 rounded-3xl shadow-2xl text-center transition-transform duration-300 hover:-translate-y-1 hover:shadow-[0_0_25px_rgba(160,68,255,0.4)]">
      <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-200 mb-2">
        {user.name}
      </h2>
      <p className="text-gray-600 dark:text-gray-300 mb-1">
        Email: {user.email}
      </p>
  
      <p className="text-gray-500 dark:text-gray-400 text-sm">
        Customize your profile, update information, and personalize your experience.
      </p>
    </div>
  </div>
</motion.div>

        <motion.div
          initial={{ opacity: 0, x: 40 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.7 }}
          className="flex-1 w-full bg-white dark:bg-[#10141F]/60 backdrop-blur-xl border border-[#A044FF]/30 rounded-3xl shadow-xl p-8 space-y-6"
        >
          <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
                         bg-clip-text text-transparent">
            Edit Profile
          </h2>

          {/* Name */}
          <div>
            <label className="block font-semibold mb-1 text-gray-800 dark:text-gray-200">Full Name</label>
            <input
              type="text"
              value={user.name}
              onChange={(e) => setUser({ ...user, name: e.target.value })}
              className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-600 
                         bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-200
                         focus:outline-none focus:ring-2 focus:ring-[#A044FF] focus:ring-offset-1 transition"
            />
          </div>

          {/* Email */}
          <div>
            <label className="block font-semibold mb-1 text-gray-800 dark:text-gray-200">Email Address</label>
            <input
              type="email"
              value={user.email}
              onChange={(e) => setUser({ ...user, email: e.target.value })}
              className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-600 
                         bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-200
                         focus:outline-none focus:ring-2 focus:ring-[#A044FF] focus:ring-offset-1 transition"
            />
          </div>

          {/* Password */}
          <div>
            <label className="block font-semibold mb-1 text-gray-800 dark:text-gray-200">New Password</label>
            <input
              type="password"
              placeholder="••••••••"
              value={user.password}
              onChange={(e) => setUser({ ...user, password: e.target.value })}
              className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-600 
                         bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-200
                         focus:outline-none focus:ring-2 focus:ring-[#A044FF] focus:ring-offset-1 transition"
            />
          </div>

      

          {/* Save Button */}
          <button
            className="w-full py-3 rounded-xl font-semibold text-white 
                       bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
                       hover:opacity-90 transition duration-300 shadow-lg transform hover:-translate-y-1"
          >
            Save Changes
          </button>
        </motion.div>
      </div>
    </div>
  );
}
