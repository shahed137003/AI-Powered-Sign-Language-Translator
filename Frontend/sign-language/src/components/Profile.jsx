import React, { useState } from "react";
import { motion } from "framer-motion";
import { BarChart3, Globe, Settings, Save, CheckCircle } from 'lucide-react';

// Replaced local image import with a dynamic placeholder URL
const PROFILE_PLACEHOLDER_URL = "https://placehold.co/300x300/A044FF/ffffff?text=User";

export default function Profile() {
  const [user, setUser] = useState({
    name: "Shahd Mohamed",
    email: "shahd@example.com",
    password: "",
    theme: "system",
    preferredLanguage: "ASL",
  });
  const [showToast, setShowToast] = useState(false);

  // Mock data for usage stats
  const usageStats = {
    totalTranslations: 452,
    learningStreak: 15, // days
  };

  const fadeUp = {
    hidden: { opacity: 0, y: 25 },
    visible: { opacity: 1, y: 0 },
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setUser((prevUser) => ({ ...prevUser, [name]: value }));
  };

  const handleSave = () => {
    // In a real application, this is where you would call an API or update Firestore.
    console.log("Saving changes for user:", user);
    
    // Show confirmation toast
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000); // Hide after 3 seconds
  };

  const Card = ({ children, className = "" }) => (
    <motion.div
      variants={fadeUp}
      transition={{ duration: 0.6, delay: 0.2 }}
      className={`
        p-8 rounded-3xl shadow-xl border border-[#A044FF]/30 
        bg-white dark:bg-[#0F1420]/60 backdrop-blur-xl
        ${className}
      `}
    >
      {children}
    </motion.div>
  );

  const InputField = ({ label, type = "text", name, value, onChange, placeholder }) => (
    <div className="space-y-1">
      <label htmlFor={name} className="font-semibold text-gray-800 dark:text-gray-200 block">
        {label}
      </label>
      <input
        id={name}
        type={type}
        name={name}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        className="
          w-full p-3 rounded-xl bg-gray-100 dark:bg-gray-800
          border border-gray-300 dark:border-gray-700
          text-gray-900 dark:text-gray-200
          focus:outline-none focus:ring-2 focus:ring-[#A044FF]/80
          transition duration-200
        "
      />
    </div>
  );

  return (
    <div className="w-full bg-gray-50 dark:bg-gray-900 py-24 px-4 sm:px-6 lg:px-20 min-h-screen font-inter">

      {/* Save Toast Notification */}
      <motion.div
        initial={{ x: "100%", opacity: 0 }}
        animate={showToast ? { x: 0, opacity: 1 } : { x: "100%", opacity: 0 }}
        transition={{ duration: 0.5, type: "spring", stiffness: 100 }}
        className="
          fixed top-20 right-6 p-4 rounded-xl shadow-2xl bg-green-500/90 backdrop-blur-md
          text-white font-semibold flex items-center gap-2 z-[60]
        "
      >
        <CheckCircle size={20} />
        Changes saved successfully!
      </motion.div>

      {/* ---- PAGE TITLE ---- */}
      <motion.h1
        initial="hidden"
        animate="visible"
        variants={fadeUp}
        transition={{ duration: 0.7 }}
        className="
          text-5xl sm:text-6xl font-extrabold mb-16 text-center
          bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
          bg-clip-text text-transparent tracking-tight
        "
      >
        Your Profile Dashboard
      </motion.h1>

      <div className="flex flex-col lg:flex-row gap-10 items-start max-w-7xl mx-auto">

        {/* ---------------- LEFT COLUMN ---------------- */}
        <div className="lg:w-1/3 w-full space-y-8">
          
          {/* PROFILE OVERVIEW CARD */}
          <Card className="flex flex-col items-center p-8 transition hover:-translate-y-1 hover:shadow-[0_0_25px_rgba(160,68,255,0.35)] duration-500">
            {/* Profile Image */}
            <div className="
              w-40 h-40 rounded-full overflow-hidden mb-6
              border-4 border-[#A044FF] shadow-2xl transition-transform hover:scale-105 duration-500
            ">
              <img 
                src={PROFILE_PLACEHOLDER_URL} 
                alt="Profile" 
                className="w-full h-full object-cover" 
                // Fallback for image loading errors
                onError={(e) => { e.target.onerror = null; e.target.src="https://placehold.co/300x300/6A3093/ffffff?text=User" }}
              />
            </div>

            <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-200 mb-2">
              {user.name}
            </h2>

            <p className="text-purple-600 dark:text-pink-400 font-medium mb-4">
              {user.email}
            </p>

            <p className="text-gray-500 dark:text-gray-400 text-sm leading-relaxed text-center border-t dark:border-gray-700 pt-4">
              Welcome back! Manage your personal details and app preferences below.
            </p>
          </Card>

          {/* USAGE STATISTICS CARD */}
          <Card className="space-y-4">
            <h3 className="flex items-center gap-2 text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4">
              <BarChart3 className="text-[#A044FF]" /> Usage Stats
            </h3>
            
            <div className="flex justify-between items-center p-3 rounded-xl bg-gray-100 dark:bg-gray-800">
              <span className="text-gray-600 dark:text-gray-400">Total Translations</span>
              <span className="text-2xl font-bold text-purple-600 dark:text-pink-400">{usageStats.totalTranslations}</span>
            </div>
            
            <div className="flex justify-between items-center p-3 rounded-xl bg-gray-100 dark:bg-gray-800">
              <span className="text-gray-600 dark:text-gray-400">Current Streak (Days)</span>
              <span className="text-2xl font-bold text-green-500">{usageStats.learningStreak} 🔥</span>
            </div>
          </Card>
        </div>

        {/* ---------------- RIGHT COLUMN: EDIT PROFILE & SETTINGS ---------------- */}
        <div className="lg:w-2/3 w-full space-y-8">

          {/* EDIT PERSONAL DETAILS */}
          <Card className="space-y-6">
            <h2 className="flex items-center gap-2 text-3xl font-bold 
              bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
              bg-clip-text text-transparent mb-4
            ">
              Personal Details
            </h2>

            <InputField 
              label="Full Name"
              name="name"
              value={user.name}
              onChange={handleInputChange}
            />

            <InputField 
              label="Email Address (Cannot be changed)"
              type="email"
              name="email"
              value={user.email}
              onChange={handleInputChange}
              readOnly 
              className="opacity-70 cursor-not-allowed"
            />
            
            <InputField 
              label="New Password"
              type="password"
              name="password"
              value={user.password}
              onChange={handleInputChange}
              placeholder="••••••••"
            />
          </Card>

          {/* APPLICATION SETTINGS */}
          <Card className="space-y-6">
            <h2 className="flex items-center gap-2 text-3xl font-bold 
              bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
              bg-clip-text text-transparent mb-4
            ">
              <Settings className="text-[#A044FF]" /> App Preferences
            </h2>

            {/* Preferred Language Dropdown */}
            <div className="space-y-1">
              <label htmlFor="preferredLanguage" className="font-semibold text-gray-800 dark:text-gray-200 block">
                <Globe size={16} className="inline mr-1 text-purple-500" /> Preferred Sign Language
              </label>
              <select
                id="preferredLanguage"
                name="preferredLanguage"
                value={user.preferredLanguage}
                onChange={handleInputChange}
                className="
                  w-full p-3 rounded-xl bg-gray-100 dark:bg-gray-800 appearance-none
                  border border-gray-300 dark:border-gray-700
                  text-gray-900 dark:text-gray-200
                  focus:outline-none focus:ring-2 focus:ring-[#A044FF]/80
                  transition duration-200 cursor-pointer
                "
              >
                <option value="ASL">American Sign Language (ASL)</option>
                <option value="BSL">British Sign Language (BSL)</option>
                <option value="JSL">Japanese Sign Language (JSL)</option>
                <option value="Custom">Custom / Other</option>
              </select>
            </div>

            {/* Theme Dropdown */}
            <div className="space-y-1">
              <label htmlFor="theme" className="font-semibold text-gray-800 dark:text-gray-200 block">
                Theme
              </label>
              <select
                id="theme"
                name="theme"
                value={user.theme}
                onChange={handleInputChange}
                className="
                  w-full p-3 rounded-xl bg-gray-100 dark:bg-gray-800 appearance-none
                  border border-gray-300 dark:border-gray-700
                  text-gray-900 dark:text-gray-200
                  focus:outline-none focus:ring-2 focus:ring-[#A044FF]/80
                  transition duration-200 cursor-pointer
                "
              >
                <option value="system">System Default</option>
                <option value="light">Light Mode</option>
                <option value="dark">Dark Mode (Midnight)</option>
              </select>
            </div>
          </Card>

          {/* Save Button (Separate card for visual weight) */}
          <motion.button
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            transition={{ type: "spring", stiffness: 300 }}
            onClick={handleSave}
            className="
              w-full py-4 rounded-xl font-extrabold text-xl text-white 
              bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
              shadow-2xl shadow-[#A044FF]/50
              hover:opacity-95 transition duration-300 transform hover:-translate-y-0.5
              flex items-center justify-center gap-3
            "
          >
            <Save size={20} />
            Save All Changes
          </motion.button>

        </div>
      </div>
    </div>
  );
}