import React, { useState } from "react";
import { motion } from "framer-motion";
import { Save, CheckCircle, User, Lock } from 'lucide-react';

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

  const fadeUp = {
    hidden: { opacity: 0, y: 25 },
    visible: { opacity: 1, y: 0 },
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setUser((prevUser) => ({ ...prevUser, [name]: value }));
  };

  const handleSave = () => {
    console.log("Saving changes for user:", user);
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000);
  };

  const Card = ({ children, className = "" }) => (
    <motion.div
      variants={fadeUp}
      transition={{ duration: 0.6, delay: 0.2 }}
      className={`p-8 dark:bg-[#1a163a]/60 backdrop-blur-xl border border-gray-200 dark:border-purple-500/20 rounded-3xl shadow-xl ${className}`}
    >
      {children}
    </motion.div>
  );

  // Updated InputField with icon support
  const InputField = ({ label, type = "text", name, value, onChange, placeholder, icon }) => (
    <div className="space-y-1">
      <label htmlFor={name} className="font-semibold text-gray-800 dark:text-gray-200 block">
        {label}
      </label>
      <div className="relative">
        <span className="absolute inset-y-0 left-3 flex items-center text-gray-400 dark:text-gray-400">
          {icon}
        </span>
        <input
          id={name}
          type={type}
          name={name}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          className="
            w-full p-3 pl-10 rounded-xl bg-white/70 dark:bg-gray-700/50
            border border-gray-300 dark:border-gray-700
            text-gray-900 dark:text-gray-200
            focus:outline-none focus:ring-2 focus:ring-[#A044FF]/80
            transition duration-200
          "
        />
      </div>
    </div>
  );

  return (
    <div className="w-full bg-gray-50 dark:bg-[#0f0c29] py-24 px-4 sm:px-6 lg:px-20 min-h-screen font-inter">

      {/* Save Toast Notification */}
      <motion.div
        initial={{ x: "100%", opacity: 0 }}
        animate={showToast ? { x: 0, opacity: 1 } : { x: "100%", opacity: 0 }}
        transition={{ duration: 0.5, type: "spring", stiffness: 100 }}
        className="fixed top-20 right-6 p-4 rounded-xl shadow-2xl bg-green-500/90 backdrop-blur-md text-white font-semibold flex items-center gap-2 z-[60]"
      >
        <CheckCircle size={20} />
        Changes saved successfully!
      </motion.div>

      {/* ---- PAGE TITLE ---- */}
{/* --- HEADER --- */}
<motion.div
  initial="hidden"
  whileInView="visible"
  viewport={{ once: true }}
  variants={fadeUp}
  transition={{ duration: 0.8 }}
  className="max-w-7xl mx-auto text-center mb-16 relative z-10"
>
  <span className="text-purple-600 dark:text-purple-400 font-bold tracking-widest uppercase text-sm mb-2 block">
    Welcome Back
  </span>
  <h2 className="text-4xl sm:text-5xl font-extrabold mb-4 text-gray-900 dark:text-white">
    <span className="bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
      Your Profile Dashboard
    </span>
  </h2>
  <p className="text-gray-600 dark:text-gray-400 text-lg sm:text-xl max-w-2xl mx-auto">
    Manage your personal details, update your preferences, and track your activity all in one place.
  </p>
</motion.div>


      <div className="flex flex-col lg:flex-row gap-10 items-start max-w-7xl mx-auto">

        {/* ---------------- LEFT COLUMN ---------------- */}
        <div className="lg:w-1/3 w-full space-y-8">
          {/* PROFILE OVERVIEW CARD */}
          <Card className="flex flex-col items-center p-8 transition hover:-translate-y-1 hover:shadow-[0_0_25px_rgba(160,68,255,0.35)] duration-500">
            <div className="w-40 h-40 rounded-full overflow-hidden mb-6 border-4 border-[#A044FF] shadow-2xl transition-transform hover:scale-105 duration-500">
              <img
                src={PROFILE_PLACEHOLDER_URL}
                alt="Profile"
                className="w-full h-full object-cover"
                onError={(e) => { e.target.onerror = null; e.target.src="https://placehold.co/300x300/6A3093/ffffff?text=User" }}
              />
            </div>
            <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-200 mb-2">{user.name}</h2>
            <p className="text-purple-600 dark:text-[#A044FF] font-medium mb-4">{user.email}</p>
            <p className="text-gray-500 dark:text-gray-400 text-sm leading-relaxed text-center border-t dark:border-gray-700 pt-4">
              Welcome back! Manage your personal details and app preferences below.
            </p>
          </Card>
        </div>

        {/* ---------------- RIGHT COLUMN ---------------- */}
        <div className="lg:w-2/3 w-full space-y-8">
          {/* EDIT PERSONAL DETAILS */}
          <Card className="space-y-6">
            <h2 className="flex items-center gap-2 text-3xl font-bold text-gray-900 dark:text-white mb-4">
              Personal Details
            </h2>

            <InputField
              label="Full Name"
              name="name"
              value={user.name}
              onChange={handleInputChange}
              icon={<User size={20} />}
            />

            <InputField
              label="New Password"
              type="password"
              name="password"
              value={user.password}
              onChange={handleInputChange}
              placeholder="••••••••"
              icon={<Lock size={20} />}
            />
          </Card>

          {/* Save Button */}
          <motion.button
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            transition={{ type: "spring", stiffness: 300 }}
            onClick={handleSave}
            className="w-full py-4 rounded-xl font-extrabold text-xl text-white bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] shadow-2xl shadow-[#A044FF]/50 hover:opacity-95 transition duration-300 transform hover:-translate-y-0.5 flex items-center justify-center gap-3"
          >
            <Save size={20} />
            Save All Changes
          </motion.button>
        </div>
      </div>
    </div>
  );
}
