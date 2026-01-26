<<<<<<< HEAD
import React from 'react'

export default function Profile() {
  return (
    <div>Profile</div>
  )
}
=======
import React, { useState } from "react";
import { motion } from "framer-motion";
import { 
  Save, 
  CheckCircle, 
  User, 
  Lock, 
  Mail, 
  Globe, 
  Palette, 
  Bell, 
  Shield,
  Eye,
  EyeOff,
  Upload,
  Camera,
  BarChart3,
  Activity,
  Award,
  Settings,
  LogOut,
  Zap,
  Sparkles
} from 'lucide-react';
import { TbSparkles } from "react-icons/tb";

const PROFILE_PLACEHOLDER_URL = "https://placehold.co/300x300/A044FF/ffffff?text=User";

export default function Profile() {
  const [user, setUser] = useState({
    name: "Shahd Mohamed",
    email: "shahd@example.com",
    password: "",
    theme: "system",
    preferredLanguage: "ASL",
    notifications: true,
    twoFactor: false,
    autoSave: true
  });
  const [showToast, setShowToast] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [stats] = useState({
    translations: 1250,
    accuracy: 98.7,
    streak: 42,
    level: "Expert"
  });

  const fadeUp = {
    hidden: { opacity: 0, y: 25 },
    visible: { opacity: 1, y: 0 },
  };

  const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setUser((prevUser) => ({ 
      ...prevUser, 
      [name]: type === 'checkbox' ? checked : value 
    }));
  };

  const handleSave = () => {
    console.log("Saving changes for user:", user);
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000);
  };

  const Card = ({ children, className = "", hover = false }) => (
    <motion.div
      variants={fadeUp}
      whileHover={hover ? { y: -5, scale: 1.02 } : {}}
      className={`
        relative p-8 bg-gradient-to-br from-white/80 to-white/60 dark:from-white/10 dark:to-white/5 
        backdrop-blur-xl border border-purple-200/50 dark:border-purple-500/20 
        rounded-3xl shadow-xl shadow-purple-100/20 dark:shadow-purple-900/20
        transition-all duration-500 overflow-hidden
        ${hover ? 'hover:shadow-2xl hover:shadow-purple-200/30 dark:hover:shadow-purple-900/40' : ''}
        ${className}
      `}
    >
      {children}
    </motion.div>
  );

  const InputField = ({ label, type = "text", name, value, onChange, placeholder, icon, options, isSelect = false }) => (
  <div className="space-y-2">
    <label htmlFor={name} className="font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-2">
      {icon && <span className="text-purple-600 dark:text-purple-400">{icon}</span>}
      {label}
    </label>
    <div className="relative group">
      {isSelect ? (
        <select
          id={name}
          name={name}
          value={value}
          onChange={onChange}
          className="
            w-full p-3 pl-12 rounded-xl bg-white/70 dark:bg-gray-800/50
            border border-gray-300 dark:border-gray-700
            text-gray-900 dark:text-gray-200
            focus:outline-none focus:ring-2 focus:ring-[#A044FF]/80
            transition-all duration-300
            group-hover:border-purple-400/50 dark:group-hover:border-purple-400/30
            appearance-none bg-no-repeat bg-[length:20px_20px] bg-[center_right_1rem]
          "
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%23A044FF' viewBox='0 0 16 16'%3E%3Cpath d='M7.247 11.14 2.451 5.658C1.885 5.013 2.345 4 3.204 4h9.592a1 1 0 0 1 .753 1.659l-4.796 5.48a1 1 0 0 1-1.506 0z'/%3E%3C/svg%3E")`
          }}
        >
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      ) : (
        <>
          <span className="absolute inset-y-0 left-3 flex items-center text-gray-400 dark:text-gray-400">
            {icon}
          </span>
          <input
            id={name}
            type={type === 'password' && showPassword ? 'text' : type}
            name={name}
            value={value}
            onChange={onChange}
            placeholder={placeholder}
            className="
              w-full p-3 pl-12 pr-12 rounded-xl bg-white/70 dark:bg-gray-800/50
              border border-gray-300 dark:border-gray-700
              text-gray-900 dark:text-gray-200
              focus:outline-none focus:ring-2 focus:ring-[#A044FF]/80
              transition-all duration-300
              group-hover:border-purple-400/50 dark:group-hover:border-purple-400/30
            "
          />
          {type === 'password' && (
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute inset-y-0 right-3 flex items-center text-gray-400 hover:text-purple-600 transition-colors"
            >
              {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
            </button>
          )}
        </>
      )}
    </div>
  </div>
);

  const Switch = ({ label, checked, onChange, name }) => (
    <div className="flex items-center justify-between py-2">
      <span className="text-gray-700 dark:text-gray-300 font-medium">{label}</span>
      <button
        type="button"
        onClick={() => onChange({ target: { name, type: 'checkbox', checked: !checked } })}
        className={`
          relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-300
          ${checked ? 'bg-purple-600' : 'bg-gray-300 dark:bg-gray-700'}
        `}
      >
        <span
          className={`
            inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-300
            ${checked ? 'translate-x-6' : 'translate-x-1'}
          `}
        />
      </button>
    </div>
  );

  const StatCard = ({ icon: Icon, label, value, unit = "", gradient = "from-purple-500 to-pink-500" }) => (
    <motion.div
      whileHover={{ scale: 1.05 }}
      className={`
        relative p-6 rounded-2xl bg-gradient-to-br from-white/80 to-white/60 dark:from-white/10 dark:to-white/5
        backdrop-blur-xl border border-purple-200/50 dark:border-purple-500/20
        shadow-lg shadow-purple-100/20 dark:shadow-purple-900/20
        group transition-all duration-500 overflow-hidden
      `}
    >
      <div className={`absolute inset-0 bg-gradient-to-r ${gradient}/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-4">
          <div className={`p-3 rounded-xl bg-gradient-to-br ${gradient}/10`}>
            <Icon className="text-gray-800 dark:text-gray-200" size={24} />
          </div>
          <TbSparkles className="text-purple-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        </div>
        <div className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
          {value}
          {unit && <span className="text-lg text-purple-600 dark:text-purple-400 ml-1">{unit}</span>}
        </div>
        <div className="text-sm text-gray-600 dark:text-gray-400">{label}</div>
      </div>
    </motion.div>
  );

  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-gray-50 via-white to-purple-50/60 dark:from-[#0a0518] dark:via-[#110a2e] dark:to-[#1e0f5c] py-24 px-4 sm:px-6 lg:px-8 font-inter">
      
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

      {/* Animated glows */}
      <div className="absolute top-1/4 left-1/4 w-[600px] h-[600px] bg-gradient-to-r from-purple-600/10 via-pink-600/10 to-indigo-600/10 rounded-full blur-[120px]" />
      <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-gradient-to-r from-indigo-600/10 via-purple-600/10 to-pink-600/10 rounded-full blur-[100px]" />

      {/* Save Toast Notification */}
      <motion.div
        initial={{ x: "100%", opacity: 0 }}
        animate={showToast ? { x: 0, opacity: 1 } : { x: "100%", opacity: 0 }}
        transition={{ duration: 0.5, type: "spring", stiffness: 100 }}
        className="fixed top-24 right-6 p-4 rounded-xl shadow-2xl bg-gradient-to-r from-green-500/90 to-emerald-500/90 backdrop-blur-md text-white font-semibold flex items-center gap-2 z-50"
      >
        <CheckCircle size={20} />
        <span>Changes saved successfully!</span>
        <div className="w-32 h-1 bg-white/50 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: "100%" }}
            animate={{ width: "0%" }}
            transition={{ duration: 3, ease: "linear" }}
            className="h-full bg-white"
          />
        </div>
      </motion.div>

      {/* Header */}
      <motion.div
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        variants={staggerContainer}
        className="max-w-7xl mx-auto text-center mb-16 relative z-10"
      >
        <motion.div
          variants={fadeUp}
          className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-gradient-to-r from-purple-500/15 via-purple-400/10 to-purple-300/10 border border-purple-200/60 dark:border-purple-700/60 backdrop-blur-xl shadow-lg shadow-purple-500/10 relative overflow-hidden group mb-8"
        >
          <div className="relative">
            <span className="absolute animate-ping inline-flex h-3.5 w-3.5 rounded-full bg-purple-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-gradient-to-r from-purple-500 to-purple-400" />
          </div>
          <span className="text-sm font-bold bg-gradient-to-r from-purple-600 via-purple-500 to-purple-400 bg-clip-text text-transparent">
            Personal Dashboard
          </span>
          <TbSparkles className="text-purple-500 ml-1" />
          <div className="absolute -inset-1 bg-gradient-to-r from-purple-500/0 via-purple-400/10 to-purple-500/0 group-hover:via-purple-400/20 transition-all duration-500" />
        </motion.div>

        <motion.h1
          variants={fadeUp}
          className="font-extrabold text-4xl sm:text-5xl lg:text-[53px] leading-tight mb-6"
        >
          <span className="block text-gray-900 dark:text-white">
            Welcome Back,
          </span>
          <span className="block bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
            {user.name}
          </span>
        </motion.h1>

        <motion.p
          variants={fadeUp}
          className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed"
        >
          Manage your personal details, preferences, and track your AI translation journey all in one place.
        </motion.p>

        {/* Decorative Elements */}
        <motion.div
          variants={fadeUp}
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

      <div className="max-w-7xl mx-auto">
    

        <div className="flex flex-col lg:flex-row gap-8">
          {/* Left Column */}
          <div className="lg:w-2/5 space-y-8">
            {/* Profile Card */}
            <Card hover={true} className="text-center">
              <div className="relative mb-6">
                <div className="w-40 h-40 mx-auto rounded-full overflow-hidden border-4 border-white dark:border-gray-800 shadow-2xl relative group">
                  <img
                    src={PROFILE_PLACEHOLDER_URL}
                    alt="Profile"
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-purple-600/30 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                  <button className="absolute bottom-2 right-2 p-2 rounded-full bg-white dark:bg-gray-800 shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    <Camera size={18} className="text-purple-600" />
                  </button>
                </div>
                <div className="absolute top-0 right-0">
                  <div className="w-12 h-12 rounded-full bg-gradient-to-r from-[#6A3093] to-[#A044FF] flex items-center justify-center shadow-lg">
                    <TbSparkles className="text-white" size={20} />
                  </div>
                </div>
              </div>
              
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">{user.name}</h2>
              <p className="text-purple-600 dark:text-purple-400 font-medium mb-4 flex items-center justify-center gap-2">
                <Mail size={16} />
                {user.email}
              </p>
              <p className="text-gray-500 dark:text-gray-400 text-sm leading-relaxed mb-6">
                AI Translation Expert • Premium Member • Joined 2024
              </p>
              
              <div className="flex flex-wrap gap-2 justify-center">
                <span className="px-3 py-1 rounded-full text-xs font-medium bg-gradient-to-r from-purple-500/10 to-pink-500/10 text-purple-700 dark:text-purple-300 border border-purple-300/30 dark:border-purple-500/30">
                  Premium
                </span>
                <span className="px-3 py-1 rounded-full text-xs font-medium bg-gradient-to-r from-blue-500/10 to-cyan-500/10 text-blue-700 dark:text-blue-300 border border-blue-300/30 dark:border-blue-500/30">
                  AI Translator
                </span>
                <span className="px-3 py-1 rounded-full text-xs font-medium bg-gradient-to-r from-green-500/10 to-emerald-500/10 text-green-700 dark:text-green-300 border border-green-300/30 dark:border-green-500/30">
                  Verified
                </span>
              </div>
            </Card>

            {/* Quick Actions */}
            <Card>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                <Zap className="text-purple-600" size={20} />
                Quick Actions
              </h3>
              <div className="space-y-3">
                <button className="w-full p-3 rounded-xl bg-gradient-to-r from-white/50 to-white/30 dark:from-white/5 dark:to-white/3 border border-purple-200/50 dark:border-purple-500/20 text-gray-700 dark:text-gray-300 font-medium hover:bg-white/70 dark:hover:bg-white/10 transition-all duration-300 flex items-center gap-3">
                  <Settings size={18} />
                  Account Settings
                </button>
                <button className="w-full p-3 rounded-xl bg-gradient-to-r from-white/50 to-white/30 dark:from-white/5 dark:to-white/3 border border-purple-200/50 dark:border-purple-500/20 text-gray-700 dark:text-gray-300 font-medium hover:bg-white/70 dark:hover:bg-white/10 transition-all duration-300 flex items-center gap-3">
                  <Shield size={18} />
                  Privacy & Security
                </button>
                <button className="w-full p-3 rounded-xl bg-gradient-to-r from-red-500/10 to-pink-500/10 border border-red-200/50 dark:border-red-500/20 text-red-600 dark:text-red-400 font-medium hover:bg-red-500/20 transition-all duration-300 flex items-center gap-3">
                  <LogOut size={18} />
                  Sign Out
                </button>
              </div>
            </Card>
          </div>

          {/* Right Column */}
          <div className="lg:w-3/5 space-y-8">
            {/* Personal Details */}
            <Card>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-8 flex items-center gap-3">
                <User className="text-purple-600" size={24} />
                Personal Information
              </h3>
              <div className="grid md:grid-cols-2 gap-6">
                <InputField
                  label="Full Name"
                  name="name"
                  value={user.name}
                  onChange={handleInputChange}
                  icon={<User size={18} />}
                />
                <InputField
                  label="Email Address"
                  type="email"
                  name="email"
                  value={user.email}
                  onChange={handleInputChange}
                  icon={<Mail size={18} />}
                />
                <InputField
                  label="Password"
                  type="password"
                  name="password"
                  value={user.password}
                  onChange={handleInputChange}
                  placeholder="Enter new password"
                  icon={<Lock size={18} />}
                />
                <InputField
                  label="Preferred Language"
                  name="preferredLanguage"
                  value={user.preferredLanguage}
                  onChange={handleInputChange}
                  isSelect={true}
                  icon={<Globe size={18} />}
                  options={[
                    { value: "ASL", label: "American Sign Language" },
                    { value: "BSL", label: "British Sign Language" },
                    { value: "LSF", label: "French Sign Language" },
                    { value: "DGS", label: "German Sign Language" },
                  ]}
                />
              </div>
            </Card>

            {/* Preferences */}
            <Card>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-8 flex items-center gap-3">
                <Palette className="text-purple-600" size={24} />
                Preferences & Settings
              </h3>
              <div className="space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <InputField
                    label="Theme"
                    name="theme"
                    value={user.theme}
                    onChange={handleInputChange}
                    isSelect={true}
                    icon={<Palette size={18} />}
                    options={[
                      { value: "light", label: "Light Mode" },
                      { value: "dark", label: "Dark Mode" },
                      { value: "system", label: "System Default" },
                    ]}
                  />
                  <div className="space-y-4">
                    <Switch
                      label="Push Notifications"
                      name="notifications"
                      checked={user.notifications}
                      onChange={handleInputChange}
                    />
                    <Switch
                      label="Two-Factor Authentication"
                      name="twoFactor"
                      checked={user.twoFactor}
                      onChange={handleInputChange}
                    />
                    <Switch
                      label="Auto Save Progress"
                      name="autoSave"
                      checked={user.autoSave}
                      onChange={handleInputChange}
                    />
                  </div>
                </div>
              </div>
            </Card>

            {/* Save Button */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleSave}
              className="w-full py-4 rounded-xl font-extrabold text-xl text-white bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] shadow-2xl shadow-purple-500/30 hover:shadow-purple-500/50 transition-all duration-300 relative overflow-hidden group"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-white/5 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
              <div className="relative z-10 flex items-center justify-center gap-3">
                <Save size={24} />
                Save All Changes
                <Sparkles className="opacity-0 group-hover:opacity-100 transition-opacity duration-300" size={18} />
              </div>
            </motion.button>
          </div>
        </div>
      </div>
    </div>
  );
}
>>>>>>> e251330 (Add frontend, backend, and ai_service)
