import React, { useState } from 'react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import axios from 'axios';
import { FaEnvelope, FaLock, FaKey, FaArrowLeft, FaEye, FaEyeSlash, FaCheckCircle, FaShieldAlt } from 'react-icons/fa';
import { BsLock, BsClock, BsCheck2Circle } from 'react-icons/bs';
import { motion } from 'framer-motion';

export default function ResetPassword() {
  const location = useLocation();
  const [formData, setFormData] = useState({
    email: location.state?.email || '',
    code: '',
    new_password: '',
    confirm_password: ''
  });
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [passwordStrength, setPasswordStrength] = useState(0);
  const navigate = useNavigate();

  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
    
    if (name === 'new_password') {
      // Calculate password strength
      let strength = 0;
      if (value.length >= 6) strength += 25;
      if (value.length >= 8) strength += 25;
      if (/[A-Z]/.test(value)) strength += 25;
      if (/[0-9]/.test(value)) strength += 25;
      setPasswordStrength(strength);
    }
    
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (formData.new_password !== formData.confirm_password) {
      setError('Passwords do not match');
      return;
    }

    if (formData.new_password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    setLoading(true);
    setError('');
    setMessage('');

    try {
      const { confirm_password, ...resetData } = formData;
      await axios.post(`${API_URL}/password/reset`, resetData);
      setMessage('Password reset successfully! Redirecting to login...');
      
      setTimeout(() => {
        navigate('/login');
      }, 2000);
    } catch (err) {
      setError(err.response?.data?.detail || 'Invalid or expired reset code. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Animation variants matching home page
  const fadeUp = {
    hidden: { opacity: 0, y: 40 },
    show: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.8,
        ease: [0.22, 1, 0.36, 1]
      }
    }
  };

  const fade = {
    hidden: { opacity: 0 },
    show: { 
      opacity: 1,
      transition: {
        duration: 1,
        ease: "easeOut"
      }
    }
  };

  const scaleIn = {
    hidden: { opacity: 0, scale: 0.8 },
    show: { 
      opacity: 1, 
      scale: 1,
      transition: {
        duration: 0.6,
        ease: "backOut"
      }
    }
  };

  const getStrengthColor = (strength) => {
    if (strength >= 75) return 'from-green-500 to-emerald-600';
    if (strength >= 50) return 'from-yellow-500 to-amber-600';
    if (strength >= 25) return 'from-orange-500 to-red-500';
    return 'from-gray-300 to-gray-400';
  };

  const getStrengthText = (strength) => {
    if (strength >= 75) return 'Strong';
    if (strength >= 50) return 'Good';
    if (strength >= 25) return 'Weak';
    return 'Very Weak';
  };

  return (
    <div className="relative w-full min-h-screen bg-gradient-to-br from-gray-50 via-white to-purple-50/60 dark:from-[#0a0518] dark:via-[#110a2e] dark:to-[#1e0f5c] overflow-hidden selection:bg-purple-500 selection:text-white transition-all duration-700">
      
      {/* Premium Geometric Grid - Same as homepage */}
      <div className="absolute inset-0 opacity-40 dark:opacity-60 pointer-events-none">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(90deg, rgba(168, 85, 247, 0.1) 1px, transparent 1px),
            linear-gradient(180deg, rgba(168, 85, 247, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px'
        }} />
      </div>

      {/* Animated gradient orbs */}
      <div className="absolute top-1/4 left-1/4 w-[400px] h-[400px] bg-gradient-to-r from-purple-600/20 via-purple-500/10 to-green-500/10 rounded-full blur-[120px] pointer-events-none animate-pulse-slow" />
      <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-gradient-to-r from-green-600/15 via-purple-400/10 to-blue-500/10 rounded-full blur-[120px] pointer-events-none" />

      <div className="relative z-10 w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-24">
        
        {/* Premium Badge */}
        <motion.div
          initial="hidden"
          animate="show"
          variants={fadeUp}
          className="text-center mb-12"
        >
          <motion.div
            variants={fadeUp}
            whileHover={{ scale: 1.05, rotate: 1 }}
            className="inline-flex items-center gap-3 px-6 py-3 rounded-full bg-gradient-to-r from-purple-500/15 via-purple-400/10 to-purple-300/10 border border-purple-200/60 dark:border-purple-700/60 backdrop-blur-xl shadow-lg shadow-purple-500/10 mb-8"
          >
            <div className="relative">
              <span className="absolute animate-ping inline-flex h-4 w-4 rounded-full bg-purple-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-4 w-4 bg-gradient-to-r from-purple-500 to-purple-400" />
            </div>
            <span className="text-sm font-bold bg-gradient-to-r from-purple-600 via-purple-500 to-purple-400 bg-clip-text text-transparent">
              SECURE PASSWORD RESET
            </span>
          </motion.div>

          {/* Main Header */}
          <motion.h1
            variants={fadeUp}
            className="font-extrabold text-4xl sm:text-5xl lg:text-[56px] leading-tight mb-6"
          >
            <motion.span
              variants={fadeUp}
              className="block text-gray-900 dark:text-white"
            >
              Reset Your
            </motion.span>
            <motion.span
              variants={fadeUp}
              transition={{ delay: 0.1 }}
              className="block bg-gradient-to-r from-purple-600 to-purple-400 bg-clip-text text-transparent"
            >
              Password Securely
            </motion.span>
          </motion.h1>

          <motion.p
            variants={fadeUp}
            transition={{ delay: 0.2 }}
            className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto"
          >
            Create a new strong password to secure your account
          </motion.p>
        </motion.div>


        {/* Content Section */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 lg:gap-12">
          
          {/* Left Column - Security Features (3/5 width) */}
          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            variants={{
              hidden: { opacity: 0, x: -50 },
              show: {
                opacity: 1,
                x: 0,
                transition: {
                  duration: 0.8,
                  ease: "easeOut"
                }
              }
            }}
            className="lg:col-span-3 space-y-8"
          >
            {/* Security Steps */}
            <div className="space-y-6">
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Security Process</h3>
              
              <div className="space-y-4">
                {[
                  {
                    step: "01",
                    title: "Email Verification",
                    description: "Enter the email associated with your account",
                    icon: <FaEnvelope className="text-xl" />,
                    color: "from-blue-500/20 to-blue-600/20",
                    iconColor: "text-blue-500",
                    status: "complete"
                  },
                  {
                    step: "02",
                    title: "Code Authentication",
                    description: "Enter the 6-digit code sent to your email",
                    icon: <FaKey className="text-xl" />,
                    color: "from-purple-500/20 to-purple-600/20",
                    iconColor: "text-purple-500",
                    status: formData.code ? "complete" : "current"
                  },
                  {
                    step: "03",
                    title: "Password Reset",
                    description: "Create a new strong password",
                    icon: <FaLock className="text-xl" />,
                    color: "from-green-500/20 to-green-600/20",
                    iconColor: "text-green-500",
                    status: "pending"
                  }
                ].map((item, i) => (
                  <motion.div
                    key={i}
                    variants={scaleIn}
                    transition={{ delay: i * 0.1 }}
                    className="flex items-center gap-4 p-4 rounded-2xl backdrop-blur-xl bg-white/80 dark:bg-white/5 border border-white/30 dark:border-white/10"
                  >
                    <div className={`flex-shrink-0 w-12 h-12 rounded-xl bg-gradient-to-br ${item.color} flex items-center justify-center`}>
                      <div className={item.iconColor}>
                        {item.icon}
                      </div>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-1">
                        <span className="text-sm font-bold text-gray-500 dark:text-gray-400">{item.step}</span>
                        <h4 className="font-semibold text-gray-900 dark:text-white">{item.title}</h4>
                        {item.status === "complete" && (
                          <BsCheck2Circle className="text-green-500 ml-auto" />
                        )}
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">{item.description}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Password Requirements */}
            <motion.div
              variants={fadeUp}
              className="p-6 rounded-2xl backdrop-blur-xl bg-gradient-to-r from-blue-50/80 to-purple-50/50 dark:from-blue-900/20 dark:to-purple-900/20 border border-blue-200/50 dark:border-blue-500/20"
            >
              <h4 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <BsLock className="text-blue-500" />
                Password Requirements
              </h4>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <div className={`w-5 h-5 rounded-full flex items-center justify-center ${
                    formData.new_password.length >= 6 
                      ? 'bg-green-500 text-white' 
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-400'
                  }`}>
                    {formData.new_password.length >= 6 && <BsCheck2Circle className="text-sm" />}
                  </div>
                  <span className={`text-sm ${formData.new_password.length >= 6 ? 'text-green-600 dark:text-green-400' : 'text-gray-600 dark:text-gray-400'}`}>
                    At least 6 characters long
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <div className={`w-5 h-5 rounded-full flex items-center justify-center ${
                    formData.new_password.length >= 8 
                      ? 'bg-green-500 text-white' 
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-400'
                  }`}>
                    {formData.new_password.length >= 8 && <BsCheck2Circle className="text-sm" />}
                  </div>
                  <span className={`text-sm ${formData.new_password.length >= 8 ? 'text-green-600 dark:text-green-400' : 'text-gray-600 dark:text-gray-400'}`}>
                    Minimum 8 characters for better security
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <div className={`w-5 h-5 rounded-full flex items-center justify-center ${
                    /[A-Z]/.test(formData.new_password) 
                      ? 'bg-green-500 text-white' 
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-400'
                  }`}>
                    {/[A-Z]/.test(formData.new_password) && <BsCheck2Circle className="text-sm" />}
                  </div>
                  <span className={`text-sm ${/[A-Z]/.test(formData.new_password) ? 'text-green-600 dark:text-green-400' : 'text-gray-600 dark:text-gray-400'}`}>
                    Include uppercase letters
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <div className={`w-5 h-5 rounded-full flex items-center justify-center ${
                    /[0-9]/.test(formData.new_password) 
                      ? 'bg-green-500 text-white' 
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-400'
                  }`}>
                    {/[0-9]/.test(formData.new_password) && <BsCheck2Circle className="text-sm" />}
                  </div>
                  <span className={`text-sm ${/[0-9]/.test(formData.new_password) ? 'text-green-600 dark:text-green-400' : 'text-gray-600 dark:text-gray-400'}`}>
                    Include numbers
                  </span>
                </div>
              </div>
            </motion.div>
          </motion.div>

          {/* Right Column - Reset Form (2/5 width) */}
          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true }}
            variants={{
              hidden: { opacity: 0, x: 50 },
              show: {
                opacity: 1,
                x: 0,
                transition: {
                  duration: 0.8,
                  ease: "easeOut",
                  delay: 0.2
                }
              }
            }}
            className="lg:col-span-2"
          >
            <div className="p-8 rounded-3xl backdrop-blur-xl bg-white/90 dark:bg-white/10 border border-white/30 dark:border-white/10 shadow-2xl shadow-purple-500/20">
              {/* Form Header Icon */}
              <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] mx-auto mb-8">
                <FaShieldAlt className="text-2xl text-white" />
              </div>

              {/* Success Message */}
              {message && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mb-6 p-4 rounded-xl bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center">
                      <FaCheckCircle className="text-white text-xs" />
                    </div>
                    <p className="text-sm text-green-700 dark:text-green-300">{message}</p>
                  </div>
                </motion.div>
              )}

              {/* Error Message */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mb-6 p-4 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center">
                      <span className="text-white text-xs">!</span>
                    </div>
                    <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
                  </div>
                </motion.div>
              )}

              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Email Field */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center gap-2">
                    <FaEnvelope className="text-purple-500" />
                    Email Address
                  </label>
                  <div className="relative">
                    <input
                      type="email"
                      name="email"
                      value={formData.email}
                      onChange={handleChange}
                      required
                      disabled={!!location.state?.email}
                      className="w-full px-4 py-3 pl-12 bg-white/50 dark:bg-gray-900/50 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 transition-all disabled:opacity-60 disabled:cursor-not-allowed"
                      placeholder="Enter your email"
                    />
                    <FaEnvelope className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500" />
                  </div>
                </div>

                {/* Verification Code Field */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center gap-2">
                    <FaKey className="text-purple-500" />
                    Verification Code
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      name="code"
                      value={formData.code}
                      onChange={handleChange}
                      required
                      maxLength="6"
                      className="w-full px-4 py-3 pl-12 text-center tracking-widest text-xl font-mono bg-white/50 dark:bg-gray-900/50 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 transition-all"
                      placeholder="123456"
                    />
                    <FaKey className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500" />
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1">
                    <BsClock className="text-gray-400" />
                    Code expires in 15 minutes
                  </p>
                </div>

                {/* New Password Field */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center gap-2">
                    <FaLock className="text-purple-500" />
                    New Password
                  </label>
                  <div className="relative">
                    <input
                      type={showPassword ? "text" : "password"}
                      name="new_password"
                      value={formData.new_password}
                      onChange={handleChange}
                      required
                      className="w-full px-4 py-3 pl-12 pr-12 bg-white/50 dark:bg-gray-900/50 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 transition-all"
                      placeholder="Create a strong password"
                    />
                    <FaLock className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500" />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-purple-500 dark:hover:text-purple-400 transition-colors"
                    >
                      {showPassword ? <FaEyeSlash /> : <FaEye />}
                    </button>
                  </div>
                  
                  {/* Password Strength Meter */}
                  {formData.new_password && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs">
                        <span className="text-gray-600 dark:text-gray-400">Password Strength</span>
                        <span className={`font-semibold ${
                          passwordStrength >= 75 ? 'text-green-600 dark:text-green-400' :
                          passwordStrength >= 50 ? 'text-yellow-600 dark:text-yellow-400' :
                          passwordStrength >= 25 ? 'text-orange-600 dark:text-orange-400' :
                          'text-red-600 dark:text-red-400'
                        }`}>
                          {getStrengthText(passwordStrength)}
                        </span>
                      </div>
                      <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${passwordStrength}%` }}
                          transition={{ duration: 0.5 }}
                          className={`h-full bg-gradient-to-r ${getStrengthColor(passwordStrength)} rounded-full`}
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Confirm Password Field */}
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center gap-2">
                    <FaLock className="text-purple-500" />
                    Confirm Password
                  </label>
                  <div className="relative">
                    <input
                      type={showConfirmPassword ? "text" : "password"}
                      name="confirm_password"
                      value={formData.confirm_password}
                      onChange={handleChange}
                      required
                      className="w-full px-4 py-3 pl-12 pr-12 bg-white/50 dark:bg-gray-900/50 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 transition-all"
                      placeholder="Re-enter your password"
                    />
                    <FaLock className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500" />
                    <button
                      type="button"
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-purple-500 dark:hover:text-purple-400 transition-colors"
                    >
                      {showConfirmPassword ? <FaEyeSlash /> : <FaEye />}
                    </button>
                  </div>
                  {formData.confirm_password && (
                    <p className={`text-xs flex items-center gap-1 ${
                      formData.new_password === formData.confirm_password 
                        ? 'text-green-600 dark:text-green-400' 
                        : 'text-red-600 dark:text-red-400'
                    }`}>
                      {formData.new_password === formData.confirm_password 
                        ? <BsCheck2Circle /> 
                        : '✗'
                      }
                      {formData.new_password === formData.confirm_password 
                        ? 'Passwords match' 
                        : 'Passwords do not match'
                      }
                    </p>
                  )}
                </div>

                {/* Submit Button */}
                <motion.button
                  type="submit"
                  disabled={loading}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="w-full px-6 py-4 bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] text-white font-bold rounded-xl shadow-lg shadow-purple-500/40 hover:shadow-purple-500/60 transition-all duration-300 flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed group"
                >
                  {loading ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Securing Account...
                    </>
                  ) : (
                    <>
                      <FaShieldAlt className="group-hover:scale-110 transition-transform" />
                      Reset Password & Secure Account
                    </>
                  )}
                </motion.button>

                {/* Links Section */}
                <div className="space-y-4">
                  <div className="text-center">
                    <Link 
                      to="/forget-password" 
                      className="text-sm text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 transition-colors"
                    >
                      Need a new verification code?
                    </Link>
                  </div>
                  
                  <div className="pt-4 border-t border-gray-200 dark:border-gray-800">
                    <Link
                      to="/login"
                      className="w-full flex items-center justify-center gap-3 px-6 py-3 border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800/50 rounded-xl font-medium transition-all duration-300 group"
                    >
                      <FaArrowLeft className="group-hover:-translate-x-1 transition-transform" />
                      Back to Sign In
                    </Link>
                  </div>
                </div>
              </form>

              {/* Security Footer */}
              <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-800">
                <div className="flex items-center justify-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                  <FaShieldAlt className="text-green-500" />
                  <span>Your password is encrypted with AES-256 encryption</span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Security Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.5 }}
          className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6"
        >
          <div className="p-6 rounded-2xl backdrop-blur-xl bg-gradient-to-r from-purple-500/5 to-transparent border border-purple-200/30 dark:border-purple-500/20">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-purple-500 to-purple-600 flex items-center justify-center">
                <FaLock className="text-white" />
              </div>
              <div>
                <div className="text-lg font-bold text-gray-900 dark:text-white">256-bit</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Military Encryption</div>
              </div>
            </div>
          </div>
          <div className="p-6 rounded-2xl backdrop-blur-xl bg-gradient-to-r from-green-500/5 to-transparent border border-green-200/30 dark:border-green-500/20">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-green-500 to-green-600 flex items-center justify-center">
                <FaShieldAlt className="text-white" />
              </div>
              <div>
                <div className="text-lg font-bold text-gray-900 dark:text-white">Zero-Knowledge</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Privacy Protocol</div>
              </div>
            </div>
          </div>
          <div className="p-6 rounded-2xl backdrop-blur-xl bg-gradient-to-r from-blue-500/5 to-transparent border border-blue-200/30 dark:border-blue-500/20">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-r from-blue-500 to-blue-600 flex items-center justify-center">
                <BsClock className="text-white" />
              </div>
              <div>
                <div className="text-lg font-bold text-gray-900 dark:text-white">15 Min</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Code Expiry</div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Custom CSS for animations */}
      <style jsx>{`
        @keyframes pulse-slow {
          0%, 100% { opacity: 0.5; }
          50% { opacity: 0.8; }
        }
        .animate-pulse-slow {
          animation: pulse-slow 4s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}