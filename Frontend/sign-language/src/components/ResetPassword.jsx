import React, { useState } from 'react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import axios from 'axios';
import { FaEnvelope, FaLock, FaKey, FaArrowLeft, FaEye, FaEyeSlash, FaCheckCircle } from 'react-icons/fa';
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
  const navigate = useNavigate();

  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setError(''); // Clear error on input change
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate passwords match
    if (formData.new_password !== formData.confirm_password) {
      setError('Passwords do not match');
      return;
    }

    // Validate password strength
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
      
      // Redirect to login after 2 seconds
      setTimeout(() => {
        navigate('/login');
      }, 2000);
    } catch (err) {
      setError(err.response?.data?.detail || 'Invalid or expired reset code. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Motion variants
  const fadeUp = { hidden: { opacity: 0, y: 30 }, visible: { opacity: 1, y: 0 } };
  const fadeIn = { hidden: { opacity: 0 }, visible: { opacity: 1 } };

  return (
    <div className="w-full bg-gray-50 dark:bg-[#0f0c29] py-16 px-6 lg:px-20 relative overflow-hidden transition-colors duration-500 min-h-screen flex items-center justify-center">
      
      {/* Background Glows */}
      <div className="absolute top-1/4 left-1/4 w-[300px] h-[300px] bg-purple-600/10 rounded-full blur-[100px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-[300px] h-[300px] bg-green-600/10 rounded-full blur-[100px] pointer-events-none" />

      <div className="max-w-lg w-full mx-auto relative z-10">
        {/* Header Section */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          transition={{ duration: 0.8 }}
          className="text-center mb-10"
        >
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-r from-green-600 to-purple-600 mb-6 shadow-lg">
            <FaCheckCircle className="text-3xl text-white" />
          </div>
          <span className="text-purple-600 dark:text-purple-400 font-bold tracking-widest uppercase text-sm mb-2 block">
            Account Security
          </span>
          <h2 className="text-4xl sm:text-5xl font-extrabold mb-4 text-gray-900 dark:text-white">
            <span className="bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
              Set New Password
            </span>
          </h2>
          <p className="text-gray-600 dark:text-gray-400 text-lg sm:text-xl max-w-2xl mx-auto">
            Enter the 6-digit verification code from your email and create a new password
          </p>
        </motion.div>

        {/* Reset Password Card */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="p-8 lg:p-10 dark:bg-[#1a163a]/60 backdrop-blur-xl border border-gray-200 dark:border-purple-500/20 rounded-3xl shadow-xl transition duration-500 hover:shadow-purple-900/40"
        >
          {/* Success Message */}
          {message && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 p-4 rounded-r-lg mb-6"
            >
              <div className="flex items-center">
                <svg className="h-5 w-5 text-green-400 mr-3" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <p className="text-sm text-green-700 dark:text-green-300">{message}</p>
              </div>
            </motion.div>
          )}

          {/* Error Message */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 p-4 rounded-r-lg mb-6"
            >
              <div className="flex items-center">
                <svg className="h-5 w-5 text-red-400 mr-3" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
              </div>
            </motion.div>
          )}

          <form className="space-y-6" onSubmit={handleSubmit}>
            {/* Email Input */}
            <div className="relative">
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                disabled={!!location.state?.email}
                placeholder="Your Email"
                className="w-full p-4 pl-12 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 placeholder-transparent peer bg-white/70 dark:bg-gray-700/50 text-gray-900 dark:text-gray-200 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
              />
              <FaEnvelope className="absolute left-4 top-1/2 transform -translate-y-1/2 text-purple-400 peer-focus:text-purple-600 transition-colors" />
              <label className="absolute left-12 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 text-base peer-placeholder-shown:top-1/2 peer-focus:top-3 peer-focus:text-purple-600 peer-focus:dark:text-purple-400 peer-focus:text-sm transition-all pointer-events-none">
                Email Address
              </label>
            </div>

            {/* Verification Code Input */}
            <div className="relative">
              <input
                type="text"
                name="code"
                value={formData.code}
                onChange={handleChange}
                required
                maxLength="6"
                placeholder="6-digit Code"
                className="w-full p-4 pl-12 text-center tracking-widest text-xl font-mono border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 placeholder-transparent peer bg-white/70 dark:bg-gray-700/50 text-gray-900 dark:text-gray-200 transition-colors"
              />
              <FaKey className="absolute left-4 top-1/2 transform -translate-y-1/2 text-purple-400 peer-focus:text-purple-600 transition-colors" />
              <label className="absolute left-12 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 text-base peer-placeholder-shown:top-1/2 peer-focus:top-3 peer-focus:text-purple-600 peer-focus:dark:text-purple-400 peer-focus:text-sm transition-all pointer-events-none">
                Verification Code
              </label>
            </div>

            {/* New Password Input */}
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                name="new_password"
                value={formData.new_password}
                onChange={handleChange}
                required
                minLength={6}
                placeholder="New Password"
                className="w-full p-4 pl-12 pr-12 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 placeholder-transparent peer bg-white/70 dark:bg-gray-700/50 text-gray-900 dark:text-gray-200 transition-colors"
              />
              <FaLock className="absolute left-4 top-1/2 transform -translate-y-1/2 text-purple-400 peer-focus:text-purple-600 transition-colors" />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-purple-600 transition-colors"
              >
                {showPassword ? <FaEyeSlash /> : <FaEye />}
              </button>
              <label className="absolute left-12 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 text-base peer-placeholder-shown:top-1/2 peer-focus:top-3 peer-focus:text-purple-600 peer-focus:dark:text-purple-400 peer-focus:text-sm transition-all pointer-events-none">
                New Password (min 6 characters)
              </label>
            </div>

            {/* Confirm Password Input */}
            <div className="relative">
              <input
                type={showConfirmPassword ? "text" : "password"}
                name="confirm_password"
                value={formData.confirm_password}
                onChange={handleChange}
                required
                minLength={6}
                placeholder="Confirm Password"
                className="w-full p-4 pl-12 pr-12 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 placeholder-transparent peer bg-white/70 dark:bg-gray-700/50 text-gray-900 dark:text-gray-200 transition-colors"
              />
              <FaLock className="absolute left-4 top-1/2 transform -translate-y-1/2 text-purple-400 peer-focus:text-purple-600 transition-colors" />
              <button
                type="button"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-purple-600 transition-colors"
              >
                {showConfirmPassword ? <FaEyeSlash /> : <FaEye />}
              </button>
              <label className="absolute left-12 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 text-base peer-placeholder-shown:top-1/2 peer-focus:top-3 peer-focus:text-purple-600 peer-focus:dark:text-purple-400 peer-focus:text-sm transition-all pointer-events-none">
                Confirm Password
              </label>
            </div>

            {/* Password Requirements */}
            <motion.div
              initial="hidden"
              whileInView="visible"
              variants={fadeIn}
              className="bg-gray-50 dark:bg-gray-800/50 p-4 rounded-xl border border-gray-200 dark:border-gray-700"
            >
              <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">Password Requirements:</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li className={`flex items-center gap-2 ${formData.new_password.length >= 6 ? 'text-green-600 dark:text-green-400' : ''}`}>
                  <svg className={`w-4 h-4 ${formData.new_password.length >= 6 ? 'text-green-500' : 'text-gray-400'}`} fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d={formData.new_password.length >= 6 ? "M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" : "M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H7z"} clipRule="evenodd" />
                  </svg>
                  At least 6 characters long
                </li>
                <li className={`flex items-center gap-2 ${formData.new_password === formData.confirm_password && formData.new_password ? 'text-green-600 dark:text-green-400' : ''}`}>
                  <svg className={`w-4 h-4 ${formData.new_password === formData.confirm_password && formData.new_password ? 'text-green-500' : 'text-gray-400'}`} fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d={formData.new_password === formData.confirm_password && formData.new_password ? "M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" : "M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H7z"} clipRule="evenodd" />
                  </svg>
                  Passwords match
                </li>
              </ul>
            </motion.div>

            {/* Submit Button */}
            <motion.button
              type="submit"
              disabled={loading}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              transition={{ type: "spring", stiffness: 300 }}
              className="w-full px-6 py-4 bg-gradient-to-r from-[#6A3093] to-[#A044FF] text-white font-bold rounded-full shadow-lg shadow-purple-500/40 transform transition duration-300 flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Resetting Password...
                </>
              ) : (
                <>
                  <FaCheckCircle />
                  Reset Password
                </>
              )}
            </motion.button>

            {/* Links Section */}
            <div className="pt-6 border-t border-gray-200 dark:border-gray-800 space-y-4">
              <div className="text-center">
                <Link 
                  to="/forget-password" 
                  className="text-purple-600 dark:text-purple-400 hover:text-purple-800 dark:hover:text-purple-300 transition-colors font-medium text-sm"
                >
                  Need a new verification code?
                </Link>
              </div>
              
              <div className="text-center">
                <Link 
                  to="/login"
                  className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-300 transition-colors font-medium"
                >
                  <FaArrowLeft />
                  Back to Login
                </Link>
              </div>
            </div>
          </form>
        </motion.div>

        {/* Security Note */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3 }}
          className="mt-8 text-center"
        >
          <div className="inline-flex items-center gap-3 bg-white/50 dark:bg-gray-800/50 px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700">
            <svg className="w-5 h-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Your new password will be encrypted and securely stored
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
}