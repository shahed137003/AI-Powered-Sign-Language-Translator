import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from "../context/useAuth";
import { FaUser, FaEnvelope, FaLock, FaUserPlus, FaSignInAlt, FaEye, FaEyeSlash } from 'react-icons/fa';
import { motion } from 'framer-motion';

export default function Register() {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [success, setSuccess] = useState(false);
  
  const { register } = useAuth();
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setError(''); // Clear error on input change
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    // Validation
    if (!formData.username.trim()) {
      setError('Username is required');
      return;
    }
    
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    
    if (formData.password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    setLoading(true);

    try {
      const result = await register({
        username: formData.username.trim(),
        email: formData.email,
        password: formData.password
      });
      
      if (result.success) {
        setSuccess(true);
        setTimeout(() => {
          navigate('/'); // Go to home page after successful registration
        }, 2000);
      } else {
        setError(result.error || 'Registration failed. Please try again.');
      }
    } catch (err) {
      setError('An unexpected error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Motion variants
  const fadeUp = { hidden: { opacity: 0, y: 30 }, visible: { opacity: 1, y: 0 } };

  return (
    <div className="w-full bg-gray-50 dark:bg-[#0f0c29] py-16 px-6 lg:px-20 relative overflow-hidden transition-colors duration-500 min-h-screen flex items-center justify-center">
      
      {/* Background Glows */}
      <div className="absolute top-1/4 left-1/4 w-[300px] h-[300px] bg-purple-600/10 rounded-full blur-[100px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-[300px] h-[300px] bg-pink-600/10 rounded-full blur-[100px] pointer-events-none" />

      <div className="max-w-4xl w-full mx-auto relative z-10">
        {/* Header Section */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          transition={{ duration: 0.8 }}
          className="text-center mb-10"
        >
          <span className="text-purple-600 dark:text-purple-400 font-bold tracking-widest uppercase text-sm mb-2 block">
            Join Our Community
          </span>
          <h2 className="text-4xl sm:text-5xl font-extrabold mb-4 text-gray-900 dark:text-white">
            <span className="bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] bg-clip-text text-transparent">
              Create Account
            </span>
          </h2>
          <p className="text-gray-600 dark:text-gray-400 text-lg sm:text-xl max-w-2xl mx-auto">
            Start your journey with LinguaSign and unlock powerful sign language translation features
          </p>
        </motion.div>

        {/* Registration Form Card */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Form Section */}
          <motion.form
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            onSubmit={handleSubmit}
            className="p-8 lg:p-10 dark:bg-[#1a163a]/60 backdrop-blur-xl border border-gray-200 dark:border-purple-500/20 rounded-3xl shadow-xl flex flex-col gap-6 
              transition duration-500 hover:shadow-purple-900/40"
          >
            {/* Username Input */}
            <div className="relative">
              <input
                type="text"
                name="username"
                value={formData.username}
                onChange={handleChange}
                required
                placeholder="Your Username"
                className="w-full p-4 pl-12 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 placeholder-transparent peer bg-white/70 dark:bg-gray-700/50 text-gray-900 dark:text-gray-200 transition-colors"
              />
              <FaUser className="absolute left-4 top-1/2 transform -translate-y-1/2 text-purple-400 peer-focus:text-purple-600 transition-colors" />
              <label className="absolute left-12 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 text-base peer-placeholder-shown:top-1/2 peer-focus:top-3 peer-focus:text-purple-600 peer-focus:dark:text-purple-400 peer-focus:text-sm transition-all pointer-events-none">
                Username
              </label>
            </div>

            {/* Email Input */}
            <div className="relative">
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                placeholder="Your Email"
                className="w-full p-4 pl-12 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 placeholder-transparent peer bg-white/70 dark:bg-gray-700/50 text-gray-900 dark:text-gray-200 transition-colors"
              />
              <FaEnvelope className="absolute left-4 top-1/2 transform -translate-y-1/2 text-purple-400 peer-focus:text-purple-600 transition-colors" />
              <label className="absolute left-12 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 text-base peer-placeholder-shown:top-1/2 peer-focus:top-3 peer-focus:text-purple-600 peer-focus:dark:text-purple-400 peer-focus:text-sm transition-all pointer-events-none">
                Email Address
              </label>
            </div>

            {/* Password Input */}
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                name="password"
                value={formData.password}
                onChange={handleChange}
                required
                minLength={6}
                placeholder="Your Password"
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
                Password (min 6 characters)
              </label>
            </div>

            {/* Confirm Password Input */}
            <div className="relative">
              <input
                type={showConfirmPassword ? "text" : "password"}
                name="confirmPassword"
                value={formData.confirmPassword}
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

            {/* Error Message */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 p-4 rounded-r-lg"
              >
                <div className="flex items-center">
                  <svg className="h-5 w-5 text-red-400 mr-3" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                  <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
                </div>
              </motion.div>
            )}

            {/* Success Message */}
            {success && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 p-4 rounded-r-lg"
              >
                <div className="flex items-center">
                  <svg className="h-5 w-5 text-green-400 mr-3" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  <p className="text-sm text-green-700 dark:text-green-300">
                    Account created successfully! Redirecting...
                  </p>
                </div>
              </motion.div>
            )}

            {/* Submit Button */}
            <motion.button
              type="submit"
              disabled={loading}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              transition={{ type: "spring", stiffness: 300 }}
              className="px-6 py-4 bg-gradient-to-r from-[#6A3093] to-[#A044FF] text-white font-bold rounded-full shadow-lg shadow-purple-500/40 transform transition duration-300 flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Creating Account...
                </>
              ) : (
                <>
                  <FaUserPlus />
                  Create Account
                </>
              )}
            </motion.button>

            {/* Login Link */}
            <div className="text-center pt-4 border-t border-gray-200 dark:border-gray-800">
              <p className="text-gray-600 dark:text-gray-400">
                Already have an account?{' '}
                <Link 
                  to="/login" 
                  className="font-medium text-purple-600 dark:text-purple-400 hover:text-purple-800 dark:hover:text-purple-300 transition-colors inline-flex items-center gap-2"
                >
                  <FaSignInAlt />
                  Sign In
                </Link>
              </p>
            </div>
          </motion.form>

          {/* Benefits Section */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="space-y-8"
          >
            <div>
              <h3 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">Why Join LinguaSign?</h3>
              
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-12 h-12 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-1">Real-time Translation</h4>
                    <p className="text-gray-600 dark:text-gray-400">Instant sign language to text conversion with AI accuracy</p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-12 h-12 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-1">Secure & Private</h4>
                    <p className="text-gray-600 dark:text-gray-400">Your data is encrypted and never shared with third parties</p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-12 h-12 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-1">Community Access</h4>
                    <p className="text-gray-600 dark:text-gray-400">Connect with other users and share your experiences</p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-12 h-12 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="text-xl font-semibold text-gray-900 dark:text-white mb-1">Priority Support</h4>
                    <p className="text-gray-600 dark:text-gray-400">Get dedicated support for any questions or issues</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Security Badge */}
            <div className="bg-gradient-to-r from-gray-50 to-purple-50 dark:from-gray-900/50 dark:to-purple-900/20 p-6 rounded-2xl border border-gray-200 dark:border-purple-500/20">
              <div className="flex items-center gap-4">
                <div className="flex-shrink-0">
                  <div className="w-12 h-12 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                    <svg className="w-6 h-6 text-green-600 dark:text-green-400" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white">Secure Registration</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">We use industry-standard encryption to protect your data</p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}