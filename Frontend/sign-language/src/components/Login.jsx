import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from "../context/useAuth";
import { FaEnvelope, FaLock, FaSignInAlt, FaUserPlus, FaEye, FaEyeSlash, FaKey, FaShieldAlt, FaCheck } from 'react-icons/fa';
import { BsRobot, BsStars, BsLightningFill } from 'react-icons/bs';
import { motion } from 'framer-motion';

export default function Login() {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    if (error) setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const result = await login(formData);
    
    if (result.success) {
      navigate('/');
    } else {
      setError(result.error || 'Login failed. Please check your credentials.');
    }
    
    setLoading(false);
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
      <div className="absolute top-1/4 left-1/4 w-[400px] h-[400px] bg-gradient-to-r from-purple-600/20 via-purple-500/10 to-pink-500/10 rounded-full blur-[120px] pointer-events-none animate-pulse-slow" />
      <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-gradient-to-r from-pink-600/15 via-purple-400/10 to-blue-500/10 rounded-full blur-[120px] pointer-events-none" />

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
              WELCOME BACK
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
              Sign In to
            </motion.span>
            <motion.span
              variants={fadeUp}
              transition={{ delay: 0.1 }}
              className="block bg-gradient-to-r from-purple-600 to-purple-400 bg-clip-text text-transparent"
            >
              Your Account
            </motion.span>
          </motion.h1>

          <motion.p
            variants={fadeUp}
            transition={{ delay: 0.2 }}
            className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto"
          >
            Access AI-powered sign language translation and premium features
          </motion.p>
        </motion.div>

        {/* Login Form Section */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 lg:gap-12">
          
          {/* Left Column - Benefits (3/5 width) */}
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
            {/* Premium Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {[
                {
                  icon: <BsRobot className="text-2xl" />,
                  title: "AI Translation Ready",
                  description: "Access real-time sign language translation with advanced AI",
                  color: "from-purple-500/20 to-purple-600/20",
                  iconColor: "text-purple-500",
                  delay: 0.1
                },
                {
                  icon: <BsLightningFill className="text-2xl" />,
                  title: "Instant Access",
                  description: "Get started immediately with all premium features",
                  color: "from-pink-500/20 to-rose-600/20",
                  iconColor: "text-pink-500",
                  delay: 0.2
                },
                {
                  icon: <FaShieldAlt className="text-2xl" />,
                  title: "Secure Session",
                  description: "Encrypted connection with enterprise-grade security",
                  color: "from-blue-500/20 to-indigo-600/20",
                  iconColor: "text-blue-500",
                  delay: 0.3
                },
                {
                  icon: <BsStars className="text-2xl" />,
                  title: "Sync Across Devices",
                  description: "Your preferences and history saved in the cloud",
                  color: "from-violet-500/20 to-purple-600/20",
                  iconColor: "text-violet-500",
                  delay: 0.4
                }
              ].map((feature, i) => (
                <motion.div
                  key={i}
                  variants={scaleIn}
                  transition={{ delay: feature.delay }}
                  whileHover={{ 
                    scale: 1.05,
                    y: -8,
                    boxShadow: "0 20px 40px -15px rgba(139, 92, 246, 0.4)"
                  }}
                  className={`group relative p-6 rounded-2xl backdrop-blur-xl border transition-all duration-300 overflow-hidden bg-white/80 dark:bg-white/5 border-white/30 dark:border-white/10 hover:border-purple-300/50 dark:hover:border-purple-500/50`}
                >
                  <div className="relative z-10">
                    <div className={`p-4 rounded-xl bg-gradient-to-br ${feature.color} w-fit mb-4 group-hover:scale-110 transition-transform duration-300`}>
                      <div className={feature.iconColor}>
                        {feature.icon}
                      </div>
                    </div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                      {feature.title}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400 text-sm">
                      {feature.description}
                    </p>
                  </div>
                  <div className="absolute inset-0 bg-gradient-to-br from-transparent via-transparent to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                </motion.div>
              ))}
            </div>

            {/* Quick Stats */}
            <motion.div
              variants={fadeUp}
              className="p-6 rounded-2xl backdrop-blur-xl bg-gradient-to-r from-gray-50/80 to-purple-50/50 dark:from-gray-900/40 dark:to-purple-900/20 border border-gray-200/50 dark:border-purple-500/20"
            >
              <div className="grid grid-cols-2 gap-6">
                <div className="text-center">
                  <div className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-purple-400 bg-clip-text text-transparent">
                    99%
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    Accuracy Rate
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-purple-400 bg-clip-text text-transparent">
                    50+
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    Languages
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Trust Badges */}
            <motion.div
              variants={fadeUp}
              className="flex flex-wrap items-center gap-6"
            >
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                  <FaCheck className="text-green-600 dark:text-green-400" />
                </div>
                <div>
                  <div className="font-semibold text-gray-900 dark:text-white">Secure Login</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Protected by SSL</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                  <BsRobot className="text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <div className="font-semibold text-gray-900 dark:text-white">AI-Powered</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Advanced Models</div>
                </div>
              </div>
            </motion.div>
          </motion.div>

          {/* Right Column - Login Form (2/5 width) */}
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
              <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] mx-auto mb-8">
                <FaSignInAlt className="text-2xl text-white" />
              </div>

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
                      className="w-full px-4 py-3 pl-12 bg-white/50 dark:bg-gray-900/50 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 transition-all"
                      placeholder="you@example.com"
                    />
                    <FaEnvelope className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500" />
                  </div>
                </div>

                {/* Password Field */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center gap-2">
                      <FaLock className="text-purple-500" />
                      Password
                    </label>
                    <Link 
                      to="/forget-password"
                      className="text-xs text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 transition-colors"
                    >
                      Forgot password?
                    </Link>
                  </div>
                  <div className="relative">
                    <input
                      type={showPassword ? "text" : "password"}
                      name="password"
                      value={formData.password}
                      onChange={handleChange}
                      required
                      className="w-full px-4 py-3 pl-12 pr-12 bg-white/50 dark:bg-gray-900/50 border border-gray-300 dark:border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 transition-all"
                      placeholder="Enter your password"
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
                </div>

                {/* Remember Me Checkbox */}
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="remember"
                    className="h-4 w-4 rounded border-gray-300 dark:border-gray-600 bg-white/50 dark:bg-gray-900/50 text-purple-600 focus:ring-purple-500 focus:ring-offset-0"
                  />
                  <label htmlFor="remember" className="ml-2 text-sm text-gray-600 dark:text-gray-400">
                    Remember this device
                  </label>
                </div>

                {/* Error Message */}
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center">
                        <span className="text-white text-xs">!</span>
                      </div>
                      <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
                    </div>
                  </motion.div>
                )}

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
                      Signing In...
                    </>
                  ) : (
                    <>
                      <FaSignInAlt className="group-hover:scale-110 transition-transform" />
                      Sign In to Your Account
                    </>
                  )}
                </motion.button>

                {/* Divider */}
                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-gray-300/50 dark:border-gray-700/50" />
                  </div>
                  <div className="relative flex justify-center text-sm">
                    <span className="px-4 bg-white dark:bg-transparent text-gray-500 dark:text-gray-400">
                      New to LinguaSign?
                    </span>
                  </div>
                </div>

                {/* Register Link */}
                <Link
                  to="/register"
                  className="w-full px-6 py-4 border-2 border-purple-600/50 dark:border-purple-500/50 text-purple-600 dark:text-purple-400 hover:bg-purple-50/50 dark:hover:bg-purple-900/20 hover:border-purple-700/50 dark:hover:border-purple-400/50 rounded-xl font-bold transition-all duration-300 flex items-center justify-center gap-3 group"
                >
                  <FaUserPlus className="group-hover:scale-110 transition-transform" />
                  Create New Account
                </Link>
              </form>

              {/* Terms */}
              <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-800">
                <p className="text-xs text-center text-gray-500 dark:text-gray-400">
                  By signing in, you agree to our{' '}
                  <a href="#" className="text-purple-600 dark:text-purple-400 hover:underline">
                    Terms of Service
                  </a>{' '}
                  and{' '}
                  <a href="#" className="text-purple-600 dark:text-purple-400 hover:underline">
                    Privacy Policy
                  </a>
                </p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Security Footer */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.5 }}
          className="mt-12 text-center"
        >
          <div className="inline-flex items-center gap-3 text-sm text-gray-500 dark:text-gray-400">
            <FaShieldAlt className="text-purple-500" />
            <span>Protected by 256-bit SSL encryption • Your data is never stored unencrypted</span>
          </div>
        </motion.div>
      </div>

      {/* Custom CSS for animated pulse */}
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