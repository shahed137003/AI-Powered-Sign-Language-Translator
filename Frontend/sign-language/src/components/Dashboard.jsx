import React, { useState, useEffect, useRef } from "react";
import { useTheme } from "../context/ThemeContext";
import { motion, AnimatePresence } from "framer-motion";
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
  Camera,
  BarChart3,
  Activity,
  Award,
  Settings,
  LogOut,
  Zap,
  Sparkles,
  TrendingUp,
  Clock,
  Users,
  MessageSquare,
  PieChart,
  ArrowUpRight,
} from "lucide-react";
import { TbSparkles, TbHandLoveYou } from "react-icons/tb";
import { BsArrowRight } from "react-icons/bs";

// Placeholder avatar (same as Profile)
const PROFILE_PLACEHOLDER_URL = "https://placehold.co/300x300/A044FF/ffffff?text=User";

export default function Dashboard() {
  const { themeColor } = useTheme();
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationFrameRef = useRef(null);
  const [isDark, setIsDark] = useState(false);
  const [showToast, setShowToast] = useState(false);

  // ---- Theme / Dark mode observer ----
  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains("dark"));
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    setIsDark(document.documentElement.classList.contains("dark"));
    return () => observer.disconnect();
  }, []);

  // ---- Particle system (identical to Profile) ----
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const themeColorsMap = {
      purple: ["#A855F7", "#9333EA", "#7C3AED", "#6D28D9", "#8B5CF6"],
      "midnight-blue": ["#6366F1", "#4F46E5", "#4338CA", "#3730A3", "#818CF8"],
    };
    const currentThemeColors = themeColorsMap[themeColor] || themeColorsMap["purple"];
    const colors = isDark ? currentThemeColors : currentThemeColors.slice().reverse();

    particlesRef.current = Array.from({ length: 100 }).map(() => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      size: Math.random() * 3 + 1,
      speedX: Math.random() * 0.5 - 0.25,
      speedY: Math.random() * 0.5 - 0.25,
      color: colors[Math.floor(Math.random() * colors.length)],
      opacity: Math.random() * 0.5 + 0.1,
      glow: Math.random() > 0.8,
    }));

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particlesRef.current.forEach((particle) => {
        particle.x += particle.speedX;
        particle.y += particle.speedY;

        if (particle.x < -10) particle.x = canvas.width + 10;
        if (particle.x > canvas.width + 10) particle.x = -10;
        if (particle.y < -10) particle.y = canvas.height + 10;
        if (particle.y > canvas.height + 10) particle.y = -10;

        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);

        if (particle.glow) {
          const glowGradient = ctx.createRadialGradient(
            particle.x,
            particle.y,
            0,
            particle.x,
            particle.y,
            particle.size * 3
          );
          glowGradient.addColorStop(0, particle.color + "88");
          glowGradient.addColorStop(1, particle.color + "00");
          ctx.fillStyle = glowGradient;
        } else {
          ctx.fillStyle =
            particle.color + Math.floor(particle.opacity * 255).toString(16).padStart(2, "0");
        }

        ctx.fill();

        particlesRef.current.forEach((otherParticle) => {
          const dx = particle.x - otherParticle.x;
          const dy = particle.y - otherParticle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 80) {
            ctx.beginPath();
            ctx.strokeStyle = particle.color + "33";
            ctx.lineWidth = 0.5 * (1 - distance / 80);
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(otherParticle.x, otherParticle.y);
            ctx.stroke();
          }
        });
      });
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    const handleResize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };

    window.addEventListener("resize", handleResize);
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      window.removeEventListener("resize", handleResize);
    };
  }, [isDark, themeColor]);

  // ---- Mock dashboard data ----
  const stats = [
    { label: "Total Translations", value: "1,284", icon: MessageSquare, change: "+12%", color: "primary" },
    { label: "Accuracy Rate", value: "98.7%", icon: TrendingUp, change: "+2.1%", color: "emerald" },
    { label: "Active Sessions", value: "47", icon: Users, change: "+8", color: "blue" },
    { label: "Satisfaction", value: "4.9/5", icon: Award, change: "+0.2", color: "purple" },
  ];

  const weeklyData = [
    { day: "Mon", value: 42 },
    { day: "Tue", value: 68 },
    { day: "Wed", value: 35 },
    { day: "Thu", value: 89 },
    { day: "Fri", value: 73 },
    { day: "Sat", value: 54 },
    { day: "Sun", value: 27 },
  ];

  const recentActivity = [
    { action: "Translated 'Hello' to ASL", time: "2 min ago", user: "Shahd" },
    { action: "Completed daily streak", time: "1 hour ago", user: "Shahd" },
    { action: "Shared a translation", time: "3 hours ago", user: "Shahd" },
    { action: "Updated profile preferences", time: "Yesterday", user: "Shahd" },
  ];

  // ---- Toast handler ----
  const handleSave = () => {
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000);
  };

  // ---- Animation variants (same as Profile) ----
  const fadeUp = {
    hidden: { opacity: 0, y: 40 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.8, ease: [0.22, 1, 0.36, 1] },
    },
  };

  const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.2 },
    },
  };

  return (
    <div className="relative w-full min-h-screen bg-gradient-to-br from-gray-50 via-white to-primary-50/60 dark:from-primary-bg-1 dark:via-primary-bg-2 dark:to-primary-bg-3 overflow-hidden selection:bg-primary-500 selection:text-white transition-all duration-700">
      {/* Canvas Particles */}
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />

      {/* Geometric Grid */}
      <div className="absolute inset-0 opacity-40 dark:opacity-60 pointer-events-none">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `
              linear-gradient(90deg, var(--theme-grid-color) 1px, transparent 1px),
              linear-gradient(180deg, var(--theme-grid-color) 1px, transparent 1px)
            `,
            backgroundSize: "40px 40px",
          }}
        />
      </div>

      {/* Animated Gradient Orbs */}
      <motion.div
        className="absolute top-0 left-0 w-[600px] h-[600px] bg-primary-600/20 rounded-full blur-[120px]"
        animate={{ x: [0, 200, 0], y: [0, -200, 0] }}
        transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
      />
      <motion.div
        className="absolute bottom-0 right-0 w-[600px] h-[600px] bg-primary-400/20 rounded-full blur-[120px]"
        animate={{ x: [0, -200, 0], y: [0, 200, 0] }}
        transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
      />

      {/* Toast Notification */}
      <AnimatePresence>
        {showToast && (
          <motion.div
            initial={{ x: "100%", opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: "100%", opacity: 0 }}
            transition={{ duration: 0.5, type: "spring", stiffness: 100 }}
            className="fixed top-24 right-6 p-4 rounded-xl shadow-2xl bg-gradient-to-r from-green-500/90 to-emerald-500/90 backdrop-blur-md text-white font-semibold flex items-center gap-2 z-50"
          >
            <CheckCircle size={20} />
            <span>Dashboard updated successfully!</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Container */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
        {/* Header */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          className="text-center mb-16"
        >
          {/* Premium Badge */}
          <motion.div
            whileHover={{ scale: 1.05, rotate: 1 }}
            className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-gradient-to-r from-primary-500/15 via-primary-400/10 to-primary-300/10 border border-primary-200/60 dark:border-primary-700/60 backdrop-blur-xl shadow-lg shadow-primary-500/10 relative overflow-hidden group mb-8"
          >
            <div className="relative">
              <span className="absolute animate-ping inline-flex h-3.5 w-3.5 rounded-full bg-primary-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-gradient-to-r from-primary-500 to-primary-400" />
            </div>
            <span className="text-sm font-bold bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 bg-clip-text text-transparent">
              Dashboard Overview
            </span>
            <TbSparkles className="text-primary-500 ml-1" />
            <div className="absolute -inset-1 bg-gradient-to-r from-primary-500/0 via-primary-400/10 to-primary-500/0 group-hover:via-primary-400/20 transition-all duration-500" />
          </motion.div>

          <motion.h1
            variants={fadeUp}
            className="font-extrabold text-4xl sm:text-5xl lg:text-[53px] leading-tight mb-6"
          >
            <span className="block text-gray-900 dark:text-white">Your AI Translation</span>
            <span className="block bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 bg-clip-text text-transparent">
              Dashboard
            </span>
          </motion.h1>

          <motion.p
            variants={fadeUp}
            className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto"
          >
            Monitor your translation activity, track progress, and manage your account from a single
            control center.
          </motion.p>

          {/* Decorative divider */}
          <motion.div
            variants={fadeUp}
            className="flex items-center justify-center gap-8 mt-10"
          >
            <div className="w-12 h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent rounded-full" />
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="w-6 h-6 rounded-full border-2 border-primary-400/50"
            />
            <div className="w-12 h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent rounded-full" />
          </motion.div>
        </motion.div>

        {/* Stats Grid */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={staggerContainer}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
        >
          {stats.map((stat, idx) => (
            <motion.div
              key={idx}
              variants={fadeUp}
              className="relative group"
            >
              <div className="relative p-6 rounded-2xl bg-white/70 dark:bg-white/5 backdrop-blur-xl border border-primary-200/50 dark:border-primary-500/20 shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-[1.02]">
                <div className="flex items-start justify-between">
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400 font-medium">{stat.label}</p>
                    <p className="text-3xl font-black text-gray-900 dark:text-white mt-1">{stat.value}</p>
                    <span className="inline-block mt-2 text-xs font-semibold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full">
                      {stat.change}
                    </span>
                  </div>
                  <div className={`p-3 rounded-xl bg-${stat.color}-500/10 text-${stat.color}-500`}>
                    <stat.icon size={24} />
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Two‑column layout: Chart + Activity */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Bar Chart Card */}
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeUp}
            className="lg:col-span-2"
          >
            <div className="relative p-6 rounded-2xl bg-white/70 dark:bg-white/5 backdrop-blur-xl border border-primary-200/50 dark:border-primary-500/20 shadow-xl hover:shadow-2xl transition-all duration-300">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
                  <BarChart3 className="text-primary-600" size={20} />
                  Weekly Activity
                </h3>
                <span className="text-xs text-gray-400 dark:text-gray-500">Translations per day</span>
              </div>
              <div className="flex items-end justify-between h-48 gap-2">
                {weeklyData.map((item, idx) => {
                  const max = Math.max(...weeklyData.map((d) => d.value));
                  const height = (item.value / max) * 100;
                  return (
                    <div key={idx} className="flex flex-col items-center flex-1">
                      <motion.div
                        initial={{ height: 0 }}
                        animate={{ height: `${height}%` }}
                        transition={{ duration: 0.8, delay: 0.1 * idx, ease: "easeOut" }}
                        className="w-full max-w-10 bg-gradient-to-t from-primary-400 to-primary-600 rounded-full relative group"
                        style={{ height: `${height}%`, minHeight: "8px" }}
                      >
                        <div className="absolute -top-8 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity bg-gray-900 dark:bg-gray-700 text-white text-xs px-2 py-1 rounded whitespace-nowrap">
                          {item.value}
                        </div>
                      </motion.div>
                      <span className="text-xs text-gray-500 dark:text-gray-400 mt-2">{item.day}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </motion.div>

          {/* Recent Activity */}
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeUp}
          >
            <div className="relative p-6 rounded-2xl bg-white/70 dark:bg-white/5 backdrop-blur-xl border border-primary-200/50 dark:border-primary-500/20 shadow-xl hover:shadow-2xl transition-all duration-300 h-full">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2 mb-4">
                <Clock className="text-primary-600" size={20} />
                Recent Activity
              </h3>
              <ul className="space-y-4">
                {recentActivity.map((item, idx) => (
                  <li key={idx} className="flex items-start gap-3 pb-3 border-b border-gray-200/50 dark:border-gray-700/50 last:border-0 last:pb-0">
                    <div className="w-2 h-2 mt-2 rounded-full bg-primary-400 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                        {item.action}
                      </p>
                      <div className="flex items-center gap-2 text-xs text-gray-400 dark:text-gray-500 mt-0.5">
                        <span>{item.user}</span>
                        <span>•</span>
                        <span>{item.time}</span>
                      </div>
                    </div>
                    <ArrowUpRight size={14} className="text-gray-400 dark:text-gray-500 flex-shrink-0" />
                  </li>
                ))}
              </ul>
            </div>
          </motion.div>
        </div>

        {/* Quick Actions */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          className="grid grid-cols-1 sm:grid-cols-3 gap-6"
        >
          <button className="relative overflow-hidden p-4 rounded-xl bg-gradient-to-r from-primary-500/20 to-primary-400/20 border border-primary-200/50 dark:border-primary-500/20 backdrop-blur-sm text-gray-900 dark:text-white font-medium hover:shadow-xl transition-all duration-300 group">
            <div className="flex items-center gap-3">
              <MessageSquare size={20} className="text-primary-600" />
              <span>New Translation</span>
            </div>
            <div className="absolute inset-0 bg-gradient-to-r from-primary-500/0 via-white/5 to-primary-500/0 group-hover:via-white/10 transition-all duration-500" />
          </button>
          <button className="relative overflow-hidden p-4 rounded-xl bg-gradient-to-r from-emerald-500/20 to-emerald-400/20 border border-emerald-200/50 dark:border-emerald-500/20 backdrop-blur-sm text-gray-900 dark:text-white font-medium hover:shadow-xl transition-all duration-300 group">
            <div className="flex items-center gap-3">
              <PieChart size={20} className="text-emerald-600" />
              <span>View Reports</span>
            </div>
            <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/0 via-white/5 to-emerald-500/0 group-hover:via-white/10 transition-all duration-500" />
          </button>
          <button className="relative overflow-hidden p-4 rounded-xl bg-gradient-to-r from-blue-500/20 to-blue-400/20 border border-blue-200/50 dark:border-blue-500/20 backdrop-blur-sm text-gray-900 dark:text-white font-medium hover:shadow-xl transition-all duration-300 group">
            <div className="flex items-center gap-3">
              <Settings size={20} className="text-blue-600" />
              <span>Manage Settings</span>
            </div>
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/0 via-white/5 to-blue-500/0 group-hover:via-white/10 transition-all duration-500" />
          </button>
        </motion.div>
      </div>
    </div>
  );
}