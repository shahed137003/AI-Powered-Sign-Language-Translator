import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useTheme } from "../context/ThemeContext";
import { motion, AnimatePresence } from "framer-motion";
import {
  FaHandPeace,
  FaHandSpock,
  FaSearch,
  FaGraduationCap,
  FaVideo,
  FaComments,
  FaBookOpen,
  FaLightbulb,
} from "react-icons/fa";
import {
  TbSparkles,
  TbHandLoveYou,
  TbArrowsShuffle,
  TbMoodSmile,
} from "react-icons/tb";
import {
  GiArtificialIntelligence,
  GiHand,
  GiTalk,
} from "react-icons/gi";
import { BsStars, BsLightningCharge, BsRobot } from "react-icons/bs";

// ─── Hardcoded category lists (used for classification only) ──────────
const classificationData = {
  emotions: ["HAPPY", "SAD", "ANGRY", "SCARED", "BORED", "CONFUSED", "EXCITED", "STRESS", "JEALOUS", "SURPRISE", "LOVE", "HATE", "WORRY", "CRAZY", "SHY", "PROUD", "TIRED"],
  actions: ["RUN", "WALK", "EAT", "DRINK", "SLEEP", "WORK", "PLAY", "STUDY", "READ", "WRITE", "LISTEN", "TALK", "LAUGH", "CRY", "DANCE", "SING", "DRIVE", "FLY", "SWIM"],
  people: ["DOCTOR", "TEACHER", "STUDENT", "MOTHER", "FATHER", "BROTHER", "SISTER", "FRIEND", "BOSS", "NURSE", "PRESIDENT", "ACTOR", "ASTRONAUT", "ACCOUNTANT", "POLICE"],
  places: ["HOSPITAL", "SCHOOL", "UNIVERSITY", "PARK", "CITY", "HOME", "OFFICE", "RESTAURANT", "STORE", "AIRPORT", "HOTEL", "CHURCH", "MUSEUM", "LIBRARY"],
  time: ["TODAY", "TOMORROW", "YESTERDAY", "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY", "MORNING", "AFTERNOON", "EVENING", "NIGHT", "WEEK", "MONTH", "YEAR", "LATER", "NOW", "SOON"],
};

export default function HelpGuide() {
  const navigate = useNavigate();
  const { themeColor } = useTheme();
  const gridColor =
    themeColor === "midnight-blue"
      ? "rgba(99, 102, 241, 0.1)"
      : "var(--theme-grid-color)";
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedGloss, setSelectedGloss] = useState(null);
  const [videoError, setVideoError] = useState(false);
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationFrameRef = useRef(null);
  const [isDark, setIsDark] = useState(false);

  // ─── Load signs from CSV ──────────────────────────────────────────────
  const [allSigns, setAllSigns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  // Map each sign to its category
  const [signCategoryMap, setSignCategoryMap] = useState({});

  useEffect(() => {
    const fetchSigns = async () => {
      try {
        const response = await fetch("/asl_classes_export.csv");
        if (!response.ok) throw new Error("Failed to load CSV");
        const text = await response.text();
        const cleanText = text.replace(/^\uFEFF/, "");
        const lines = cleanText.split(/\r?\n/).filter((line) => line.trim() !== "");
        if (lines.length === 0) throw new Error("CSV is empty");

        const firstLine = lines[0];
        const delimiter = firstLine.includes("\t") ? "\t" : ",";
        const headers = firstLine.split(delimiter).map((h) => h.trim());

        const classIndex = headers.findIndex(
          (h) => h.toLowerCase() === "class_name"
        );
        if (classIndex === -1) {
          console.error("Headers found:", headers);
          throw new Error("Missing class_name column");
        }

        const signs = lines.slice(1).map((line) => {
          const cols = line.split(delimiter);
          return cols[classIndex]?.trim() || "";
        }).filter((s) => s.length > 0);

        setAllSigns(signs);

        // ─── Classify each sign ──────────────────────────────────────
        const map = {};
        const categoryKeys = ["emotions", "actions", "people", "places", "time"];
        signs.forEach((sign) => {
          let assigned = "basic"; // default
          for (const cat of categoryKeys) {
            if (classificationData[cat].includes(sign)) {
              assigned = cat;
              break;
            }
          }
          map[sign] = assigned;
        });
        setSignCategoryMap(map);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };
    fetchSigns();
  }, []);

  // ─── Dark mode detection ─────────────────────────────────────────────
  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains("dark"));
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
    setIsDark(document.documentElement.classList.contains("dark"));
    return () => observer.disconnect();
  }, []);

  // ─── Particle system (unchanged) ────────────────────────────────────
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
    const currentThemeColors =
      themeColorsMap[themeColor] || themeColorsMap["purple"];
    const colors = isDark
      ? currentThemeColors
      : currentThemeColors.slice().reverse();

    particlesRef.current = Array.from({ length: 120 }).map(() => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      size: Math.random() * 4 + 1,
      speedX: Math.random() * 0.5 - 0.25,
      speedY: Math.random() * 0.5 - 0.25,
      color: colors[Math.floor(Math.random() * colors.length)],
      opacity: Math.random() * 0.6 + 0.2,
      glow: Math.random() > 0.7,
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
            particle.size * 4
          );
          glowGradient.addColorStop(0, particle.color + "99");
          glowGradient.addColorStop(1, particle.color + "00");
          ctx.fillStyle = glowGradient;
        } else {
          ctx.fillStyle =
            particle.color +
            Math.floor(particle.opacity * 255)
              .toString(16)
              .padStart(2, "0");
        }

        ctx.fill();
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
      if (animationFrameRef.current)
        cancelAnimationFrame(animationFrameRef.current);
      window.removeEventListener("resize", handleResize);
    };
  }, [isDark, themeColor]);

  // ─── Build categories with real counts ──────────────────────────────
  const getCategoryCount = (catId) => {
    if (catId === "all") return allSigns.length;
    return Object.values(signCategoryMap).filter((c) => c === catId).length;
  };

  const categories = [
    { id: "all", name: "All Glosses", icon: <FaBookOpen />, count: getCategoryCount("all") },
    { id: "basic", name: "Basic", icon: <FaHandPeace />, count: getCategoryCount("basic") },
    { id: "emotions", name: "Emotions", icon: <TbMoodSmile />, count: getCategoryCount("emotions") },
    { id: "actions", name: "Actions", icon: <FaHandSpock />, count: getCategoryCount("actions") },
    { id: "people", name: "People", icon: <FaComments />, count: getCategoryCount("people") },
    { id: "places", name: "Places", icon: <FaVideo />, count: getCategoryCount("places") },
    { id: "time", name: "Time", icon: <FaGraduationCap />, count: getCategoryCount("time") },
  ];

  // ─── Filtered glosses (category + search) ──────────────────────────
  const filteredGlosses = allSigns.filter((gloss) => {
    const matchesCategory =
      selectedCategory === "all" || signCategoryMap[gloss] === selectedCategory;
    const matchesSearch = gloss.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  // ─── Steps & animations (unchanged) ─────────────────────────────────
  const steps = [
    {
      title: "Show Your Hand",
      icon: <GiHand className="text-3xl" />,
      description:
        "Position your hand clearly in front of your camera. Make sure your hand is well‑lit and visible.",
      tip: "Good lighting and clear background improve recognition accuracy!",
    },
    {
      title: "Perform the Sign",
      icon: <TbHandLoveYou className="text-3xl" />,
      description:
        "Perform the sign language gesture for the word you want to translate.",
      tip: "Start with basic signs like 'Hello', 'Thank you', or 'How are you?'",
    },
    {
      title: "AI Recognition",
      icon: <GiArtificialIntelligence className="text-3xl" />,
      description:
        "Our AI analyzes your hand movements and matches them with our extensive gloss database.",
      tip: "The AI recognizes over 800+ different signs with 99% accuracy!",
    },
    {
      title: "Instant Translation",
      icon: <BsLightningCharge className="text-3xl" />,
      description:
        "The translated text appears on screen and the sign disappears as the translation completes.",
      tip: "You can save, share, or practice the translation immediately!",
    },
  ];

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

  const scaleIn = {
    hidden: { opacity: 0, scale: 0.9 },
    visible: {
      opacity: 1,
      scale: 1,
      transition: { duration: 0.6, ease: "backOut" },
    },
  };

  // ─── Render ──────────────────────────────────────────────────────────
  return (
    <div
      id="guide"
      className="relative w-full min-h-screen bg-gradient-to-br from-gray-50 via-white to-primary-50/60 dark:from-primary-bg-1 dark:via-primary-bg-2 dark:to-primary-bg-3 overflow-hidden selection:bg-primary-500 selection:text-white transition-all duration-700"
    >
      {/* Canvas Particles */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />

      {/* Premium Geometric Grid */}
      <div className="absolute inset-0 opacity-40 dark:opacity-60 pointer-events-none">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `
              linear-gradient(90deg, ${gridColor} 1px, transparent 1px),
              linear-gradient(180deg, ${gridColor} 1px, transparent 1px)
            `,
            backgroundSize: "40px 40px",
          }}
        />
      </div>

      {/* Animated gradient orbs */}
      <motion.div
        className="absolute top-0 left-0 w-[600px] h-[600px] bg-primary-600/20 rounded-full blur-[120px]"
        animate={{ x: [0, 200, 0], y: [0, -200, 0] }}
        transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
      />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
        {/* Header */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          className="text-center mb-12"
        >
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-gradient-to-r from-primary-500/15 via-primary-400/10 to-primary-300/10 border border-primary-200/60 dark:border-primary-700/60 backdrop-blur-xl shadow-lg shadow-primary-500/10 relative overflow-hidden group mb-8"
          >
            <div className="relative">
              <span className="absolute animate-ping inline-flex h-3.5 w-3.5 rounded-full bg-primary-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-gradient-to-r from-primary-500 to-primary-400" />
            </div>
            <span className="text-sm font-bold bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 bg-clip-text text-transparent">
              USER GUIDE
            </span>
            <TbSparkles className="text-primary-500 ml-1" />
          </motion.div>

          <motion.h1 className="font-extrabold text-4xl sm:text-5xl lg:text-[53px] leading-tight mb-6">
            <span className="block text-gray-900 dark:text-white">
              How to Use
            </span>
            <span className="block bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 bg-clip-text text-transparent">
              LinguaSign
            </span>
          </motion.h1>

          <motion.p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Your complete guide to real‑time sign language translation. Start
            signing and watch as your gestures come to life!
          </motion.p>

          {/* Decorative Elements */}
          <motion.div className="flex items-center justify-center gap-8 mt-10">
            <div className="w-12 h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent rounded-full" />
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="w-6 h-6 rounded-full border-2 border-primary-400/50"
            />
            <div className="w-12 h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent rounded-full" />
          </motion.div>
        </motion.div>

        {/* How It Works Steps */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={staggerContainer}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-center text-gray-900 dark:text-white mb-10 flex items-center justify-center gap-3">
            <BsStars className="text-primary-500" />
            How It Works
            <BsLightningCharge className="text-primary-500" />
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {steps.map((step, index) => (
              <motion.div
                key={index}
                variants={scaleIn}
                whileHover={{ y: -8, scale: 1.02 }}
                className="relative group"
              >
                <div className="absolute -inset-0.5 bg-gradient-to-r from-primary-500/30 to-primary-600/30 rounded-2xl blur opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                <div className="relative p-6 rounded-2xl bg-white/70 dark:bg-white/5 backdrop-blur-xl border border-primary-200/50 dark:border-primary-500/20 shadow-lg text-center">
                  <div className="absolute -top-4 -left-4 w-8 h-8 rounded-full bg-gradient-to-r from-primary-500 to-primary-400 text-white flex items-center justify-center font-bold text-sm shadow-lg">
                    {index + 1}
                  </div>
                  <div className="text-5xl mb-4 text-primary-500 flex justify-center">
                    {step.icon}
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                    {step.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 text-sm mb-3">
                    {step.description}
                  </p>
                  <div className="p-3 rounded-xl bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-700/30">
                    <p className="text-xs text-primary-600 dark:text-primary-300 flex items-center gap-2">
                      <FaLightbulb className="text-xs" />
                      {step.tip}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Available Glosses Section */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={fadeUp}
          className="mb-20"
        >
          <div className="text-center mb-10">
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4 flex items-center justify-center gap-3">
              <FaBookOpen className="text-primary-500" />
              Available Signs
              <GiTalk className="text-primary-500" />
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              {loading
                ? "Loading signs..."
                : error
                ? "Error loading signs"
                : `Explore our extensive library of ${allSigns.length} sign language glosses`}
            </p>
          </div>

          {/* Search Bar */}
          <div className="max-w-md mx-auto mb-8">
            <div className="relative">
              <FaSearch className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search for a sign..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-12 pr-4 py-3 rounded-xl bg-white/70 dark:bg-white/5 border border-primary-200/50 dark:border-primary-500/20 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500/50 transition-all"
              />
            </div>
          </div>

          {/* Category Filters – now with real counts and filtering */}
          <div className="flex flex-wrap justify-center gap-3 mb-10">
            {categories.map((cat) => (
              <motion.button
                key={cat.id}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setSelectedCategory(cat.id)}
                className={`px-4 py-2 rounded-full font-medium transition-all duration-300 flex items-center gap-2 ${
                  selectedCategory === cat.id
                    ? "bg-gradient-to-r from-primary-600 to-primary-500 text-white shadow-lg shadow-primary-500/30"
                    : "bg-white/50 dark:bg-white/5 text-gray-700 dark:text-gray-300 border border-primary-200/50 dark:border-primary-500/20 hover:bg-primary-50 dark:hover:bg-primary-900/20"
                }`}
              >
                {cat.icon}
                {cat.name}
                <span
                  className={`text-xs px-1.5 py-0.5 rounded-full ${
                    selectedCategory === cat.id
                      ? "bg-white/20 text-white"
                      : "bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400"
                  }`}
                >
                  {cat.count}
                </span>
              </motion.button>
            ))}
          </div>

          {/* Glosses Grid */}
          {loading ? (
            <div className="text-center py-12">
              <p className="text-gray-500 dark:text-gray-400">Loading signs…</p>
            </div>
          ) : error ? (
            <div className="text-center py-12">
              <p className="text-red-500 dark:text-red-400">
                Failed to load signs: {error}
              </p>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
                {filteredGlosses.slice(0, 60).map((gloss, index) => (
                  <motion.button
                    key={index}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.01 }}
                    whileHover={{ scale: 1.05, y: -2 }}
                    onClick={() => {
                        setVideoError(false);
                        setSelectedGloss(gloss);
                    }}
                    className="p-3 rounded-xl bg-white/60 dark:bg-white/5 border border-primary-200/50 dark:border-primary-500/20 hover:border-primary-400 dark:hover:border-primary-400 transition-all duration-300 group"
                  >
                    <div className="flex flex-col items-center text-center">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-500/20 to-primary-400/20 flex items-center justify-center mb-2 group-hover:scale-110 transition-transform">
                        <GiHand className="text-primary-500 text-xl" />
                      </div>
                      <span className="text-sm font-medium text-gray-800 dark:text-gray-200">
                        {gloss}
                      </span>
                    </div>
                  </motion.button>
                ))}
              </div>

              {filteredGlosses.length === 0 && (
                <div className="text-center py-12">
                  <p className="text-gray-500 dark:text-gray-400">
                    No signs found matching your search and category.
                  </p>
                </div>
              )}

              {filteredGlosses.length > 60 && (
                <div className="text-center mt-8">
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Showing 60 of {filteredGlosses.length} signs. Use search to find specific signs.
                  </p>
                </div>
              )}
            </>
          )}
        </motion.div>

        {/* Tips Section */}
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={staggerContainer}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12"
        >
          <motion.div
            variants={scaleIn}
            className="p-6 rounded-2xl bg-gradient-to-r from-primary-50/80 to-primary-100/50 dark:from-primary-900/20 dark:to-primary-800/10 border border-primary-200 dark:border-primary-700/30"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-full bg-primary-500/20 flex items-center justify-center">
                <FaLightbulb className="text-primary-600 text-xl" />
              </div>
              <h3 className="font-bold text-gray-900 dark:text-white">
                Pro Tip #1
              </h3>
            </div>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Make sure your hand is fully visible and well‑lit. The AI
              recognises signs best when there's good contrast.
            </p>
          </motion.div>

          <motion.div
            variants={scaleIn}
            className="p-6 rounded-2xl bg-gradient-to-r from-primary-50/80 to-primary-100/50 dark:from-primary-900/20 dark:to-primary-800/10 border border-primary-200 dark:border-primary-700/30"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-full bg-primary-500/20 flex items-center justify-center">
                <BsRobot className="text-primary-600 text-xl" />
              </div>
              <h3 className="font-bold text-gray-900 dark:text-white">
                Pro Tip #2
              </h3>
            </div>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Practice with basic signs first. The AI learns from your signing
              style and improves over time.
            </p>
          </motion.div>

          <motion.div
            variants={scaleIn}
            className="p-6 rounded-2xl bg-gradient-to-r from-primary-50/80 to-primary-100/50 dark:from-primary-900/20 dark:to-primary-800/10 border border-primary-200 dark:border-primary-700/30"
          >
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-full bg-primary-500/20 flex items-center justify-center">
                <TbArrowsShuffle className="text-primary-600 text-xl" />
              </div>
              <h3 className="font-bold text-gray-900 dark:text-white">
                Pro Tip #3
              </h3>
            </div>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Hold each sign for a moment. The translation starts when it
              recognises your hand and ends when you lower it.
            </p>
          </motion.div>
        </motion.div>

        {/* Start Translating CTA */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center"
        >
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate("/translate")}
            className="relative overflow-hidden px-10 py-4 rounded-full bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400 text-white font-bold text-lg shadow-lg shadow-primary-500/30 hover:shadow-primary-500/50 transition-all duration-300 inline-flex items-center gap-3 group"
          >
            <span className="relative z-10">Start Translating Now</span>
            <TbHandLoveYou className="text-2xl group-hover:scale-110 transition-transform" />
          </motion.button>
        </motion.div>
      </div>
        <AnimatePresence>
    {selectedGloss && (
      <motion.div
        className="fixed inset-0 bg-black/70 flex items-center justify-center z-50"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <motion.div
          initial={{ scale: 0.8 }}
          animate={{ scale: 1 }}
          exit={{ scale: 0.8 }}
          className="bg-white dark:bg-gray-900 rounded-2xl p-6 w-[650px] max-w-[95%] shadow-2xl"
        >
          <h2 className="text-3xl font-bold text-center mb-5 dark:text-white">
            {selectedGloss}
          </h2>

          {!videoError ? (
            <video
              controls
              autoPlay
              className="w-full rounded-xl"
              onError={() => setVideoError(true)}
            >
              <source
                src={`/videos/${selectedGloss
                  .toLowerCase()
                  .replace(/\s+/g, "_")}.mp4`}
                type="video/mp4"
              />
            </video>
          ) : (
            <div className="text-center py-16">
              <p className="text-red-500 text-lg">
                No video available for "{selectedGloss}"
              </p>
            </div>
          )}

          <button
            onClick={() => setSelectedGloss(null)}
            className="mt-6 w-full py-3 rounded-xl bg-primary-600 text-white hover:bg-primary-700 transition"
          >
            Close
          </button>
        </motion.div>
      </motion.div>
    )}
  </AnimatePresence>
    </div>
  );
}