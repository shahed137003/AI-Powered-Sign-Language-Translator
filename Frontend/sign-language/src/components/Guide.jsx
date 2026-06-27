import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useTheme } from "../context/ThemeContext";
import { motion, AnimatePresence } from "framer-motion";
import { 
  FaHandPeace, 
  FaHandSpock, 
  FaHandPointUp, 
  FaHandPointRight,
  FaSearch,
  FaGraduationCap,
  FaVideo,
  FaMicrophone,
  FaComments,
  FaBookOpen,
  FaLightbulb,
  FaInfoCircle,
  FaPlay,
  FaPause,
  FaStepForward,
  FaStepBackward
} from "react-icons/fa";
import { TbSparkles, TbHandLoveYou, TbArrowsShuffle, TbMoodSmile, TbMoodSad, TbMoodAngry, TbMoodHappy } from "react-icons/tb";
import { GiArtificialIntelligence, GiHand, GiTalk } from "react-icons/gi";
import { BsStars, BsLightningCharge, BsRobot } from "react-icons/bs";

export default function HelpGuide() {
  const navigate = useNavigate();
  const { themeColor } = useTheme();
  const gridColor = themeColor === "midnight-blue" ? "rgba(99, 102, 241, 0.1)" : "rgba(168, 85, 247, 0.1)";
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedGloss, setSelectedGloss] = useState(null);
  const [currentPage, setCurrentPage] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationFrameRef = useRef(null);
  const [isDark, setIsDark] = useState(false);

  // Detect dark mode
  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains("dark"));
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    setIsDark(document.documentElement.classList.contains("dark"));
    return () => observer.disconnect();
  }, []);
  // Particle system
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const themeColorsMap = {
      purple: ['#A855F7', '#9333EA', '#7C3AED', '#6D28D9', '#8B5CF6'],
      'midnight-blue': ['#6366F1', '#4F46E5', '#4338CA', '#3730A3', '#818CF8'],
    };
    const currentThemeColors = themeColorsMap[themeColor] || themeColorsMap['purple'];
    const colors = isDark ? currentThemeColors : currentThemeColors.slice().reverse();

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

      particlesRef.current.forEach(particle => {
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
            particle.x, particle.y, 0,
            particle.x, particle.y, particle.size * 4
          );
          glowGradient.addColorStop(0, particle.color + '99');
          glowGradient.addColorStop(1, particle.color + '00');
          ctx.fillStyle = glowGradient;
        } else {
          ctx.fillStyle = particle.color + Math.floor(particle.opacity * 255).toString(16).padStart(2, '0');
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

    window.addEventListener('resize', handleResize);
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      window.removeEventListener('resize', handleResize);
    };
  }, [isDark, themeColor]);

  // Available glosses data
  const glossesData = {
    basic: [
      "JACKET", "RIGHT", "PATIENT", "WHAT FOR", "SHOP", "CITY", "EAT", "FALL", 
      "WHAT", "HOSPITAL", "SAME", "WALK", "TEACH", "FINE", "WASH", "IMAGINE",
      "HOW", "WELCOME", "CAN", "DOCTOR", "DRINK", "MILK", "MEAN", "ARM", "RUN",
      "STUPID", "WANT", "LETTER", "DECIDE", "BORED", "BABY", "TALL", "STAND",
      "CONFUSED", "TYPE", "CALL", "EMPTY", "KNIFE", "LIVE", "THEY", "MODEL",
      "BROTHER", "MORNING", "MAYBE", "LATER", "TWO", "TAKE CARE", "TEAM", "THAT",
      "SURPRISE", "SLOW", "PAIN", "CHOCOLATE", "BECAUSE", "PERSON", "SCARED",
      "RAIN", "GONE", "FRIENDLY", "WHICH", "WHO", "FAVORITE", "FOUR", "EXPERIENCE",
      "IMPORTANT", "INVITE", "LAUGH", "HELLO", "HEARING", "UNIVERSITY", "BORROW",
      "BOY", "PLAY", "ANALYZE", "UNDERSTAND", "TRAIN", "TRAVEL", "ASIA", "SON",
      "ANSWER", "ANGRY", "STRESS", "AND", "TEACHER", "ARTICLE", "ATTENTION",
      "ASTRONAUT", "SMART", "SMALL", "SIT", "STATE", "SISTER", "SICK", "STILL",
      "TIME", "THREE", "UNCLE", "VACATION", "CHILD", "CHEAP", "CHAT", "CHALLENGE",
      "COLLEGE", "COLD", "CLEAN", "NONE", "PEN", "NICE", "ONION", "NEW", "PITY",
      "PLEASE", "PARENTS", "AWKWARD", "AUSTRALIA", "PENCIL", "PEOPLE", "AUNT",
      "QUIET", "PRACTICE", "BOOK", "BOSS", "BREAD", "BEHIND", "BIG", "BLUE",
      "ARGUE", "ARREST1", "SHY", "SELF", "REASON", "RESTAURANT", "RICH", "READ",
      "PRINT", "AMERICA", "ALWAYS", "MONDAY", "DREAM", "MAGAZINE", "MACHINE",
      "LOUD", "MISUNDERSTAND", "MORE", "BUY", "MUCH", "MOST", "BUTTER", "BUS",
      "CAMERA", "BREAKDOWN", "KEY", "JEALOUS", "LATE", "LEARN", "LIBRARY",
      "LIGHT", "KNOW", "LAST", "LEFT", "COOK", "DOOR", "COUSIN", "DIE",
      "COMFORTABLE", "EASY", "DIVORCE", "DONT KNOW", "DONT WANT", "DONT LIKE",
      "COUNTRY", "MUST", "MAN", "MAKE", "FIGHT", "FLOOR", "GET", "FEEL", "FAST",
      "HELP", "HEAVY", "HEART", "HOT", "HOTDOG", "EMOTION", "EXERCISE", "HONOR",
      "IN", "IMPOSSIBLE", "GLASSES", "HOCKEY", "HEADACHE", "HARD", "GO", "HAIR",
      "WINDOW", "WRITE", "WORLD", "GROUP", "WONDER", "GRADUATE", "WHERE", "WEEK",
      "WARM", "WARN", "WE", "WEAK", "WEDNESDAY", "5 DOLLARS", "ACCORDION",
      "A LITTLE BIT", "WISE", "WILL", "HONEST", "GRANDFATHER", "GROW UP", "YES",
      "HE", "HAVE", "HAMBURGER", "HOME", "WATER", "YESTERDAY", "CARRY", "CANADA",
      "CHURCH", "CHECK", "CHILDREN", "COMPUTER", "DRIVE", "DIRTY", "DROP",
      "LONELY", "EMAIL", "DRUNK", "MARRY", "LOOK FOR", "LIKE", "LEND", "LEAVE",
      "LISTEN", "GIRL", "GIVE", "KID", "IF", "LANGUAGE", "FRIDAY", "GAME",
      "FUNNY", "FIX", "FOR", "FINGERSPELL", "FAMILY", "EXPENSIVE", "EXCITED",
      "EQUAL", "HOUSE", "MONTH", "BUT", "LONG", "CHAIR", "MISS", "CENTER",
      "MONEY1", "SORRY", "STRONG", "TEA", "TAKE", "TABLE", "SURE", "SUMMER",
      "STUDY", "STUDENT", "WAIT", "VISIT", "THIN", "TEST", "ANNOUNCE", "TIRED",
      "UGLY", "TOMORROW", "TODAY", "ALPHABET", "UP", "THURSDAY", "SHOULD",
      "APART", "ARREST2", "POTATO", "PRETTY", "READY", "RELATIONSHIP", "SAD",
      "ROOM", "SALAD", "SHOW", "SHOWER", "BAD", "BICYCLE", "BEDROOM", "BEFORE",
      "BREAK", "BODY", "BIRTH", "PAST", "OH I SEE", "PARK", "OTHER", "NURSE",
      "OFFICE", "NUMBERS", "NO", "NOT", "COFFEE", "CLASS", "ANGEL", "ANIMAL",
      "STAY", "STUBBORN", "SLEEP", "AT", "START", "POOR", "ALL", "ALIGN",
      "THIRSTY", "AUTOMATIC", "AUTISM1", "RELAX", "ARIZONA", "ARTICULATE SIGN",
      "ART", "APRIL", "ABSOLUTELY NOTHING", "ACCESS", "ADMIRE", "PRESIDENT",
      "ANTLERS", "ANY", "AGAINST", "AUDITORIUM", "AIRPLANE", "ASSIGN", "FRIEND",
      "FULL", "FATHER", "FINISH", "ADULT", "WORRY", "ADDRESS", "HIGH", "DOWN",
      "FRANCE", "LECTURE", "COMMUNICATION", "CRAZY", "MEETING", "DAY", "ACTION",
      "AFRICA", "ADOPT", "ADVANTAGE", "ADD", "ALL GONE", "ALL OVER BODY",
      "ALL OF SUDDEN", "ATHLETE", "STRANGE", "ARCHEOLOGY", "MONEY2", "ACCIDENT",
      "ABOVE", "ACCOMPLISH", "ALASKA", "AGAIN", "AGENCY", "AUDIENCE", "AUTISM2",
      "NAME", "OUT", "PHONE", "APPEAR", "ARMY", "TELL", "MY", "ATTITUDE",
      "ARREST3", "ACT", "AFTER", "WOLF", "ADDICT", "I LOVE YOU", "MOTHER",
      "CAR", "AREA", "ACCEPT", "ACCENT", "ACCOUNTANT", "AGREEMENT", "YEAR",
      "ACTOR", "A LINE BOB", "ARRIVE", "ALCOHOL", "AGE", "ALARM", "ALL TOGETHER",
      "ALONE", "APPOINTMENT", "HAPPY", "ADVERTISE", "WORK", "INTERNET",
      "9 OCLOCK", "1 DOLLAR", "WEAR", "ALL WAY", "HUNGRY", "ABBREVIATE",
      "ACQUIRE", "WHY", "ADMIT", "STOP", "AMAZING", "ALL DAY", "YOUR", "MEET",
      "ME", "YOUNG", "YOU", "CANNOT", "COOL", "CANDY", "DONT CARE", "DONT NEED",
      "TRY", "WIFE", "ENJOY", "FREE", "SELL", "TV", "BALL"
    ],
    emotions: ["HAPPY", "SAD", "ANGRY", "SCARED", "BORED", "CONFUSED", "EXCITED", "STRESS", "JEALOUS", "SURPRISE", "LOVE", "HATE", "WORRY", "CRAZY", "SHY", "PROUD", "TIRED"],
    actions: ["RUN", "WALK", "EAT", "DRINK", "SLEEP", "WORK", "PLAY", "STUDY", "READ", "WRITE", "LISTEN", "TALK", "LAUGH", "CRY", "DANCE", "SING", "DRIVE", "FLY", "SWIM"],
    people: ["DOCTOR", "TEACHER", "STUDENT", "MOTHER", "FATHER", "BROTHER", "SISTER", "FRIEND", "BOSS", "NURSE", "PRESIDENT", "ACTOR", "ASTRONAUT", "ACCOUNTANT", "POLICE"],
    places: ["HOSPITAL", "SCHOOL", "UNIVERSITY", "PARK", "CITY", "HOME", "OFFICE", "RESTAURANT", "STORE", "AIRPORT", "HOTEL", "CHURCH", "MUSEUM", "LIBRARY"],
    time: ["TODAY", "TOMORROW", "YESTERDAY", "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY", "MORNING", "AFTERNOON", "EVENING", "NIGHT", "WEEK", "MONTH", "YEAR", "LATER", "NOW", "SOON"]
  };

  const allGlosses = [...new Set(Object.values(glossesData).flat())];
  
  const filteredGlosses = allGlosses.filter(gloss => 
    (selectedCategory === "all" || 
     (selectedCategory === "basic" && glossesData.basic.includes(gloss)) ||
     (selectedCategory === "emotions" && glossesData.emotions.includes(gloss)) ||
     (selectedCategory === "actions" && glossesData.actions.includes(gloss)) ||
     (selectedCategory === "people" && glossesData.people.includes(gloss)) ||
     (selectedCategory === "places" && glossesData.places.includes(gloss)) ||
     (selectedCategory === "time" && glossesData.time.includes(gloss))) &&
    gloss.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const categories = [
    { id: "all", name: "All Glosses", icon: <FaBookOpen />, count: allGlosses.length },
    { id: "basic", name: "Basic", icon: <FaHandPeace />, count: glossesData.basic.length },
    { id: "emotions", name: "Emotions", icon: <TbMoodSmile />, count: glossesData.emotions.length },
    { id: "actions", name: "Actions", icon: <FaHandSpock />, count: glossesData.actions.length },
    { id: "people", name: "People", icon: <FaComments />, count: glossesData.people.length },
    { id: "places", name: "Places", icon: <FaVideo />, count: glossesData.places.length },
    { id: "time", name: "Time", icon: <FaGraduationCap />, count: glossesData.time.length }
  ];

  const steps = [
    {
      title: "Show Your Hand",
      icon: <GiHand className="text-3xl" />,
      description: "Position your hand clearly in front of your camera. Make sure your hand is well-lit and visible.",
      tip: "Good lighting and clear background improve recognition accuracy!"
    },
    {
      title: "Perform the Sign",
      icon: <TbHandLoveYou className="text-3xl" />,
      description: "Perform the sign language gesture for the word you want to translate.",
      tip: "Start with basic signs like 'Hello', 'Thank you', or 'How are you?'"
    },
    {
      title: "AI Recognition",
      icon: <GiArtificialIntelligence className="text-3xl" />,
      description: "Our AI analyzes your hand movements and matches them with our extensive gloss database.",
      tip: "The AI recognizes over 800+ different signs with 99% accuracy!"
    },
    {
      title: "Instant Translation",
      icon: <BsLightningCharge className="text-3xl" />,
      description: "The translated text appears on screen and the sign disappears as the translation completes.",
      tip: "You can save, share, or practice the translation immediately!"
    }
  ];

  const fadeUp = {
    hidden: { opacity: 0, y: 40 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.8, ease: [0.22, 1, 0.36, 1] }
    }
  };

  const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.2 }
    }
  };

  const scaleIn = {
    hidden: { opacity: 0, scale: 0.9 },
    visible: { 
      opacity: 1, 
      scale: 1,
      transition: { duration: 0.6, ease: "backOut" }
    }
  };

  return (
      <div id="guide" className="relative w-full min-h-screen bg-gradient-to-br from-gray-50 via-white to-primary-50/60 dark:from-primary-bg-1 dark:via-primary-bg-2 dark:to-primary-bg-3 overflow-hidden selection:bg-primary-500 selection:text-white transition-all duration-700">
      
      {/* Canvas Particles */}
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />

      {/* Premium Geometric Grid */}
      <div className="absolute inset-0 opacity-40 dark:opacity-60 pointer-events-none">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(90deg, ${gridColor} 1px, transparent 1px),
            linear-gradient(180deg, ${gridColor} 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px'
        }} />
      </div>

      {/* Animated gradient orbs */}
      <motion.div
        className="absolute top-0 left-0 w-[600px] h-[600px] bg-primary-600/20 rounded-full blur-[120px]"
        animate={{ x: [0, 200, 0], y: [0, -200, 0] }}
        transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
      />
      {/* <motion.div
        className="absolute bottom-0 right-0 w-[600px] h-[600px] bg-primary-400/20 rounded-full blur-[120px]"
        animate={{ x: [0, -200, 0], y: [0, 200, 0] }}
        transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
      /> */}

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
            Your complete guide to real-time sign language translation. Start signing and watch as your gestures come to life!
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
              Explore our extensive library of {allGlosses.length}+ sign language glosses
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

          {/* Category Filters */}
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
                <span className={`text-xs px-1.5 py-0.5 rounded-full ${
                  selectedCategory === cat.id
                    ? "bg-white/20 text-white"
                    : "bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400"
                }`}>
                  {cat.count}
                </span>
              </motion.button>
            ))}
          </div>

          {/* Glosses Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {filteredGlosses.map((gloss, index) => (
              <motion.button
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.01 }}
                whileHover={{ scale: 1.05, y: -2 }}
                onClick={() => setSelectedGloss(gloss)}
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
              <p className="text-gray-500 dark:text-gray-400">No signs found matching your search.</p>
            </div>
          )}

          {filteredGlosses.length > 60 && (
            <div className="text-center mt-8">
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Showing All {filteredGlosses.length} signs. Use search to find specific signs.
              </p>
            </div>
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
          <motion.div variants={scaleIn} className="p-6 rounded-2xl bg-gradient-to-r from-primary-50/80 to-primary-100/50 dark:from-primary-900/20 dark:to-primary-800/10 border border-primary-200 dark:border-primary-700/30">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-full bg-primary-500/20 flex items-center justify-center">
                <FaLightbulb className="text-primary-600 text-xl" />
              </div>
              <h3 className="font-bold text-gray-900 dark:text-white">Pro Tip #1</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Make sure your hand is fully visible and well-lit. The AI recognizes signs best when there's good contrast.
            </p>
          </motion.div>

          <motion.div variants={scaleIn} className="p-6 rounded-2xl bg-gradient-to-r from-primary-50/80 to-primary-100/50 dark:from-primary-900/20 dark:to-primary-800/10 border border-primary-200 dark:border-primary-700/30">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-full bg-primary-500/20 flex items-center justify-center">
                <BsRobot className="text-primary-600 text-xl" />
              </div>
              <h3 className="font-bold text-gray-900 dark:text-white">Pro Tip #2</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Practice with basic signs first. The AI learns from your signing style and improves over time.
            </p>
          </motion.div>

          <motion.div variants={scaleIn} className="p-6 rounded-2xl bg-gradient-to-r from-primary-50/80 to-primary-100/50 dark:from-primary-900/20 dark:to-primary-800/10 border border-primary-200 dark:border-primary-700/30">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-full bg-primary-500/20 flex items-center justify-center">
                <TbArrowsShuffle className="text-primary-600 text-xl" />
              </div>
              <h3 className="font-bold text-gray-900 dark:text-white">Pro Tip #3</h3>
            </div>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Hold each sign for a moment. The translation starts when it recognizes your hand and ends when you lower it.
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



    </div>
  );
}
