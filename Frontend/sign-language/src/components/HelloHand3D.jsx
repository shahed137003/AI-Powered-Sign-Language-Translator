import { motion } from "framer-motion";

export default function HelloHand3D() {
  return (
    <div className="relative w-[180px] h-[180px] flex items-center justify-center">
      {/* Glow */}
      <div className="absolute inset-0 rounded-full blur-2xl bg-gradient-to-br from-[#A044FF55] to-[#BF5AE055]" />

      {/* 3D Illusion Hand */}
      <motion.svg
        width="150"
        height="150"
        viewBox="0 0 300 300"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        animate={{ y: [0, -10, 0] }}
        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        className="drop-shadow-2xl"
      >
        {/* Hand Outline (3D illusion gradient) */}
        <motion.path
          d="M150 40C162 40 170 55 170 70V130H180V65C180 50 195 45 205 55C212 62 215 70 215 90V150H225V110C225 95 240 90 250 100C258 108 260 120 260 145V200C260 240 245 260 220 275C200 288 170 290 150 290C130 290 100 288 80 275C55 260 40 240 40 200V140C40 120 45 110 55 105C70 96 85 105 85 125V170H95V80C95 60 110 40 130 40H150Z"
          stroke="url(#grad)"
          strokeWidth="12"
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: 1.8, ease: "easeOut" }}
        />

        {/* Gradient Definition */}
        <defs>
          <linearGradient id="grad" x1="0" y1="0" x2="300" y2="300">
            <stop offset="0%" stopColor="#6A3093" />
            <stop offset="50%" stopColor="#A044FF" />
            <stop offset="100%" stopColor="#BF5AE0" />
          </linearGradient>
        </defs>
      </motion.svg>
    </div>
  );
}