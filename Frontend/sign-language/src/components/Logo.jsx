export default function Logo({ size = 42 }) {
  return (
    <div
      className="flex items-center justify-center rounded-xl"
      style={{ width: size, height: size }}
    >
      <svg
        viewBox="0 0 100 100"
        className="w-full h-full"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Gradient */}
        <defs>
          <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#6A3093" />
            <stop offset="50%" stopColor="#A044FF" />
            <stop offset="100%" stopColor="#BF5AE0" />
          </linearGradient>
        </defs>

        {/* Background */}
        <rect
          x="5"
          y="5"
          width="90"
          height="90"
          rx="22"
          fill="url(#logoGradient)"
        />

        {/* AI dots */}
        <circle cx="30" cy="35" r="3" fill="white" />
        <circle cx="70" cy="35" r="3" fill="white" />
        <circle cx="50" cy="55" r="3" fill="white" />

        {/* Hand (minimal sign shape) */}
        <path
          d="M45 30 v28 M55 30 v28 M35 42 v20 M65 42 v20"
          stroke="white"
          strokeWidth="4"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
}
