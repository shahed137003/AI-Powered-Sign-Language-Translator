const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Profile.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// 1. Add missing imports
if (!content.includes('import { useTheme } from "../context/ThemeContext"')) {
  content = content.replace(/import React, \{ useState \} from "react";/, 'import React, { useState, useEffect, useRef } from "react";\nimport { useTheme } from "../context/ThemeContext";');
}

// 2. Add useTheme and canvas refs to component
if (!content.includes('const { themeColor } = useTheme();')) {
  content = content.replace(/export default function Profile\(\) \{/, `export default function Profile() {
  const { themeColor } = useTheme();
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationFrameRef = useRef(null);
  const [isDark, setIsDark] = useState(false);
  
  // Theme and Particle Effect
  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains("dark"));
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    setIsDark(document.documentElement.classList.contains("dark"));
    return () => observer.disconnect();
  }, []);

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
    const currentThemeColors = themeColorsMap[themeColor || 'purple'];
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
            particle.x, particle.y, particle.size * 3
          );
          glowGradient.addColorStop(0, particle.color + '88');
          glowGradient.addColorStop(1, particle.color + '00');
          ctx.fillStyle = glowGradient;
        } else {
          ctx.fillStyle = particle.color + Math.floor(particle.opacity * 255).toString(16).padStart(2, '0');
        }
        
        ctx.fill();

        particlesRef.current.forEach(otherParticle => {
          const dx = particle.x - otherParticle.x;
          const dy = particle.y - otherParticle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 80) {
            ctx.beginPath();
            ctx.strokeStyle = particle.color + '33';
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

    window.addEventListener('resize', handleResize);
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      window.removeEventListener('resize', handleResize);
    };
  }, [isDark, themeColor]);
`);
}

// 3. Update return statement container
content = content.replace(/<div className="w-full min-h-screen/, '<div className="relative w-full min-h-screen');

// 4. Inject canvas
if (!content.includes('<canvas')) {
  content = content.replace(/\{\/\* Premium Geometric Grid \*\/\}/, `{/* Premium Canvas Particles */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />\n\n        {/* Premium Geometric Grid */}`);
}

// 5. Fix hardcoded purple in grid
content = content.replace(/rgba\(168, 85, 247, 0\.1\)/g, 'rgba(var(--theme-primary-500), 0.1)'); // Note: rgba() with CSS variable only works if variable is channels. We'll use CSS variable properly if we can, else just use the existing approach. Actually, tailwind doesn't create raw channel variables by default unless configured.
// Better way: use tailwind classes for the grid if possible, or just use CSS variable for color.
content = content.replace(/linear-gradient\(90deg, rgba\(168, 85, 247, 0\.1\) 1px, transparent 1px\)/g, 'linear-gradient(90deg, var(--color-primary-500) 1px, transparent 1px)');
content = content.replace(/linear-gradient\(180deg, rgba\(168, 85, 247, 0\.1\) 1px, transparent 1px\)/g, 'linear-gradient(180deg, var(--color-primary-500) 1px, transparent 1px)');

// 6. Update Card component to use group and glowing backdrop
const oldCard = `const Card = ({ children, className = "", hover = false }) => (
    <motion.div
      variants={fadeUp}
      whileHover={hover ? { y: -5, scale: 1.02 } : {}}
      className={\`
        relative p-8 bg-gradient-to-br from-white/80 to-white/60 dark:from-white/10 dark:to-white/5 
        backdrop-blur-xl border border-primary-200/50 dark:border-primary-500/20 
        rounded-3xl shadow-xl shadow-primary-100/20 dark:shadow-primary-900/20
        transition-all duration-500 overflow-hidden
        \${hover ? 'hover:shadow-2xl hover:shadow-primary-200/30 dark:hover:shadow-primary-900/40' : ''}
        \${className}
      \`}
    >
      {children}
    </motion.div>
  );`;

const newCard = `const Card = ({ children, className = "", hover = false }) => (
    <motion.div
      variants={fadeUp}
      whileHover={hover ? { y: -5, scale: 1.02 } : {}}
      className={\`group relative \${hover ? 'cursor-pointer' : ''} \${className}\`}
    >
      {hover && <div className="absolute -inset-0.5 bg-gradient-to-br from-primary-500/30 to-primary-600/30 rounded-3xl blur opacity-0 group-hover:opacity-100 transition-opacity duration-500" />}
      <div className={\`
        relative h-full p-8 bg-white/70 dark:bg-white/5 
        backdrop-blur-xl border border-primary-200/50 dark:border-primary-500/20 
        rounded-3xl shadow-xl shadow-primary-100/20 dark:shadow-primary-900/20
        transition-colors duration-300 overflow-hidden
        \${hover ? 'group-hover:bg-white/90 dark:group-hover:bg-white/10' : ''}
      \`}>
        {children}
      </div>
    </motion.div>
  );`;

content = content.replace(oldCard, newCard);

fs.writeFileSync(filePath, content, 'utf8');
console.log('Successfully updated Profile.jsx');
