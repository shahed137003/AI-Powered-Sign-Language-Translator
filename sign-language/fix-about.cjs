const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/About.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// 1. Import useTheme
if (!content.includes('useTheme')) {
  content = content.replace(
    /import \{ motion \} from "framer-motion";/,
    'import { motion } from "framer-motion";\nimport { useTheme } from "../context/ThemeContext";'
  );
}

// 2. Add useTheme hook inside component
content = content.replace(
  /const \[hoveredCard, setHoveredCard\] = useState\(null\);/,
  'const [hoveredCard, setHoveredCard] = useState(null);\n  const { themeColor } = useTheme();'
);

// 3. Fix Particle colors based on themeColor
content = content.replace(
  /const colors = isDark[\s\S]*?\];/g,
  `const colors = themeColor === 'midnight-blue'
      ? (isDark ? ['#3b82f6', '#2563eb', '#1d4ed8', '#1e40af', '#60a5fa'] : ['#60a5fa', '#3b82f6', '#2563eb', '#1d4ed8', '#93c5fd'])
      : (isDark ? ['#A855F7', '#9333EA', '#7C3AED', '#6D28D9', '#8B5CF6'] : ['#8B5CF6', '#7C3AED', '#6D28D9', '#9333EA', '#A855F7']);`
);

// 4. Remove connecting lines in particles
content = content.replace(
  /particlesRef\.current\.forEach\(otherParticle => \{[\s\S]*?\}\);/g,
  '// Connected lines removed for cleaner look'
);

// 5. Replace purple with primary
content = content.replace(/purple/g, 'primary');
content = content.replace(/selection:bg-primary-500/g, 'selection:bg-primary-500');

// Fix theme context dependency for useEffect
content = content.replace(/}, \[isDark\]\);/, '}, [isDark, themeColor]);');

fs.writeFileSync(filePath, content, 'utf8');
console.log('Fixed About.jsx');
