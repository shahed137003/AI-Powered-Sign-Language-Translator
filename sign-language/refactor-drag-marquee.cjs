const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Features.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// 1. Add refs and state
if (!content.includes('const carouselRef = useRef(null);')) {
  content = content.replace(
    /const \[hoveredCard, setHoveredCard\] = useState\(null\);/,
    `const carouselRef = useRef(null);\n  const [isCarouselHovered, setIsCarouselHovered] = useState(false);\n  const [isDragging, setIsDragging] = useState(false);\n  let startX = 0;\n  let scrollLeftState = 0;\n\n  const [hoveredCard, setHoveredCard] = useState(null);`
  );
}

// 2. Add useEffect for continuous scroll
const scrollEffect = `
  useEffect(() => {
    if (!carouselRef.current || isCarouselHovered || isDragging) return;
    
    let animationId;
    const scroll = () => {
      if (carouselRef.current) {
        carouselRef.current.scrollLeft += 0.8;
        if (carouselRef.current.scrollLeft >= carouselRef.current.scrollWidth / 2) {
          carouselRef.current.scrollLeft = 0;
        }
      }
      animationId = requestAnimationFrame(scroll);
    };
    animationId = requestAnimationFrame(scroll);
    return () => cancelAnimationFrame(animationId);
  }, [isCarouselHovered, isDragging]);
`;
if (!content.includes('carouselRef.current.scrollLeft +=')) {
  content = content.replace(
    /useEffect\(\(\) => \{\n    const handleThemeChange/,
    scrollEffect + '\n  useEffect(() => {\n    const handleThemeChange'
  );
}

// 3. Replace the <motion.div animate={{x}}> with a native draggable container
const motionDivRegex = /<motion\.div\s+className="flex gap-8 w-max px-4"\s+animate=\{\{ x: \["0%", "-50%"\] \}\}\s+transition=\{\{\s+duration: 120,\s+repeat: Infinity,\s+ease: "linear",\s+\}\}\s+>/;

const nativeDiv = `
          <div 
            ref={carouselRef}
            className="flex gap-8 w-max px-4 overflow-x-auto hide-scrollbar cursor-grab active:cursor-grabbing pb-8"
            onMouseEnter={() => setIsCarouselHovered(true)} 
            onMouseLeave={() => { setIsCarouselHovered(false); setIsDragging(false); }}
            onTouchStart={() => setIsCarouselHovered(true)} 
            onTouchEnd={() => { setIsCarouselHovered(false); setIsDragging(false); }}
            onMouseDown={(e) => {
              setIsDragging(true);
              startX = e.pageX - carouselRef.current.offsetLeft;
              scrollLeftState = carouselRef.current.scrollLeft;
            }}
            onMouseUp={() => setIsDragging(false)}
            onMouseMove={(e) => {
              if (!isDragging) return;
              e.preventDefault();
              const x = e.pageX - carouselRef.current.offsetLeft;
              const walk = (x - startX) * 2;
              carouselRef.current.scrollLeft = scrollLeftState - walk;
            }}
          >
`;

if (motionDivRegex.test(content)) {
  content = content.replace(motionDivRegex, nativeDiv);
  // Replace the closing </motion.div> for this container with </div>
  // It's located right after the closing map
  content = content.replace(/                <\/motion\.div>\n              \);\n            \}\)}\n          <\/motion\.div>/, `                </motion.div>\n              );\n            })}\n          </div>`);
}

// 4. Add hide-scrollbar css
if (!content.includes('.hide-scrollbar')) {
  content = content.replace(
    /<style>\{`/,
    `<style>{\`
        .hide-scrollbar::-webkit-scrollbar { display: none; }
        .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }`
  );
}

fs.writeFileSync(filePath, content, 'utf8');
console.log('Successfully refactored Features.jsx to native draggable scroll.');
