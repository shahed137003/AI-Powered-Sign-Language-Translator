const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Features.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// 1. Remove the state and refs
content = content.replace(
  /  const carouselRef = useRef\(null\);\n  const \[isCarouselHovered, setIsCarouselHovered\] = useState\(false\);\n  const \[isDragging, setIsDragging\] = useState\(false\);\n  const dragState = useRef\(\{ startX: 0, scrollLeft: 0, exactScrollLeft: 0 \}\);\n/,
  ''
);
// Also remove the old ones if they were left over
content = content.replace(
  /  const carouselRef = useRef\(null\);\n  const \[isCarouselHovered, setIsCarouselHovered\] = useState\(false\);\n  const \[isDragging, setIsDragging\] = useState\(false\);\n  let startX = 0;\n  let scrollLeftState = 0;\n/,
  ''
);


// 2. Remove the scrollEffect
const scrollEffectRegex = /  useEffect\(\(\) => \{\n    if \(!carouselRef\.current.*?\n    \}, \[isCarouselHovered, isDragging\]\);\n/s;
content = content.replace(scrollEffectRegex, '');

// 3. Revert the wrapper to framer-motion
const nativeDivRegex = /<div \n              ref=\{carouselRef\}.*?className="flex gap-8 w-max px-4 overflow-x-auto hide-scrollbar cursor-grab active:cursor-grabbing pb-8".*?onMouseMove=\{.*?\}\n            >/s;

const motionDiv = `<motion.div
            className="flex gap-8 w-max px-4"
            animate={{ x: ["0%", "-50%"] }}
            transition={{ 
              duration: 120, 
              repeat: Infinity, 
              ease: "linear",
            }}
          >`;

if (nativeDivRegex.test(content)) {
  content = content.replace(nativeDivRegex, motionDiv);
  // Revert closing div
  content = content.replace(
    /                <\/motion\.div>\n              \);\n            \}\)}\n          <\/div>/,
    `                </motion.div>\n              );\n            })}\n          </motion.div>`
  );
}

// 4. Remove hide-scrollbar CSS
content = content.replace(/        \.hide-scrollbar::-webkit-scrollbar \{ display: none; \}\n        \.hide-scrollbar \{ -ms-overflow-style: none; scrollbar-width: none; \}\n/, '');


fs.writeFileSync(filePath, content, 'utf8');
console.log('Successfully reverted Features.jsx to framer-motion marquee.');
