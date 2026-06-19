const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Features.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// Replace top state
content = content.replace(
  /  const carouselRef = useRef\(null\);\n  const \[isCarouselHovered, setIsCarouselHovered\] = useState\(false\);\n  const \[isDragging, setIsDragging\] = useState\(false\);\n  let startX = 0;\n  let scrollLeftState = 0;/,
  `  const carouselRef = useRef(null);
  const [isCarouselHovered, setIsCarouselHovered] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const dragState = useRef({ startX: 0, scrollLeft: 0, exactScrollLeft: 0 });`
);

// Inject useEffect
const scrollEffect = `
  useEffect(() => {
    if (!carouselRef.current || isCarouselHovered || isDragging) return;
    
    let animationId;
    const scroll = () => {
      if (carouselRef.current) {
        dragState.current.exactScrollLeft += 1;
        
        if (dragState.current.exactScrollLeft >= carouselRef.current.scrollWidth / 2) {
          dragState.current.exactScrollLeft = 0;
        }
        
        carouselRef.current.scrollLeft = Math.floor(dragState.current.exactScrollLeft);
      }
      animationId = requestAnimationFrame(scroll);
    };
    animationId = requestAnimationFrame(scroll);
    return () => cancelAnimationFrame(animationId);
  }, [isCarouselHovered, isDragging]);
`;

// Only inject if not already present
if (!content.includes('dragState.current.exactScrollLeft +=')) {
  content = content.replace(
    /  useEffect\(\(\) => \{\n    const observer = new MutationObserver\(\(\) => \{/,
    scrollEffect + '\n  useEffect(() => {\n    const observer = new MutationObserver(() => {'
  );
}

// Update handlers
content = content.replace(
  /onMouseDown=\{\(e\) => \{\n                setIsDragging\(true\);\n                startX = e\.pageX - carouselRef\.current\.offsetLeft;\n                scrollLeftState = carouselRef\.current\.scrollLeft;\n              \}\}/,
  `onMouseDown={(e) => {
                setIsDragging(true);
                dragState.current.startX = e.pageX - carouselRef.current.offsetLeft;
                dragState.current.scrollLeft = carouselRef.current.scrollLeft;
              }}`
);

content = content.replace(
  /onMouseMove=\{\(e\) => \{\n                if \(!isDragging\) return;\n                e\.preventDefault\(\);\n                const x = e\.pageX - carouselRef\.current\.offsetLeft;\n                const walk = \(x - startX\) \* 2;\n                carouselRef\.current\.scrollLeft = scrollLeftState - walk;\n              \}\}/,
  `onMouseMove={(e) => {
                if (!isDragging) return;
                e.preventDefault();
                const x = e.pageX - carouselRef.current.offsetLeft;
                const walk = (x - dragState.current.startX) * 1.5;
                const newScroll = dragState.current.scrollLeft - walk;
                carouselRef.current.scrollLeft = newScroll;
                dragState.current.exactScrollLeft = newScroll; // sync exact tracker
              }}`
);

fs.writeFileSync(filePath, content, 'utf8');
console.log('Fixed drag and auto scroll in Features.jsx');
