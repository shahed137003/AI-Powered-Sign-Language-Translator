const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Features.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// The problematic block:
//                   animate="visible"
//                   exit="exit"
//                   onMouseEnter={() => setHoveredCard(index)}
//                   onMouseLeave={() => setHoveredCard(null)}
//                   // Continuous floating animation
//                   animate={{

// We want to replace it by wrapping the inner content with <motion.div animate={{ ... }} transition={{ ... }}>

// 1. First, remove the continuous animate and transition from the outer motion.div
const blockToReplace = `                  animate={{
                    y: [0, -cardAnim?.floatAmplitude || -8, 0, cardAnim?.floatAmplitude || 8, 0],
                    rotate: [0, cardAnim?.rotateAmplitude || 2, -cardAnim?.rotateAmplitude || 2, cardAnim?.rotateAmplitude || 2, 0],
                    scale: [1, 1.01, 1],
                  }}
                  transition={{
                    y: {
                      duration: cardAnim?.floatDuration || 4,
                      delay: cardAnim?.floatDelay || 0,
                      repeat: Infinity,
                      ease: "easeInOut",
                    },
                    rotate: {
                      duration: cardAnim?.rotateDuration || 5,
                      delay: cardAnim?.rotateDelay || 0,
                      repeat: Infinity,
                      ease: "easeInOut",
                    },
                    scale: {
                      duration: cardAnim?.scaleDuration || 6,
                      delay: cardAnim?.scaleDelay || 0,
                      repeat: Infinity,
                      ease: "easeInOut",
                    }
                  }}
                  className={\`relative p-8 rounded-3xl h-full flex flex-col justify-between transition-all duration-500 overflow-hidden \${
                    hoveredCard === index 
                      ? 'bg-white/90 dark:bg-primary-bg-2/90 border-2 border-primary-400 dark:border-primary-400 shadow-2xl scale-105 z-20' 
                      : 'bg-white/60 dark:bg-primary-bg-2/40 border border-primary-200/50 dark:border-primary-600/30 shadow-xl z-10'
                  } backdrop-blur-xl group\`}`;

const newBlock = `                  className="h-full relative"
                >
                  <motion.div
                    animate={{
                      y: [0, -(cardAnim?.floatAmplitude || 8), 0, (cardAnim?.floatAmplitude || 8), 0],
                      rotate: [0, (cardAnim?.rotateAmplitude || 2), -(cardAnim?.rotateAmplitude || 2), (cardAnim?.rotateAmplitude || 2), 0],
                      scale: [1, 1.01, 1],
                    }}
                    transition={{
                      y: {
                        duration: cardAnim?.floatDuration || 4,
                        delay: cardAnim?.floatDelay || 0,
                        repeat: Infinity,
                        ease: "easeInOut",
                      },
                      rotate: {
                        duration: cardAnim?.rotateDuration || 5,
                        delay: cardAnim?.rotateDelay || 0,
                        repeat: Infinity,
                        ease: "easeInOut",
                      },
                      scale: {
                        duration: cardAnim?.scaleDuration || 6,
                        delay: cardAnim?.scaleDelay || 0,
                        repeat: Infinity,
                        ease: "easeInOut",
                      }
                    }}
                    className={\`relative p-8 rounded-3xl h-full flex flex-col justify-between transition-all duration-500 overflow-hidden \${
                      hoveredCard === index 
                        ? 'bg-white/90 dark:bg-primary-bg-2/90 border-2 border-primary-400 dark:border-primary-400 shadow-2xl z-20' 
                        : 'bg-white/60 dark:bg-primary-bg-2/40 border border-primary-200/50 dark:border-primary-600/30 shadow-xl z-10'
                    } backdrop-blur-xl group\`}
                  >`;

if(content.includes(blockToReplace)) {
  content = content.replace(blockToReplace, newBlock);
} else {
  console.log("Could not find the block to replace. It might be formatted slightly differently.");
}

// Now we need to close the inner <motion.div> at the very end of the card mapping.
// The end of the card is:
//                 </motion.div>
//               );
//             })}
// We need to change the last </motion.div> of the card to </motion.div></motion.div>

content = content.replace(
  /                  {hoveredCard === index && \(\n                    <motion.div\n                      initial=\{\{ opacity: 0 \}\}\n                      animate=\{\{ opacity: 1 \}\}\n                      className="absolute inset-0 border-2 border-primary-400 rounded-3xl pointer-events-none"\n                      layoutId="card-border"\n                    \/>\n                  \)}\n                <\/motion.div>/,
  `                  {hoveredCard === index && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="absolute inset-0 border-2 border-primary-400 rounded-3xl pointer-events-none"
                      layoutId="card-border"
                    />
                  )}
                  </motion.div>
                </motion.div>`
);

fs.writeFileSync(filePath, content, 'utf8');
console.log('Fixed double animate props');
