const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Features.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// 1. Remove state variables related to pagination
content = content.replace(/const \[currentPage, setCurrentPage\] = useState\(0\);\n/, '');
content = content.replace(/const \[hoveredNav, setHoveredNav\] = useState\(null\);\n/, '');
content = content.replace(/const CARDS_PER_PAGE = 3;\n/, '');

// 2. Remove pagination derivation logic
const paginationDerivationRegex = /const totalPages = Math\.ceil\(features\.length \/ CARDS_PER_PAGE\);\n  const currentFeatures = features\.slice\(currentPage \* CARDS_PER_PAGE, \(currentPage \+ 1\) \* CARDS_PER_PAGE\);\n\n  const goToPage = \(pageIndex\) => {\n    if \(pageIndex >= 0 && pageIndex < totalPages\) {\n      setCurrentPage\(pageIndex\);\n    }\n  };\n/;
content = content.replace(paginationDerivationRegex, '');

// 3. Find the exact string from `<div className="flex justify-center items-center gap-3 mb-12 flex-wrap">` down to the end of the feature cards rendering block and pagination UI.
// We will replace everything from `{/* Hover-Based Navigation - Thumbnail Gallery */}` to `{/* Feature Counter */}` and its closing div.

const startMarker = "{/* Hover-Based Navigation - Thumbnail Gallery */}";
const endMarker = "</motion.div>\n      </div>\n\n      <style>";

const startIndex = content.indexOf(startMarker);
const endIndex = content.indexOf(endMarker);

if (startIndex !== -1 && endIndex !== -1) {
  const replacement = `
        {/* Continuous Auto-Scrolling Carousel */}
        <div className="relative z-10 w-full overflow-hidden py-12 px-0 mt-8 group mask-image-linear-edges">
          <motion.div
            className="flex gap-8 w-max px-4"
            animate={{ x: ["0%", "-50%"] }}
            transition={{ 
              duration: 40, 
              repeat: Infinity, 
              ease: "linear",
            }}
          >
            {[...features, ...features].map((feature, index) => {
              const cardAnim = cardAnimations[index % cardAnimations.length];
              
              return (
                <motion.div
                  key={\`\${feature.title}-\${index}\`}
                  custom={index}
                  variants={cardVariants}
                  initial="hidden"
                  whileInView="visible"
                  viewport={{ once: true, margin: "100px" }}
                  onMouseEnter={() => setHoveredCard(index)}
                  onMouseLeave={() => setHoveredCard(null)}
                  style={{
                    boxShadow: "0 20px 40px -15px rgba(139, 92, 246, 0.15)",
                  }}
                  whileHover={{
                    y: -15,
                    scale: 1.03,
                    transition: { type: "spring", stiffness: 400, damping: 25 }
                  }}
                  className="w-[340px] md:w-[380px] shrink-0 group relative bg-white/80 dark:bg-white/5 backdrop-blur-xl rounded-3xl border-2 border-primary-200/50 dark:border-primary-800/50 shadow-xl overflow-hidden flex flex-col transition-all duration-300 hover:border-primary-300 dark:hover:border-primary-600 cursor-pointer"
                >
                  <motion.div
                    className="h-full flex flex-col"
                    animate={{
                      y: [0, -cardAnim?.floatAmplitude || -8, 0, cardAnim?.floatAmplitude || 8, 0],
                      rotate: [0, cardAnim?.rotateAmplitude || 2, -cardAnim?.rotateAmplitude || 2, cardAnim?.rotateAmplitude || 2, 0],
                      scale: [1, 1.01, 1],
                    }}
                    transition={{
                      y: { duration: cardAnim?.floatDuration || 4, delay: cardAnim?.floatDelay || 0, repeat: Infinity, ease: "easeInOut" },
                      rotate: { duration: cardAnim?.rotateDuration || 5, delay: cardAnim?.rotateDelay || 0, repeat: Infinity, ease: "easeInOut" },
                      scale: { duration: cardAnim?.pulseDuration || 3, delay: cardAnim?.pulseDelay || 0, repeat: Infinity, ease: "easeInOut" }
                    }}
                  >
                    {/* Animated Border Glow */}
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-primary-600 to-primary-400 rounded-3xl opacity-0 group-hover:opacity-30 transition-opacity duration-500 blur-xl" />
                    
                    {/* Card Hover Gradients */}
                    <div className="absolute inset-0 bg-gradient-to-br from-primary-400 to-primary-600 opacity-0 group-hover:opacity-[0.08] transition-opacity duration-700" />
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
                    
                    {/* Corner Accents */}
                    <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-br from-primary-400 to-primary-600 opacity-0 group-hover:opacity-20 rounded-bl-3xl transition-all duration-500" />
                    <div className="absolute bottom-0 left-0 w-24 h-24 bg-gradient-to-tr from-primary-400 to-primary-600 opacity-0 group-hover:opacity-20 rounded-tr-3xl transition-all duration-500" />

                    <div className="relative p-8 pb-6 flex-grow flex flex-col z-10 h-full">
                      
                      <div className="flex justify-between items-start mb-6">
                        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary-100 to-primary-50 dark:from-primary-900/50 dark:to-primary-800/50 border border-primary-200 dark:border-primary-700 flex items-center justify-center shadow-inner group-hover:scale-110 transition-transform duration-500">
                          <span className="text-3xl font-black bg-gradient-to-br from-primary-600 to-primary-400 bg-clip-text text-transparent">
                            {(index % features.length) + 1}
                          </span>
                        </div>
                        <div className="px-3 py-1.5 rounded-full bg-primary-100/50 dark:bg-primary-900/30 border border-primary-200 dark:border-primary-700 backdrop-blur-md">
                          <span className="text-xs font-bold text-primary-700 dark:text-primary-300">AI Powered</span>
                        </div>
                      </div>

                      {/* Icon Badge */}
                      <div className="absolute top-4 left-4 z-20">
                        <motion.div 
                          className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 text-white shadow-lg group-hover:scale-110 transition-transform duration-300"
                          animate={hoveredCard === index ? { rotate: [0, -5, 5, 0] } : {}}
                          transition={{ duration: 0.5 }}
                        >
                          {feature.icon}
                        </motion.div>
                      </div>

                      {/* Feature Image Container */}
                      <div className="relative pt-12 pb-4 px-6 flex items-center justify-center">
                        <motion.div 
                          className="absolute inset-0 bg-primary-500/20 blur-3xl rounded-full scale-75 opacity-0 group-hover:opacity-100 transition-opacity duration-700"
                          animate={hoveredCard === index ? { scale: [0.75, 1, 0.75] } : {}}
                          transition={{ duration: 2, repeat: Infinity }}
                        />
                        <motion.img
                          src={allAssets[\`../assets/\${themeFolder}/\${feature.imgName}\`]}
                          alt={feature.title}
                          className="relative z-10 w-36 h-36 object-contain drop-shadow-xl"
                          animate={hoveredCard === index ? { scale: 1.1, rotate: [0, -3, 3, 0] } : { scale: 1 }}
                          transition={{ duration: 0.5 }}
                        />
                      </div>

                      {/* Feature Text */}
                      <div className="relative z-10 p-6 pt-2 flex-grow flex flex-col">
                        <h2 className="text-xl font-black mb-3 bg-gradient-to-r from-primary-700 via-primary-600 to-primary-500 dark:from-primary-300 dark:via-primary-200 dark:to-primary-100 bg-clip-text text-transparent">
                          {feature.title}
                        </h2>
                        
                        <p className="text-gray-600 dark:text-gray-300 leading-relaxed text-sm mb-4">
                          {feature.description.length > 100 
                            ? \`\${feature.description.substring(0, 100)}...\` 
                            : feature.description}
                        </p>

                        {/* Stats Grid */}
                        <div className="grid grid-cols-4 gap-2 mb-4 p-3 rounded-xl bg-primary-50/50 dark:bg-primary-900/20 border border-primary-200/30 dark:border-primary-700/30">
                          <motion.div className="text-center" whileHover={{ scale: 1.05 }}>
                            <div className="text-sm font-black bg-gradient-to-r from-primary-600 to-primary-500 bg-clip-text text-transparent">
                              {feature.statValue}
                            </div>
                            <div className="text-[9px] text-primary-600 dark:text-primary-400 font-medium">{feature.statLabel}</div>
                          </motion.div>
                          <motion.div className="text-center" whileHover={{ scale: 1.05 }}>
                            <div className="flex items-center justify-center gap-0.5">
                              <FaStar className="text-primary-500 text-[10px]" />
                              <span className="text-sm font-black text-gray-800 dark:text-white">{feature.rating}</span>
                            </div>
                            <div className="text-[9px] text-primary-600 dark:text-primary-400 font-medium">Rating</div>
                          </motion.div>
                          <motion.div className="text-center" whileHover={{ scale: 1.05 }}>
                            <div className="text-sm font-black text-gray-800 dark:text-white">
                              {feature.users}
                            </div>
                            <div className="text-[9px] text-primary-600 dark:text-primary-400 font-medium">Users</div>
                          </motion.div>
                          <motion.div className="text-center" whileHover={{ scale: 1.05 }}>
                            <div className="flex items-center justify-center gap-0.5">
                              <TbClock className="text-primary-500 text-[10px]" />
                              <span className="text-sm font-black text-gray-800 dark:text-white">{feature.speed}</span>
                            </div>
                            <div className="text-[9px] text-primary-600 dark:text-primary-400 font-medium">Response</div>
                          </motion.div>
                        </div>

                        {/* Tags */}
                        <div className="flex flex-wrap gap-2 mt-auto">
                          {feature.tags.map((tag, i) => (
                            <motion.span
                              key={i}
                              initial={{ opacity: 0, scale: 0.8 }}
                              whileInView={{ opacity: 1, scale: 1 }}
                              transition={{ delay: i * 0.05 }}
                              whileHover={{ scale: 1.05 }}
                              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-primary-100/80 dark:bg-primary-800/30 text-xs font-semibold text-primary-700 dark:text-primary-300 border border-primary-200/50 dark:border-primary-700/50"
                            >
                              <BsCheckCircleFill className="text-[8px] text-primary-500" />
                              {tag}
                            </motion.span>
                          ))}
                        </div>
                      </div>

                      {/* Animated Bottom Line */}
                      <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-primary-400 via-primary-500 to-primary-600 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-700 origin-left rounded-b-3xl" />
                      
                      {/* Pulse Ring on Hover */}
                      <motion.div
                        className="absolute inset-0 rounded-3xl border-2 border-primary-400/0"
                        animate={hoveredCard === index ? { scale: [1, 1.02, 1], opacity: [0, 0.2, 0] } : {}}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      />
                    </div>
                  </motion.div>
                </motion.div>
              );
            })}
          </motion.div>
        </div>
      </div>

      <style>`;
  
  content = content.substring(0, startIndex) + replacement + content.substring(endIndex + "</motion.div>\n      </div>\n\n      <style>".length - "<style>".length);
  
  // Need to make sure max-w-7xl mx-auto isn't wrapping our marquee, otherwise it gets cut off. 
  // Let's change the header container to close BEFORE the carousel.
  content = content.replace(
    /        <\/motion\.div>\n\n        \{\/\* Continuous Auto-Scrolling Carousel \*\/\}/,
    `        </motion.div>\n      </div>\n\n      {/* Continuous Auto-Scrolling Carousel */}`
  );
  
  // And remove the extra closing div since we closed it early
  content = content.replace(
    /        <\/div>\n      <\/div>\n\n      <style>/,
    `        </div>\n\n      <style>`
  );

  // Add the CSS for edge fading
  content = content.replace(
    /<style>\{`/,
    `<style>{\`
        .mask-image-linear-edges {
          -webkit-mask-image: linear-gradient(to right, transparent, black 5%, black 95%, transparent);
          mask-image: linear-gradient(to right, transparent, black 5%, black 95%, transparent);
        }`
  );

  fs.writeFileSync(filePath, content, 'utf8');
  console.log('Successfully refactored Features.jsx to an infinite marquee.');
} else {
  console.log('Could not find markers', startIndex, endIndex);
}
