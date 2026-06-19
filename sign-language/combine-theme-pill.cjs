const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Navbar.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// The block to replace is from "Professional Theme Color Swatches" to the end of "Dark Mode Toggle"
const regex = /\{\/\* Professional Theme Color Swatches \*\/\}[\s\S]*?(?=\{\/\* Mobile Menu Button - Enhanced \*\/\})/g;

const newBlock = `{/* Unified Appearance Control Pill (Theme + Dark Mode) */}
        <div className="hidden sm:flex items-center gap-3 px-2 py-1.5 bg-white/80 dark:bg-white/5 backdrop-blur-xl rounded-full border border-primary-200/50 dark:border-primary-800/50 shadow-lg shadow-primary-500/10 dark:shadow-primary-900/20">
          
          {/* Theme Color Swatches */}
          <div className="flex items-center gap-2 pl-1">
            <button
              onClick={() => setThemeColor('purple')}
              aria-label="Purple Theme"
              className={\`w-6 h-6 rounded-full bg-gradient-to-br from-purple-400 to-purple-600 shadow-md transition-all duration-300 \${
                themeColor !== 'midnight-blue' 
                  ? 'ring-2 ring-offset-2 ring-purple-500 scale-110 dark:ring-offset-gray-900' 
                  : 'opacity-50 hover:opacity-100 hover:scale-105'
              }\`}
            />
            <button
              onClick={() => setThemeColor('midnight-blue')}
              aria-label="Midnight Blue Theme"
              className={\`w-6 h-6 rounded-full bg-gradient-to-br from-indigo-400 to-indigo-600 shadow-md transition-all duration-300 \${
                themeColor === 'midnight-blue' 
                  ? 'ring-2 ring-offset-2 ring-indigo-500 scale-110 dark:ring-offset-gray-900' 
                  : 'opacity-50 hover:opacity-100 hover:scale-105'
              }\`}
            />
          </div>

          {/* Divider */}
          <div className="w-px h-6 bg-gray-300/50 dark:bg-gray-700/50 mx-1"></div>

          {/* Dark Mode Toggle */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={toggleDarkMode}
            aria-label="Toggle Dark Mode"
            className="
              relative w-14 h-7 flex items-center
              bg-gray-200/50 dark:bg-gray-800/50
              rounded-full p-1
              shadow-inner shadow-gray-400/30 dark:shadow-gray-900
              border border-primary-300/20 dark:border-primary-700/20
              transition-all duration-500
              group mr-1
            "
          >
            <motion.div
              layout
              transition={{ type: "spring", stiffness: 500, damping: 30 }}
              className={\`
                absolute w-5 h-5 rounded-full flex items-center justify-center
                shadow-md
                \${darkMode 
                  ? "translate-x-7 bg-gradient-to-br from-yellow-400 to-orange-500" 
                  : "translate-x-0 bg-gradient-to-br from-primary-500 to-pink-500"
                }
              \`}
            >
              {darkMode ? (
                <FaSun className="text-white text-[10px]" />
              ) : (
                <FaMoon className="text-white text-[10px]" />
              )}
            </motion.div>
          </motion.button>
        </div>

        `;

if (content.match(regex)) {
  content = content.replace(regex, newBlock);
  fs.writeFileSync(filePath, content, 'utf8');
  console.log('Combined theme swatches and dark mode toggle into a unified pill.');
} else {
  console.log('Could not find the target blocks.');
}
