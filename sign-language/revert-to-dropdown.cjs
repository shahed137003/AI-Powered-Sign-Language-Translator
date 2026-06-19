const fs = require('fs');
const filePath = 'd:/GP/Frontend/sign-language/src/components/Navbar.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// 1. Add state variable
if (!content.includes('appearanceMenuOpen')) {
  content = content.replace(
    /const \[userMenuOpen, setUserMenuOpen\] = useState\(false\);/,
    'const [userMenuOpen, setUserMenuOpen] = useState(false);\n  const [appearanceMenuOpen, setAppearanceMenuOpen] = useState(false);'
  );
}

// 2. Make userMenu toggle close appearance menu, and add appearance toggle
content = content.replace(
  /const toggleUserMenu = \(\) => \{\n\s*setUserMenuOpen\(!userMenuOpen\);\n\s*\};/,
  'const toggleUserMenu = () => {\n    setUserMenuOpen(!userMenuOpen);\n    setAppearanceMenuOpen(false);\n  };\n\n  const toggleAppearanceMenu = () => {\n    setAppearanceMenuOpen(!appearanceMenuOpen);\n    setUserMenuOpen(false);\n  };'
);

// 3. Replace the entire Unified Appearance Control Pill
const oldPillRegex = /\{\/\* Unified Appearance Control Pill \(Theme \+ Dark Mode\) \*\/\}[\s\S]*?(?=\{\/\* Mobile Menu Button - Enhanced \*\/)/g;

const newDropdown = `{/* Professional Appearance Dropdown */}
        <div className="relative hidden sm:block">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={toggleAppearanceMenu}
            aria-label="Appearance Settings"
            className="
              relative w-10 h-10 flex items-center justify-center
              bg-white/80 dark:bg-white/5
              backdrop-blur-xl
              rounded-full
              border border-primary-200/50 dark:border-primary-800/50
              shadow-md hover:shadow-lg transition-all duration-300
              text-gray-600 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400
            "
          >
            <FaPalette size={16} />
          </motion.button>
          
          <AnimatePresence>
            {appearanceMenuOpen && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.95 }}
                transition={{ duration: 0.2 }}
                className="
                  absolute right-0 mt-3 p-4 w-64
                  bg-white/95 dark:bg-[#0f0920]/95
                  backdrop-blur-xl
                  border border-primary-200/50 dark:border-primary-800/50
                  rounded-2xl
                  shadow-2xl shadow-primary-500/10 dark:shadow-primary-900/40
                  flex flex-col gap-4 z-50
                "
              >
                {/* Theme Mode Segment */}
                <div>
                  <div className="text-xs font-bold text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wider">
                    Appearance
                  </div>
                  <div className="flex bg-gray-100 dark:bg-gray-800/50 p-1 rounded-xl">
                    <button
                      onClick={() => { if(darkMode) toggleDarkMode(); }}
                      className={\`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-semibold transition-all \${
                        !darkMode 
                          ? 'bg-white dark:bg-gray-700 shadow text-primary-600 dark:text-primary-400' 
                          : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                      }\`}
                    >
                      <FaSun size={14} /> Light
                    </button>
                    <button
                      onClick={() => { if(!darkMode) toggleDarkMode(); }}
                      className={\`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-semibold transition-all \${
                        darkMode 
                          ? 'bg-gray-800 shadow text-primary-400' 
                          : 'text-gray-500 hover:text-gray-700'
                      }\`}
                    >
                      <FaMoon size={14} /> Dark
                    </button>
                  </div>
                </div>

                {/* Accent Color Segment */}
                <div>
                  <div className="text-xs font-bold text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wider">
                    Accent Color
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => setThemeColor('purple')}
                      className={\`flex items-center gap-2 p-2 rounded-xl border transition-all \${
                        themeColor === 'purple' 
                          ? 'border-purple-500 bg-purple-50 dark:bg-purple-500/10' 
                          : 'border-transparent hover:bg-gray-50 dark:hover:bg-white/5'
                      }\`}
                    >
                      <div className="w-4 h-4 rounded-full bg-purple-500 shadow-sm"></div>
                      <span className={\`text-sm font-medium \${themeColor === 'purple' ? 'text-purple-700 dark:text-purple-300' : 'text-gray-600 dark:text-gray-400'}\`}>Purple</span>
                    </button>
                    <button
                      onClick={() => setThemeColor('midnight-blue')}
                      className={\`flex items-center gap-2 p-2 rounded-xl border transition-all \${
                        themeColor === 'midnight-blue' 
                          ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-500/10' 
                          : 'border-transparent hover:bg-gray-50 dark:hover:bg-white/5'
                      }\`}
                    >
                      <div className="w-4 h-4 rounded-full bg-indigo-600 shadow-sm"></div>
                      <span className={\`text-sm font-medium \${themeColor === 'midnight-blue' ? 'text-indigo-700 dark:text-indigo-300' : 'text-gray-600 dark:text-gray-400'}\`}>Blue</span>
                    </button>
                  </div>
                </div>

              </motion.div>
            )}
          </AnimatePresence>
        </div>

        `;

if (content.match(oldPillRegex)) {
  content = content.replace(oldPillRegex, newDropdown);
  fs.writeFileSync(filePath, content, 'utf8');
  console.log('Successfully reverted to a professional Appearance Dropdown.');
} else {
  console.log('Regex did not match.');
}
