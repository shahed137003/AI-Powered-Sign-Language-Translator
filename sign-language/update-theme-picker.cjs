const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Navbar.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// The block to replace is:
//         {/* Theme Picker Dropdown */}
//         <div className="relative"> ... </div>

const regex = /\{\/\* Theme Picker Dropdown \*\/\}[\s\S]*?(?=\{\/\* Dark Mode Toggle - Enhanced with home page styling \*\/\})/g;

const newBlock = `{/* Professional Theme Color Swatches */}
        <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-white/80 dark:bg-white/5 backdrop-blur-xl rounded-full border-2 border-primary-200/50 dark:border-primary-800/50 shadow-lg shadow-primary-500/10 dark:shadow-primary-900/20">
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

        `;

if (content.match(regex)) {
  content = content.replace(regex, newBlock);
  fs.writeFileSync(filePath, content, 'utf8');
  console.log('Replaced dropdown with professional color swatches.');
} else {
  console.log('Could not find Theme Picker block.');
}
