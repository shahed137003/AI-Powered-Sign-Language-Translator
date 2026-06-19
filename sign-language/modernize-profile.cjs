const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Profile.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// Add subtle glow to the inputs on focus, making them more modern
const oldInputClass = 'className="w-full p-3 rounded-xl bg-white/40 dark:bg-gray-900/40 border border-gray-300/50 dark:border-gray-700/50 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500/50 transition-all duration-300"';
const newInputClass = 'className="w-full p-3 rounded-xl bg-white/50 dark:bg-gray-900/50 border border-gray-300/50 dark:border-gray-700/50 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500/50 hover:bg-white/70 dark:hover:bg-gray-800/60 transition-all duration-300 shadow-sm"';
content = content.replace(new RegExp(oldInputClass.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), newInputClass);

// Handle password input separately because it has pr-10
const oldPasswordClass = 'className="w-full p-3 rounded-xl bg-white/40 dark:bg-gray-900/40 border border-gray-300/50 dark:border-gray-700/50 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500/50 transition-all duration-300 pr-10"';
const newPasswordClass = 'className="w-full p-3 rounded-xl bg-white/50 dark:bg-gray-900/50 border border-gray-300/50 dark:border-gray-700/50 text-gray-900 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500/50 hover:bg-white/70 dark:hover:bg-gray-800/60 transition-all duration-300 shadow-sm pr-10"';
content = content.replace(new RegExp(oldPasswordClass.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), newPasswordClass);

fs.writeFileSync(filePath, content, 'utf8');
console.log('Modernized input fields in Profile.jsx');
