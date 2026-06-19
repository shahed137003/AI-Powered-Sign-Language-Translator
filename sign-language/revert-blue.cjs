const fs = require('fs');

// 1. Revert index.css midnight-blue block
let cssContent = fs.readFileSync('d:/GP/Frontend/sign-language/src/index.css', 'utf8');

const midnightBlueRegex = /html\[data-theme="midnight-blue"\] \{[\s\S]*?\n\}\n?/g;
const originalMidnightBlue = `html[data-theme="midnight-blue"] {
  /* Midnight Blue Theme */
  --theme-primary-50: #eef2ff;
  --theme-primary-100: #e0e7ff;
  --theme-primary-200: #c7d2fe;
  --theme-primary-300: #a5b4fc;
  --theme-primary-400: #818cf8;
  --theme-primary-500: #6366f1;
  --theme-primary-600: #4f46e5;
  --theme-primary-700: #4338ca;
  --theme-primary-800: #3730a3;
  --theme-primary-900: #312e81;
  --theme-primary-950: #1e1b4b;
  --theme-primary-custom-1: #312e81;
  --theme-primary-custom-2: #4f46e5;
  --theme-primary-custom-3: #818cf8;
  
  --theme-primary-bg-1: #0B1120;
  --theme-primary-bg-2: #1e1b4b;
  --theme-primary-bg-3: #312e81;
  --theme-primary-bg-4: #0B1120;
  --theme-primary-bg-5: #1e1b4b;
}
`;

if (cssContent.match(midnightBlueRegex)) {
  cssContent = cssContent.replace(midnightBlueRegex, originalMidnightBlue);
  fs.writeFileSync('d:/GP/Frontend/sign-language/src/index.css', cssContent, 'utf8');
}

// 2. Revert particle components midnight-blue arrays
const files = [
  'd:/GP/Frontend/sign-language/src/components/Chat.jsx',
  'd:/GP/Frontend/sign-language/src/components/Footer.jsx',
  'd:/GP/Frontend/sign-language/src/components/Home.jsx',
  'd:/GP/Frontend/sign-language/src/components/Login.jsx',
  'd:/GP/Frontend/sign-language/src/components/Register.jsx',
  'd:/GP/Frontend/sign-language/src/components/Translate.jsx',
  'd:/GP/Frontend/sign-language/src/components/Contact.jsx'
];

files.forEach(file => {
  if (fs.existsSync(file)) {
    let content = fs.readFileSync(file, 'utf8');
    // Revert blue hex array to original indigo hex array
    const oldArray = /'midnight-blue': \['#3B82F6', '#2563EB', '#1D4ED8', '#1E40AF', '#60A5FA'\],?/g;
    content = content.replace(oldArray, `'midnight-blue': ['#6366F1', '#4F46E5', '#4338CA', '#3730A3', '#818CF8'],`);
    fs.writeFileSync(file, content, 'utf8');
  }
});

console.log('Reverted Midnight Blue to original indigo colors.');
