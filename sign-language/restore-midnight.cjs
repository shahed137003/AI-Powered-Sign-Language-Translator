const fs = require('fs');

const midnightBlue = `
html[data-theme="midnight-blue"] {
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
  
  --theme-primary-bg-1: #090914;
  --theme-primary-bg-2: #101026;
  --theme-primary-bg-3: #18183d;
  --theme-primary-bg-4: #0a0a1a;
  --theme-primary-bg-5: #111124;
}
`;

let cssContent = fs.readFileSync('d:/GP/Frontend/sign-language/src/index.css', 'utf8');

if (!cssContent.includes('data-theme="midnight-blue"')) {
  cssContent += midnightBlue;
  fs.writeFileSync('d:/GP/Frontend/sign-language/src/index.css', cssContent, 'utf8');
  console.log('Restored midnight-blue CSS block');
} else {
  console.log('midnight-blue CSS block already exists');
}
