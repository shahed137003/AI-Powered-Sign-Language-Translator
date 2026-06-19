const fs = require('fs');

// 1. Update index.css
let cssContent = fs.readFileSync('d:/GP/Frontend/sign-language/src/index.css', 'utf8');
const redBlockRegex = /html\[data-theme="red"\] \{[\s\S]*?\n\}\n/g;
const wineBlock = `html[data-theme="wine"] {
  /* Wine Red Theme */
  --theme-primary-50: #fff1f2;
  --theme-primary-100: #ffe4e6;
  --theme-primary-200: #fecdd3;
  --theme-primary-300: #fda4af;
  --theme-primary-400: #fb7185;
  --theme-primary-500: #f43f5e;
  --theme-primary-600: #e11d48;
  --theme-primary-700: #be123c;
  --theme-primary-800: #9f1239;
  --theme-primary-900: #881337;
  --theme-primary-950: #4c0519;
  --theme-primary-custom-1: #881337;
  --theme-primary-custom-2: #e11d48;
  --theme-primary-custom-3: #fb7185;
  
  --theme-primary-bg-1: #1a050d;
  --theme-primary-bg-2: #2d0a16;
  --theme-primary-bg-3: #590f2b;
  --theme-primary-bg-4: #260914;
  --theme-primary-bg-5: #3a0d1e;
}
`;
cssContent = cssContent.replace(redBlockRegex, wineBlock);
fs.writeFileSync('d:/GP/Frontend/sign-language/src/index.css', cssContent, 'utf8');

// 2. Update Navbar.jsx
let navbarContent = fs.readFileSync('d:/GP/Frontend/sign-language/src/components/Navbar.jsx', 'utf8');
const navbarRegex = /\{\s*name:\s*"Red",\s*value:\s*"red",\s*color:\s*"bg-red-500"\s*\}/g;
navbarContent = navbarContent.replace(navbarRegex, '{ name: "Wine Red", value: "wine", color: "bg-rose-700" }');
fs.writeFileSync('d:/GP/Frontend/sign-language/src/components/Navbar.jsx', navbarContent, 'utf8');

// 3. Update particle components
const files = [
  'd:/GP/Frontend/sign-language/src/components/Chat.jsx',
  'd:/GP/Frontend/sign-language/src/components/Footer.jsx',
  'd:/GP/Frontend/sign-language/src/components/Home.jsx',
  'd:/GP/Frontend/sign-language/src/components/Login.jsx',
  'd:/GP/Frontend/sign-language/src/components/Register.jsx',
  'd:/GP/Frontend/sign-language/src/components/Translate.jsx'
];

files.forEach(file => {
  let content = fs.readFileSync(file, 'utf8');
  const arrayRegex = /\s*red:\s*\[.*?\]/g;
  content = content.replace(arrayRegex, `
        wine: ['#F43F5E', '#E11D48', '#BE123C', '#9F1239', '#FB7185']`);
  fs.writeFileSync(file, content, 'utf8');
});

console.log('Red replaced with Wine Red.');
