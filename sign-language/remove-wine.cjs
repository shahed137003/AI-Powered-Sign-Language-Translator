const fs = require('fs');

// 1. Remove from index.css
let cssContent = fs.readFileSync('d:/GP/Frontend/sign-language/src/index.css', 'utf8');
const wineBlockRegex = /html\[data-theme="wine"\] \{[\s\S]*?\n\}\n\n?/g;
cssContent = cssContent.replace(wineBlockRegex, '');
fs.writeFileSync('d:/GP/Frontend/sign-language/src/index.css', cssContent, 'utf8');

// 2. Remove from Navbar.jsx
let navbarContent = fs.readFileSync('d:/GP/Frontend/sign-language/src/components/Navbar.jsx', 'utf8');
const navbarRegex = /\s*\{\s*name:\s*"Wine Red",\s*value:\s*"wine",\s*color:\s*"bg-rose-700"\s*\},?/g;
navbarContent = navbarContent.replace(navbarRegex, '');
fs.writeFileSync('d:/GP/Frontend/sign-language/src/components/Navbar.jsx', navbarContent, 'utf8');

// 3. Remove from particle components
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
    const arrayRegex = /\s*wine:\s*\[.*?\]\,?/g;
    content = content.replace(arrayRegex, '');
    fs.writeFileSync(file, content, 'utf8');
  }
});

console.log('Wine theme removed.');
