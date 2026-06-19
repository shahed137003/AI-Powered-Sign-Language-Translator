const fs = require('fs');

// 1. Update index.css midnight-blue block
let cssContent = fs.readFileSync('d:/GP/Frontend/sign-language/src/index.css', 'utf8');

const midnightBlueRegex = /html\[data-theme="midnight-blue"\] \{[\s\S]*?\n\}\n?/g;
const newMidnightBlue = `html[data-theme="midnight-blue"] {
  /* Midnight Blue Theme */
  --theme-primary-50: #eff6ff;
  --theme-primary-100: #dbeafe;
  --theme-primary-200: #bfdbfe;
  --theme-primary-300: #93c5fd;
  --theme-primary-400: #60a5fa;
  --theme-primary-500: #3b82f6;
  --theme-primary-600: #2563eb;
  --theme-primary-700: #1d4ed8;
  --theme-primary-800: #1e40af;
  --theme-primary-900: #1e3a8a;
  --theme-primary-950: #172554;
  --theme-primary-custom-1: #1e3a8a;
  --theme-primary-custom-2: #2563eb;
  --theme-primary-custom-3: #60a5fa;
  
  --theme-primary-bg-1: #020617;
  --theme-primary-bg-2: #0f172a;
  --theme-primary-bg-3: #1e3a8a;
  --theme-primary-bg-4: #0B1120;
  --theme-primary-bg-5: #172554;
}
`;

if (cssContent.match(midnightBlueRegex)) {
  cssContent = cssContent.replace(midnightBlueRegex, newMidnightBlue);
  fs.writeFileSync('d:/GP/Frontend/sign-language/src/index.css', cssContent, 'utf8');
} else {
  console.log("Could not find midnight-blue block in index.css");
}

// 2. Update particle components midnight-blue arrays
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
    // Replace the indigo hex array with blue hex array
    const oldArray = /'midnight-blue': \['#6366F1', '#4F46E5', '#4338CA', '#3730A3', '#818CF8'\],?/g;
    content = content.replace(oldArray, `'midnight-blue': ['#3B82F6', '#2563EB', '#1D4ED8', '#1E40AF', '#60A5FA'],`);
    fs.writeFileSync(file, content, 'utf8');
  }
});

// 3. Update Contact.jsx hardcoded card colors to theme colors
let contactContent = fs.readFileSync('d:/GP/Frontend/sign-language/src/components/Contact.jsx', 'utf8');

contactContent = contactContent.replace(
  /color: "from-primary-500 to-pink-500"/,
  'color: "from-primary-400 to-primary-600"'
);
contactContent = contactContent.replace(
  /color: "from-blue-500 to-cyan-500"/,
  'color: "from-primary-500 to-primary-700"'
);
contactContent = contactContent.replace(
  /color: "from-green-500 to-emerald-500"/,
  'color: "from-primary-600 to-primary-800"'
);
contactContent = contactContent.replace(
  /color: "from-orange-500 to-yellow-500"/,
  'color: "from-primary-700 to-primary-900"'
);

fs.writeFileSync('d:/GP/Frontend/sign-language/src/components/Contact.jsx', contactContent, 'utf8');

console.log('Done tweaking midnight-blue and contact cards.');
