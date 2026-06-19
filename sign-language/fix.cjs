const fs = require('fs');
const path = require('path');
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
  const name = path.basename(file, '.jsx');
  
  // Fix imports
  if (content.trim().startsWith('{ useTheme } from \'../context/ThemeContext\';')) {
    content = content.replace(/^.*?\{ useTheme \} from \'..\/context\/ThemeContext\';\n/, 'import React, { useState, useEffect, useRef } from "react";\nimport { useTheme } from "../context/ThemeContext";\n');
  }
  
  // Fix export
  if (!content.includes('export default function ' + name)) {
    content = content.replace(/  const \{ themeColor \} = useTheme\(\);\n/, 'export default function ' + name + '() {\n  const { themeColor } = useTheme();\n');
  }
  
  fs.writeFileSync(file, content, 'utf8');
});
console.log('Fixed 6 files');
