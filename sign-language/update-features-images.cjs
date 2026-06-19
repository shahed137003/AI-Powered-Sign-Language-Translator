const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Features.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// 1. Remove static imports
content = content.replace(/import Sign from "\.\.\/assets\/feature1\.svg";\n/, '');
content = content.replace(/import Avatar from "\.\.\/assets\/feature2\.svg";\n/, '');
content = content.replace(/import Chatbot from "\.\.\/assets\/feature3\.svg";\n/, '');
content = content.replace(/import Mobile from "\.\.\/assets\/feature4\.svg";\n/, '');
content = content.replace(/import Profile from "\.\.\/assets\/profile\.svg";\n/, '');
content = content.replace(/import Customize from "\.\.\/assets\/customize\.svg";\n/, '');
content = content.replace(/import Ai from "\.\.\/assets\/Ai2\.svg";\n/, '');

// 2. Add useTheme import if it doesn't exist
if (!content.includes('import { useTheme } from "../context/ThemeContext"')) {
  content = content.replace(/import React, \{[^\}]+\} from "react";\n/, '$&\nimport { useTheme } from "../context/ThemeContext";\n');
}

// 3. Update the array to use imgName
content = content.replace(/img: Sign,/g, 'imgName: "feature1.svg",');
content = content.replace(/img: Avatar,/g, 'imgName: "feature2.svg",');
content = content.replace(/img: Chatbot,/g, 'imgName: "feature3.svg",');
content = content.replace(/img: Mobile,/g, 'imgName: "feature4.svg",');
content = content.replace(/img: Profile,/g, 'imgName: "profile.svg",');
content = content.replace(/img: Customize,/g, 'imgName: "customize.svg",');
content = content.replace(/img: Ai,/g, 'imgName: "Ai2.svg",');

// 4. Update component to get themeColor and compute the URL
// We need to inject const { themeColor } = useTheme(); into the top of the component if it isn't there.
// Features component starts like: export default function Features() {
if (!content.includes('const { themeColor } = useTheme();')) {
  content = content.replace(/export default function Features\(\) \{/, 'export default function Features() {\n  const { themeColor } = useTheme();\n  const themeFolder = themeColor === "midnight-blue" ? "blue" : "purple";\n');
}

// 5. Update the <img src={feature.img} to use dynamic URL
// We will replace `src={feature.img}` with `src={new URL(\`../assets/\${themeFolder}/\${feature.imgName}\`, import.meta.url).href}`
content = content.replace(/src=\{feature\.img\}/g, 'src={new URL(`../assets/${themeFolder}/${feature.imgName}`, import.meta.url).href}');

fs.writeFileSync(filePath, content, 'utf8');
console.log('Features.jsx updated successfully');
