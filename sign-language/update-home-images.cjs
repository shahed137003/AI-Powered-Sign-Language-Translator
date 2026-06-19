const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Home.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// 1. Remove static imports
content = content.replace(/import Hero from "\.\.\/assets\/hero\.svg";\n/, '');
content = content.replace(/import HeroDark from "\.\.\/assets\/heroDark\.svg";\n/, '');
content = content.replace(/import HeroBlue from "\.\.\/assets\/image\.svg";\n/, '');

// 2. Add themeFolder definition
if (!content.includes('const themeFolder = themeColor === "midnight-blue" ? "blue" : "purple";')) {
  content = content.replace(/const \{ themeColor \} = useTheme\(\);/, 'const { themeColor } = useTheme();\n  const themeFolder = themeColor === "midnight-blue" ? "blue" : "purple";');
}

// 3. Update the image src
content = content.replace(/src=\{themeColor === 'midnight-blue' \? HeroBlue : \(isDark && HeroDark \? HeroDark : Hero\)\}/g, 'src={new URL(`../assets/${themeFolder}/${isDark ? "heroDark.svg" : "hero.svg"}`, import.meta.url).href}');

fs.writeFileSync(filePath, content, 'utf8');
console.log('Home.jsx updated successfully');
