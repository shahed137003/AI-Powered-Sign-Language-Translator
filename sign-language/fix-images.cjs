const fs = require('fs');

function fixComponent(filePath) {
  let content = fs.readFileSync(filePath, 'utf8');

  // 1. Add glob import
  if (!content.includes('const allAssets = import.meta.glob')) {
    content = content.replace(/export default function \w+\(\) \{/, 'const allAssets = import.meta.glob("../assets/**/*.{svg,png,jpg,jpeg}", { eager: true, import: "default" });\n\n$&');
  }

  // 2. Replace new URL usage in Home.jsx
  if (filePath.includes('Home.jsx')) {
    content = content.replace(/new URL\(\`\.\.\/assets\/\$\{themeFolder\}\/\$\{isDark \? "heroDark\.svg" : "hero\.svg"\}\`, import\.meta\.url\)\.href/g, 
      'allAssets[`../assets/${themeFolder}/${isDark ? "heroDark.svg" : "hero.svg"}`]');
  }

  // 3. Replace new URL usage in Features.jsx
  if (filePath.includes('Features.jsx')) {
    content = content.replace(/new URL\(\`\.\.\/assets\/\$\{themeFolder\}\/\$\{feature\.imgName\}\`, import\.meta\.url\)\.href/g, 
      'allAssets[`../assets/${themeFolder}/${feature.imgName}`]');
  }

  fs.writeFileSync(filePath, content, 'utf8');
  console.log('Fixed ' + filePath);
}

fixComponent('d:/GP/Frontend/sign-language/src/components/Home.jsx');
fixComponent('d:/GP/Frontend/sign-language/src/components/Features.jsx');
