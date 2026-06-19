const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Home.jsx';
let content = fs.readFileSync(filePath, 'utf8');

const replacement = `const getHeroImage = () => {
    const darkPath = \`../assets/\${themeFolder}/heroDark.svg\`;
    const lightPath = \`../assets/\${themeFolder}/hero.svg\`;
    const imagePath = \`../assets/\${themeFolder}/image.svg\`;
    
    if (isDark) {
      return allAssets[darkPath] || allAssets[imagePath] || allAssets[lightPath];
    }
    return allAssets[lightPath] || allAssets[imagePath] || allAssets[darkPath];
  };`;

// Insert the helper function right before the return statement
content = content.replace(/return \(/, replacement + '\n\n  return (');

// Update the src to use the helper function
content = content.replace(/src=\{allAssets\[\`\.\.\/assets\/\$\{themeFolder\}\/\$\{isDark \? "heroDark\.svg" : "hero\.svg"\}\`\]\}/g, 'src={getHeroImage()}');

fs.writeFileSync(filePath, content, 'utf8');
console.log('Fixed Home.jsx robust image loading');
