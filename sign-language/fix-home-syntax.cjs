const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Home.jsx';
let content = fs.readFileSync(filePath, 'utf8');

const badFuncRegex = /\s*const getHeroImage = \(\) => \{\s*const darkPath = `\.\.\/assets\/\$\{themeFolder\}\/heroDark\.svg`;\s*const lightPath = `\.\.\/assets\/\$\{themeFolder\}\/hero\.svg`;\s*const imagePath = `\.\.\/assets\/\$\{themeFolder\}\/image\.svg`;\s*if \(isDark\) \{\s*return allAssets\[darkPath\] \|\| allAssets\[imagePath\] \|\| allAssets\[lightPath\];\s*\}\s*return allAssets\[lightPath\] \|\| allAssets\[imagePath\] \|\| allAssets\[darkPath\];\s*\};\s*/;

content = content.replace(badFuncRegex, '');

const correctPlacement = `
  const getHeroImage = () => {
    const darkPath = \`../assets/\${themeFolder}/heroDark.svg\`;
    const lightPath = \`../assets/\${themeFolder}/hero.svg\`;
    const imagePath = \`../assets/\${themeFolder}/image.svg\`;
    
    if (isDark) {
      return allAssets[darkPath] || allAssets[imagePath] || allAssets[lightPath];
    }
    return allAssets[lightPath] || allAssets[imagePath] || allAssets[darkPath];
  };

  return (
    <div className="relative w-full`;

// replace the actual render return
content = content.replace(/\s*return \(\s*<div className="relative w-full/, correctPlacement);

fs.writeFileSync(filePath, content, 'utf8');
console.log('Fixed Home.jsx return syntax error');
