const fs = require('fs');

function fixFile(filePath) {
  let content = fs.readFileSync(filePath, 'utf8');

  // Replace grid inline styles
  const gridStyleOld1 = /rgba\(168, 85, 247, 0\.1\)/g;
  const gridStyleOld2 = /rgba\(139, 92, 246, 0\.1\)/g;
  content = content.replace(gridStyleOld1, '\${gridColor}');
  content = content.replace(gridStyleOld2, '\${gridColor}');

  // Inject gridColor variable if needed
  if (!content.includes('const gridColor =')) {
    content = content.replace(/(const \{ themeColor \} = useTheme\(\);)/, '$1\n  const gridColor = themeColor === "midnight-blue" ? "rgba(99, 102, 241, 0.1)" : "rgba(168, 85, 247, 0.1)";');
  }

  fs.writeFileSync(filePath, content, 'utf8');
}

fixFile('d:/GP/Frontend/sign-language/src/components/Home.jsx');

console.log('Fixed grid color in Home.jsx');
