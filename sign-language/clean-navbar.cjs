const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Navbar.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// Remove unused state
content = content.replace(/const \[themeMenuOpen, setThemeMenuOpen\] = useState\(false\);\n?\s*/g, '');

// Remove toggleThemeMenu function completely
content = content.replace(/const toggleThemeMenu = \(\) => \{[\s\S]*?\};\n?\s*/g, '');

// Remove setThemeMenuOpen(false) from toggleUserMenu
content = content.replace(/setThemeMenuOpen\(false\);\n?\s*/g, '');

fs.writeFileSync(filePath, content, 'utf8');
console.log('Cleaned up unused theme menu state variables.');
