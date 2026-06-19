const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Footer.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// Add import Link
if (!content.includes('import { Link } from "react-router-dom";')) {
  content = content.replace(
    /import \{ motion \} from "framer-motion";/,
    'import { motion } from "framer-motion";\nimport { Link } from "react-router-dom";'
  );
}

// Replace <a href={link.href} with <Link to={link.href} for quickLinks
// Replace </a> with </Link>
// The quickLinks render is roughly: <a href={link.href} ...> ... </a>
content = content.replace(/<a\s+href=\{link\.href\}/g, '<Link to={link.href}');
content = content.replace(/<\/a>/g, '</Link>');

fs.writeFileSync(filePath, content, 'utf8');
console.log('Replaced <a> tags with <Link> in Footer.jsx');
