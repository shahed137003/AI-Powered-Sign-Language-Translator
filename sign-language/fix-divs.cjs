const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/About.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// Fix the globally applied wrong div tags
content = content.replace(/<\/div><\/div><\/motion\.div>/g, '</div></motion.div>');

fs.writeFileSync(filePath, content, 'utf8');
console.log('Fixed div tags');
