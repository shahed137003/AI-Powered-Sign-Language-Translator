const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Profile.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// Replace hardcoded purple with primary
content = content.replace(/purple-500/g, 'primary-500');
content = content.replace(/purple-600/g, 'primary-600');
content = content.replace(/purple-400/g, 'primary-400');
content = content.replace(/purple-700/g, 'primary-700');
content = content.replace(/purple-300/g, 'primary-300');
content = content.replace(/purple-200/g, 'primary-200');
content = content.replace(/purple-100/g, 'primary-100');
content = content.replace(/purple-50/g, 'primary-50');
content = content.replace(/purple-900/g, 'primary-900');
content = content.replace(/purple-800/g, 'primary-800');

fs.writeFileSync(filePath, content, 'utf8');
console.log('Fixed Profile.jsx purple colors');
