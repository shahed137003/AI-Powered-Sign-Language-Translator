const fs = require('fs');
const files = [
  'd:/GP/Frontend/sign-language/src/components/Chat.jsx',
  'd:/GP/Frontend/sign-language/src/components/Footer.jsx',
  'd:/GP/Frontend/sign-language/src/components/Home.jsx',
  'd:/GP/Frontend/sign-language/src/components/Login.jsx',
  'd:/GP/Frontend/sign-language/src/components/Register.jsx',
  'd:/GP/Frontend/sign-language/src/components/Translate.jsx'
];

files.forEach(file => {
  let content = fs.readFileSync(file, 'utf8');
  
  const searchStr = `blue: ['#3B82F6', '#2563EB', '#1D4ED8', '#1E40AF', '#60A5FA'],`;
  const replacementStr = `blue: ['#3B82F6', '#2563EB', '#1D4ED8', '#1E40AF', '#60A5FA'],
        'midnight-blue': ['#6366F1', '#4F46E5', '#4338CA', '#3730A3', '#818CF8'],`;

  if (content.includes(searchStr) && !content.includes('midnight-blue')) {
    content = content.replace(searchStr, replacementStr);
    fs.writeFileSync(file, content, 'utf8');
  }
});
console.log('Midnight blue added to particles.');
