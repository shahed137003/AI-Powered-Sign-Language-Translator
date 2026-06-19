const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/About.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// Update Mission/Vision cards
content = content.replace(
  /<div className="relative p-8 rounded-2xl bg-white\/70 dark:bg-white\/5 backdrop-blur-xl border border-primary-200\/50 dark:border-primary-500\/20 shadow-xl hover:shadow-2xl transition-all duration-300">/g,
  `<div className="absolute -inset-0.5 bg-gradient-to-r from-primary-500/30 to-primary-600/30 rounded-2xl blur opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              <div className="relative p-8 rounded-2xl bg-white/80 dark:bg-white/10 backdrop-blur-xl border-2 border-primary-200/50 dark:border-primary-500/20 shadow-xl group-hover:border-primary-400 dark:group-hover:border-primary-400 transition-all duration-500">`
);

// Update Features cards
content = content.replace(
  /className=\{`relative p-6 rounded-2xl bg-white\/70 dark:bg-white\/5 backdrop-blur-xl border transition-all duration-300 \$\{[\s\S]*?`\}/g,
  `className={\`relative p-6 rounded-2xl bg-white/80 dark:bg-white/10 backdrop-blur-xl border-2 transition-all duration-500 overflow-hidden \${
                  hoveredCard === i 
                    ? 'border-primary-400 dark:border-primary-400 shadow-2xl' 
                    : 'border-primary-200/50 dark:border-primary-500/20 shadow-xl'
                }\`}`
);

// Add the inner hover gradients to Features cards
content = content.replace(
  /className=\{`relative p-6 rounded-2xl bg-white\/80 dark:bg-white\/10 backdrop-blur-xl border-2 transition-all duration-500 overflow-hidden \$\{[\s\S]*?`\}>/g,
  `$&
                <div className="absolute inset-0 bg-gradient-to-br from-primary-400 to-primary-600 opacity-0 group-hover:opacity-10 transition-opacity duration-700" />
                <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-br from-primary-400 to-primary-600 opacity-0 group-hover:opacity-20 rounded-bl-2xl transition-all duration-500" />
                <div className="absolute bottom-0 left-0 w-16 h-16 bg-gradient-to-tr from-primary-400 to-primary-600 opacity-0 group-hover:opacity-20 rounded-tr-2xl transition-all duration-500" />
                <div className="relative z-10">`
);

// Close the relative z-10 div in Features
content = content.replace(
  /<\/div>\s*<\/motion\.div>/g,
  '</div></div></motion.div>'
);

// Update Team cards
content = content.replace(
  /<div className="relative p-6 rounded-2xl bg-white\/70 dark:bg-white\/5 backdrop-blur-xl border border-primary-200\/50 dark:border-primary-500\/20 shadow-lg hover:shadow-xl transition-all duration-300">/g,
  `<div className="absolute -inset-0.5 bg-gradient-to-r from-primary-500/30 to-primary-600/30 rounded-2xl blur opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                <div className="relative p-6 rounded-2xl bg-white/80 dark:bg-white/10 backdrop-blur-xl border-2 border-primary-200/50 dark:border-primary-500/20 shadow-xl group-hover:border-primary-400 dark:group-hover:border-primary-400 transition-all duration-500">`
);

fs.writeFileSync(filePath, content, 'utf8');
console.log('Fixed About.jsx card styling');
