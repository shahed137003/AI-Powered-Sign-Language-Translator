const fs = require('fs');

let content = fs.readFileSync('d:/GP/Frontend/sign-language/src/components/Contact.jsx', 'utf8');

const searchRegex = /\{\/\* Contact Information Cards \*\/\}[\s\S]*?\{\/\* Social Links \*\/\}/;

const replacement = `{/* Contact Information Cards */}
          <div className="lg:col-span-1 space-y-8 flex flex-col">
            <div className="grid grid-cols-2 gap-4">
              {contactInfo.map((info, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  whileHover={{ y: -5, scale: 1.05 }}
                  className="group relative cursor-pointer h-full"
                >
                  <div className={\`absolute -inset-0.5 bg-gradient-to-br \${info.color} rounded-3xl blur opacity-0 group-hover:opacity-60 transition-opacity duration-500\`} />
                  
                  <div className="relative h-full p-4 sm:p-5 flex flex-col items-center text-center justify-center rounded-3xl bg-white/70 dark:bg-white/5 backdrop-blur-xl border border-primary-200/50 dark:border-primary-500/20 shadow-lg shadow-primary-100/20 dark:shadow-primary-900/20 group-hover:bg-white/90 dark:group-hover:bg-white/10 transition-colors duration-300 overflow-hidden">
                    
                    {/* Big background icon */}
                    <info.icon className="absolute -bottom-4 -right-4 text-7xl sm:text-8xl text-gray-400 dark:text-gray-500 opacity-[0.06] group-hover:opacity-[0.12] group-hover:scale-110 group-hover:-rotate-12 transition-all duration-500 pointer-events-none" />
                    
                    <div className={\`p-3 sm:p-4 rounded-2xl bg-gradient-to-br \${info.color} shadow-lg mb-3 group-hover:-translate-y-1 transition-transform duration-300\`}>
                      <info.icon className="text-2xl sm:text-3xl text-white drop-shadow-md group-hover:scale-110 transition-transform duration-300" />
                    </div>
                    
                    <h3 className="font-bold text-gray-900 dark:text-white mb-1 text-xs sm:text-sm z-10">{info.title}</h3>
                    <p className="text-gray-600 dark:text-gray-300 text-[10px] sm:text-xs font-medium z-10 break-words w-full">{info.value}</p>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Social Links */}`;

if (content.match(searchRegex)) {
  content = content.replace(searchRegex, replacement);
  fs.writeFileSync('d:/GP/Frontend/sign-language/src/components/Contact.jsx', content, 'utf8');
  console.log('Cards replaced successfully.');
} else {
  console.log('Could not find target block to replace.');
}
