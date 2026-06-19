const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Contact.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// 1. Replace hardcoded purple with primary
content = content.replace(/purple-500/g, 'primary-500');
content = content.replace(/purple-600/g, 'primary-600');
content = content.replace(/purple-400/g, 'primary-400');
content = content.replace(/purple-700/g, 'primary-700');
content = content.replace(/purple-300/g, 'primary-300');
content = content.replace(/purple-200/g, 'primary-200');
content = content.replace(/purple-50/g, 'primary-50');

// 2. Fix the dark mode background in the wrapper div
content = content.replace(/dark:from-\[#0a0518\] dark:via-\[#110a2e\] dark:to-\[#1e0f5c\]/g, 'dark:from-primary-bg-1 dark:via-primary-bg-2 dark:to-primary-bg-3');

// 3. Update the contactInfo array to include colors
const oldContactInfo = `  const contactInfo = [
    { icon: FaEnvelope, title: "Email Support", value: "support@linguasign.io" },
    { icon: FaPhone, title: "Phone Support", value: "+1 (555) 123-4567" },
    { icon: FaMapMarkerAlt, title: "Headquarters", value: "San Francisco, CA" },
    { icon: FaRegClock, title: "Business Hours", value: "Mon-Fri, 9AM-6PM PST" },
  ];`;

const newContactInfo = `  const contactInfo = [
    { icon: FaEnvelope, title: "Email Support", value: "support@linguasign.io", color: "from-primary-400 to-primary-600" },
    { icon: FaPhone, title: "Phone Support", value: "+1 (555) 123-4567", color: "from-primary-500 to-primary-700" },
    { icon: FaMapMarkerAlt, title: "Headquarters", value: "San Francisco, CA", color: "from-primary-600 to-primary-800" },
    { icon: FaRegClock, title: "Business Hours", value: "Mon-Fri, 9AM-6PM PST", color: "from-primary-700 to-primary-900" },
  ];`;

content = content.replace(oldContactInfo, newContactInfo);

// 4. Update the contactInfo mapping section to match Login.jsx layout
const oldMapRegex = /\{\s*contactInfo\.map\(\(info, index\) => \([\s\S]*?<\/motion\.div>\s*\)\)\s*\}/;
const newMap = `{contactInfo.map((info, index) => (
                  <motion.div
                    key={index}
                    variants={scaleIn}
                    whileHover={{ y: -5, scale: 1.05 }}
                    className="group relative cursor-pointer h-full"
                  >
                    <div className={\`absolute -inset-0.5 bg-gradient-to-br \${info.color} rounded-3xl blur opacity-0 group-hover:opacity-60 transition-opacity duration-500\`} />
                    <div className="relative h-full p-4 sm:p-5 flex flex-col items-center text-center justify-center rounded-3xl bg-white/70 dark:bg-white/5 backdrop-blur-xl border border-primary-200/50 dark:border-primary-500/20 shadow-lg shadow-primary-100/20 dark:shadow-primary-900/20 group-hover:bg-white/90 dark:group-hover:bg-white/10 transition-colors duration-300 overflow-hidden">
                      <info.icon className="absolute -bottom-4 -right-4 text-7xl sm:text-8xl text-gray-400 dark:text-gray-500 opacity-[0.06] group-hover:opacity-[0.12] group-hover:scale-110 group-hover:-rotate-12 transition-all duration-500 pointer-events-none" />
                      
                      <div className={\`p-3 sm:p-4 rounded-2xl bg-gradient-to-br \${info.color} shadow-lg mb-3 group-hover:-translate-y-1 transition-transform duration-300\`}>
                        <info.icon className="text-2xl sm:text-3xl text-white drop-shadow-md group-hover:scale-110 transition-transform duration-300" />
                      </div>
                      <h3 className="font-bold text-gray-900 dark:text-white text-sm mb-1 z-10">{info.title}</h3>
                      <p className="text-gray-600 dark:text-gray-300 text-xs font-medium z-10">{info.value}</p>
                    </div>
                  </motion.div>
                ))}`;

content = content.replace(oldMapRegex, newMap);

fs.writeFileSync(filePath, content, 'utf8');
console.log('Successfully updated Contact.jsx');
