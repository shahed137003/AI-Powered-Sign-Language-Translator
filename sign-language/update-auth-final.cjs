const fs = require('fs');

const loginReplacement = `            {/* Premium Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                {
                  icon: BsRobot,
                  title: "AI Translation Ready",
                  description: "Access real-time sign language translation with advanced AI",
                  color: "from-primary-400 to-primary-600",
                  delay: 0.1
                },
                {
                  icon: BsLightningFill,
                  title: "Instant Access",
                  description: "Get started immediately with all premium features",
                  color: "from-primary-500 to-primary-700",
                  delay: 0.2
                },
                {
                  icon: FaShieldAlt,
                  title: "Secure Session",
                  description: "Encrypted connection with enterprise-grade security",
                  color: "from-primary-600 to-primary-800",
                  delay: 0.3
                },
                {
                  icon: BsStars,
                  title: "Sync Across Devices",
                  description: "Your preferences and history saved in the cloud",
                  color: "from-primary-700 to-primary-900",
                  delay: 0.4
                }
              ].map((feature, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: feature.delay }}
                  whileHover={{ y: -5, scale: 1.05 }}
                  className="group relative cursor-pointer h-full"
                >
                  <div className={\`absolute -inset-0.5 bg-gradient-to-br \${feature.color} rounded-3xl blur opacity-0 group-hover:opacity-60 transition-opacity duration-500\`} />
                  
                  <div className="relative h-full p-4 flex flex-col items-center text-center justify-center rounded-3xl bg-white/70 dark:bg-white/5 backdrop-blur-xl border border-primary-200/50 dark:border-primary-500/20 shadow-lg shadow-primary-100/20 dark:shadow-primary-900/20 group-hover:bg-white/90 dark:group-hover:bg-white/10 transition-colors duration-300 overflow-hidden">
                    <feature.icon className="absolute -bottom-4 -right-4 text-7xl text-gray-400 dark:text-gray-500 opacity-[0.06] group-hover:opacity-[0.12] group-hover:scale-110 group-hover:-rotate-12 transition-all duration-500 pointer-events-none" />
                    <div className={\`p-3 rounded-2xl bg-gradient-to-br \${feature.color} shadow-lg mb-3 group-hover:-translate-y-1 transition-transform duration-300\`}>
                      <feature.icon className="text-2xl text-white drop-shadow-md group-hover:scale-110 transition-transform duration-300" />
                    </div>
                    <h3 className="font-bold text-gray-900 dark:text-white mb-1 text-sm z-10">{feature.title}</h3>
                    <p className="text-gray-600 dark:text-gray-300 text-xs font-medium z-10 break-words w-full">{feature.description}</p>
                  </div>
                </motion.div>
              ))}
            </div>`;

const registerReplacement = `            {/* Premium Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                {
                  icon: GiArtificialIntelligence,
                  title: "AI-Powered Translation",
                  description: "Advanced neural networks with 99% gesture recognition accuracy",
                  color: "from-primary-400 to-primary-600",
                  delay: 0.1
                },
                {
                  icon: BsLightningFill,
                  title: "Real-time Processing",
                  description: "Instant translation with sub-second latency",
                  color: "from-primary-500 to-primary-700",
                  delay: 0.2
                },
                {
                  icon: FaShieldAlt,
                  title: "Enterprise Security",
                  description: "End-to-end encryption and privacy-first design",
                  color: "from-primary-600 to-primary-800",
                  delay: 0.3
                },
                {
                  icon: BsStars,
                  title: "Premium Features",
                  description: "Access to all advanced tools and customizations",
                  color: "from-primary-700 to-primary-900",
                  delay: 0.4
                }
              ].map((feature, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: feature.delay }}
                  whileHover={{ y: -5, scale: 1.05 }}
                  className="group relative cursor-pointer h-full"
                >
                  <div className={\`absolute -inset-0.5 bg-gradient-to-br \${feature.color} rounded-3xl blur opacity-0 group-hover:opacity-60 transition-opacity duration-500\`} />
                  
                  <div className="relative h-full p-4 flex flex-col items-center text-center justify-center rounded-3xl bg-white/70 dark:bg-white/5 backdrop-blur-xl border border-primary-200/50 dark:border-primary-500/20 shadow-lg shadow-primary-100/20 dark:shadow-primary-900/20 group-hover:bg-white/90 dark:group-hover:bg-white/10 transition-colors duration-300 overflow-hidden">
                    <feature.icon className="absolute -bottom-4 -right-4 text-7xl text-gray-400 dark:text-gray-500 opacity-[0.06] group-hover:opacity-[0.12] group-hover:scale-110 group-hover:-rotate-12 transition-all duration-500 pointer-events-none" />
                    <div className={\`p-3 rounded-2xl bg-gradient-to-br \${feature.color} shadow-lg mb-3 group-hover:-translate-y-1 transition-transform duration-300\`}>
                      <feature.icon className="text-2xl text-white drop-shadow-md group-hover:scale-110 transition-transform duration-300" />
                    </div>
                    <h3 className="font-bold text-gray-900 dark:text-white mb-1 text-sm z-10">{feature.title}</h3>
                    <p className="text-gray-600 dark:text-gray-300 text-xs font-medium z-10 break-words w-full">{feature.description}</p>
                  </div>
                </motion.div>
              ))}
            </div>`;


function processFile(filePath, isLogin) {
  let content = fs.readFileSync(filePath, 'utf8');
  let startIndex = content.indexOf('{/* Premium Feature Cards */}');
  let endIndex = isLogin ? content.indexOf('{/* Quick Stats */}') : content.indexOf('{/* Trust Badges */}');
  
  if (startIndex !== -1 && endIndex !== -1) {
    let rep = isLogin ? loginReplacement : registerReplacement;
    let newContent = content.substring(0, startIndex) + rep + "\n\n            " + content.substring(endIndex);
    fs.writeFileSync(filePath, newContent, 'utf8');
    console.log("Updated " + filePath);
  } else {
    console.log("Could not find markers in " + filePath);
  }
}

processFile('d:/GP/Frontend/sign-language/src/components/Login.jsx', true);
processFile('d:/GP/Frontend/sign-language/src/components/Register.jsx', false);
