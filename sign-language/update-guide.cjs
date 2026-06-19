const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Guide.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// 1. Add missing import for useTheme
if (!content.includes('import { useTheme } from "../context/ThemeContext"')) {
  content = content.replace(/import React, \{ useState, useEffect, useRef \} from "react";/, `import React, { useState, useEffect, useRef } from "react";\nimport { useTheme } from "../context/ThemeContext";`);
}

// 2. Add themeColor hook inside the component
if (!content.includes('const { themeColor } = useTheme();')) {
  content = content.replace(/export default function HelpGuide\(\) \{/, `export default function HelpGuide() {\n  const { themeColor } = useTheme();`);
}

// 3. Fix the particle colors and dependency
const oldColors = `    const colors = isDark 
      ? ['#A855F7', '#9333EA', '#7C3AED', '#6D28D9', '#8B5CF6'] 
      : ['#8B5CF6', '#7C3AED', '#6D28D9', '#9333EA', '#A855F7'];`;

const newColors = `    const themeColorsMap = {
      purple: ['#A855F7', '#9333EA', '#7C3AED', '#6D28D9', '#8B5CF6'],
      'midnight-blue': ['#6366F1', '#4F46E5', '#4338CA', '#3730A3', '#818CF8'],
    };
    const currentThemeColors = themeColorsMap[themeColor || 'purple'];
    const colors = isDark ? currentThemeColors : currentThemeColors.slice().reverse();`;

if (content.includes(oldColors)) {
  content = content.replace(oldColors, newColors);
  content = content.replace(/  \}, \[isDark\]\);/g, `  }, [isDark, themeColor]);`);
}

// 4. Fix background classes
content = content.replace(/dark:from-\[#0a0518\] dark:via-\[#110a2e\] dark:to-\[#1e0f5c\]/g, 'dark:from-primary-bg-1 dark:via-primary-bg-2 dark:to-primary-bg-3');

// 5. Global replacement of purple with primary
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
console.log('Fixed Guide.jsx theme');
