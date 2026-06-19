const fs = require('fs');

const filePath = 'd:/GP/Frontend/sign-language/src/components/Guide.jsx';
let content = fs.readFileSync(filePath, 'utf8');

// 1. Import useNavigate if it doesn't exist
if (!content.includes('useNavigate')) {
  if (content.includes('react-router-dom')) {
    content = content.replace(/import \{.*\} from 'react-router-dom';/, function(match) {
      if (!match.includes('useNavigate')) {
        return match.replace('import {', 'import { useNavigate,');
      }
      return match;
    });
  } else {
    content = content.replace(/(import React.*?from "react";)/, '$1\nimport { useNavigate } from "react-router-dom";');
  }
}

// 2. Add navigate = useNavigate()
if (!content.includes('const navigate = useNavigate();')) {
  content = content.replace(/(export default function HelpGuide\(\) \{)/, '$1\n  const navigate = useNavigate();');
}

// 3. Replace onClick={() => window.location.href = "/translate"}
content = content.replace(/onClick=\{\(\) => window\.location\.href = "\/translate"\}/g, 'onClick={() => navigate("/translate")}');

fs.writeFileSync(filePath, content, 'utf8');
console.log('Fixed Guide.jsx navigation');
