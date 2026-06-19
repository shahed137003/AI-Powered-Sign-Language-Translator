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
  content = content.replace('import React, { useState, useEffect, useRef } from "react";', 'import React, { useState, useEffect, useRef, useContext, useCallback, useMemo } from "react";');
  fs.writeFileSync(file, content, 'utf8');
});
console.log('Hooks added.');
