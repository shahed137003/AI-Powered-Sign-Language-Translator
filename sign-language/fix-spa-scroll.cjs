const fs = require('fs');

// 1. Add id="features" to Features.jsx
const featuresPath = 'd:/GP/Frontend/sign-language/src/components/Features.jsx';
let featuresCode = fs.readFileSync(featuresPath, 'utf8');
featuresCode = featuresCode.replace(
  /return \(\s*<div\s+className="/,
  'return (\n      <div id="features"\n        className="'
);
fs.writeFileSync(featuresPath, featuresCode);

// 2. Add id="guide" to Guide.jsx
const guidePath = 'd:/GP/Frontend/sign-language/src/components/Guide.jsx';
let guideCode = fs.readFileSync(guidePath, 'utf8');
guideCode = guideCode.replace(
  /return \(\s*<div className="relative w-full min-h-screen/,
  'return (\n      <div id="guide" className="relative w-full min-h-screen'
);
fs.writeFileSync(guidePath, guideCode);

// 3. Add Help Guide to Footer.jsx
const footerPath = 'd:/GP/Frontend/sign-language/src/components/Footer.jsx';
let footerCode = fs.readFileSync(footerPath, 'utf8');
footerCode = footerCode.replace(
  /\{ name: "Features", href: "\/#features", icon: <BsLightningFill className="text-sm" \/> \},/,
  '{ name: "Features", href: "/#features", icon: <BsLightningFill className="text-sm" /> },\n      { name: "Help Guide", href: "/#guide", icon: <TbMessageChatbot className="text-sm" /> },'
);
fs.writeFileSync(footerPath, footerCode);

// 4. Add Scroll Logic to App.jsx
const appPath = 'd:/GP/Frontend/sign-language/src/App.jsx';
let appCode = fs.readFileSync(appPath, 'utf8');

// import useEffect and useLocation
if (!appCode.includes('useLocation')) {
  appCode = appCode.replace(
    /import \{ Routes, Route \} from "react-router-dom";/,
    'import { Routes, Route, useLocation } from "react-router-dom";\nimport { useEffect } from "react";'
  );
}

// Add scroll effect inside App()
if (!appCode.includes('const location = useLocation()')) {
  appCode = appCode.replace(
    /function App\(\) \{\n\s*return \(/,
    `function App() {
  const location = useLocation();

  useEffect(() => {
    if (location.hash) {
      const id = location.hash.substring(1);
      setTimeout(() => {
        const el = document.getElementById(id);
        if (el) {
          el.scrollIntoView({ behavior: 'smooth' });
        }
      }, 100);
    } else {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, [location]);

  return (`
  );
}
fs.writeFileSync(appPath, appCode);

console.log('Successfully fixed all SPA scroll issues and added Guide link to Footer');
