const fs = require('fs');

let cssContent = fs.readFileSync('d:/GP/Frontend/sign-language/src/index.css', 'utf8');

const regexBg1 = /--theme-primary-bg-1:\s*#0B1120;/g;
const regexBg2 = /--theme-primary-bg-2:\s*#1e1b4b;/g;
const regexBg3 = /--theme-primary-bg-3:\s*#312e81;/g;
const regexBg4 = /--theme-primary-bg-4:\s*#0B1120;/g;
const regexBg5 = /--theme-primary-bg-5:\s*#1e1b4b;/g;

// Only replace inside the midnight-blue block.
// Since these exact hexes are mostly just in midnight-blue, but to be safe:

let midnightBlockMatch = cssContent.match(/html\[data-theme="midnight-blue"\] \{[\s\S]*?\n\}/);
if (midnightBlockMatch) {
  let block = midnightBlockMatch[0];
  block = block.replace(/--theme-primary-bg-1:\s*#[0-9a-fA-F]+;/, '--theme-primary-bg-1: #02040a;');
  block = block.replace(/--theme-primary-bg-2:\s*#[0-9a-fA-F]+;/, '--theme-primary-bg-2: #0f0d2a;');
  block = block.replace(/--theme-primary-bg-3:\s*#[0-9a-fA-F]+;/, '--theme-primary-bg-3: #1c1955;');
  block = block.replace(/--theme-primary-bg-4:\s*#[0-9a-fA-F]+;/, '--theme-primary-bg-4: #02040a;');
  block = block.replace(/--theme-primary-bg-5:\s*#[0-9a-fA-F]+;/, '--theme-primary-bg-5: #0f0d2a;');
  
  cssContent = cssContent.replace(midnightBlockMatch[0], block);
  fs.writeFileSync('d:/GP/Frontend/sign-language/src/index.css', cssContent, 'utf8');
  console.log("Darkened midnight-blue backgrounds.");
} else {
  console.log("Could not find midnight-blue block.");
}
