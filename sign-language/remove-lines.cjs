const fs = require('fs');

function removeConnections(filePath) {
  let content = fs.readFileSync(filePath, 'utf8');

  // The code block to remove:
  /*
        particlesRef.current.forEach(otherParticle => {
          const dx = particle.x - otherParticle.x;
          const dy = particle.y - otherParticle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 100) {
            ctx.beginPath();
            ctx.strokeStyle = particle.color + '44';
            ctx.lineWidth = 0.6 * (1 - distance / 100);
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(otherParticle.x, otherParticle.y);
            ctx.stroke();
          }
        });
  */

  const removeRegex = /\s*particlesRef\.current\.forEach\(otherParticle => \{[\s\S]*?ctx\.stroke\(\);\s*\}\s*\}\);\s*/g;
  content = content.replace(removeRegex, '\n        ');

  fs.writeFileSync(filePath, content, 'utf8');
  console.log('Removed connecting lines from ' + filePath);
}

removeConnections('d:/GP/Frontend/sign-language/src/components/Features.jsx');
removeConnections('d:/GP/Frontend/sign-language/src/components/Guide.jsx');
