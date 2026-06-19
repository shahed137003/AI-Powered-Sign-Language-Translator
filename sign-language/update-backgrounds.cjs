const fs = require('fs');

const homeParticlesLogic = `    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const themeColorsMap = {
      purple: ['#A855F7', '#9333EA', '#7C3AED', '#6D28D9', '#8B5CF6'],
      'midnight-blue': ['#6366F1', '#4F46E5', '#4338CA', '#3730A3', '#818CF8'],
    };
    const currentThemeColors = themeColorsMap[themeColor || 'purple'];
    const colors = isDark ? currentThemeColors : currentThemeColors.slice().reverse();

    particlesRef.current = Array.from({ length: 120 }).map(() => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      size: Math.random() * 4 + 1,
      speedX: Math.random() * 0.5 - 0.25,
      speedY: Math.random() * 0.5 - 0.25,
      color: colors[Math.floor(Math.random() * colors.length)],
      opacity: Math.random() * 0.6 + 0.2,
      glow: Math.random() > 0.7,
    }));

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particlesRef.current.forEach(particle => {
        particle.x += particle.speedX;
        particle.y += particle.speedY;

        if (particle.x < -10) particle.x = canvas.width + 10;
        if (particle.x > canvas.width + 10) particle.x = -10;
        if (particle.y < -10) particle.y = canvas.height + 10;
        if (particle.y > canvas.height + 10) particle.y = -10;

        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        
        if (particle.glow) {
          const glowGradient = ctx.createRadialGradient(
            particle.x, particle.y, 0,
            particle.x, particle.y, particle.size * 4
          );
          glowGradient.addColorStop(0, particle.color + '99');
          glowGradient.addColorStop(1, particle.color + '00');
          ctx.fillStyle = glowGradient;
        } else {
          ctx.fillStyle = particle.color + Math.floor(particle.opacity * 255).toString(16).padStart(2, '0');
        }
        
        ctx.fill();

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
      });
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    const handleResize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };

    window.addEventListener('resize', handleResize);
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      window.removeEventListener('resize', handleResize);
    };`;

function fixFile(filePath) {
  let content = fs.readFileSync(filePath, 'utf8');

  // Replace grid inline styles
  const gridStyleOld1 = /rgba\(168, 85, 247, 0\.1\)/g;
  const gridStyleOld2 = /rgba\(139, 92, 246, 0\.1\)/g;
  content = content.replace(gridStyleOld1, '\${gridColor}');
  content = content.replace(gridStyleOld2, '\${gridColor}');

  // Inject gridColor variable if needed
  if (!content.includes('const gridColor =')) {
    content = content.replace(/(const \{ themeColor \} = useTheme\(\);)/, '$1\n  const gridColor = themeColor === "midnight-blue" ? "rgba(99, 102, 241, 0.1)" : "rgba(168, 85, 247, 0.1)";');
  }

  // Find the useEffect for particles and replace it
  const particleRegex = /const canvas = canvasRef\.current;[\s\S]*?(?=const |return \(\s*<div)/;
  content = content.replace(particleRegex, homeParticlesLogic + '\n\n  ');

  // Make sure to add `themeColor` to particle useEffect dependency if it's missing or just `[isDark]`
  content = content.replace(/\}, \[isDark\]\);/g, '}, [isDark, themeColor]);');

  fs.writeFileSync(filePath, content, 'utf8');
}

fixFile('d:/GP/Frontend/sign-language/src/components/Features.jsx');
fixFile('d:/GP/Frontend/sign-language/src/components/Guide.jsx');

console.log('Fixed backgrounds in Features and Guide to match Home exactly.');
