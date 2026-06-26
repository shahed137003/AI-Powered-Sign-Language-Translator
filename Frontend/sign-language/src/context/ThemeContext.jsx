import React, { createContext, useContext, useEffect, useState } from "react";

const ThemeContext = createContext();

export const ThemeProvider = ({ children }) => {
  const [themeColor, setThemeColor] = useState(() => {
    return localStorage.getItem("themeColor") || "purple";
  });

  useEffect(() => {
    const root = document.documentElement;
    if (themeColor === "purple") {
      root.removeAttribute("data-theme");
    } else {
      root.setAttribute("data-theme", themeColor);
    }
    localStorage.setItem("themeColor", themeColor);
  }, [themeColor]);

  return (
    <ThemeContext.Provider value={{ themeColor, setThemeColor }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => useContext(ThemeContext);
