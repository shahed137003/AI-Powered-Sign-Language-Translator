import React, { useEffect, useState } from "react";
import { FaHandBackFist, FaBars, FaXmark } from "react-icons/fa6"; 
import { FaSun, FaMoon, FaSignInAlt } from "react-icons/fa";
import { NavLink } from "react-router-dom";

export default function Navbar() {
  const [darkMode, setDarkMode] = useState(() => {
    // Initialize dark mode from localStorage or system preference
    if (localStorage.getItem("theme") === "dark" || 
        (!("theme" in localStorage) && window.matchMedia("(prefers-color-scheme: dark)").matches)) {
      return true;
    }
    return false;
  });
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
    // Set background of body for smooth transition
    document.body.classList.toggle('dark', darkMode);
  }, [darkMode]);

  const toggleDarkMode = () => setDarkMode(!darkMode);
  const toggleMobileMenu = () => setMobileMenuOpen(!mobileMenuOpen);

  const navItems = [
    { name: "Home", to: "/" },
    { name: "Translate", to: "/translate" },
    { name: "Profile", to: "/profile" },
  ];

  // Base classes for NavLinks
  const navLinkClass = "relative px-2 py-1 transition-all duration-300 group text-lg font-medium";
  
  // Custom active link underline style (neon glow)
  const activeStyles = "text-purple-600  font-semibold";
  const underlineClass = (isActive) => `
    absolute left-0 -bottom-1 h-[3px] rounded-full
    bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
    transition-all duration-300
    group-hover:w-full
    ${isActive ? "w-full shadow-lg shadow-purple-500/50" : "w-0"}
  `;

  return (
    // Fixed navbar with enhanced glassmorphism and deep dark background
    <div
      className="
        fixed top-0 left-0 w-full px-6 lg:px-16 py-4 flex items-center justify-between
        bg-white/50 dark:bg-[#0f0c29]/70 backdrop-blur-lg 
        border-b border-gray-200/40 
        dark:shadow-2xl dark:shadow-purple-900/40 z-50
        dark:border-purple-500/30 transition-colors duration-500 
        shadow-2xl shadow-purple-900/35 
      "
    >
      {/* Logo */}
      <NavLink
        to="/"
        className="
          flex items-center gap-3 text-3xl lg:text-4xl font-bold dark:from-[#6A3093] dark:to-[#A044FF]
          bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0]
          bg-clip-text text-transparent  italic 
          hover:scale-[1.02] transition-transform duration-300
        "
      >
        <FaHandBackFist className="text-4xl lg:text-5xl dark:text-[#6A3093] text-[#BF5AE0]  hover:rotate-6 transition-transform" />
        LinguaSign
      </NavLink>

      {/* Desktop Nav Links */}
      <div className="hidden md:flex items-center gap-10 text-gray-700 dark:text-gray-300">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => `${navLinkClass} ${isActive ? activeStyles : "hover:text-purple-500"}`}
          >
            {({ isActive }) => (
              <>
                {item.name}
                <span className={underlineClass(isActive)} />
              </>
            )}
          </NavLink>
        ))}
      </div>

      {/* Right Section (Login + Dark Mode) */}
      <div className="flex items-center gap-4 lg:gap-6">
        {/* Login Button */}
        <NavLink
          to="/login"
          className="
            hidden sm:flex items-center gap-2 
            bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF]
            text-white text-lg px-6 py-2 rounded-full
            shadow-lg shadow-purple-500/40 hover:shadow-2xl hover:scale-105 active:scale-95
            transition-all duration-300 font-semibold
          "
        >
          Login
          <FaSignInAlt className="text-xl" /> 
        </NavLink>

        {/* Dark Mode Switch */}
        <button
          onClick={toggleDarkMode}
          aria-label="Toggle Dark Mode"
          className="
            relative w-14 h-7 flex items-center rounded-full
            bg-gray-300 dark:bg-gray-700 p-0.5
            transition-all duration-300 shadow-inner
          "
        >
          <div
            className={`
              absolute w-6 h-6 rounded-full 
              flex items-center justify-center shadow-md
              transition-transform duration-300
              ${darkMode ? "translate-x-7 bg-gray-900" : "translate-x-0 bg-white"}
            `}
          >
            {darkMode ? (
              <FaSun className="text-yellow-400 text-base" />
            ) : (
              <FaMoon className="text-purple-600 text-base" />
            )}
          </div>
        </button>

        {/* Mobile Menu Button */}
        <button 
          onClick={toggleMobileMenu} 
          className="md:hidden text-2xl text-gray-700 dark:text-gray-300 p-2 rounded-full hover:bg-gray-200/50 dark:hover:bg-gray-700/50 transition"
          aria-label="Toggle navigation menu"
        >
          {mobileMenuOpen ? <FaXmark /> : <FaBars />}
        </button>
      </div>
      
      {/* Mobile Menu Drawer */}
      <div
        className={`
          absolute top-full left-0 w-full 
          bg-white/90 dark:bg-[#0f0c29]/95 backdrop-blur-lg 
          border-b border-gray-200 dark:border-purple-500/30
          transition-all duration-500 ease-in-out
          md:hidden
          ${mobileMenuOpen ? "max-h-80 opacity-100 py-4" : "max-h-0 opacity-0 overflow-hidden"}
        `}
      >
        <div className="flex flex-col items-start px-6 space-y-4">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={toggleMobileMenu} // Close menu on link click
              className={({ isActive }) => `
                w-full py-2 text-xl font-medium border-b border-gray-200 dark:border-gray-700
                ${isActive ? activeStyles : "text-gray-700 dark:text-gray-200"}
                hover:text-purple-500  transition-colors
              `}
            >
              {item.name}
            </NavLink>
          ))}
     <NavLink
  to="/login"
  onClick={toggleMobileMenu}
  className="
    relative /* Essential: Position inner div */
    overflow-hidden /* Essential: Clip sliding effect */
    group /* Essential: Enable group-hover */
    w-full py-3 mt-2 text-center text-xl font-semibold 
    bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF] text-white rounded-full 
    shadow-md transition-all /* Use transition-all instead of opacity */
    focus:outline-none focus:ring-4 focus:ring-purple-500/50
  "
>
  {/* Ensure text is above the sliding overlay */}
  <span className="relative z-10">
    Login
  </span>
  
  {/* The glass overlay effect */}
  <div 
    className="
      absolute top-0 left-0 w-full h-full 
      bg-white/20 
      translate-y-full /* Start hidden below */
      group-hover:translate-y-0 /* Slide up on hover */
      transition-transform duration-300
      z-0
      rounded-full 
    "
  />
</NavLink>
        </div>
      </div>
    </div>
  );
}