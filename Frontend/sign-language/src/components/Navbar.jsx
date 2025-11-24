import React, { useEffect, useState } from "react";
import { FaHandBackFist } from "react-icons/fa6";
import { IoMdLogIn } from "react-icons/io";
import { FaSun, FaMoon } from "react-icons/fa";
import { NavLink } from "react-router-dom";

export default function Navbar() {
  const [darkMode, setDarkMode] = useState(
    localStorage.getItem("theme") === "dark"
  );

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [darkMode]);

  const navLinkClass =
    "relative px-2 py-1 transition-colors duration-300 group";

  return (
    <div
      className="
        fixed top-0 left-0 w-full px-8 lg:px-16 py-4 flex items-center justify-between
        bg-white/40 dark:bg-gray-900/50 backdrop-blur-xl 
        border-b border-gray-200/30 
        shadow-md z-50
        dark:border-[#6A3093]/60
      "
    >
      {/* Logo */}
      <NavLink
        to="/"
        className="
          flex items-center gap-3 text-3xl lg:text-4xl font-serif 
          bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] 
          dark:from-[#6A3093] dark:to-[#A044FF]
          bg-clip-text text-transparent font-bold 
          hover:scale-105 transition-transform duration-300
        "
      >
        <FaHandBackFist className="text-4xl lg:text-5xl text-[#BF5AE0] dark:text-[#6A3093]" />
        LinguaSign
      </NavLink>

      {/* Nav Links */}
      <div className="hidden md:flex items-center gap-10 text-lg font-medium dark:text-gray-200 text-gray-700">

        {/* HOME */}
   <NavLink
  to="/"
  className={({ isActive }) =>
    `${navLinkClass} ${isActive ? "text-[#6A3093] dark:text-[#BF5AE0]" : ""}`
  }
>
  {({ isActive }) => (
    <>
      Home
      <span
        className={`
          absolute left-0 -bottom-1 h-0.5 
          bg-gradient-to-r from-[#6A3093] to-[#BF5AE0]
          transition-all duration-300
          group-hover:w-full
          ${isActive ? "w-full" : "w-0"}
        `}
      />
    </>
  )}
</NavLink>


        {/* TRANSLATE */}
     {/* TRANSLATE */}
<NavLink
  to="/translate"
  className={({ isActive }) =>
    `${navLinkClass} ${
      isActive ? "text-[#6A3093] dark:text-[#BF5AE0]" : ""
    }`
  }
>
  {({ isActive }) => (
    <>
      Translate
      <span
        className={`
          absolute left-0 -bottom-1 h-0.5 
          bg-gradient-to-r from-[#6A3093] to-[#BF5AE0]
          transition-all duration-300
          group-hover:w-full
          ${isActive ? "w-full" : "w-0"}
        `}
      ></span>
    </>
  )}
</NavLink>


        {/* PROFILE */}
{/* PROFILE */}
<NavLink
  to="/profile"
  className={({ isActive }) =>
    `${navLinkClass} ${
      isActive ? "text-[#6A3093] dark:text-[#BF5AE0]" : ""
    }`
  }
>
  {({ isActive }) => (
    <>
      Profile
      <span
        className={`
          absolute left-0 -bottom-1 h-0.5 
          bg-gradient-to-r from-[#6A3093] to-[#BF5AE0]
          transition-all duration-300
          group-hover:w-full
          ${isActive ? "w-full" : "w-0"}
        `}
      ></span>
    </>
  )}
</NavLink>


      </div>

      {/* Right Section */}
      <div className="flex items-center gap-6">
        {/* Login Button */}
        <NavLink
          to="/login"
          className="
            flex items-center gap-2 
            bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] 
            dark:from-[#6A3093] dark:to-[#A044FF]
            text-white text-lg px-6 py-2 rounded-xl
            shadow-lg hover:shadow-2xl hover:scale-105 active:scale-95
            transition-all duration-300
          "
        >
          Login
          <IoMdLogIn className="text-2xl" />
        </NavLink>

        {/* Dark Mode Switch */}
        <button
          onClick={() => setDarkMode(!darkMode)}
          className="
            relative w-14 h-7 flex items-center rounded-full
            bg-gray-200 dark:bg-gray-700
            transition-all duration-300 shadow-inner
          "
        >
          <div
            className={`
              absolute w-6 h-6 rounded-full bg-white dark:bg-gray-900
              flex items-center justify-center shadow-md
              transition-transform duration-300
              ${darkMode ? "translate-x-7" : "translate-x-1"}
            `}
          >
            {darkMode ? (
              <FaSun className="text-yellow-400 text-lg" />
            ) : (
              <FaMoon className="text-[#6A3093] text-lg" />
            )}
          </div>
        </button>
      </div>

      {/* Mobile Menu */}
      <div className="md:hidden text-3xl text-gray-700 dark:text-gray-300 cursor-pointer">
        ☰
      </div>
    </div>
  );
}
