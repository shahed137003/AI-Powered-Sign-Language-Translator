import React, { useEffect, useState } from "react";
import { FaHandBackFist } from "react-icons/fa6";
import { IoMdLogIn } from "react-icons/io";
import { FaSun, FaMoon } from "react-icons/fa";
import { Link ,Navigate } from "react-router-dom";

export default function Navbar() {
  const [darkMode, setDarkMode] = useState(
    localStorage.getItem("theme") === "dark"
  );
//  const nav=Navigate();
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [darkMode]);

  return (
    <div
      className="
        fixed top-0 left-0 w-full px-8 lg:px-16 py-4 flex items-center justify-between
        bg-white/40 dark:bg-gray-900/50 backdrop-blur-xl 
        border-b border-gray-200/30 
        shadow-md z-50
        dark:border-b dark:border-[#6A3093]/60
      "
    >
      {/* Logo */}
      <Link
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
      </Link>

      {/* Nav Links */}
      <div className="hidden md:flex items-center gap-10 text-lg font-medium dark:text-gray-200 text-gray-700">
        <Link
          to="/"
          className="relative group px-2 py-1 hover:text-[#6A3093] dark:hover:text-[#BF5AE0] transition-colors duration-300"
        >
          Home
          <span
            className="
              absolute left-0 -bottom-1 w-0 h-0.5 
              bg-gradient-to-r from-[#6A3093] to-[#BF5AE0] 
              group-hover:w-full transition-all duration-300
            "
          ></span>
        </Link>

        <Link
          to="/translate"
          className="relative group px-2 py-1 hover:text-[#6A3093] dark:hover:text-[#BF5AE0] transition-colors duration-300"
        >
          Translate
          <span
            className="
              absolute left-0 -bottom-1 w-0 h-0.5 
              bg-gradient-to-r from-[#6A3093] to-[#BF5AE0] 
              group-hover:w-full transition-all duration-300
            "
          ></span>
        </Link>

        <Link
          to="/profile"
          className="relative group px-2 py-1 hover:text-[#6A3093] dark:hover:text-[#BF5AE0] transition-colors duration-300"
        >
          Profile
          <span
            className="
              absolute left-0 -bottom-1 w-0 h-0.5 
              bg-gradient-to-r from-[#6A3093] to-[#BF5AE0] 
              group-hover:w-full transition-all duration-300
            "
          ></span>
        </Link>
      </div>

      {/* Right Section */}
      <div className="flex items-center gap-6">
        {/* Login Button */}
        <Link
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
        </Link>

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

      {/* Mobile Menu Icon */}
      <div className="md:hidden text-3xl text-gray-700 dark:text-gray-300 cursor-pointer">
        ☰
      </div>
    </div>
  );
}
