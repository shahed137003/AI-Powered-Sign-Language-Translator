import React, { useEffect, useState } from "react";
import { FaHandBackFist, FaBars, FaXmark } from "react-icons/fa6";
import { FaSun, FaMoon, FaSignInAlt, FaSignOutAlt, FaUser } from "react-icons/fa";
import { NavLink, useNavigate } from "react-router-dom";
import {useAuth} from "../context/useAuth";
 
export default function Navbar() {
  const [darkMode, setDarkMode] = useState(() => {
    if (
      localStorage.getItem("theme") === "dark" ||
      (!("theme" in localStorage) &&
        window.matchMedia("(prefers-color-scheme: dark)").matches)
    ) {
      return true;
    }
    return false;
  });

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // ✅ GET AUTH INFO
  const { user, logout, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  // Apply dark mode to document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
    document.body.classList.toggle("dark", darkMode);
  }, [darkMode]);

  const toggleDarkMode = () => setDarkMode(!darkMode);
  const toggleMobileMenu = () => setMobileMenuOpen(!mobileMenuOpen);

  // ✅ LOGOUT FUNCTION
  const handleLogout = () => {
    logout();
    navigate('/');
    if (mobileMenuOpen) setMobileMenuOpen(false);
  };

  const navItems = [
    { name: "Home", to: "/" },
    { name: "Translate", to: "/translate" },
    { name: "Profile", to: "/profile" },
    { name: "Contact Us", to: "/contactus" },
  ];

  // NavLink Base
  const navLinkClass =
    "relative px-2 py-1 transition-all duration-300 group text-lg font-medium";

  // Underline animation
  const underlineClass = (isActive) =>
    `absolute left-0 -bottom-1 h-[3px] rounded-full bg-gradient-to-r 
     from-[#6A3093] via-[#A044FF] to-[#BF5AE0] transition-all duration-300 
     group-hover:w-full ${isActive ? "w-full shadow-lg shadow-purple-500/50" : "w-0"}`;

  return (
    <div
      className="
      fixed top-0 left-0 w-full px-6 lg:px-16 py-4 flex items-center justify-between
      bg-white/50 dark:bg-[#0f0c29]/70 backdrop-blur-lg 
      border-b border-gray-200/40 dark:border-purple-500/30
      shadow-2xl dark:shadow-purple-900/40
      transition-colors duration-500 z-50"
    >
      {/* Logo */}
      <NavLink
        to="/"
        className="
        flex items-center gap-3 text-3xl lg:text-4xl font-bold italic
        bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF]
        bg-clip-text text-transparent hover:scale-[1.02] transition-transform duration-300"
      >
        <FaHandBackFist className="text-4xl lg:text-5xl text-[#BF5AE0] dark:text-[#6A3093] hover:rotate-6 transition-transform" />
        LinguaSign
      </NavLink>

      {/* Desktop Nav */}
      <div className="hidden md:flex items-center gap-10 text-gray-700 dark:text-gray-300">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `${navLinkClass} ${isActive ? "text-purple-600 font-semibold" : "hover:text-purple-500"}`
            }
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

      {/* Right side buttons */}
<div className="flex items-center gap-4 lg:gap-6">
  {/* ✅ SHOW USER INFO IF LOGGED IN */}
  {isAuthenticated ? (
    <>
      <div className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-full bg-white/80 dark:bg-white/10 border border-white/50 dark:border-white/20 backdrop-blur-sm">
        <div className="w-8 h-8 rounded-full bg-gradient-to-r from-[#6A3093] to-[#A044FF] flex items-center justify-center">
          <FaUser className="text-white text-sm" />
        </div>
        <span className="text-sm font-medium text-gray-700 dark:text-white">
          {user?.email?.split('@')[0] || 'User'}
        </span>
      </div>
      
      {/* Logout Button - Updated Design */}
      <button
        onClick={handleLogout}
        className="
          hidden sm:flex items-center gap-2
          bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF]
          text-white text-sm font-medium px-5 py-2.5 rounded-xl
          shadow-lg shadow-purple-500/30 dark:shadow-purple-900/40 
          hover:shadow-xl hover:shadow-purple-500/40 dark:hover:shadow-purple-900/60
          hover:scale-[1.02] active:scale-[0.98]
          transition-all duration-300 group relative overflow-hidden"
      >
        {/* Shimmer effect overlay */}
        <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></span>
        
        <FaSignOutAlt className="text-base group-hover:rotate-180 transition-transform duration-300" />
        <span>Logout</span>
      </button>
    </>
  ) : (
    /* ✅ SHOW LOGIN BUTTON IF NOT LOGGED IN */
    <NavLink
      to="/login"
      className="
        hidden sm:flex items-center gap-2
        bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF]
        text-white text-sm font-medium px-5 py-2.5 rounded-xl
        shadow-lg shadow-purple-500/30 dark:shadow-purple-900/40
        hover:shadow-xl hover:shadow-purple-500/40 dark:hover:shadow-purple-900/60
        hover:scale-[1.02] active:scale-[0.98]
        transition-all duration-300 group relative overflow-hidden"
    >
      {/* Shimmer effect overlay */}
      <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></span>
      
      <FaSignInAlt className="text-base group-hover:translate-x-1 transition-transform duration-300" />
      <span>Login</span>
    </NavLink>
  )}

  {/* Dark Mode Switch */}
  <button
    onClick={toggleDarkMode}
    aria-label="Toggle Dark Mode"
    className="
      relative w-14 h-7 flex items-center rounded-full
      bg-gradient-to-r from-gray-300 to-gray-400 dark:from-gray-700 dark:to-gray-800 
      p-0.5 transition-all duration-300 shadow-inner"
  >
    <div
      className={`
        absolute w-6 h-6 rounded-full flex items-center justify-center
        shadow-lg transition-all duration-300
        ${darkMode ? "translate-x-7 bg-gradient-to-r from-yellow-400 to-orange-400" : "translate-x-0 bg-gradient-to-r from-white to-gray-100"}
      `}
    >
      {darkMode ? (
        <FaSun className="text-yellow-600 text-sm" />
      ) : (
        <FaMoon className="text-purple-600 text-sm" />
      )}
    </div>
  </button>

  {/* Mobile Menu Button */}
  <button
    onClick={toggleMobileMenu}
    className="md:hidden text-2xl text-gray-700 dark:text-gray-300 p-2 rounded-xl bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm hover:bg-white/80 dark:hover:bg-gray-700/80 transition-all duration-300 border border-gray-200/50 dark:border-gray-700/50"
  >
    {mobileMenuOpen ? <FaXmark /> : <FaBars />}
  </button>
</div>

{/* Mobile Drawer */}
<div
  className={`
    absolute top-full left-0 w-full bg-white/95 dark:bg-[#0f0c29]/98
    backdrop-blur-xl border-b border-gray-200/50 dark:border-purple-500/30
    shadow-xl dark:shadow-purple-900/20
    transition-all duration-500 ease-in-out md:hidden
    ${mobileMenuOpen ? "max-h-80 opacity-100 py-6" : "max-h-0 opacity-0 overflow-hidden"}
  `}
>
  <div className="flex flex-col items-start px-8 space-y-5">
    {navItems.map((item) => (
      <NavLink
        key={item.to}
        to={item.to}
        onClick={toggleMobileMenu}
        className={({ isActive }) =>
          `w-full py-3 text-lg font-medium border-b border-gray-100 dark:border-gray-800 
          transition-all duration-300 ${isActive ? "text-purple-600 dark:text-purple-400 font-semibold" : "text-gray-700 dark:text-gray-200"}
          hover:text-purple-500 dark:hover:text-purple-300 hover:translate-x-2`
        }
      >
        {item.name}
      </NavLink>
    ))}

    {/* ✅ MOBILE AUTH BUTTONS - UPDATED */}
    {isAuthenticated ? (
      <>
        <div className="w-full py-3 flex items-center gap-3 text-lg font-medium border-b border-gray-100 dark:border-gray-800 text-gray-700 dark:text-gray-200">
          <div className="w-10 h-10 rounded-full bg-gradient-to-r from-[#6A3093] to-[#A044FF] flex items-center justify-center">
            <FaUser className="text-white" />
          </div>
          <div>
            <p className="font-semibold">{user?.email?.split('@')[0] || 'User'}</p>
            <p className="text-sm text-gray-500 dark:text-gray-400">{user?.email}</p>
          </div>
        </div>
        
        <button
          onClick={handleLogout}
          className="
            w-full py-3.5 mt-2 text-center text-lg font-semibold
            bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF]
            text-white rounded-xl shadow-lg shadow-purple-500/30 dark:shadow-purple-900/40
            transition-all duration-300 hover:shadow-xl hover:shadow-purple-500/40 
            active:scale-[0.98] flex items-center justify-center gap-3"
        >
          <FaSignOutAlt className="text-xl" />
          Logout
        </button>
      </>
    ) : (
      <NavLink
        to="/login"
        onClick={toggleMobileMenu}
        className="
          w-full py-3.5 mt-2 text-center text-lg font-semibold
          bg-gradient-to-r from-[#6A3093] via-[#A044FF] to-[#BF5AE0] dark:from-[#6A3093] dark:to-[#A044FF]
          text-white rounded-xl shadow-lg shadow-purple-500/30 dark:shadow-purple-900/40
          transition-all duration-300 hover:shadow-xl hover:shadow-purple-500/40 
          active:scale-[0.98] flex items-center justify-center gap-3"
      >
        <FaSignInAlt className="text-xl" />
        Login
      </NavLink>
    )}
  </div>
</div>
    </div>
  );
}
