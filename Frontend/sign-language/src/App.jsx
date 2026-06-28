import React from "react";

import Home from "./components/Home";
import Navbar from "./components/Navbar";
import Features from "./components/Features";
//import About from "./components/About";
import Contact from "./components/Contact";
import Footer from "./components/Footer";
import Chat from "./components/Chat";
import Translate from "./components/Translate";
import Profile from "./components/Profile";
import Login from "./components/Login";
import HelpGuide from "./components/Guide";
import Register from "./components/Register";
import HelloHand3D from "./components/HelloHand3D";
import { Routes, Route, useLocation } from "react-router-dom";
import { useEffect } from "react";
import Chatbot from "./components/Chatbot";
import ForgetPassword from "./components/ForgetPassword";
import ResetPassword from "./components/ResetPassword";
import ProtectedRoute from "./components/ProtectedRoute";
import Dashboard from "./components/Dashboard";
function App() {
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

  return (
    <>
      <Navbar />

      <Routes>
        <Route
          path="/"
          element={
            <>
              <Home />
              <Features />
              
              
                <Footer />
    
            </>
          }
        />

        <Route path="/translate" element={
          <ProtectedRoute>
            <Translate />
          </ProtectedRoute>
        } />
        <Route path="/profile" element={
          <ProtectedRoute>
            <Profile />
          </ProtectedRoute>
        } />
        <Route path="/chat" element={<Chat />} /> 
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/contactus" element={<Contact />} />
        <Route path="/guide" element={<HelpGuide />} />
        <Route path="/forget-password" element={<ForgetPassword />} />
        <Route path="/chatbot" element={<ProtectedRoute><Chatbot /></ProtectedRoute>} />
        <Route path="/reset-password" element={<ResetPassword />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />
        
      </Routes>

    
    </>
  );
}

export default App;