import React from "react";

import Home from "./components/Home";
import Navbar from "./components/Navbar";
import Features from "./components/Features";
import About from "./components/About";
import Contact from "./components/Contact";
import Footer from "./components/Footer";
import Chat from "./components/Chat";
import Translate from "./components/Translate";
import Profile from "./components/Profile";
import Login from "./components/Login";
import Chatbot from "./components/Chatbot";
import Register from "./components/Register";
import HelloHand3D from "./components/HelloHand3D";
import { Routes, Route } from "react-router-dom";

import ForgetPassword from "./components/ForgetPassword";
import ResetPassword from "./components/ResetPassword";
import ServerSettingsModal from "./components/ServerSettingsModal";
import ProtectedRoute from "./components/ProtectedRoute";

function App() {
  return (
    <>
      <Navbar />
      <ServerSettingsModal />

      <Routes>
        <Route
          path="/"
          element={
            <>
              <Home />
              <Features />
              <About />
                <Footer />
    
            </>
          }
        />

        <Route path="/translate" element={<ProtectedRoute><Translate /></ProtectedRoute>} />
        <Route path="/profile" element={<ProtectedRoute><Profile /></ProtectedRoute>} />
        <Route path="/chat" element={<ProtectedRoute><Chat /></ProtectedRoute>} /> 
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/contactus" element={<Contact />} />
        <Route path="/chatbot" element={<ProtectedRoute><Chatbot /></ProtectedRoute>} />
        <Route path="/forget-password" element={<ForgetPassword />} />
        <Route path="/reset-password" element={<ResetPassword />} />
      </Routes>

    
    </>
  );
}

export default App;