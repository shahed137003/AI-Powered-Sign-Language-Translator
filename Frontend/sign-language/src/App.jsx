import React from "react";

import Home from "./components/Home";
import Navbar from "./components/Navbar";
import Features from "./components/Features";
import About from "./components/About";
import Contact from "./components/Contact";
import Footer from "./components/Footer";

import Translate from "./components/Translate";
import Profile from "./components/Profile";
import Login from "./components/Login";
import Chatbot from "./components/Chatbot";
import Register from "./components/Register";
<<<<<<< HEAD

import { Routes, Route } from "react-router-dom";

=======
import HelloHand3D from "./components/HelloHand3D";
import { Routes, Route } from "react-router-dom";

import ForgetPassword from "./components/ForgetPassword";
import ResetPassword from "./components/ResetPassword";
>>>>>>> e251330 (Add frontend, backend, and ai_service)
function App() {
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
              <About />
<<<<<<< HEAD
              <Contact />
=======
                <Footer />
    
>>>>>>> e251330 (Add frontend, backend, and ai_service)
            </>
          }
        />

        <Route path="/translate" element={<Translate />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
<<<<<<< HEAD
        <Route path="/chatbot" element={<Chatbot />} />
      </Routes>

      <Footer />
=======
        <Route path="/contactus" element={<Contact />} />
        <Route path="/chatbot" element={<Chatbot />} />
        <Route path="/forget-password" element={<ForgetPassword />} />
        <Route path="/reset-password" element={<ResetPassword />} />
      </Routes>

    
>>>>>>> e251330 (Add frontend, backend, and ai_service)
    </>
  );
}

export default App;
