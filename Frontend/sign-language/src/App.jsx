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

import { Routes, Route } from "react-router-dom";

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
    
            </>
          }
        />

        <Route path="/translate" element={<Translate />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
          <Route path="/contactus" element={<Contact />} />
        <Route path="/chatbot" element={<Chatbot />} />
      </Routes>

      <Footer />
    </>
  );
}

export default App;
