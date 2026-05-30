import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import './index.css';
import { HashRouter } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext.jsx';

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <HashRouter>
      <AuthProvider>
         <App />
      </AuthProvider>
    </HashRouter>

  </React.StrictMode>
);
// import React from "react";
// import ReactDOM from "react-dom/client";
// import App from "./App.jsx";
// import './index.css';
// import { HashRouter } from 'react-router-dom';
// <<<<<<< HEAD

// =======
// import { AuthProvider } from './context/AuthContext.jsx';
// >>>>>>> e251330 (Add frontend, backend, and ai_service)

// ReactDOM.createRoot(document.getElementById("root")).render(
//   <React.StrictMode>
//     <HashRouter>
// <<<<<<< HEAD
//          <App />
// =======
//       <AuthProvider>
//          <App />
//       </AuthProvider>
// >>>>>>> e251330 (Add frontend, backend, and ai_service)
//     </HashRouter>

//   </React.StrictMode>
// );