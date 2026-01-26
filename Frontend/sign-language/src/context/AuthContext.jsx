import { createContext, useEffect, useState } from "react";
import axios from "axios";

/* ✅ EXPORT CONTEXT */
export const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(() => {
    const storedUser = localStorage.getItem("user");
    return storedUser ? JSON.parse(storedUser) : null;
  });
  const [token, setToken] = useState(
    localStorage.getItem("token") || null
  );
  

  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

  useEffect(() => {
    if (token) {
      axios.defaults.headers.common.Authorization = `Bearer ${token}`;
      localStorage.setItem("token", token);
    } else {
      delete axios.defaults.headers.common.Authorization;
      localStorage.removeItem("token");
    }
    
  }, [token]);

  const login = async (credentials) => {
    const res = await axios.post(`${API_URL}/users/login`, credentials);
    const { access_token, role } = res.data;

    const userInfo = {
      email: credentials.email,
      role: role || "user",
    };

    setToken(access_token);
    setUser(userInfo);
    localStorage.setItem("user", JSON.stringify(userInfo));

    return { success: true };
  };

  const register = async (data) => {
    await axios.post(`${API_URL}/users/register`, data);
    return login({ email: data.email, password: data.password });
  };

  const logout = () => {
    setUser(null);
    setToken(null);
    localStorage.clear();
  };
  /* ✅ loading is derived, not stored */
  const loading = token === undefined;
  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        login,
        register,
        logout,
        isAuthenticated: !!token,
      }}
    >
      {!loading && children}
    </AuthContext.Provider>
  );
};
