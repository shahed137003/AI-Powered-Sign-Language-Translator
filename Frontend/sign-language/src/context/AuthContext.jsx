import { createContext, useEffect, useState } from "react";
import axios from "axios";
import { getApiUrl } from "../lib/api";

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
  

  // Use the Capacitor-aware resolver so mobile uses the device's actual server IP
  const API_URL = getApiUrl();

  useEffect(() => {
    const fetchCurrentUser = async () => {
      if (token) {
        axios.defaults.headers.common.Authorization = `Bearer ${token}`;
        localStorage.setItem("token", token);
        try {
          const res = await axios.get(`${API_URL}/users/me`);
          const backendUser = res.data;
          const userInfo = {
            id: backendUser.id,
            username: backendUser.username,
            email: backendUser.email,
            role: backendUser.role,
          };
          setUser(userInfo);
          localStorage.setItem("user", JSON.stringify(userInfo));
        } catch (err) {
          console.error("Failed to fetch current user profile:", err);
          if (err?.response?.status === 401) {
            logout();
          }
        }
      } else {
        delete axios.defaults.headers.common.Authorization;
        localStorage.removeItem("token");
      }
    };
    fetchCurrentUser();
  }, [token]);

  const login = async (credentials) => {
    try {
      const res = await axios.post(`${API_URL}/users/login`, credentials);
      const { access_token, user: backendUser } = res.data;

      const userInfo = {
        id: backendUser?.id,
        username: backendUser?.username || credentials.email.split('@')[0],
        email: backendUser?.email || credentials.email,
        role: backendUser?.role || "user",
      };

      setToken(access_token);
      setUser(userInfo);
      localStorage.setItem("user", JSON.stringify(userInfo));

      return { success: true };
    } catch (err) {
      const message =
        err?.response?.data?.detail ||
        err?.message ||
        "Login failed. Please check your credentials.";
      return { success: false, error: message };
    }
  };

  const register = async (data) => {
    try {
      await axios.post(`${API_URL}/users/register`, data);
      return login({ email: data.email, password: data.password });
    } catch (err) {
      const message =
        err?.response?.data?.detail ||
        err?.message ||
        "Registration failed. Please try again.";
      return { success: false, error: message };
    }
  };

  const logout = () => {
    setUser(null);
    setToken(null);
    localStorage.removeItem("token");
    localStorage.removeItem("user");
  };

  const updateUser = (newUserData) => {
    setUser((prev) => {
      const updated = prev ? { ...prev, ...newUserData } : null;
      if (updated) {
        localStorage.setItem("user", JSON.stringify(updated));
      }
      return updated;
    });
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
        updateUser,
        isAuthenticated: !!token,
      }}
    >
      {!loading && children}
    </AuthContext.Provider>
  );
};
