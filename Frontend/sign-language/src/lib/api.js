// Dynamic API and WebSocket endpoint resolver for web & mobile/Capacitor

// Helper to check if the app is running in a Capacitor native container
export const isCapacitor = () => {
  return window.Capacitor !== undefined || !!window.android || !!window.webkit?.messageHandlers;
};

// Retrieve base IP/Host. Defaults to localhost if not specified in localStorage
export const getServerHost = () => {
  if (isCapacitor()) {
    const savedHost = localStorage.getItem("CUSTOM_SERVER_HOST");
    if (savedHost) {
      return savedHost.trim();
    }
    // Standard android emulator fallback host (10.0.2.2 points to host PC localhost)
    if (navigator.userAgent.toLowerCase().includes("android")) {
      return "10.0.2.2";
    }
  }

  // Web default: extract host from Vite env or window.location
  const envUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
  try {
    const parsed = new URL(envUrl);
    return parsed.hostname;
  } catch (e) {
    return "localhost";
  }
};

// Retrieve API Base URL (HTTP/HTTPS)
export const getApiUrl = () => {
  const host = getServerHost();
  // If the user entered a full URL, use it directly
  if (host.startsWith("http://") || host.startsWith("https://")) {
    return host;
  }
  return `http://${host}:8000`;
};

// Retrieve WebSocket URL for backend (port 8000) or AI service (port 8001)
export const getWsUrl = (port = 8000, path = "") => {
  const host = getServerHost();
  const rawHost = host.replace(/^https?:\/\//, ""); // strip http/https if present

  // Under Capacitor, use ws:// for local IPs/hosts unless explicitly starting with https://
  let protocol = "ws:";
  if (isCapacitor()) {
    protocol = host.startsWith("https://") ? "wss:" : "ws:";
  } else {
    protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  }

  const cleanPath = path.startsWith("/") ? path : `/${path}`;

  return `${protocol}//${rawHost}:${port}${cleanPath}`;
};

// Save a custom host to localStorage (e.g., "192.168.1.100")
export const saveServerHost = (host) => {
  if (!host) {
    localStorage.removeItem("CUSTOM_SERVER_HOST");
  } else {
    // Strip trailing slashes, http:// or ws:// to keep it clean
    let cleanHost = host.trim();
    cleanHost = cleanHost.replace(/^(https?:\/\/|wss?:\/\/)/, "");
    cleanHost = cleanHost.split(":")[0]; // keep only host part
    localStorage.setItem("CUSTOM_SERVER_HOST", cleanHost);
  }
  // Force reload to apply changes
  window.location.reload();
};
