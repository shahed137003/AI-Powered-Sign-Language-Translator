import React, { useState, useEffect } from "react";
import { FaServer, FaCheck, FaTimes, FaUndo } from "react-icons/fa";
import { isCapacitor, getServerHost, saveServerHost } from "../lib/api";
import { motion, AnimatePresence } from "framer-motion";

export default function ServerSettingsModal() {
  const [isOpen, setIsOpen] = useState(false);
  const [host, setHost] = useState("");
  const [showButton, setShowButton] = useState(false);

  useEffect(() => {
    // Only show this server connection utility when running in Capacitor
    if (isCapacitor()) {
      setShowButton(true);
      setHost(getServerHost());
    }
  }, []);

  const handleSave = (e) => {
    e.preventDefault();
    saveServerHost(host);
    setIsOpen(false);
  };

  const handleReset = () => {
    saveServerHost("");
    setHost(getServerHost());
    setIsOpen(false);
  };

  if (!showButton) return null;

  return (
    <>
      {/* Floating Gear Button */}
      <motion.button
        onClick={() => setIsOpen(true)}
        whileHover={{ scale: 1.1, rotate: 15 }}
        whileTap={{ scale: 0.9 }}
        className="fixed bottom-6 left-6 z-50 p-4 bg-gradient-to-r from-[#6A3093] to-[#BF5AE0] text-white rounded-full shadow-2xl shadow-purple-500/50 border border-purple-300/30 flex items-center justify-center"
        title="Server Connection Settings"
      >
        <FaServer className="text-xl" />
      </motion.button>

      {/* Settings Modal */}
      <AnimatePresence>
        {isOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="bg-white dark:bg-[#1a163a] border border-gray-200 dark:border-purple-500/20 rounded-3xl p-6 w-full max-w-md shadow-2xl"
            >
              <div className="flex justify-between items-center mb-6 border-b border-gray-100 dark:border-purple-900/30 pb-3">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                  <FaServer className="text-purple-500" />
                  Mobile API Server Configuration
                </h3>
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
                >
                  <FaTimes className="text-gray-500 dark:text-gray-400" />
                </button>
              </div>

              <form onSubmit={handleSave} className="space-y-4">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    Host IP / Domain Address
                  </label>
                  <input
                    type="text"
                    value={host}
                    onChange={(e) => setHost(e.target.value)}
                    placeholder="e.g. 192.168.1.100 or 10.0.2.2"
                    className="w-full p-4 border border-gray-300 dark:border-purple-500/30 rounded-xl bg-gray-50 dark:bg-gray-900/50 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-600 focus:ring-2 focus:ring-purple-500 focus:outline-none"
                    required
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                    Enter your host computer's local IP address (run <code>ipconfig</code> in terminal). 
                    If using Android Emulator, use <code>10.0.2.2</code> to connect to your local PC.
                  </p>
                </div>

                <div className="flex gap-3 pt-4 border-t border-gray-100 dark:border-purple-900/30">
                  <button
                    type="button"
                    onClick={handleReset}
                    className="flex-1 py-3 px-4 bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200 rounded-xl font-semibold hover:bg-gray-300 dark:hover:bg-gray-700 transition-colors flex items-center justify-center gap-2"
                  >
                    <FaUndo /> Reset Default
                  </button>
                  <button
                    type="submit"
                    className="flex-1 py-3 px-4 bg-gradient-to-r from-[#6A3093] to-[#BF5AE0] text-white rounded-xl font-semibold hover:shadow-lg hover:shadow-purple-500/30 transition-all flex items-center justify-center gap-2"
                  >
                    <FaCheck /> Save & Reload
                  </button>
                </div>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  );
}
