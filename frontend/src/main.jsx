import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./index.css";

async function boot() {
  // Static GitHub Pages build: swap the FastAPI backend for an in-browser one
  if (import.meta.env.VITE_DEMO === "true") {
    const { install } = await import("./demo/mockBackend.js");
    install();
  }

  ReactDOM.createRoot(document.getElementById("root")).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}

boot();
