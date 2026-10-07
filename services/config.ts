// Backend base URL. Set VITE_API_URL at build time for production
// (e.g. https://neurograph-api.onrender.com). Falls back to local dev.
export const API_URL: string = (
  import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
).replace(/\/+$/, '');
