import axios from "axios";

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

/** Set VITE_API_KEY when the backend runs with API_KEYS. Unset = open API (local development). */
export const API_KEY: string | undefined = import.meta.env.VITE_API_KEY || undefined;

export const apiClient = axios.create({
  baseURL: API_BASE,
  headers: API_KEY ? { "X-API-Key": API_KEY } : undefined,
});

/** Report downloads are plain browser links that cannot send headers, so they carry the key
 * as a query parameter instead. */
export function withApiKey(url: string): string {
  if (!API_KEY) return url;
  return `${url}${url.includes("?") ? "&" : "?"}api_key=${encodeURIComponent(API_KEY)}`;
}
