import { useEffect, useRef, useState } from "react";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/live";
const MAX_EVENTS = 200;

export function useLiveFeed() {
  const [events, setEvents] = useState([]);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef(null);
  const retryRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    let retryTimeout;

    function connect() {
      if (cancelled) return;
      const ws = new WebSocket(WS_URL);
      socketRef.current = ws;

      ws.onopen = () => {
        retryRef.current = 0;
        setConnected(true);
      };

      ws.onmessage = (msg) => {
        try {
          const event = JSON.parse(msg.data);
          setEvents((prev) => {
            if (prev.some((e) => e.event_id === event.event_id)) return prev;
            return [event, ...prev].slice(0, MAX_EVENTS);
          });
        } catch (_) {
          /* ignore malformed frame */
        }
      };

      ws.onclose = () => {
        setConnected(false);
        if (cancelled) return;
        const delay = Math.min(1000 * 2 ** retryRef.current, 15000);
        retryRef.current += 1;
        retryTimeout = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();
    return () => {
      cancelled = true;
      clearTimeout(retryTimeout);
      socketRef.current?.close();
    };
  }, []);

  return { events, connected };
}
