"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { simulateStreamUrl } from "@/lib/api";
import type { RiskBand, ScoredOrder } from "@/lib/types";

export interface SimulationRun {
  requested: number;
  received: number;
  running: boolean;
  error: string | null;
  // Set once the whole batch has been scored and saved server-side.
  bandCounts: Record<RiskBand, number> | null;
}

interface OrdersContextValue {
  orders: ScoredOrder[];
  addOrders: (newOrders: ScoredOrder[]) => void;
  selectedOrder: ScoredOrder | null;
  selectOrder: (order: ScoredOrder | null) => void;
  simulation: SimulationRun | null;
  startSimulation: (n: number, riskShift: number) => void;
  isAwaitingSave: (orderId: string) => boolean;
}

const OrdersContext = createContext<OrdersContextValue | null>(null);

const MAX_FEED_SIZE = 500;
const STORAGE_KEY = "riskguard.feed";
// Streamed orders reach the feed in small batches rather than one render
// (and one sessionStorage write of the whole feed) per order.
const STREAM_FLUSH_MS = 120;

// The live feed is persisted to sessionStorage so a page refresh or a hard
// navigation between dashboard pages (not just Next.js soft nav) doesn't wipe
// the demo's in-progress state. It's read through useSyncExternalStore: the
// server and the hydrating client both render the empty feed, then React
// switches to the stored one -- no hydration mismatch, and no mount effect
// juggling a "hydrated" flag (which raced under StrictMode's double effects).
const EMPTY_FEED: ScoredOrder[] = [];
let feed: ScoredOrder[] | null = null; // loaded on the first client-side read
const feedListeners = new Set<() => void>();

function readFeed(): ScoredOrder[] {
  if (feed === null) {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      feed = raw ? (JSON.parse(raw) as ScoredOrder[]) : EMPTY_FEED;
    } catch {
      feed = EMPTY_FEED; // malformed or unavailable storage
    }
  }
  return feed;
}

function updateFeed(update: (prev: ScoredOrder[]) => ScoredOrder[]) {
  const next = update(readFeed());
  if (next === feed) return;
  feed = next;
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // storage unavailable (private mode, quota) -- feed still works in-memory
  }
  feedListeners.forEach((listener) => listener());
}

// Newest first by order time, not by arrival. Timestamps are ISO-8601 in a
// single format, so they sort chronologically as plain strings.
function byNewestFirst(a: ScoredOrder, b: ScoredOrder) {
  return a.order_timestamp < b.order_timestamp ? 1 : a.order_timestamp > b.order_timestamp ? -1 : 0;
}

function subscribeFeed(listener: () => void) {
  feedListeners.add(listener);
  return () => {
    feedListeners.delete(listener);
  };
}

export function OrdersProvider({ children }: { children: React.ReactNode }) {
  const orders = useSyncExternalStore(subscribeFeed, readFeed, () => EMPTY_FEED);
  const [selectedOrder, setSelectedOrder] = useState<ScoredOrder | null>(null);

  // The simulation stream lives here in the root layout, not in the
  // Simulation Console page, so navigating to the dashboard mid-batch keeps
  // it running -- the feed fills live there instead of the stream (and, with
  // it, the batch) being cut off.
  const [simulation, setSimulation] = useState<SimulationRun | null>(null);
  // Orders of the batch still streaming: already shown, but the server saves
  // a batch only once all of it is scored, so they can't take a decision yet.
  const [unsavedIds, setUnsavedIds] = useState<ReadonlySet<string>>(() => new Set());
  const sourceRef = useRef<EventSource | null>(null);
  const bufferRef = useRef<ScoredOrder[]>([]);
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      sourceRef.current?.close();
      if (flushTimerRef.current) clearTimeout(flushTimerRef.current);
    },
    []
  );

  // Dedup by order_id: the same order can legitimately reach addOrders
  // twice (e.g. simulated on the Simulation Console, then re-fetched by the
  // dashboard's cold-start GET /orders load) -- without this, it would
  // appear twice in the feed. Sorted rather than prepended: that cold-start
  // load of older, saved orders can arrive after newer live-streamed ones,
  // and prepending would bury the newest orders beneath it.
  const addOrders = useCallback((newOrders: ScoredOrder[]) => {
    updateFeed((prev) => {
      const existingIds = new Set(prev.map((o) => o.order_id));
      const deduped = newOrders.filter((o) => !existingIds.has(o.order_id));
      if (deduped.length === 0) return prev;
      return [...deduped, ...prev].sort(byNewestFirst).slice(0, MAX_FEED_SIZE);
    });
  }, []);

  const selectOrder = useCallback((order: ScoredOrder | null) => setSelectedOrder(order), []);

  const startSimulation = useCallback(
    (n: number, riskShift: number) => {
      sourceRef.current?.close();
      if (flushTimerRef.current) clearTimeout(flushTimerRef.current);
      flushTimerRef.current = null;
      bufferRef.current = [];

      const batchIds: string[] = [];
      const source = new EventSource(simulateStreamUrl(n, riskShift));
      sourceRef.current = source;
      setSimulation({ requested: n, received: 0, running: true, error: null, bandCounts: null });
      setUnsavedIds(new Set());

      const flush = () => {
        flushTimerRef.current = null;
        const arrived = bufferRef.current;
        bufferRef.current = [];
        if (arrived.length === 0) return;
        addOrders(arrived);
        setUnsavedIds(new Set(batchIds));
        setSimulation((s) => (s ? { ...s, received: batchIds.length } : s));
      };

      // Always close explicitly: an EventSource left open after an error
      // auto-reconnects, which would silently start a second batch.
      const finish = (update: Partial<SimulationRun>) => {
        source.close();
        if (sourceRef.current === source) sourceRef.current = null;
        if (flushTimerRef.current) clearTimeout(flushTimerRef.current);
        flush();
        setUnsavedIds(new Set());
        setSimulation((s) => (s ? { ...s, ...update, running: false } : s));
      };

      source.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.error) {
          // The server reported the batch failed, and a batch is saved only
          // as a whole -- none of it exists, so take it back out of the feed.
          bufferRef.current = [];
          finish({ error: data.error });
          const dropped = new Set(batchIds);
          updateFeed((prev) => prev.filter((o) => !dropped.has(o.order_id)));
          setSelectedOrder((sel) => (sel && dropped.has(sel.order_id) ? null : sel));
          return;
        }
        if (data.done) {
          finish({ bandCounts: data.band_counts });
          return;
        }
        bufferRef.current.push(data as ScoredOrder);
        batchIds.push(data.order_id);
        if (flushTimerRef.current === null) flushTimerRef.current = setTimeout(flush, STREAM_FLUSH_MS);
      };

      source.onerror = () => {
        // Connection lost (or the API isn't running). The server finishes and
        // saves a batch it has started even without a listener, so orders
        // already shown stay; a dashboard reload picks up the rest.
        finish({ error: "Streaming connection to the API failed." });
      };
    },
    [addOrders]
  );

  const isAwaitingSave = useCallback((orderId: string) => unsavedIds.has(orderId), [unsavedIds]);

  const value = useMemo(
    () => ({ orders, addOrders, selectedOrder, selectOrder, simulation, startSimulation, isAwaitingSave }),
    [orders, addOrders, selectedOrder, selectOrder, simulation, startSimulation, isAwaitingSave]
  );

  return <OrdersContext.Provider value={value}>{children}</OrdersContext.Provider>;
}

export function useOrders() {
  const ctx = useContext(OrdersContext);
  if (!ctx) throw new Error("useOrders must be used within OrdersProvider");
  return ctx;
}
