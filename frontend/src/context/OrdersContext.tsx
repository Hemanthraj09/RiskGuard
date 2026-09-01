"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ScoredOrder } from "@/lib/types";

interface OrdersContextValue {
  orders: ScoredOrder[];
  addOrders: (newOrders: ScoredOrder[]) => void;
  selectedOrder: ScoredOrder | null;
  selectOrder: (order: ScoredOrder | null) => void;
}

const OrdersContext = createContext<OrdersContextValue | null>(null);

const MAX_FEED_SIZE = 500;
const STORAGE_KEY = "riskguard.feed";

export function OrdersProvider({ children }: { children: React.ReactNode }) {
  const [orders, setOrders] = useState<ScoredOrder[]>([]);
  const [selectedOrder, setSelectedOrder] = useState<ScoredOrder | null>(null);
  // A useState (not useRef) guard: under React StrictMode's dev-mode double
  // effect invocation, a ref-based guard doesn't reset between the
  // simulated mount/unmount/remount, so the persist-effect below can see
  // hydrated=true with a still-stale (pre-hydration) `orders` closure and
  // clobber sessionStorage with []. State-based guards re-render before the
  // persist-effect observes them, which avoids that race.
  const [isHydrated, setIsHydrated] = useState(false);

  // Persist the live feed to sessionStorage so a page refresh or a hard
  // navigation between dashboard pages (not just Next.js soft nav) doesn't
  // wipe the demo's in-progress state.
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (raw) setOrders(JSON.parse(raw));
    } catch {
      // ignore malformed/unavailable storage
    }
    setIsHydrated(true);
  }, []);

  useEffect(() => {
    if (!isHydrated) return;
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(orders));
    } catch {
      // storage unavailable (private mode, quota) -- feed still works in-memory
    }
  }, [orders, isHydrated]);

  // Dedup by order_id: the same order can legitimately reach addOrders
  // twice (e.g. simulated on the Simulation Console, then re-fetched by the
  // dashboard's cold-start GET /orders load) -- without this, it would
  // appear twice in the feed.
  const addOrders = useCallback((newOrders: ScoredOrder[]) => {
    setOrders((prev) => {
      const existingIds = new Set(prev.map((o) => o.order_id));
      const deduped = newOrders.filter((o) => !existingIds.has(o.order_id));
      if (deduped.length === 0) return prev;
      return [...deduped, ...prev].slice(0, MAX_FEED_SIZE);
    });
  }, []);

  const selectOrder = useCallback((order: ScoredOrder | null) => setSelectedOrder(order), []);

  const value = useMemo(
    () => ({ orders, addOrders, selectedOrder, selectOrder }),
    [orders, addOrders, selectedOrder, selectOrder]
  );

  return <OrdersContext.Provider value={value}>{children}</OrdersContext.Provider>;
}

export function useOrders() {
  const ctx = useContext(OrdersContext);
  if (!ctx) throw new Error("useOrders must be used within OrdersProvider");
  return ctx;
}
