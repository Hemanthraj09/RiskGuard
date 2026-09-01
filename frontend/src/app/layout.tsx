import type { Metadata } from "next";
import "./globals.css";
import { OrdersProvider } from "@/context/OrdersContext";
import { NavBar } from "@/components/NavBar";

export const metadata: Metadata = {
  title: "RiskGuard — Return Risk Scorer",
  description: "AI Risk Manager for e-commerce return risk (Razorpay Buildathon, Track 02)",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col" style={{ background: "var(--background)" }}>
        <OrdersProvider>
          <NavBar />
          <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-8">{children}</main>
        </OrdersProvider>
      </body>
    </html>
  );
}
