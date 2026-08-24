"use client";

import { useState, useRef, useEffect } from "react";

interface Device {
  device_id: number;
  category: string;
  brand: string;
  model: string;
  processor: string;
  gpu: string | null;
  ram_gb: number | null;
  storage_gb: number | null;
  display: string;
  battery: string | null;
  price_bdt: number;
  chunk_text?: string;
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const [maxPrice, setMaxPrice] = useState("");
  const [brand, setBrand] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<{ role: "user" | "assistant"; content: string; devices?: Device[] }[]>([]);
  const [selectedForComparison, setSelectedForComparison] = useState<Device[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = query;
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setQuery("");
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userMessage,
          category: category || null,
          max_price_bdt: maxPrice ? parseFloat(maxPrice) : null,
          brand: brand || null,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch recommendation");
      }

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, devices: data.retrieved_devices },
      ]);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError(String(err));
      }
    } finally {
      setLoading(false);
    }
  };

  const toggleComparison = (device: Device) => {
    setSelectedForComparison((prev) => {
      const exists = prev.find((d) => d.device_id === device.device_id);
      if (exists) {
        return prev.filter((d) => d.device_id !== device.device_id);
      }
      if (prev.length >= 3) {
        alert("You can only compare up to 3 devices.");
        return prev;
      }
      return [...prev, device];
    });
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center p-4">
      <h1 className="text-3xl font-bold my-6 text-gray-800">Device Recommender</h1>

      <div className="w-full max-w-5xl flex flex-col md:flex-row gap-6">
        {/* Chat / Main Section */}
        <div className="flex-1 bg-white p-6 rounded-xl shadow flex flex-col h-[80vh]">
          
          {/* Filters */}
          <div className="flex gap-4 mb-4 pb-4 border-b">
            <input
              type="text"
              placeholder="Brand (e.g. Apple)"
              className="border p-2 rounded flex-1 text-black"
              value={brand}
              onChange={(e) => setBrand(e.target.value)}
            />
            <select
              className="border p-2 rounded flex-1 text-black"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              <option value="">Any Category</option>
              <option value="Mobile">Mobile</option>
              <option value="Laptop">Laptop</option>
            </select>
            <input
              type="number"
              placeholder="Max Price (BDT)"
              className="border p-2 rounded flex-1 text-black"
              value={maxPrice}
              onChange={(e) => setMaxPrice(e.target.value)}
            />
          </div>

          {/* Chat History */}
          <div className="flex-1 overflow-y-auto mb-4 space-y-4 pr-2 text-black">
            {messages.map((m, i) => (
              <div key={i} className={`flex flex-col ${m.role === "user" ? "items-end" : "items-start"}`}>
                <div className={`p-4 max-w-3xl rounded-lg ${m.role === "user" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800"}`}>
                  <p className="whitespace-pre-wrap">{m.content}</p>
                </div>
                
                {/* Device Cards for Assistant */}
                {m.role === "assistant" && m.devices && m.devices.length > 0 && (
                  <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4 w-full">
                    {m.devices.map((device) => (
                      <div key={device.device_id} className="border p-4 rounded-lg bg-white shadow-sm flex flex-col">
                        <h3 className="font-bold text-lg">{device.brand} {device.model}</h3>
                        <div className="text-sm text-gray-600 mt-2 flex-1">
                          <p><strong>Price:</strong> {device.price_bdt} BDT</p>
                          <p><strong>Processor:</strong> {device.processor}</p>
                          <p><strong>RAM:</strong> {device.ram_gb} GB</p>
                          <p><strong>Storage:</strong> {device.storage_gb} GB</p>
                        </div>
                        <button 
                          onClick={() => toggleComparison(device)}
                          className="mt-4 bg-gray-200 hover:bg-gray-300 text-gray-800 py-1 px-3 rounded text-sm transition"
                        >
                          {selectedForComparison.find(d => d.device_id === device.device_id) ? "Remove from Compare" : "Compare"}
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {loading && <div className="text-gray-500 italic">Thinking...</div>}
            {error && <div className="text-red-500 bg-red-50 p-3 rounded">{error}</div>}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <form onSubmit={handleSearch} className="flex gap-2">
            <input
              type="text"
              placeholder="E.g., I need a gaming laptop under 150000 BDT..."
              className="border border-gray-300 p-3 flex-1 rounded text-black shadow-inner"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded shadow transition"
            >
              Send
            </button>
          </form>
        </div>

        {/* Comparison Section */}
        {selectedForComparison.length > 0 && (
          <div className="w-full md:w-1/3 bg-white p-6 rounded-xl shadow h-[80vh] overflow-y-auto">
            <h2 className="text-xl font-bold mb-4 text-gray-800">Compare Devices</h2>
            <div className="flex gap-2 overflow-x-auto pb-4">
              {selectedForComparison.map(device => (
                <div key={device.device_id} className="min-w-[200px] border rounded p-3 text-sm text-gray-800 flex flex-col relative">
                  <button 
                    onClick={() => toggleComparison(device)}
                    className="absolute top-1 right-2 text-gray-400 hover:text-red-500 font-bold"
                  >
                    ×
                  </button>
                  <h3 className="font-bold mb-2 border-b pb-1 pr-4">{device.brand} {device.model}</h3>
                  <p className="py-1 border-b"><strong>Price:</strong><br/>{device.price_bdt} BDT</p>
                  <p className="py-1 border-b"><strong>CPU:</strong><br/>{device.processor}</p>
                  <p className="py-1 border-b"><strong>GPU:</strong><br/>{device.gpu || "N/A"}</p>
                  <p className="py-1 border-b"><strong>RAM:</strong><br/>{device.ram_gb} GB</p>
                  <p className="py-1 border-b"><strong>Storage:</strong><br/>{device.storage_gb} GB</p>
                  <p className="py-1 border-b"><strong>Display:</strong><br/>{device.display}</p>
                  <p className="py-1"><strong>Battery:</strong><br/>{device.battery || "N/A"}</p>
                </div>
              ))}
            </div>
            {selectedForComparison.length < 2 && (
              <p className="text-sm text-gray-500 mt-4 italic">Select at least 2 devices to compare.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
