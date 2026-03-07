import urllib.request, json

BASE = "http://localhost:8000/mcp"

def init():
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    data = json.dumps({"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"demo","version":"1.0"}},"id":0}).encode()
    req = urllib.request.Request(BASE, data, headers)
    with urllib.request.urlopen(req) as r:
        return r.headers.get("mcp-session-id")

def call(name, arguments, session, id):
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", "mcp-session-id": session}
    data = json.dumps({"jsonrpc":"2.0","method":"tools/call","params":{"name":name,"arguments":arguments},"id":id}).encode()
    req = urllib.request.Request(BASE, data, headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read().decode()
    for line in body.splitlines():
        if line.startswith("data:"):
            result = json.loads(line[5:])
            try:
                return result["result"]["content"][0]["text"]
            except:
                return str(result)

def scene(number, title):
    print("\n" + "=" * 60)
    print(f"  TOOL {number}: {title}")
    print("=" * 60)
    input("  🎬 Press Enter to run...")

SESSION = init()
print(f"\n✅ SERVER CONNECTED | Session: {SESSION}")
print("🎬 Press Enter to run each tool one by one\n")

scene(1, "LIST ALL AVAILABLE dLLM MODELS")
print(call("dllm_list_models", {}, SESSION, 1))
input("\n✅ Done. Press Enter for next tool...")

scene(2, "LOAD MODEL INTO GPU MEMORY")
print("⏳ Loading modernbert-large-chat — takes 2-3 mins on first load...")
print(call("dllm_load_model", {"params": {"model_key": "modernbert-large-chat"}}, SESSION, 2))
input("\n✅ Done. Press Enter for next tool...")

scene(3, "MODEL INFO — ARCHITECTURE + VRAM STATS")
print(call("dllm_model_info", {"params": {"model_key": "modernbert-large-chat"}}, SESSION, 3))
input("\n✅ Done. Press Enter for next tool...")

scene(4, "STANDARD DIFFUSION TEXT GENERATION")
print("⏳ Running diffusion sampling...")
print(call("dllm_generate", {"params": {"prompt": "What makes diffusion language models unique?", "max_new_tokens": 80}}, SESSION, 4))
input("\n✅ Done. Press Enter for next tool...")

scene(5, "TEXT INFILLING — NATIVE [MASK] FILLING")
print("⏳ Filling masked tokens using bidirectional context...")
print(call("dllm_infill", {"params": {"text_with_masks": "The [MASK] of France is Paris, famous for the [MASK] Tower and its [MASK] cuisine.", "steps": 256}}, SESSION, 5))
input("\n✅ Done. Press Enter for next tool...")

scene(6, "FAST GENERATE — KV CACHE + PARALLEL DECODING")
print("⏳ Running Fast-dLLM with 2-4x speedup...")
print(call("dllm_fast_generate", {"params": {"prompt": "Explain the future of AI in 3 sentences.", "max_new_tokens": 60}}, SESSION, 6))
input("\n✅ Done. Press Enter for next tool...")

scene(7, "TRACE DIFFUSION STEPS — TOKEN EVOLUTION")
print("⏳ Capturing token evolution from noise to meaning...")
print(call("dllm_trace_steps", {"params": {"prompt": "The meaning of life is", "max_new_tokens": 30, "total_steps": 64}}, SESSION, 7))
input("\n✅ Done. Press Enter for next tool...")

scene(8, "AR vs DIFFUSION — SAME PROMPT, DIFFERENT PARADIGM")
print("⏳ Loading AR model + diffusion model for comparison...")
print(call("dllm_compare_ar_vs_diffusion", {"params": {"prompt": "Write a short poem about artificial intelligence.", "max_new_tokens": 60}}, SESSION, 8))

print("\n" + "🎬" * 30)
print("  ✅ ALL 8 TOOLS DEMONSTRATED SUCCESSFULLY")
print("  dLLM MCP SERVER — LIVE ON RUNPOD RTX 4090")
print("🎬" * 30)
