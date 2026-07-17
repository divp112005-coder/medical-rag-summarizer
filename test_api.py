"""
test_api.py — Quick end-to-end test for the Phase 2 RAG pipeline.
Run:  python test_api.py
"""
import json
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"

def separator(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

# ---------------------------------------------------------------
# Test 1: Health check
# ---------------------------------------------------------------
separator("GET /  (health check)")
req = urllib.request.Request(BASE + "/", headers={"Accept": "application/json"})
health = json.loads(urllib.request.urlopen(req).read())
print(json.dumps(health, indent=2))

# ---------------------------------------------------------------
# Test 2: Full RAG query with LLM
# ---------------------------------------------------------------
separator("POST /api/query  (full RAG pipeline)")

payload = json.dumps({
    "question": "What are the admission deadlines for graduate programs?",
    "top_k": 3
}).encode()

req2 = urllib.request.Request(
    BASE + "/api/query",
    data=payload,
    headers={"Content-Type": "application/json"}
)

try:
    resp = json.loads(urllib.request.urlopen(req2, timeout=60).read())
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code, e.read().decode())
    raise

print(f"Question         : {resp['question']}")
print(f"Chunks retrieved : {resp['total_chunks_retrieved']}")
print()
print("--- Retrieved Chunks ---")
for c in resp["retrieved_chunks"]:
    print(f"  Rank {c['rank']} | Score {c['score']:.4f} | Page {c['page_number']} | {c['chunk_id']}")

print()
print("--- LLM Summary ---")
print(resp["llm_summary"])

print()
print("--- Disclaimer ---")
print(resp["disclaimer"])

# ---------------------------------------------------------------
# Test 3: Refusal test — question clearly outside the document
# ---------------------------------------------------------------
separator("POST /api/query  (refusal test — off-topic question)")

payload2 = json.dumps({
    "question": "What is the capital of France?",
    "top_k": 2
}).encode()

req3 = urllib.request.Request(
    BASE + "/api/query",
    data=payload2,
    headers={"Content-Type": "application/json"}
)
resp3 = json.loads(urllib.request.urlopen(req3, timeout=60).read())
print(f"Question  : {resp3['question']}")
print(f"LLM Reply : {resp3['llm_summary']}")
print()
