#!/usr/bin/env python3
"""Build NEXUS PoC corpus with provenance manifest.

Sources (all licensing-clean):
 1. Project Gutenberg books (public domain) - writing/English ability
 2. CPython stdlib source files (PSF license, present on this machine) - Python code
 3. Locally authored sample files (HTML/CSS/SQL/shell/JS/Kotlin) - domain coverage
 4. Synthetic instruction-tuning examples (authored here, labeled synthetic)
"""
import json
import os
import re
import shutil
import urllib.request

ROOT = "/home/z/my-project/nexus-500m"
CORPUS = os.path.join(ROOT, "training", "data", "corpus")
os.makedirs(CORPUS, exist_ok=True)

MANIFEST = []

def add_file(relpath, content, source, license_note):
    path = os.path.join(CORPUS, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    MANIFEST.append({"file": relpath, "source": source, "license": license_note,
                     "bytes": len(content.encode('utf-8'))})

# ---------- 1. Gutenberg books ----------
BOOKS = {
    "alice.txt": "https://www.gutenberg.org/files/11/11-0.txt",
    "sherlock.txt": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "franken.txt": "https://www.gutenberg.org/files/84/84-0.txt",
    "moby.txt": "https://www.gutenberg.org/files/2701/2701-0.txt",
    "dorian.txt": "https://www.gutenberg.org/files/174/174-0.txt",
    "janeeyre.txt": "https://www.gutenberg.org/files/1260/1260-0.txt",
    "treasure.txt": "https://www.gutenberg.org/files/120/120-0.txt",
    "warworlds.txt": "https://www.gutenberg.org/files/36/36-0.txt",
    "timemachine.txt": "https://www.gutenberg.org/files/35/35-0.txt",
    "artofwar.txt": "https://www.gutenberg.org/files/132/132-0.txt",
}
ok_books = 0
for fn, url in BOOKS.items():
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "nexus-corpus/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8", errors="replace")
        # strip gutendberg header/footer
        m = re.search(r"\*\*\* ?START OF (THE |THIS )?PROJECT GUTENBERG.*?\*\*\*", raw, re.S)
        if m:
            raw = raw[m.end():]
        m = re.search(r"\*\*\* ?END OF (THE |THIS )?PROJECT GUTENBERG", raw)
        if m:
            raw = raw[:m.start()]
        add_file(os.path.join("books", fn), raw, url, "public domain (Project Gutenberg)")
        ok_books += 1
    except Exception as e:
        print(f"  skip {fn}: {e}")
print(f"gutenberg: {ok_books} books")

# ---------- 2. CPython stdlib ----------
py_src_root = "/home/z/.local/share/uv/python/cpython-3.12.15-linux-x86_64-gnu/lib/python3.12"
picked = 0
for dirpath, dirnames, filenames in os.walk(py_src_root):
    dirnames[:] = [d for d in dirnames if d not in ("site-packages", "__pycache__", "test",
                                                    "idlelib", "lib2to3", "turtledemo")]
    for fn in sorted(filenames):
        if not fn.endswith(".py"):
            continue
        p = os.path.join(dirpath, fn)
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                src = f.read()
        except Exception:
            continue
        if len(src) < 500 or len(src) > 120000:
            continue
        rel = os.path.relpath(p, py_src_root)
        add_file(os.path.join("code_py", rel.replace("/", "__")), src,
                 f"CPython 3.12 stdlib:{rel}", "PSF License (Python Software Foundation)")
        picked += 1
    if picked >= 160:
        break
print(f"stdlib: {picked} files")

# ---------- 3. Authored domain samples ----------
html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Nexus Demo</title>
  <style>
    body { font-family: sans-serif; margin: 2rem; background: #fafafa; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; }
    button.primary { background: #2563eb; color: white; border: none; padding: 0.5rem 1rem; }
  </style>
</head>
<body>
  <h1>NEXUS Assistant</h1>
  <div class="card">
    <p>An offline-first assistant with an on-device language model.</p>
    <button class="primary" onclick="alert('offline ok')">Start</button>
  </div>
  <script>
    const model = { name: "nexus", params: 12567360, quantized: false };
    function greet(user) {
      return "Hello, " + user + "! I run entirely on this device.";
    }
  </script>
</body>
</html>
"""
add_file("authored/sample_app.html", html, "authored for NEXUS", "MIT (this repo)")

sql = """-- NEXUS demo schema
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    approved INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO conversations (title) VALUES ('First offline chat');

SELECT c.title, COUNT(m.id) AS memory_count
FROM conversations c
LEFT JOIN memories m ON m.id % 2 = 0
GROUP BY c.title
HAVING memory_count >= 0
ORDER BY c.title DESC;
"""
add_file("authored/demo_schema.sql", sql, "authored for NEXUS", "MIT (this repo)")

shell = """#!/usr/bin/env bash
# NEXUS build helper
set -euo pipefail

MODEL_DIR="training/runs/poc"
OUT_DIR="android/NexusAssistant/app/src/main/assets"

echo "[1/3] exporting model for android..."
python3 -m nexus_lm.export_android --ckpt "$MODEL_DIR/nexus_ckpt.pt" \
  --tokenizer training/tokenizer.json --outdir "$OUT_DIR"

echo "[2/3] verifying export..."
python3 - <<'PY'
import json, os
cfg = json.load(open("assets/nexus_config.json"))
assert cfg["bin_bytes"] == os.path.getsize("assets/nexus_model.bin"), "size mismatch"
print("export verified:", cfg["params_total"], "params")
PY

echo "[3/3] done"
"""
add_file("authored/build.sh", shell, "authored for NEXUS", "MIT (this repo)")

ktl = """package com.nexus.assistant

/* Minimal on-device memory store example (Kotlin) */
class MemoryStore(private val db: MemDb) {
    fun save(kind: String, content: String): Long =
        db.insert("memories", mapOf("kind" to kind, "content" to content))

    fun search(query: String): List<String> =
        db.query("SELECT content FROM memories WHERE content LIKE '%' || ? || '%'", query)
            .map { it.getString("content") }

    fun approve(id: Long) { db.update("memories", id, mapOf("approved" to 1)) }
}
"""
add_file("authored/MemoryStore.kt", ktl, "authored for NEXUS", "MIT (this repo)")

js = r"""// NEXUS tokenizer helpers (JS mirror of the Kotlin implementation)
const UNK = 0;

function pretokenize(text) {
  const out = [];
  let i = 0;
  while (i < text.length) {
    const ws = /^\s+/.exec(text.slice(i));
    if (ws) { out.push(ws[0]); i += ws[0].length; continue; }
    const word = /^\S+/.exec(text.slice(i));
    if (word) { out.push(word[0]); i += word[0].length; }
  }
  return out;
}

function encodeWord(syms, ranks) {
  let cur = syms.slice();
  while (cur.length > 1) {
    let best = -1, bestRank = Infinity;
    for (let j = 0; j + 1 < cur.length; j++) {
      const r = ranks[cur[j] + "\u0000" + cur[j + 1]];
      if (r !== undefined && r < bestRank) { bestRank = r; best = j; }
    }
    if (best < 0) break;
    cur.splice(best, 2, cur[best] + cur[best + 1]);
  }
  return cur;
}
"""
add_file("authored/tokenizer_helpers.js", js, "authored for NEXUS", "MIT (this repo)")

# ---------- 4. Synthetic instruction examples ----------
INSTRUCTIONS = []
def instr(task, prompt, answer):
    INSTRUCTIONS.append(
        "### Instruction:\n" + task + "\n### Input:\n" + prompt + "\n### Response:\n" + answer
    )

instr("Write a short professional email to schedule a meeting with a client.",
      "Client: Dana Cole. Topic: Q3 roadmap review.",
      "Subject: Q3 Roadmap Review - Meeting Request\n\n"
      "Dear Ms. Cole,\n\nI hope this message finds you well. I would like to schedule a "
      "brief meeting to walk through our Q3 roadmap and gather your feedback on the "
      "proposed priorities.\n\nWould Tuesday or Thursday next week work for you? I am "
      "happy to accommodate your schedule and can prepare an agenda in advance.\n\n"
      "Best regards,\nAlex")
instr("Write a Python function that reverses a string without slicing.",
      "reverse('nexus')",
      "def reverse(s: str) -> str:\n"
      "    result = []\n"
      "    for ch in s:\n"
      "        result.insert(0, ch)\n"
      "    return ''.join(result)\n\n"
      "print(reverse('nexus'))  # 'suxen'")
instr("Summarize the following sentence in plain words.",
      "The mechanism halts oscillation of the shaft when rotational speed exceeds tolerance.",
      "The device stops the shaft from shaking when it spins too fast.")
instr("Debug this Python snippet.",
      "def add(a b): return a + b",
      "The function definition is missing a comma between parameters. Fixed:\n"
      "def add(a, b):\n    return a + b")
instr("Write a bash one-liner to count lines in all .py files under the current folder.",
      "",
      "find . -name '*.py' -type f -exec cat {} + | wc -l")
instr("Write a SQL query selecting users created after 2024 ordered by name.",
      "table: users(id, name, created_at)",
      "SELECT id, name, created_at FROM users\nWHERE created_at > '2024-01-01'\nORDER BY name;")
instr("Write a friendly thank-you note to a colleague who helped with a launch.",
      "Colleague: Priya. Launch: new offline AI assistant app.",
      "Hi Priya,\n\nThank you so much for the extra hours you put into the launch of our "
      "offline assistant. Your calm handling of the release checklist kept the whole team "
      "on track, and the app shipped without a single critical issue.\n\nI am grateful to "
      "have you on this team.\n\nWarmly,\nSam")
instr("Explain what a transformer attention head does in one sentence for a beginner.",
      "",
      "An attention head looks at every word in the sentence and decides which other words "
      "it should gather information from before updating its own meaning.")
instr("Refactor this JavaScript to use const.",
      "var x = 5; var y = x * 2;",
      "const x = 5;\nconst y = x * 2;")
instr("Write a Kotlin function checking if a string is a palindrome.",
      "",
      "fun isPalindrome(s: String): Boolean {\n"
      "    val t = s.filter { it.isLetterOrDigit() }.lowercase()\n"
      "    return t == t.reversed()\n}")
for i in range(30):
    instr("Complete the sentence with a sensible continuation.",
          "The little robot booted its offline language model and",
          " waited patiently for its owner to type the first message of the day.")
    instr("Answer the user question directly and briefly.",
          "Can you work without internet?",
          "Yes. Everything I use to reply runs locally on this device, including the "
          "language model and your stored memories.")
    instr("Generate a unit test for a Python adder.",
          "def add(a, b): return a + b",
          "import unittest\nfrom mymod import add\n\nclass TestAdd(unittest.TestCase):\n"
          "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n"
          "        self.assertEqual(add(-1, 1), 0)\n\nif __name__ == '__main__':\n"
          "    unittest.main()")

add_file("instruct/synthetic_instructions.txt", "\n<|end|>\n".join(INSTRUCTIONS) + "\n<|end|>\n",
         "authored for NEXUS (synthetic, labeled)", "MIT (this repo)")

with open(os.path.join(CORPUS, "..", "corpus_manifest.json"), "w") as f:
    json.dump(MANIFEST, f, indent=2)
total = sum(m["bytes"] for m in MANIFEST)
print(f"CORPUS READY: {len(MANIFEST)} files, {total/1e6:.2f} MB")
