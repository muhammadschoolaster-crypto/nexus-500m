// NEXUS tokenizer helpers (JS mirror of the Kotlin implementation)
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
