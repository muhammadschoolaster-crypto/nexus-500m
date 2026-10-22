"""Byte-level BPE tokenizer for NEXUS models (trained from scratch).

Design decisions (deliberate, for Android parity):

- Base vocabulary: the 256 possible bytes, using the GPT-2 byte-to-unicode
  mapping so every token is printable text. Base token id == byte value (0..255).
- Learned merges: id 256 + merge_index. The first merge that produces a given
  string wins if two merges ever produce the same string (decode stays exact).
- One reserved special token: ``<eos>`` with id ``vocab_size - 1``.
- No regex pre-tokenization at encode time: merges apply to the whole byte-char
  stream. This makes the encoder a ~40-line loop that is trivially re-implemented
  identically in Kotlin (see android ... model/BpeTokenizer.kt) so on-device
  tokenization matches training tokenization exactly.
"""
from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List

BASE_VOCAB = 256


def bytes_to_unicode() -> dict:
    """GPT-2's reversible byte <-> unicode-char mapping (identical logic in Kotlin)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_ENCODER = bytes_to_unicode()                     # byte value -> char
BYTE_DECODER = {c: b for b, c in BYTE_ENCODER.items()}


def _split_chunks(text: str) -> List[str]:
    """Split into chunks, keeping whitespace attached to the preceding chunk."""
    chunks, cur = [], []
    for ch in text:
        cur.append(ch)
        if ch.isspace():
            chunks.append("".join(cur))
            cur = []
    if cur:
        chunks.append("".join(cur))
    return chunks


def _apply_merge(word: tuple, pair: tuple, new_sym: str) -> tuple:
    out, i = [], 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_sym)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train_bpe(texts: Iterable[str], num_merges: int, max_train_chars: int = 400_000) -> List[tuple]:
    """Greedy BPE merge training over whitespace-chunked words.

    Standard max-count BPE with a lazy max-heap; deterministic.
    """
    words: Counter = Counter()
    seen = 0
    stop = False
    for text in texts:
        if stop:
            break
        for w in _split_chunks(text):
            key = tuple(BYTE_ENCODER[b] for b in w.encode("utf-8"))
            words[key] += 1
            seen += len(w.encode("utf-8"))
            if seen >= max_train_chars:
                stop = True
                break

    pair_counts: defaultdict = defaultdict(int)
    pair_words: defaultdict = defaultdict(set)
    for w, c in words.items():
        for i in range(len(w) - 1):
            p = (w[i], w[i + 1])
            pair_counts[p] += c
            pair_words[p].add(w)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: List[tuple] = []
    while len(merges) < num_merges and heap:
        negc, pair = heapq.heappop(heap)
        c = pair_counts.get(pair, 0)
        if c <= 0:
            continue
        if -negc != c:                      # stale heap entry -> reinsert with true count
            heapq.heappush(heap, (-c, pair))
            continue

        new_sym = pair[0] + pair[1]
        for w in list(pair_words[pair]):
            c_w = words.pop(w)
            for i in range(len(w) - 1):
                p_old = (w[i], w[i + 1])
                pair_counts[p_old] -= c_w
                if pair_counts[p_old] <= 0:
                    pair_counts.pop(p_old, None)
                    pair_words.pop(p_old, None)
                else:
                    pair_words[p_old].discard(w)
            nw = _apply_merge(w, pair, new_sym)
            for i in range(len(nw) - 1):
                p_new = (nw[i], nw[i + 1])
                pair_counts[p_new] += c_w
                pair_words[p_new].add(nw)
                heapq.heappush(heap, (-pair_counts[p_new], p_new))
            words[nw] = words.get(nw, 0) + c_w
        merges.append(pair)

    return merges


class BPETokenizer:
    def __init__(self, merges: List[tuple] | None = None, version: int = 1):
        self.version = version
        self.merges: List[tuple] = [tuple(m) for m in (merges or [])]
        self.ranks = {tuple(m): i for i, m in enumerate(self.merges)}
        # merged string -> token id (first merge that produced the string wins)
        self.merge_ids: dict = {}
        for i, (a, b) in enumerate(self.merges):
            self.merge_ids.setdefault(a + b, BASE_VOCAB + i)
        # token id -> string (for decoding)
        self.id_to_sym: dict = {b: s for b, s in BYTE_ENCODER.items()}
        for i, (a, b) in enumerate(self.merges):
            self.id_to_sym.setdefault(BASE_VOCAB + i, a + b)
        self.eos_token_id = BASE_VOCAB + len(self.merges)
        self.vocab_size = self.eos_token_id + 1

    # ------------------------------------------------------------------
    def train(self, texts: Iterable[str], vocab_size: int, max_train_chars: int = 400_000) -> None:
        if vocab_size < BASE_VOCAB + 1:
            raise ValueError("vocab_size must reserve 256 byte tokens + 1 EOS token")
        num_merges = vocab_size - BASE_VOCAB - 1
        self.__init__(merges=train_bpe(texts, num_merges, max_train_chars))

    def _bpe(self, symbols: List[str]) -> List[str]:
        """Apply merges by rank, all occurrences of the best pair per round."""
        while len(symbols) >= 2:
            best_pair, best_rank = None, None
            for i in range(len(symbols) - 1):
                r = self.ranks.get((symbols[i], symbols[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_pair, best_rank = (symbols[i], symbols[i + 1]), r
            if best_pair is None:
                break
            new_sym = best_pair[0] + best_pair[1]
            symbols = list(_apply_merge(tuple(symbols), best_pair, new_sym))
        return symbols

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        syms = [BYTE_ENCODER[b] for b in text.encode("utf-8")]
        syms = self._bpe(syms)
        ids = []
        for s in syms:
            if len(s) == 1:
                ids.append(BYTE_DECODER[s])
            else:
                ids.append(self.merge_ids[s])   # guaranteed present: it was produced by a merge
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        parts = []
        for i in ids:
            sym = self.id_to_sym.get(i)
            if sym is None:                      # eos / unknown -> dropped
                continue
            for ch in sym:                       # expand multi-char merged symbols
                parts.append(BYTE_DECODER[ch])
        data = bytes(parts)
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    def save_json(self, path: str | Path) -> None:
        payload = {
            "version": self.version,
            "merges": [list(m) for m in self.merges],
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(merges=[tuple(m) for m in d["merges"]], version=d.get("version", 1))
        if d.get("vocab_size") and d["vocab_size"] != tok.vocab_size:
            raise ValueError(
                f"tokenizer file declares vocab_size={d['vocab_size']} "
                f"but merges imply {tok.vocab_size}"
            )
        return tok
