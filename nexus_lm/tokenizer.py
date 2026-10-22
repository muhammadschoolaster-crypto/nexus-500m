"""NEXUS tokenizer: character-level BPE trained on the corpus.

Implemented from scratch (pure stdlib). Design is deliberately simple and
exactly portable to on-device Kotlin inference:
  - Pretokenize: split text into runs of non-whitespace / whitespace.
  - Base vocab: every character seen at train time (id 1..N).
  - Merges: learned pair merges, stored in rank order.
  - Encode: per pretoken, repeatedly apply lowest-rank applicable merge.
  - Decode: concatenate token strings (lossless).
Unknown characters at inference map to <unk> (id 0).
"""
import json
import re
import time
from collections import Counter, defaultdict

PRETOK_RE = re.compile(r"\s+|\S+")
UNK_ID = 0


class NexusTokenizer:
    def __init__(self, vocab=None, merges=None):
        # vocab: list of strings, index = id. vocab[0] = "<unk>"
        self.vocab = vocab if vocab is not None else ["<unk>"]
        self.merges = merges if merges is not None else []  # list of [left, right]
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.ranks = {tuple(m): i for i, m in enumerate(self.merges)}

    # ---------------- training ----------------
    @classmethod
    def train(cls, texts, target_vocab: int, verbose: bool = True):
        t0 = time.time()
        pretok_counts = Counter()
        base_chars = set()
        for t in texts:
            for w in PRETOK_RE.findall(t):
                pretok_counts[w] += 1
                base_chars.update(w)

        base = sorted(base_chars)
        vocab = ["<unk>"] + base
        assert len(vocab) < target_vocab, "corpus has more unique chars than target vocab"

        # words as tuples of token ids -> strings for merging
        words = []  # list of (tuple_of_symbols, freq)
        for w, f in pretok_counts.items():
            words.append((tuple(w), f))

        # pair counts + index pair -> set of word indices
        pair_counts = Counter()
        pair_to_words = defaultdict(set)
        for i, (syms, f) in enumerate(words):
            for a, b in zip(syms, syms[1:]):
                pair_counts[(a, b)] += f
                pair_to_words[(a, b)].add(i)

        merges = []
        n_merges = target_vocab - len(vocab)
        for mi in range(n_merges):
            if not pair_counts:
                break
            (a, b), cnt = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            if cnt < 2:
                break
            new_sym = a + b
            merges.append([a, b])
            vocab.append(new_sym)
            affected = pair_to_words.get((a, b), set())
            for wi in list(affected):
                syms, f = words[wi]
                # rebuild word with the new merge (all occurrences)
                out = []
                j = 0
                changed = False
                while j < len(syms):
                    if j < len(syms) - 1 and syms[j] == a and syms[j + 1] == b:
                        out.append(new_sym)
                        j += 2
                        changed = True
                    else:
                        out.append(syms[j])
                        j += 1
                if not changed:
                    continue
                new_syms = tuple(out)
                # decrement old pairs / increment new pairs
                for x, y in zip(syms, syms[1:]):
                    pair_counts[(x, y)] -= f
                    if pair_counts[(x, y)] <= 0:
                        pair_counts.pop((x, y), None)
                        pair_to_words.pop((x, y), None)
                for x, y in zip(new_syms, new_syms[1:]):
                    pair_counts[(x, y)] += f
                    pair_to_words[(x, y)].add(wi)
                words[wi] = (new_syms, f)
            pair_counts.pop((a, b), None)
            pair_to_words.pop((a, b), None)
            if verbose and (mi + 1) % 200 == 0:
                print(f"  merge {mi+1}/{n_merges} (+{new_sym!r} count={cnt}) "
                      f"vocab={len(vocab)} elapsed={time.time()-t0:.1f}s")
        tok = cls(vocab=vocab, merges=merges)
        if verbose:
            print(f"tokenizer trained: vocab={len(tok.vocab)} merges={len(tok.merges)} "
                  f"in {time.time()-t0:.1f}s")
        return tok

    # ---------------- encoding ----------------
    def _encode_pretoken(self, syms: tuple) -> list:
        if len(syms) == 1:
            s = syms[0]
            return [self.stoi.get(s, UNK_ID)]
        # repeatedly apply the lowest-rank applicable merge
        out = list(syms)
        while True:
            best_i, best_rank = None, None
            for i in range(len(out) - 1):
                r = self.ranks.get((out[i], out[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_i, best_rank = i, r
            if best_i is None:
                break
            left, right = self.merges[best_rank]
            merged = left + right
            out = out[:best_i] + [merged] + out[best_i + 2:]
        return [self.stoi.get(s, UNK_ID) for s in out]

    def encode(self, text: str) -> list:
        ids = []
        for w in PRETOK_RE.findall(text):
            ids.extend(self._encode_pretoken(tuple(w)))
        return ids

    def decode(self, ids) -> str:
        parts = []
        for i in ids:
            if 0 < i < len(self.vocab):
                parts.append(self.vocab[i])
        return "".join(parts)

    # ---------------- persistence ----------------
    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> "NexusTokenizer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return NexusTokenizer(vocab=d["vocab"], merges=[tuple(m) for m in d["merges"]])

    def stats(self, texts):
        n_tok = sum(len(self.encode(t)) for t in texts)
        n_char = sum(len(t) for t in texts)
        return {"tokens": n_tok, "chars": n_char, "chars_per_token": n_char / max(n_tok, 1)}
