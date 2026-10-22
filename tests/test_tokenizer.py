import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus.tokenizer import BPETokenizer, BASE_VOCAB, train_bpe


def test_bytes_mapping_is_reversible():
    from nexus.tokenizer import BYTE_ENCODER, BYTE_DECODER
    assert len(BYTE_ENCODER) == 256
    for b, ch in BYTE_ENCODER.items():
        assert BYTE_DECODER[ch] == b
    # all chars distinct
    assert len(set(BYTE_ENCODER.values())) == 256


def test_train_bpe_small_case():
    merges = train_bpe(["ababab cdcdcd"], num_merges=2, max_train_chars=1000)
    assert ("a", "b") == tuple(merges[0])   # most frequent pair first
    assert ("c", "d") == tuple(merges[1])
    tok = BPETokenizer(merges)
    assert tok.vocab_size == BASE_VOCAB + len(merges) + 1
    ids = tok.encode("ababab")
    # every "ab" should be one merged token
    assert ids == [BASE_VOCAB + 0, BASE_VOCAB + 0, BASE_VOCAB + 0]


def test_encode_decode_roundtrip_text():
    text = "Hello, World! def add(a, b): return a + b  \n tabs\tand\nnewlines 123 @#$%^&*()"
    merges = train_bpe([text * 10], num_merges=32, max_train_chars=20000)
    tok = BPETokenizer(merges)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.encode(text) == ids  # deterministic


def test_eos_token():
    tok = BPETokenizer()
    assert tok.vocab_size == BASE_VOCAB + 1
    assert tok.eos_token_id == BASE_VOCAB
    ids = tok.encode("hi", add_eos=True)
    assert ids[-1] == tok.eos_token_id


def test_merges_apply_by_rank():
    # ranks: ("a","b") is merge 0 -> must apply before ("ab","c") even if the
    # text exposes both pairs
    merges = [("a", "b"), ("ab", "c")]
    tok = BPETokenizer(merges)
    ids = tok.encode("abc")
    # "ab" merges first, then "ab"+"c" -> single token "abc" (id 257)
    assert ids == [BASE_VOCAB + 1]


def test_json_roundtrip(tmp_path):
    merges = train_bpe(["the quick brown fox jumps"], num_merges=16, max_train_chars=5000)
    tok = BPETokenizer(merges)
    p = tmp_path / "tok.json"
    tok.save_json(p)
    tok2 = BPETokenizer.load_json(p)
    assert tok2.merges == tok.merges
    assert tok2.vocab_size == tok.vocab_size
    s = "the quick brown fox"
    assert tok2.encode(s) == tok.encode(s)


def test_unknowable_bytes_never_crash():
    # every possible byte must be encodable (emoji, control chars, etc.)
    tok = BPETokenizer()
    for b in range(256):
        s = bytes([b]).decode("utf-8", errors="ignore")
        if s:
            ids = tok.encode(s)
            assert all(0 <= i < tok.vocab_size for i in ids)
