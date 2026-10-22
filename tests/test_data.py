import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nexus.data import (clean_text, dedupe, find_sensitive, instruction_text,
                        load_instructions, pack_bins, split_docs)
from nexus.tokenizer import BPETokenizer


def test_sensitive_filter_catches_realistic_secrets():
    assert find_sensitive("contact me at ali.example@company.com")
    assert find_sensitive("token: hf_bsKdl90sJdkfMslqpQzXvBnLaTkLmNoPq")
    assert find_sensitive("aws key AKIAIOSFODNN7EXAMPLE here")
    assert find_sensitive("github_pat_11ABCDEFGH0123456789_abcdefghijklmnopqrstuvwxyz1234567890")
    assert find_sensitive("-----BEGIN RSA PRIVATE KEY-----")
    assert find_sensitive("my password = hunter2please")
    assert find_sensitive("call +92 300 1234567 now")
    assert not find_sensitive("the quick brown fox jumps over the lazy dog")
    assert not find_sensitive("def add(a, b): return a + b")


def test_clean_text_normalizes():
    assert clean_text("a  b\r\nc\n\n\n\nd") == "a b\nc\n\n\nd"


def test_dedupe_exact_and_normalized():
    docs = [
        {"text": "hello world this is a test document with enough length", "source": "s", "license": "l"},
        {"text": "hello world this is a test document with enough length", "source": "s", "license": "l"},
        {"text": "Hello, World! This is a test document-with enough length...", "source": "s", "license": "l"},
        {"text": "completely different content that is long enough to keep around", "source": "s", "license": "l"},
    ]
    kept = dedupe(docs)
    assert len(kept) == 2  # exact dup + normalized near-dup removed
    assert all("sha256" in d for d in kept)


def test_split_deterministic_and_disjoint():
    docs = [
        {"text": f"document number {i} with plenty of unique words to be long enough", "source": "s", "license": "l"}
        for i in range(200)
    ]
    docs = dedupe(docs)
    s1 = split_docs(docs)
    s2 = split_docs(docs)
    assert [d["sha256"] for d in s1["train"]] == [d["sha256"] for d in s2["train"]]
    all_ids = [d["sha256"] for split in s1.values() for d in split]
    assert len(all_ids) == len(set(all_ids)) == len(docs)  # disjoint + complete


def test_pack_bins_shapes_and_bounds(tmp_path):
    tok = BPETokenizer()  # raw bytes + EOS = 257
    docs = [
        {"text": f"sample document {i} " + "words here " * 30, "source": "s", "license": "l"}
        for i in range(10)
    ]
    docs = dedupe(docs)
    splits = {"train": docs[:7], "val": docs[7:], "test": []}
    paths = pack_bins(splits, tok, ctx_len=32, out_dir=tmp_path)
    arr = np.load(paths["train"])
    assert arr.shape[1] == 33  # ctx + 1
    assert arr.dtype == np.uint16
    assert arr.max() < tok.vocab_size
    assert np.load(paths["test"]).shape[0] == 0  # empty split handled


def test_instruction_template():
    s = instruction_text("Write a function.", "def f(): pass")
    assert s.startswith("### Instruction:\nWrite a function.\n### Response:\n")
    assert s.endswith("def f(): pass\n")


def test_load_instructions_validates(tmp_path):
    p = tmp_path / "sft.jsonl"
    p.write_text('{"instruction":"a","response":"b"}\n\nbad row\n', encoding="utf-8")
    import json
    p.write_text('{"instruction":"a","response":"b"}\n', encoding="utf-8")
    rows = load_instructions(p)
    assert rows == [{"instruction": "a", "response": "b"}]
    p.write_text('{"instruction":"a"}\n', encoding="utf-8")
    import pytest
    with pytest.raises(AssertionError):
        load_instructions(p)
