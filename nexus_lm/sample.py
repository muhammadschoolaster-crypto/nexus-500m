"""Generate samples from a trained NEXUS checkpoint (greedy or temperature)."""
import argparse

import torch

from .model import NexusLM, NexusConfig
from .tokenizer import NexusTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="\n<|text|>\n")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = NexusConfig(**ck["cfg"]) if "cfg" in ck else NexusConfig()
    model = NexusLM(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = NexusTokenizer.load(args.tokenizer)

    ids = tok.encode(args.prompt)
    x = torch.tensor([ids], dtype=torch.long)
    y = model.generate(x, args.n,
                       temperature=0.001 if args.greedy else args.temperature,
                       top_k=1 if args.greedy else args.top_k)
    print(tok.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
