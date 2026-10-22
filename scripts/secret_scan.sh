#!/usr/bin/env bash
# Secret scan: blocks a push if anything secret-looking is staged.
# Used manually before pushes and by CI. Exits non-zero on findings.
set -uo pipefail
cd "$(dirname "$0")/.."

PATTERNS=(
  'github_pat_[A-Za-z0-9_]\{16,\}'
  'ghp_[A-Za-z0-9]\{30,\}'
  'gho_[A-Za-z0-9]\{30,\}'
  'AKIA[0-9A-Z]\{16\}'
  'sk-[A-Za-z0-9]\{32,\}'
  'xoxb-[A-Za-z0-9-]\{10,\}'
  '-----BEGIN [A-Z ]*PRIVATE KEY-----'
  'AIza[0-9A-Za-z_-]\{35\}'
)
# Whitelisted: the SFT seeds intentionally contain the *shape* of a key pattern
# in educational text only if needed; none do today. Keep list explicit:
WHITELIST=()

fail=0
for pat in "${PATTERNS[@]}"; do
  while IFS= read -r line; do
    file="${line%%:*}"
    hit=1
    for w in "${WHITELIST[@]:-}"; do
      [[ "$line" == *"$w"* ]] && hit=0 && break
    done
    if [[ $hit -eq 1 ]]; then
      echo "SECRET SCAN FAILED: $line" >&2
      fail=1
    fi
  done < <(git grep -nE "$pat" -- . ':(exclude).git' 2>/dev/null || true)
done

if [[ $fail -ne 0 ]]; then
  echo "Refusing to proceed: remove the secret(s) first." >&2
  exit 1
fi
echo "secret scan: clean"
