#!/usr/bin/env bash
# Correctness test suite for tachyon-pi (all independent of the internal tree).
set -u
cd "$(dirname "$0")/.."
BIN=${BIN:-./build/tachyon-pi}
MACH=${MACH:-./build/machin-check}
fail=0

echo "== smoke: 101 digits + verify =="
$BIN --digits 101 --verify || fail=1

echo "== 10k digits + verify =="
$BIN --digits 10000 --verify --quiet || fail=1

echo "== 100k digits + verify (BBP) =="
$BIN --digits 100000 --verify --quiet || fail=1

echo "== 1M digits: independent Machin cross-check =="
if [ "${SKIP_MACHIN:-0}" != 1 ]; then
  $BIN --digits 1000000 --quiet --out /tmp/tachyon_1m.txt
  $MACH 1000000 > /tmp/machin_1m.txt
  if cmp -s <(head -c 1000002 /tmp/tachyon_1m.txt) <(head -c 1000002 /tmp/machin_1m.txt); then
    echo "  1M-digit Machin vs Chudnovsky: MATCH"
  else
    echo "  1M-digit Machin vs Chudnovsky: MISMATCH"; fail=1
  fi
fi

echo "== optional: reference pi-million.txt diff (skipped if offline) =="
if [ -f /tmp/pi-million.txt ]; then
  $BIN --digits 1000001 --quiet --out /tmp/tachyon_ref.txt
  if cmp -s <(tr -d '.\n' < /tmp/tachyon_ref.txt | head -c 1000001) \
            <(tr -d '.\n' < /tmp/pi-million.txt | head -c 1000001); then
    echo "  reference file: MATCH"
  else
    echo "  reference file: MISMATCH"; fail=1
  fi
else
  echo "  (no /tmp/pi-million.txt; skipping)"
fi

echo "== leaf-size invariance (result must not depend on leaf) =="
a=$($BIN --digits 5000 --leaf 1 --quiet | head -c 200)
b=$($BIN --digits 5000 --leaf 64 --quiet | head -c 200)
[ "$a" = "$b" ] && echo "  ok" || { echo "  FAIL"; fail=1; }

echo
[ $fail = 0 ] && echo "ALL TESTS PASSED" || echo "TESTS FAILED"
exit $fail
