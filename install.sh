#!/usr/bin/env bash
# Installs ~/vllm-dspark-env for bare-metal DeepSeek V4 Flash DSpark (no Docker).
#
# What it does:
#   1. Creates ~/vllm-dspark-env (new isolated venv) by cloning site-packages
#      from ~/vllm-env (which has torch 2.11.0+cu130, flashinfer 0.6.13, etc.)
#   2. Wires up the editable vllm install so the venv points at
#      ~/Code/_forks/vllm (gb10-dspark branch, pre-compiled .so files).
#   3. Runs verify.py to confirm the stack is healthy.
#
# nvfp4_ds_mla support lives in ~/Code/_forks/vllm (gb10-dspark branch) —
# no runtime patching needed here.
#
# Idempotent — safe to re-run. Works on sparky and scenty.
set -euo pipefail

DSPARK_ENV="${HOME}/vllm-dspark-env"
BASE_ENV="${HOME}/vllm-env"
VLLM_SRC="${HOME}/Code/_forks/vllm"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

die() { echo "ERROR: $*" >&2; exit 1; }
ok()  { echo "[OK] $*"; }
info(){ echo ">>> $*"; }

# ── 0. pre-flight ─────────────────────────────────────────────────────────────
[ -d "$BASE_ENV" ]   || die "$BASE_ENV not found — is this sparky or scenty?"
[ -d "$VLLM_SRC" ]   || die "$VLLM_SRC not found — ~/Code/_forks/vllm must exist"
[ -f "$VLLM_SRC/vllm/_C_stable_libtorch.abi3.so" ] || \
  die "Compiled vllm extensions not found in $VLLM_SRC/vllm/. Pre-built .so files required."

BRANCH=$(git -C "$VLLM_SRC" branch --show-current 2>/dev/null || echo "unknown")
if [ "$BRANCH" != "gb10-dspark" ]; then
  die "$VLLM_SRC is on branch '$BRANCH', expected 'gb10-dspark'. Run: git -C $VLLM_SRC checkout gb10-dspark"
fi

# ── 1. create venv ────────────────────────────────────────────────────────────
if [ ! -d "$DSPARK_ENV" ]; then
  info "Creating $DSPARK_ENV..."
  python3.12 -m venv "$DSPARK_ENV"
else
  ok "$DSPARK_ENV already exists — skipping venv creation"
fi

SITE_DST="$DSPARK_ENV/lib/python3.12/site-packages"
SITE_SRC="$BASE_ENV/lib/python3.12/site-packages"

# ── 2. sync site-packages from base venv ──────────────────────────────────────
if ! "$DSPARK_ENV/bin/python" -c "import torch" 2>/dev/null; then
  info "Syncing site-packages from $BASE_ENV (torch, flashinfer, xgrammar, etc.)..."
  # Exclude vllm/ — only .pre-* backup files that create a namespace package
  # collision shadowing the editable install. Also exclude dist-info; step 3 copies it.
  rsync -a --exclude='__pycache__' --exclude='vllm/' --exclude='vllm-*.dist-info/' \
    "$SITE_SRC/" "$SITE_DST/"
  ok "site-packages synced"
else
  ok "torch already importable in $DSPARK_ENV — skipping rsync"
fi

# Remove any stale vllm/ namespace directory (idempotent).
rm -rf "$SITE_DST/vllm/"

# ── 3. ensure editable vllm install ───────────────────────────────────────────
if ! "$DSPARK_ENV/bin/python" -c "import vllm; assert vllm.__file__" 2>/dev/null; then
  info "Linking editable vllm install ($VLLM_SRC)..."
  EDITABLE_PTH=$(ls "$SITE_SRC"/__editable__*.pth 2>/dev/null | head -1 || true)
  if [ -n "$EDITABLE_PTH" ]; then
    # sparky-style: base venv has editable install metadata — copy it
    for f in "$SITE_SRC"/__editable__*.pth "$SITE_SRC"/__editable__*finder.py; do
      [ -f "$f" ] && cp -f "$f" "$SITE_DST/"
    done
    for d in "$SITE_SRC"/vllm-*.dist-info; do
      [ -d "$d" ] && cp -rf "$d" "$SITE_DST/"
    done
    ok "editable vllm install metadata copied from $BASE_ENV"
  else
    # scenty-style: base venv has a wheel install — add VLLM_SRC to sys.path directly
    echo "$VLLM_SRC" > "$SITE_DST/vllm-dspark-editable.pth"
    ok "vllm path entry created ($VLLM_SRC)"
  fi
else
  ok "vllm already importable in $DSPARK_ENV — skipping"
fi

"$DSPARK_ENV/bin/python" -c "
import vllm, pathlib
p = pathlib.Path(vllm.__file__)
assert str(p).startswith('$VLLM_SRC'), f'vllm loaded from unexpected path: {p}'
" || die "vllm import check failed"
ok "vllm loads from $VLLM_SRC"

# Create vllm console script if not present (not copied by the .pth approach).
if [ ! -x "$DSPARK_ENV/bin/vllm" ]; then
  cat > "$DSPARK_ENV/bin/vllm" <<SCRIPT
#!$DSPARK_ENV/bin/python3
# -*- coding: utf-8 -*-
import re, sys
from vllm.entrypoints.cli.main import main
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?\$', '', sys.argv[0])
    sys.exit(main())
SCRIPT
  chmod +x "$DSPARK_ENV/bin/vllm"
  ok "vllm console script created"
else
  ok "vllm console script already present"
fi

# ── 4. verify ─────────────────────────────────────────────────────────────────
info "Running verify.py..."
"$DSPARK_ENV/bin/python" "$SCRIPT_DIR/verify.py"
