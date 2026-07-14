# vllm-dspark-build

Bare-metal venv setup for **DeepSeek V4 Flash DSpark** on 2x DGX Spark (sparky + scenty, GB10 Blackwell SM121a). No Docker.

## What this does

`install.sh` builds `~/vllm-dspark-env` — an isolated Python 3.12 venv that:

- Clones site-packages from `~/vllm-env` (torch 2.11.0+cu130, flashinfer 0.6.13, etc.) without rsync'ing from PyPI
- Wires an editable install pointing at `~/Code/_forks/vllm` (pre-compiled `.so` files, no cmake rebuild)
- Runs `verify.py` to confirm DSpark + B12X + nvfp4_ds_mla are all healthy

## Prerequisites

Both nodes must have `~/Code/_forks/vllm` checked out on `gb10-dspark`:

```bash
# If sdougbrown/vllm isn't already a remote:
git -C ~/Code/_forks/vllm remote add sdougbrown git@github.com:sdougbrown/vllm.git
git -C ~/Code/_forks/vllm fetch sdougbrown
git -C ~/Code/_forks/vllm checkout gb10-dspark
```

The branch must have compiled `.so` files in place (`vllm/_C_stable_libtorch.abi3.so`). They're checked in at the branch base SHA — don't delete them.

## Install

```bash
bash ~/Code/vllm-dspark-build/install.sh
```

Idempotent. Re-running is safe and fast (skips steps already done).

## Serve

```bash
~/Serve/serve-dspark-b12x-venv.sh
```

Run in a tmux window. The head process runs in the foreground on sparky; the worker is SSH-detached on scenty (logs at `/tmp/dspark-worker.log` on scenty).

## The `gb10-dspark` branch

`sdougbrown/vllm:gb10-dspark` is the canonical vllm for this stack:

| Commit | What |
|---|---|
| `3f99883d9` | Base — upstream vllm, SHA the `.so` files were compiled at |
| `2c72e4d72` | Spec decode prefill misclassification fix (#47381) |
| `2aef50168` | DSV4 TP16 garbage output fix (#47493) |
| `3fc9b39ee` | DSpark block-size validation in speculative config (#47419) |
| `dfec64e37` | Free out-of-window blocks under async scheduling (#47728) |
| `8f8e61c3b` | Batched DeepGEMM fix (#47884) |
| `2fa4f6d7a` | Hybrid SWA + full attention DFlash drafters + dspark utils (#47914) |
| `e95277198` | **nvfp4_ds_mla KV cache support** (DSpark B12X GB10) |

### Adding upstream fixes

```bash
git -C ~/Code/_forks/vllm fetch upstream
git -C ~/Code/_forks/vllm cherry-pick <sha>
git -C ~/Code/_forks/vllm push origin gb10-dspark
```

### Rebasing to a new upstream base

Only do this when a new upstream release includes compiled `.so` files (or after recompiling both nodes):

```bash
git -C ~/Code/_forks/vllm fetch upstream
git -C ~/Code/_forks/vllm rebase --onto <new-base> 3f99883d9 gb10-dspark
# resolve any conflicts, then:
git -C ~/Code/_forks/vllm push --force-with-lease origin gb10-dspark
# after confirming .so files are valid on both nodes, re-run install.sh on both
```

Cherry-picked upstream commits that are now in the new base will drop off automatically during the rebase.

### What's intentionally excluded

The `fix/deterministic-marlin-moe-align` personal commits (deterministic Marlin MoE alignment, new `moe_align_block_size_stable_small`/`_radix` C++ ops) are not on this branch. They only affect `MarlinMoEExperts`, which is not in the B12X path — `FlashInferB12xExperts` uses `flashinfer.fused_moe.B12xMoEWrapper` directly.

## Verify

```bash
~/vllm-dspark-env/bin/python ~/Code/vllm-dspark-build/verify.py
```

Checks: vllm source path, nvfp4_ds_mla in CacheDType, DSparkSpeculator, FlashInferB12xExperts, B12xMoEWrapper, DSparkDraftModel registry, sparse_mla supported dtypes, has_flashinfer_b12x_moe().
