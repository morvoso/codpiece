#!/usr/bin/env bash
# Fetch vendored dependencies at the pinned revision.
# Pin policy: exactly the llama.cpp build production runs (see docs/ARCHITECTURE.md).
set -euo pipefail

PIN_TAG="b10423"
PIN_COMMIT="a94d563ed801d1da1b8c2432946de07d0231bb3d"
REPO="https://github.com/ggml-org/llama.cpp"

root="$(cd "$(dirname "$0")/.." && pwd)"
dst="$root/third_party/llama.cpp"

if [ -d "$dst/.git" ]; then
    have="$(git -C "$dst" rev-parse HEAD)"
    if [ "$have" = "$PIN_COMMIT" ]; then
        echo "already at pin $PIN_TAG ($PIN_COMMIT)"
        exit 0
    fi
    echo "existing checkout at $have, refetching pin $PIN_TAG"
    git -C "$dst" fetch --depth 1 origin "refs/tags/$PIN_TAG:refs/tags/$PIN_TAG"
    git -C "$dst" checkout -q "refs/tags/$PIN_TAG"
else
    mkdir -p "$root/third_party"
    git clone --depth 1 --branch "$PIN_TAG" "$REPO" "$dst"
fi

have="$(git -C "$dst" rev-parse HEAD)"
if [ "$have" != "$PIN_COMMIT" ]; then
    echo "ERROR: pin mismatch: tag $PIN_TAG resolved to $have, expected $PIN_COMMIT" >&2
    exit 1
fi
echo "vendored llama.cpp @ $PIN_TAG ($PIN_COMMIT)"

# codpiece's fixes to the vendored ggml — the meta backend's graph-reuse lifetime
# bugs, ggml_graph_set_new_uid(), and a flash-attention shape diagnostic. The
# engine's measured numbers depend on these; a build without them is a different
# engine. See notes/results-2026-08-19.md for what each hunk fixes.
patch="$root/patches/ggml-codpiece.patch"
if [ -f "$patch" ]; then
    if git -C "$dst" apply --check "$patch" 2>/dev/null; then
        git -C "$dst" apply "$patch"
        echo "applied patches/ggml-codpiece.patch"
    elif git -C "$dst" apply --check --reverse "$patch" 2>/dev/null; then
        echo "patches/ggml-codpiece.patch already applied"
    else
        echo "ERROR: patches/ggml-codpiece.patch applies neither forward nor reverse" >&2
        exit 1
    fi
fi
