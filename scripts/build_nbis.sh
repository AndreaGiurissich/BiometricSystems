#!/usr/bin/env bash
# Build NBIS (mindtct + bozorth3) from a maintained mirror on a modern Ubuntu
# (e.g. Kaggle). Injects -fcommon (+ implicit-decl warning downgrades) so it
# compiles on gcc >= 10 (and >= 14). Headless build (--without-X11).
#
# Usage: bash scripts/build_nbis.sh [INSTALL_DIR] [SRC_DIR]
#   INSTALL_DIR default: /kaggle/working/nbis/install   (binaries -> $INSTALL_DIR/bin)
#   SRC_DIR     default: /kaggle/working/nbis/src
#
# Prints compile wall-time, the resolved binary paths, and PASS/FAIL. The full
# compiler log goes to <parent>/build.log; on failure the tail is printed.
# A 30-minute cap is enforced on the make step (timeout 1800).
set -uo pipefail

INSTALL_DIR="${1:-/kaggle/working/nbis/install}"
SRC_DIR="${2:-/kaggle/working/nbis/src}"
PRIMARY_REPO="https://github.com/lessandro/nbis.git"
FALLBACK_REPO="https://github.com/biometric-technologies/nist-biometric-image-software-nbis.git"
EXTRA_FLAGS="-fcommon -Wno-implicit-function-declaration -Wno-int-conversion -Wno-implicit-int"
BUILD_LOG="$(dirname "$SRC_DIR")/build.log"

echo "== NBIS build =="
echo "install : $INSTALL_DIR"
echo "src     : $SRC_DIR"
echo "log     : $BUILD_LOG"

MINDTCT="$INSTALL_DIR/bin/mindtct"
BOZORTH3="$INSTALL_DIR/bin/bozorth3"
if [ -x "$MINDTCT" ] && [ -x "$BOZORTH3" ]; then
  echo "Binaries already present, skipping build."
  echo "MINDTCT=$MINDTCT"
  echo "BOZORTH3=$BOZORTH3"
  exit 0
fi

# --- toolchain ----------------------------------------------------------------
if ! command -v gcc >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
  echo "Installing build-essential..."
  apt-get update -qq && apt-get install -y -qq build-essential || {
    echo "FAIL: cannot install build-essential"; exit 2; }
fi
command -v git >/dev/null 2>&1 || apt-get install -y -qq git || true
# Optional image libs; NBIS bundles its own copies, so failures here are non-fatal.
apt-get install -y -qq libpng-dev libjpeg-dev libopenjp2-7-dev zlib1g-dev unzip 2>/dev/null || true
echo "gcc: $(gcc --version | head -1)"

# --- fetch source -------------------------------------------------------------
mkdir -p "$(dirname "$SRC_DIR")"
if [ ! -f "$SRC_DIR/setup.sh" ]; then
  rm -rf "$SRC_DIR"
  echo "Cloning $PRIMARY_REPO ..."
  if ! git clone --depth 1 "$PRIMARY_REPO" "$SRC_DIR"; then
    echo "Primary clone failed, trying fallback $FALLBACK_REPO ..."
    git clone --depth 1 "$FALLBACK_REPO" "$SRC_DIR" || { echo "FAIL: cannot clone NBIS"; exit 3; }
  fi
fi
cd "$SRC_DIR" || { echo "FAIL: cannot cd $SRC_DIR"; exit 3; }
[ -f setup.sh ] || { echo "FAIL: setup.sh not in $SRC_DIR (mirror layout differs):"; ls -la; exit 4; }

# --- configure ----------------------------------------------------------------
chmod +x setup.sh
mkdir -p "$INSTALL_DIR"   # NBIS setup.sh aborts with "Directory doesn't exist!" otherwise
echo "Running setup.sh $INSTALL_DIR --without-X11 ..."
./setup.sh "$INSTALL_DIR" --without-X11 || { echo "FAIL: setup.sh"; exit 5; }

patch_flags() {
  # Append EXTRA_FLAGS to every CFLAGS line and the ARCH_FLAG line we can find.
  while IFS= read -r f; do
    sed -i "s|^CFLAGS\(.*\)|CFLAGS\1 $EXTRA_FLAGS|" "$f"
  done < <(find . -name 'rules.mak' 2>/dev/null)
  [ -f arch.mak ] && sed -i "s|^ARCH_FLAG = \(.*\)|ARCH_FLAG = \1 $EXTRA_FLAGS|" arch.mak
}

echo "Injecting flags (pre-config): $EXTRA_FLAGS"
patch_flags

# --- build (capped at 30 min) -------------------------------------------------
echo "make config / it / install (capped 1800s) -> $BUILD_LOG"
START=$(date +%s)
timeout 1800 bash -c "make config" >"$BUILD_LOG" 2>&1
echo "Injecting flags (post-config) ..."
patch_flags
timeout 1800 bash -c "make it && make install" >>"$BUILD_LOG" 2>&1
RC=$?
END=$(date +%s)
echo "Build wall time: $((END - START))s"

if [ "$RC" -eq 124 ]; then
  echo "FAIL: build exceeded the 30-minute cap. Surface this; do not auto-retry plan B."
  tail -n 40 "$BUILD_LOG"; exit 8
fi
if [ "$RC" -ne 0 ]; then
  echo "FAIL: make returned $RC. Last 50 log lines:"
  tail -n 50 "$BUILD_LOG"; exit 6
fi

if [ -x "$MINDTCT" ] && [ -x "$BOZORTH3" ]; then
  echo "PASS"
  echo "MINDTCT=$MINDTCT"
  echo "BOZORTH3=$BOZORTH3"
  exit 0
fi
echo "FAIL: install finished but binaries not at expected paths. Searching..."
find "$INSTALL_DIR" "$SRC_DIR" \( -name mindtct -o -name bozorth3 \) -type f 2>/dev/null
exit 7
