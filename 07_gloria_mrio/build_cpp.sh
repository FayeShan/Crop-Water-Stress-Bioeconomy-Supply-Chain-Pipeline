#!/bin/bash
# Build the C++ compute kernel
# Run from the project root: bash build_cpp.sh
#
# Prerequisites:
#   pip install pybind11
#   apt-get install -y build-essential  (if using Docker slim image)
#
# With CBLAS (recommended, ~2x faster):
#   apt-get install -y libopenblas-dev
#   USE_CBLAS=1 bash build_cpp.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_DIR="$SCRIPT_DIR/src/gloria_mrio/_cpp"
TARGET_DIR="$SCRIPT_DIR/src/gloria_mrio"

echo "=== Building GLORIA C++ Compute Kernel ==="

# Check pybind11
python -c "import pybind11" 2>/dev/null || {
    echo "Installing pybind11..."
    pip install pybind11
}

cd "$CPP_DIR"

# Set CBLAS flag if openblas is available
if [ -z "$USE_CBLAS" ]; then
    if ldconfig -p 2>/dev/null | grep -q libopenblas; then
        echo "OpenBLAS detected, enabling CBLAS"
        export USE_CBLAS=1
    fi
fi

if [ "$USE_CBLAS" = "1" ]; then
    export CFLAGS="${CFLAGS:-} -DUSE_CBLAS"
    export LDFLAGS="${LDFLAGS:-} -lopenblas"
    echo "  CBLAS: enabled"
else
    echo "  CBLAS: disabled (manual matmul, still fast for small matrices)"
fi

python setup.py build_ext --inplace 2>&1

# Copy .so to the package directory
SO_FILE=$(ls _compute_kernel*.so 2>/dev/null || ls _compute_kernel*.pyd 2>/dev/null || true)
if [ -n "$SO_FILE" ]; then
    cp "$SO_FILE" "$TARGET_DIR/"
    echo ""
    echo "=== Build successful ==="
    echo "  Installed: $TARGET_DIR/$SO_FILE"
    echo ""
    echo "  Verify: python -c \"from gloria_mrio._compute_kernel import compute_indirect_batched; print('OK')\""
else
    echo "ERROR: Build failed - no .so file produced"
    exit 1
fi
