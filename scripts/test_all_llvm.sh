#!/bin/bash

# Test script for all LLVM versions with chipStar
# Tests LLVM 19, 20, 21, 22: configure, build, install, build chipStar, run tests
# Continues testing all versions even if one fails

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CHIPSTAR_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
INSTALL_BASE="$HOME/install/llvm"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
declare -A TEST_RESULTS
declare -A BUILD_RESULTS

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to test a single LLVM version
test_llvm_version() {
    local VERSION=$1
    local INSTALL_DIR="$INSTALL_BASE/${VERSION}.0"
    local BUILD_DIR="$CHIPSTAR_ROOT/build-llvm${VERSION}"
    
    log_info "========================================="
    log_info "Testing LLVM ${VERSION}"
    log_info "========================================="
    
    # Step 1: Configure LLVM (test patches)
    log_info "Step 1: Configuring LLVM ${VERSION} (testing patches)..."
    if ! (cd "$CHIPSTAR_ROOT" && "$SCRIPT_DIR/configure_llvm.sh" --version "$VERSION" --install-dir "$INSTALL_DIR" --configure-only 2>&1); then
        log_error "LLVM ${VERSION} configuration failed!"
        BUILD_RESULTS[$VERSION]="CONFIG_FAILED"
        return 1
    fi
    log_info "LLVM ${VERSION} configuration successful"
    
    # Step 2: Build and install LLVM
    log_info "Step 2: Building and installing LLVM ${VERSION}..."
    if ! (cd "$CHIPSTAR_ROOT" && "$SCRIPT_DIR/configure_llvm.sh" --version "$VERSION" --install-dir "$INSTALL_DIR" 2>&1); then
        log_error "LLVM ${VERSION} build/install failed!"
        BUILD_RESULTS[$VERSION]="BUILD_FAILED"
        return 1
    fi
    log_info "LLVM ${VERSION} build/install successful"
    
    # Step 3: Build chipStar
    log_info "Step 3: Building chipStar with LLVM ${VERSION}..."
    
    # Load LLVM module if available
    if command -v module >/dev/null 2>&1; then
        module load "llvm/${VERSION}.0" 2>/dev/null || true
    fi
    
    # Create build directory if it doesn't exist
    if [ ! -d "$BUILD_DIR" ]; then
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        
        # Configure chipStar
        cmake -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLVM_CONFIG_BIN="$INSTALL_DIR/bin/llvm-config" \
            -DCHIP_BACKEND_OPENCL=ON \
            -DCHIP_BACKEND_LEVEL0=ON \
            "$CHIPSTAR_ROOT" || {
            log_error "chipStar CMake configuration failed for LLVM ${VERSION}!"
            BUILD_RESULTS[$VERSION]="CHIPSTAR_CONFIG_FAILED"
            return 1
        }
    else
        cd "$BUILD_DIR"
    fi
    
    # Build chipStar
    if ! ninja CHIP; then
        log_error "chipStar build failed for LLVM ${VERSION}!"
        BUILD_RESULTS[$VERSION]="CHIPSTAR_BUILD_FAILED"
        return 1
    fi
    log_info "chipStar build successful for LLVM ${VERSION}"
    
    # Step 4: Build tests
    log_info "Step 4: Building chipStar unit tests..."
    if ! ninja build_tests; then
        log_error "chipStar test build failed for LLVM ${VERSION}!"
        BUILD_RESULTS[$VERSION]="TESTS_BUILD_FAILED"
        return 1
    fi
    log_info "chipStar tests built successfully for LLVM ${VERSION}"
    
    # Step 5: Run tests
    log_info "Step 5: Running chipStar unit tests for LLVM ${VERSION}..."
    
    local TEST_FAILED=0
    
    # Test OpenCL backend
    log_info "Running OpenCL backend tests..."
    if "$CHIPSTAR_ROOT/scripts/check.py" "$BUILD_DIR" dgpu opencl > "$BUILD_DIR/test_results_opencl_llvm${VERSION}.log" 2>&1; then
        local OPENCL_PASSED=$(grep -E "tests passed|tests failed" "$BUILD_DIR/test_results_opencl_llvm${VERSION}.log" | tail -1)
        log_info "OpenCL tests completed for LLVM ${VERSION}: $OPENCL_PASSED"
    else
        log_error "OpenCL tests failed for LLVM ${VERSION}"
        TEST_FAILED=1
    fi
    
    # Test Level0 backend
    log_info "Running Level0 backend tests..."
    if "$CHIPSTAR_ROOT/scripts/check.py" "$BUILD_DIR" dgpu level0 > "$BUILD_DIR/test_results_level0_llvm${VERSION}.log" 2>&1; then
        local LEVEL0_PASSED=$(grep -E "tests passed|tests failed" "$BUILD_DIR/test_results_level0_llvm${VERSION}.log" | tail -1)
        log_info "Level0 tests completed for LLVM ${VERSION}: $LEVEL0_PASSED"
    else
        log_error "Level0 tests failed for LLVM ${VERSION}"
        TEST_FAILED=1
    fi
    
    if [ $TEST_FAILED -eq 0 ]; then
        TEST_RESULTS[$VERSION]="PASSED"
        BUILD_RESULTS[$VERSION]="SUCCESS"
        log_info "All tests passed for LLVM ${VERSION}!"
    else
        TEST_RESULTS[$VERSION]="FAILED"
        BUILD_RESULTS[$VERSION]="TESTS_FAILED"
        log_warn "Some tests failed for LLVM ${VERSION}"
    fi
    
    return $TEST_FAILED
}

# Main execution
main() {
    local VERSIONS=(19 20 21 22)
    local FAILED_VERSIONS=()
    
    log_info "Starting comprehensive LLVM testing for chipStar"
    log_info "Testing versions: ${VERSIONS[*]}"
    log_info "Install base: $INSTALL_BASE"
    log_info "chipStar root: $CHIPSTAR_ROOT"
    
    # Test each version
    for VERSION in "${VERSIONS[@]}"; do
        if ! test_llvm_version "$VERSION"; then
            FAILED_VERSIONS+=("$VERSION")
        fi
        echo ""
    done
    
    # Print summary
    log_info "========================================="
    log_info "Test Summary"
    log_info "========================================="
    
    for VERSION in "${VERSIONS[@]}"; do
        local BUILD_STATUS="${BUILD_RESULTS[$VERSION]:-UNKNOWN}"
        local TEST_STATUS="${TEST_RESULTS[$VERSION]:-NOT_RUN}"
        
        if [ "$BUILD_STATUS" = "SUCCESS" ] && [ "$TEST_STATUS" = "PASSED" ]; then
            log_info "LLVM ${VERSION}: ${GREEN}✓ PASSED${NC}"
        elif [ "$BUILD_STATUS" = "SUCCESS" ] && [ "$TEST_STATUS" = "FAILED" ]; then
            log_warn "LLVM ${VERSION}: ${YELLOW}⚠ BUILD OK, TESTS FAILED${NC}"
        else
            log_error "LLVM ${VERSION}: ${RED}✗ FAILED (${BUILD_STATUS})${NC}"
        fi
    done
    
    if [ ${#FAILED_VERSIONS[@]} -eq 0 ]; then
        log_info ""
        log_info "All LLVM versions passed!"
        return 0
    else
        log_error ""
        log_error "Failed versions: ${FAILED_VERSIONS[*]}"
        return 1
    fi
}

# Run main function
main "$@"
