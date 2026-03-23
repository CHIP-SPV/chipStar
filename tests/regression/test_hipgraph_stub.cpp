// Reproducer: hipStreamEndCapture returns hipErrorIllegalState (401)
// because getCaptureStatus() was hardcoded to return None.
//
// Expected: begin + end capture succeeds with hipSuccess
// Bug: hipStreamEndCapture returns 401

#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
    hipStream_t stream;
    hipError_t err;

    err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    if (err != hipSuccess) {
        printf("FAIL: hipStreamCreateWithFlags returned %d\n", err);
        return 1;
    }

    err = hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    if (err != hipSuccess) {
        printf("FAIL: hipStreamBeginCapture returned %d\n", err);
        hipStreamDestroy(stream);
        return 1;
    }

    hipGraph_t graph;
    err = hipStreamEndCapture(stream, &graph);
    if (err != hipSuccess) {
        printf("FAIL: hipStreamEndCapture returned %d (%s)\n", err, hipGetErrorString(err));
        hipStreamDestroy(stream);
        return 1;
    }

    printf("PASS\n");
    if (graph) hipGraphDestroy(graph);
    hipStreamDestroy(stream);
    return 0;
}
