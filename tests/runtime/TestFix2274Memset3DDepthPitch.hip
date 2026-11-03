// Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2274:
// hipMemset3D rejects a depth larger than the row pitch.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static bool check(const char *What, hipError_t Got, hipError_t Want) {
  std::printf("%s: err=%d want=%d %s\n", What, (int)Got, (int)Want,
              Got == Want ? "ok" : "FAIL");
  return Got == Want;
}

int main() {
  hipPitchedPtr P;
  hipStream_t S;
  hipExtent E = make_hipExtent(1, 1, 129), Big = E;
  bool Ok = check("malloc3D", hipMalloc3D(&P, E), hipSuccess) &&
            check("stream", hipStreamCreate(&S), hipSuccess);
  unsigned char H = 0;
  Ok = Ok && check("memset3D", hipMemset3D(P, 0x11, E), hipSuccess);
  Ok = Ok && check("read",
                   hipMemcpy(&H, (char *)P.ptr + P.pitch * 128, 1,
                             hipMemcpyDeviceToHost),
                   hipSuccess);
  std::printf("memset3D last slice byte=%#x %s\n", H,
              H == 0x11 ? "ok" : "FAIL");
  Ok = Ok && H == 0x11;
  Ok =
      Ok && check("memset3DAsync", hipMemset3DAsync(P, 0x22, E, S), hipSuccess);
  Ok = Ok && check("sync", hipStreamSynchronize(S), hipSuccess);
  Ok = Ok && check("read",
                   hipMemcpy(&H, (char *)P.ptr + P.pitch * 128, 1,
                             hipMemcpyDeviceToHost),
                   hipSuccess);
  std::printf("memset3DAsync last slice byte=%#x %s\n", H,
              H == 0x22 ? "ok" : "FAIL");
  Ok = Ok && H == 0x22;
  for (size_t D : {(size_t)130, (size_t)SIZE_MAX}) {
    Big.depth = D;
    std::printf("depth=%zu\n", D);
    Ok &= check("memset3D over", hipMemset3D(P, 1, Big), hipErrorInvalidValue);
    Ok &= check("memset3DAsync over", hipMemset3DAsync(P, 1, Big, S),
                hipErrorInvalidValue);
  }
  // Overflowing extent (2,1,2^63) on a (2,1,1) allocation.
  hipPitchedPtr Q;
  Ok = Ok && check("malloc3D small", hipMalloc3D(&Q, make_hipExtent(2, 1, 1)),
                   hipSuccess);
  Ok = Ok && check("memset3D wrap",
                   hipMemset3D(Q, 1, make_hipExtent(2, 1, 1ull << 63)),
                   hipErrorInvalidValue);
  // pitch * ysize wraps to zero.
  hipPitchedPtr W = P;
  W.ysize = 1ull << 57;
  Ok = Ok &&
       check("memset3D ysize wrap", hipMemset3D(W, 1, E), hipErrorInvalidValue);
  Ok = Ok && check("memset3DAsync ysize wrap", hipMemset3DAsync(W, 1, E, S),
                   hipErrorInvalidValue);
  Ok &= check("memset3DAsync wrap",
              hipMemset3DAsync(Q, 1, make_hipExtent(2, 1, 1ull << 63), S),
              hipErrorInvalidValue);
  // Validation must happen before the memset is captured into a graph.
  hipStream_t C;
  hipGraph_t G;
  Ok = Ok && check("capture stream", hipStreamCreate(&C), hipSuccess) &&
       check("begin capture",
             hipStreamBeginCapture(C, hipStreamCaptureModeGlobal), hipSuccess);
  Big.depth = SIZE_MAX;
  Ok = Ok && check("captured over", hipMemset3DAsync(P, 1, Big, C),
                   hipErrorInvalidValue);
  Ok = Ok && check("captured valid", hipMemset3DAsync(P, 1, E, C), hipSuccess);
  Ok = Ok && check("end capture", hipStreamEndCapture(C, &G), hipSuccess);
  std::printf("%s\n", Ok ? "PASS" : "FAIL");
  return !Ok;
}
