// Test that the HIPRTC output cache is populated on the first compilation
// and reused (not recompiled) on subsequent calls with identical inputs (#1142).
//
// The cache key is computed from source + headers + options.  With a
// non-deterministic hash (std::hash<std::string>) the key would differ between
// runs, making the cache write-only.  After switching to FNV-1a, the key is
// stable and a second compilation with the same inputs returns the cached SPIR-V.

#include "TestCommon.hh"
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <unistd.h>

namespace fs = std::filesystem;

static constexpr auto KernelSrc = R"---(
__global__ void add_one(int *out, const int *in) {
  *out = *in + 1;
}
)---";

static void runKernel(hipModule_t Mod) {
  hipFunction_t Fn;
  HIP_CHECK(hipModuleGetFunction(&Fn, Mod, "add_one"));

  int h_in = 41, h_out = 0;
  int *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_out, sizeof(int)));
  HIP_CHECK(hipMemcpy(d_in, &h_in, sizeof(int), hipMemcpyHostToDevice));

  void *args[] = {&d_out, &d_in};
  HIP_CHECK(hipModuleLaunchKernel(Fn, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(int), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  TEST_ASSERT(h_out == 42);
}

static hipModule_t compileAndLoad(const char *cacheDirEnv) {
  hiprtcProgram Prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&Prog, KernelSrc, "test.hip", 0, nullptr, nullptr));
  HIPRTC_CHECK(hiprtcCompileProgram(Prog, 0, nullptr));

  size_t CodeSz;
  HIPRTC_CHECK(hiprtcGetCodeSize(Prog, &CodeSz));
  std::vector<char> Code(CodeSz);
  HIPRTC_CHECK(hiprtcGetCode(Prog, Code.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&Prog));

  hipModule_t Mod;
  HIP_CHECK(hipModuleLoadData(&Mod, Code.data()));
  return Mod;
}

int main() {
  // Use a per-process temporary directory so the test is hermetic.
  std::string cacheDir = "/tmp/chipstar_hiprtc_cache_" + std::to_string(getpid());
  fs::create_directories(cacheDir);

  setenv("CHIP_MODULE_CACHE_DIR", cacheDir.c_str(), 1);

  HIP_CHECK(hipInit(0));
  HIP_CHECK(hipSetDevice(0));

  // --- First compilation: cache miss, SPIR-V written to cache. ---
  auto cacheEntriesBefore = 0;
  for (auto &e : fs::directory_iterator(cacheDir / "hiprtc"))
    cacheEntriesBefore++;
  (void)cacheEntriesBefore; // may not exist yet

  hipModule_t Mod1 = compileAndLoad(cacheDir.c_str());
  runKernel(Mod1);
  HIP_CHECK(hipModuleUnload(Mod1));

  // Count cache entries after first compilation.
  int filesAfterFirst = 0;
  if (fs::exists(cacheDir + "/hiprtc"))
    for (auto &e : fs::directory_iterator(cacheDir + "/hiprtc"))
      filesAfterFirst++;
  TEST_ASSERT(filesAfterFirst >= 1 && "Cache should have been written after first compilation");

  // --- Second compilation: cache hit, no recompilation needed. ---
  hipModule_t Mod2 = compileAndLoad(cacheDir.c_str());
  runKernel(Mod2);
  HIP_CHECK(hipModuleUnload(Mod2));

  // The number of cache files should not have increased (same key → same file).
  int filesAfterSecond = 0;
  for (auto &e : fs::directory_iterator(cacheDir + "/hiprtc"))
    filesAfterSecond++;
  TEST_ASSERT(filesAfterSecond == filesAfterFirst &&
              "Second compilation should reuse the cached entry, not create a new one");

  fs::remove_all(cacheDir);
  unsetenv("CHIP_MODULE_CACHE_DIR");

  std::cerr << "PASSED\n";
  return 0;
}
