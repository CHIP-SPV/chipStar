// Unit test for the shared module cache: key builder framing, loader-delta
// digest, and the CHM2 file format. Makes zero HIP calls and needs no GPU
// (chipstar::Backend is only constructed on the first HIP API call).
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

#include "ModuleCache.hh"

using namespace chipstar::cache;

namespace {

// A fully populated baseline key. Every mutation-table row re-derives this
// with exactly one field changed.
struct KeyInputs {
  std::string BackendTag = "opencl";
  std::string Il = std::string("\x03\x02\x23\x07SPV!", 8);
  std::string BuildOptions = "-cl-kernel-arg-info -cl-std=CL3.0";
  std::string LinkFlags = "-cl-fast-relaxed-math";
  bool NeedsRtDevLib = true;
  std::vector<std::pair<std::string, std::string>> RtDevLib = {
      {"atomicAddFloat_native", "BC00"},
      {"ballot_native", "BC11"},
  };
  std::string DeviceName = "Intel(R) Arc(TM) B580";
  std::string DriverVersion = "26.27.39122.11";
  std::string LoaderDelta = "0123456789abcdef";
  std::string Environment = "IGC_ShaderDumpEnable=1;OverrideDefaultFP64Settings=1";
};

std::string buildKey(const KeyInputs &In) {
  KeyBuilder KB;
  KB.add(KeyField::BackendTag, In.BackendTag)
      .add(KeyField::Il, In.Il)
      .add(KeyField::BuildOptions, In.BuildOptions)
      .add(KeyField::LinkFlags, In.LinkFlags)
      .add(KeyField::BranchFlag, In.NeedsRtDevLib);
  for (const auto &[Name, Bytes] : In.RtDevLib)
    KB.add(KeyField::RtDevLibName, Name).add(KeyField::RtDevLibBytes, Bytes);
  KB.add(KeyField::DeviceName, In.DeviceName)
      .add(KeyField::DriverVersion, In.DriverVersion)
      .add(KeyField::LoaderDelta, In.LoaderDelta)
      .add(KeyField::Environment, In.Environment);
  return KB.finish();
}

int Failures = 0;
void check(bool Cond, const char *What) {
  if (!Cond) {
    std::cout << "FAIL: " << What << "\n";
    ++Failures;
  }
}

} // namespace

int main() {
  const KeyInputs Base;
  const std::string BaseKey = buildKey(Base);

  // --- shape: usable directly as a filename
  check(BaseKey.size() == 16, "key is 16 chars");
  check(BaseKey.find_first_not_of("0123456789abcdef") == std::string::npos,
        "key is lowercase hex");

  // --- determinism: two independently constructed builders agree
  check(buildKey(Base) == BaseKey, "identical inputs give identical keys");

  // --- mutation table: every single-field change must move the key, and all
  // rows must be pairwise distinct (catches a duplicated KeyField tag).
  std::map<std::string, std::string> Keys;
  Keys["baseline"] = BaseKey;
  {
    auto M = Base; M.Il[3] ^= 1;                    Keys["il-flip-byte"] = buildKey(M);
  }
  { auto M = Base; M.Il += '\0';                    Keys["il-append-byte"] = buildKey(M); }
  { auto M = Base; M.Il.pop_back();                 Keys["il-drop-byte"] = buildKey(M); }
  { auto M = Base; M.BuildOptions = "-cl-std=CL2.0";Keys["build-options"] = buildKey(M); }
  { auto M = Base; M.LinkFlags += " -cl-opt-disable"; Keys["link-flags"] = buildKey(M); }
  { auto M = Base; M.NeedsRtDevLib = false;         Keys["branch-flag"] = buildKey(M); }
  { auto M = Base; M.RtDevLib[1].second[3] ^= 1;    Keys["rtdevlib-byte"] = buildKey(M); }
  { auto M = Base; M.RtDevLib[0].first = "atomicAddFloat_emulation";
                                                    Keys["rtdevlib-name"] = buildKey(M); }
  { auto M = Base; M.RtDevLib.pop_back();           Keys["rtdevlib-drop"] = buildKey(M); }
  { auto M = Base; std::swap(M.RtDevLib[0], M.RtDevLib[1]);
                                                    Keys["rtdevlib-order"] = buildKey(M); }
  { auto M = Base; M.DeviceName = "Intel(R) Arc(TM) B570";
                                                    Keys["device-name"] = buildKey(M); }
  { auto M = Base; M.DriverVersion = "26.27.39122.12";
                                                    Keys["driver-version"] = buildKey(M); }
  { auto M = Base; M.LoaderDelta = "0123456789abcdee";
                                                    Keys["loader-delta"] = buildKey(M); }
  { auto M = Base; M.Environment += ";NEOReadDebugKeys=1";
                                                    Keys["env-add-var"] = buildKey(M); }
  { auto M = Base; M.Environment[4] ^= 1;           Keys["env-change"] = buildKey(M); }
  { auto M = Base; M.BackendTag = "level0";         Keys["backend-tag"] = buildKey(M); }

  for (auto It = Keys.begin(); It != Keys.end(); ++It)
    for (auto Jt = std::next(It); Jt != Keys.end(); ++Jt)
      if (It->second == Jt->second) {
        std::cout << "FAIL: keys collide: " << It->first << " vs " << Jt->first
                  << "\n";
        ++Failures;
      }

  // --- framing injections: the exact ambiguities the length-prefixed
  // encoding exists to rule out.
  {
    std::string A = KeyBuilder().add(KeyField::Il, "AB")
                        .add(KeyField::BuildOptions, "").finish();
    std::string B = KeyBuilder().add(KeyField::Il, "A")
                        .add(KeyField::BuildOptions, "B").finish();
    check(A != B, "field boundary cannot shift between fields");
  }
  {
    std::string A = KeyBuilder().add(KeyField::Il, "X")
                        .add(KeyField::BuildOptions, "---device---Y").finish();
    std::string B = KeyBuilder().add(KeyField::Il, "X---device---Y").finish();
    check(A != B, "marker text inside data cannot inject a field");
  }
  {
    std::string A = KeyBuilder().add(KeyField::RtDevLibBytes, "A")
                        .add(KeyField::RtDevLibBytes, "B").finish();
    std::string B = KeyBuilder().add(KeyField::RtDevLibBytes, "AB").finish();
    check(A != B, "two modules cannot merge into one");
  }
  // --- empty is not absent
  {
    std::string A = KeyBuilder().add(KeyField::Il, "X")
                        .add(KeyField::LinkFlags, "").finish();
    std::string B = KeyBuilder().add(KeyField::Il, "X").finish();
    check(A != B, "empty field differs from absent field");
  }

  // --- loader-delta digest (pure function over stamps)
  {
    std::vector<LibStamp> S1 = {{"/usr/lib/libigc.so.2.38", 100, 1111},
                                {"/usr/lib/libze.so.1", 200, 2222}};
    auto D1 = loaderDeltaDigestFrom(S1);
    check(D1 == loaderDeltaDigestFrom(S1), "delta digest deterministic");
    auto S2 = S1; S2[0].Size = 101;
    check(loaderDeltaDigestFrom(S2) != D1, "size change moves delta digest");
    auto S3 = S1; S3[0].MTimeNs = 1112;
    check(loaderDeltaDigestFrom(S3) != D1, "mtime change moves delta digest");
    auto S4 = S1; S4[0].RealPath = "/opt/igc/libigc.so.2.40";
    check(loaderDeltaDigestFrom(S4) != D1, "path change moves delta digest");
    check(loaderDeltaDigestFrom({}) == LoaderDeltaUnavailable,
          "empty delta degrades to the fixed token");
  }

  // --- CHM2 file format, against a temp directory
  const char *TmpEnv = std::getenv("TMPDIR");
  std::string Dir = std::string(TmpEnv ? TmpEnv : "/tmp") +
                    "/testmodulecachekey-" + std::to_string(getpid());
  const std::string Payload = "compiled-device-binary-bytes\x01\x02\x03";
  const std::string Key = BaseKey;

  check(store(Dir, "opencl", Key, Payload.data(), Payload.size()),
        "store succeeds");
  {
    auto E = load(Dir, "opencl", Key);
    check(static_cast<bool>(E), "round-trip loads");
    check(E && std::string(E.data().begin(), E.data().end()) == Payload,
          "round-trip preserves bytes");
  }
  { // wrong backend tag = different path = miss
    auto E = load(Dir, "level0", Key);
    check(!E, "backend tag separates entries");
  }
  { // absent key
    auto E = load(Dir, "opencl", std::string(16, '0'));
    check(!E, "absent key is a miss");
  }

  const std::string File = Dir + "/opencl/" + Key;
  auto ReadFile = [&]() {
    std::ifstream In(File, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(In)),
                       std::istreambuf_iterator<char>());
  };
  auto WriteFile = [&](const std::string &Bytes) {
    std::ofstream Out(File, std::ios::binary | std::ios::trunc);
    Out.write(Bytes.data(), Bytes.size());
  };
  const std::string Good = ReadFile();

  { // truncate by one byte
    WriteFile(Good.substr(0, Good.size() - 1));
    check(!load(Dir, "opencl", Key), "truncated file is a miss");
  }
  { // extend by one byte
    WriteFile(Good + 'x');
    check(!load(Dir, "opencl", Key), "extended file is a miss");
  }
  { // flip one payload byte
    auto Bad = Good; Bad[Good.size() - 3] ^= 1; WriteFile(Bad);
    check(!load(Dir, "opencl", Key), "payload corruption is a miss");
  }
  { // flip a magic byte
    auto Bad = Good; Bad[0] ^= 1; WriteFile(Bad);
    check(!load(Dir, "opencl", Key), "bad magic is a miss");
  }
  { // bump format version
    auto Bad = Good; Bad[4] ^= 1; WriteFile(Bad);
    check(!load(Dir, "opencl", Key), "wrong format version is a miss");
  }
  { // corrupt payload_len towards UINT64_MAX: must not wrap or allocate
    auto Bad = Good;
    for (int I = 8; I < 16; ++I) Bad[I] = '\xff';
    WriteFile(Bad);
    check(!load(Dir, "opencl", Key), "huge payload_len is a clean miss");
  }
  { // restore and confirm it hits again (the negative cases above are not
    // an artifact of a broken reader that rejects everything)
    WriteFile(Good);
    check(static_cast<bool>(load(Dir, "opencl", Key)),
          "restored file hits again");
  }
  { // store into an unwritable location: returns false, no throw, no abort
    check(!store("/proc/no-such-dir", "opencl", Key, Payload.data(),
                 Payload.size()),
          "store into unwritable dir fails gracefully");
  }
  { // leftover temp file from a crashed writer is invisible to load
    std::ofstream Out(File + ".tmp.12345.0", std::ios::binary);
    Out << "garbage";
    Out.close();
    check(static_cast<bool>(load(Dir, "opencl", Key)),
          "temp files do not shadow entries");
  }

  std::error_code EC;
  std::filesystem::remove_all(Dir, EC);

  if (Failures) {
    std::cout << Failures << " check(s) FAILed\n";
    return 1;
  }
  std::cout << "PASSED\n";
  return 0;
}
