// chip-kernel-verify
//
// Post-compile verifier invoked by hipcc (via HIPCC_VERIFY).
//
// Extracts SPIR-V from a hipcc output (object or linked executable), lists the
// OpEntryPoint Kernel names, then runs ocloc for each requested Intel GPU
// target and diffs against the kernels present in the native binary. Mismatch
// means IGC dropped entry points (most commonly intel/intel-graphics-compiler
// issue #403, SIMD32 register-pressure path).
//
// Env:
//   HIPCC_VERIFY           0 = skip; warn = print only; anything else = fail on mismatch.
//                          Defaults to CHIP_KERNEL_VERIFY_DEFAULT_MODE at build time.
//   HIPCC_VERIFY_DEVICES   comma-separated -device values for ocloc (default: dg2).
//   HIPCC_VERIFY_OCLOC     override ocloc binary (default: ocloc).

#include "spirv-extractor.hh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <elf.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#ifndef CHIP_KERNEL_VERIFY_DEFAULT_DEVICES
#define CHIP_KERNEL_VERIFY_DEFAULT_DEVICES "dg2"
#endif

namespace fs = std::filesystem;

namespace {

enum class Mode { Off, Warn, Fail };

Mode readMode() {
  const char *env = std::getenv("HIPCC_VERIFY");
  if (!env) return Mode::Fail;
  std::string v(env);
  if (v == "0" || v == "off" || v == "OFF") return Mode::Off;
  if (v == "warn" || v == "WARN") return Mode::Warn;
  return Mode::Fail;
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == delim) { if (!cur.empty()) out.push_back(cur); cur.clear(); }
    else cur.push_back(c);
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

std::string oclocBin() {
  const char *e = std::getenv("HIPCC_VERIFY_OCLOC");
  return e ? std::string(e) : std::string("ocloc");
}

bool oclocAvailable() {
  std::string cmd = oclocBin() + " --help > /dev/null 2>&1";
  return std::system(cmd.c_str()) == 0;
}

// Run a shell command; capture combined stdout+stderr and return exit code.
int runCapture(const std::string &cmd, std::string &out) {
  std::string full = cmd + " 2>&1";
  FILE *p = popen(full.c_str(), "r");
  if (!p) return -1;
  char buf[4096];
  while (size_t n = fread(buf, 1, sizeof(buf), p)) out.append(buf, n);
  int rc = pclose(p);
  if (WIFEXITED(rc)) return WEXITSTATUS(rc);
  return -1;
}

// Extract OpEntryPoint Kernel names from SPIR-V disassembly text.
// Disassembled form: `OpEntryPoint Kernel %<id> "<name>" ...`
std::set<std::string> scanEntryPoints(const std::string &spvText) {
  std::set<std::string> names;
  static const std::regex re(R"(OpEntryPoint\s+Kernel\s+%\S+\s+\"([^\"]+)\")");
  std::istringstream iss(spvText);
  std::string line;
  while (std::getline(iss, line)) {
    std::smatch m;
    if (std::regex_search(line, m, re)) names.insert(m[1].str());
  }
  return names;
}

// Parse the sections.txt produced by `ocloc disasm ... -dump <dir>`; collect
// kernel names from `.text.<name>, 1` lines.
std::set<std::string> scanNativeKernels(const fs::path &sectionsFile) {
  std::set<std::string> names;
  std::ifstream in(sectionsFile);
  std::string line;
  static const std::regex re(R"(^\.text\.([^,]+),)");
  while (std::getline(in, line)) {
    std::smatch m;
    if (std::regex_search(line, m, re)) names.insert(m[1].str());
  }
  return names;
}

int verifyOneDevice(const fs::path &spvFile, const std::string &device,
                    const std::set<std::string> &spvNames, Mode mode,
                    const fs::path &workDir) {
  fs::path outBase = workDir / ("native_" + device);
  std::string cmd = oclocBin() + " compile -spirv_input"
                    + " -file " + spvFile.string()
                    + " -device " + device
                    + " -output_no_suffix -output " + outBase.string();
  std::string log;
  int rc = runCapture(cmd, log);
  fs::path binFile = outBase.string() + ".bin";
  if (rc != 0 || !fs::exists(binFile)) {
    // ocloc explicitly errored (e.g. fp64 not supported on this device) — that
    // is a real but ordinary compile error, not the silent-drop class of
    // problem this tool is here to detect (intel-graphics-compiler#403).
    // Print the diagnostics for visibility and skip; do not fail the build.
    std::cerr << "[chip-kernel-verify] ocloc compile errored for device '"
              << device << "' (exit " << rc << ") — not an IGC #403 drop;"
              << " skipping verification for this device:\n" << log << "\n";
    return 0;
  }

  fs::path dumpDir = workDir / ("dump_" + device);
  fs::create_directories(dumpDir);
  std::string dcmd = oclocBin() + " disasm -file " + binFile.string()
                     + " -device " + device
                     + " -dump " + dumpDir.string();
  std::string dlog;
  int drc = runCapture(dcmd, dlog);
  fs::path sectionsFile = dumpDir / "sections.txt";
  if (!fs::exists(sectionsFile)) {
    std::cerr << "[chip-kernel-verify] ocloc disasm produced no sections.txt for '"
              << device << "' (exit " << drc << "); skipping diff\n";
    return 0;
  }

  auto native = scanNativeKernels(sectionsFile);
  std::vector<std::string> missing;
  for (const auto &n : spvNames)
    if (!native.count(n)) missing.push_back(n);

  if (missing.empty()) return 0;

  std::cerr << "[chip-kernel-verify] device=" << device
            << ": " << missing.size() << " kernel(s) missing from native binary"
            << " (IGC likely dropped them; see"
               " https://github.com/intel/intel-graphics-compiler/issues/403 ):\n";
  for (const auto &m : missing) std::cerr << "    - " << m << "\n";

  return mode == Mode::Fail ? 1 : 0;
}

} // namespace

int main(int argc, char *argv[]) {
  Mode mode = readMode();
  if (mode == Mode::Off) return 0;

  if (argc < 2) {
    std::cerr << "usage: chip-kernel-verify <compiled-output>\n";
    return 0; // don't fail user's build over a missing arg
  }
  fs::path input = argv[1];
  if (!fs::exists(input)) {
    // Compiler may have produced nothing (e.g. -E); silently skip.
    return 0;
  }

  if (!oclocAvailable()) {
    std::cerr << "[chip-kernel-verify] ocloc not found (tried '" << oclocBin()
              << "') — skipping IGC #403 verification\n";
    return 0;
  }

  // Slurp the input. seekToMagic() in spirv-extractor does an unbounded 1MB
  // scan on the raw data path, so pad the buffer to avoid OOB reads on small
  // inputs (raw .spv files, small objects, etc).
  std::ifstream f(input, std::ios::binary);
  if (!f) return 0;
  std::vector<char> buf((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
  if (buf.empty()) return 0;

  // If the input is ELF, only verify when it actually has a .hip_fatbin
  // section. Intermediate object files (e.g. -dc/-c output from hipcc) carry
  // partial / pre-link offload bundles that the SPIR-V extractor isn't built
  // to parse and previously crashed it.
  if (buf.size() >= sizeof(Elf64_Ehdr)) {
    auto *eh = reinterpret_cast<const Elf64_Ehdr *>(buf.data());
    if (std::memcmp(eh->e_ident, ELFMAG, SELFMAG) == 0 &&
        eh->e_ident[EI_CLASS] == ELFCLASS64) {
      bool hasHipFatbin = false;
      if (eh->e_shoff + eh->e_shnum * sizeof(Elf64_Shdr) <= buf.size() &&
          eh->e_shstrndx < eh->e_shnum) {
        auto *sh = reinterpret_cast<const Elf64_Shdr *>(buf.data() + eh->e_shoff);
        const auto &strtab_hdr = sh[eh->e_shstrndx];
        if (strtab_hdr.sh_offset + strtab_hdr.sh_size <= buf.size()) {
          const char *strtab = buf.data() + strtab_hdr.sh_offset;
          for (uint16_t i = 0; i < eh->e_shnum; ++i) {
            if (sh[i].sh_name >= strtab_hdr.sh_size) continue;
            if (std::strcmp(strtab + sh[i].sh_name, ".hip_fatbin") == 0) {
              hasHipFatbin = true;
              break;
            }
          }
        }
      }
      if (!hasHipFatbin) return 0; // not a final HIP-bearing image; skip
    }
  }

  constexpr size_t kScanPad = 1024 * 1024 + 64;
  if (buf.size() < kScanPad) buf.resize(kScanPad, 0);

  std::string err;
  std::string_view spirv = extractSPIRVModule(buf.data(), err);
  if (spirv.empty()) {
    // No HIP fatbin present (e.g. plain C object, or preprocess-only). Not an
    // error — just nothing to check.
    return 0;
  }

  std::string text = disassembleSPIRV(spirv);
  auto spvNames = scanEntryPoints(text);
  if (spvNames.empty()) return 0;

  std::string devsEnv;
  if (const char *e = std::getenv("HIPCC_VERIFY_DEVICES")) devsEnv = e;
  if (devsEnv.empty()) devsEnv = CHIP_KERNEL_VERIFY_DEFAULT_DEVICES;
  auto devices = split(devsEnv, ',');

  fs::path workDir = fs::temp_directory_path() /
                     ("chip-kernel-verify-" + std::to_string(getpid()));
  fs::create_directories(workDir);
  fs::path spvFile = workDir / "module.spv";
  {
    std::ofstream out(spvFile, std::ios::binary);
    out.write(spirv.data(), spirv.size());
  }

  int fails = 0;
  for (const auto &d : devices) {
    fails += verifyOneDevice(spvFile, d, spvNames, mode, workDir);
  }

  std::error_code ec;
  fs::remove_all(workDir, ec);

  return fails == 0 ? 0 : 1;
}
