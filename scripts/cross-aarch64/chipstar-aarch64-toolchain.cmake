# Cross toolchain for building chipStar on x86_64 for aarch64-linux-gnu.
#
# Unlike aarch64-toolchain.cmake (used for the LLVM cross build, which is
# plain C++), this must keep clang as the compiler: chipStar's CMake rejects
# anything else (cmake/LLVMCheck.cmake) and bitcode/ invokes
# ${CMAKE_CXX_COMPILER} --target=spirv64* to emit device code, which only
# clang can do. So the compiler stays the x86-hosted clang and only the target
# changes.
#
# The target is carried by CMAKE_<LANG>_COMPILER_TARGET, which CMake emits as
# --target= on every compile and link line itself. It is not put in
# CMAKE_<LANG>_FLAGS_INIT because chipStar's CMakeLists.txt assigns
# CMAKE_CXX_FLAGS outright and would drop it, leaving x86 objects for an
# aarch64 link ("file in wrong format").
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER_TARGET   aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)
# Cross gcc's sysroot supplies libc/libstdc++/crt for the aarch64 target.
set(CMAKE_SYSROOT /)
set(_ld "--gcc-toolchain=/usr -B/usr/aarch64-linux-gnu/bin -fuse-ld=/usr/bin/aarch64-linux-gnu-ld -L/opt/spirv-tools-aarch64/lib")
set(CMAKE_EXE_LINKER_FLAGS_INIT    "${_ld}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_ld}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${_ld}")
# The prebuilt aarch64 SPIRV-Tools lives outside the compiler's default
# include search, and chipStar's found-package branch relies on the default
# path rather than propagating one. Applied at the toolchain level so
# chipStar's own CMAKE_CXX_FLAGS assignments cannot drop it.
set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES   /opt/spirv-tools-aarch64/include)
set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES /opt/spirv-tools-aarch64/include)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu /opt/spirv-tools-aarch64)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
