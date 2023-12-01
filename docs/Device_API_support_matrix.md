
## chipStar support matrix for device side

 * categories are roughly from HIP kernel guide:
 * https://github.com/ROCm-Developer-Tools/HIP/blob/develop/docs/markdown/hip_kernel_language.md

| Feature                       | HIP API # of funcs  | # of impl in chipStar |  chipStar notes |
|-------------------------------|---------------------|-----------------------|---------------------------|
| Coordinate Built-Ins          |        12           |       12              | |
| Warp Size variable            |      supported      |     supported         | implemented, but requires support from driver side to respect warpSize (= cl_intel_required_subgroup_size extension) |
| Timer functions               |        2            |        2              | currently only fallback implementations of clock, clock64 are available |
| Atomic functions              |      ~30            |      ~30               | all supported; the implementation is efficient only if cl_ext_float_atomics is present & supported by backend & HW|
| Vector Types                  |       48            |       48              | |
| Memory-Fence Instructions     |        3            |        2              | \_\_threadfence_system is unsupported |
| Synchronization Functions     |        4            |        4              | |
| Float math functions          |       94            |       94              | |
| Float math intrinsics         |        9            |        2              | 45 in CUDA, 9 in HIP; what's currently possible, is mapped to OpenCL's native_XYZ functions; the rest requires an OpenCL extension + SPIR-V + HW + driver support |
| Double math functions         |       94            |       94              | |
| Double math intrinsics        |        1            |        0              | 28 in CUDA, 1 in HIP; same as float intrinsics |
| Integer Intrinsics            |       14            |       14              | |
| Half math funcs + intrin      |       96            |       81              | atomicAdd, __hadd_rn, __hfma_relu, __hmul_rn, __hsub_rn, __hmax_nan, __hmin_nan, __shfl{down,up,xor,sync}, ldcv, ldlu, stwb, stwt |
| Half2 math funcs + intrin     |      115            |       99              | same as ^^ + double2half |
| Texture Functions             |        ?            |        ?              | partially supported (1D/2D texture types, other types unsupported) |
| Surface Functions             |     unsupported     |     unsupported       | unsupported in both HIP  & chipStar |
| Cooperative Groups Functions  |      ~30            |        0              | all missing, pathfinding effort and HW features required for efficient support |
| Warp Vote & Ballot            |        3            |        3              | |
| Warp Shuffle                  |        8            |        8              | Supported in some circumstances (on Intel GPUs, when warp/subgroup=32 and ids map to lanes correctly). Also "width" argument is ignored. |
| Device-Side Dynamic Global Memory Allocation |  3   |        0              | medium difficulty to implement, likely no special hardware/software stack support required except atomics |
| In-Line Assembly              |   supports GCN asm  |     unsupported       | requires SPIR-V and driver support |
| Warp Matrix Functions         |    5, unsupported   |     unsupported       | unsupported in both HIP  & chipStar |
| Profiler Counter Function     |    1, unsupported   |     unsupported       | unsupported in both HIP  & chipStar |
| Independent Thread Scheduling |     unsupported     |     unsupported       | unsupported in both HIP  & chipStar |
| Pragma Unroll                 |      supported      |      supported        | Clang feature |
| Assert                        |      supported      |      supported        | |
| Printf                        |        1            |       1               | fully supported |
| advanced C++ features (RTTI, virtual, exceptions) | unsupported | unsupported | |


| Total (countable)             |     733   |     approx 495    | ~67%  |
