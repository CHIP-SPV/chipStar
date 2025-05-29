/*
 * Copyright (c) 2021-24 chipStar developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "CHIPBackendOpenCL.hh"
#include "Utils.hh"

#include <sstream>

#include "Utils.hh"
#include <fstream>
#include <chrono>

// Auto-generated header that lives in <build-dir>/bitcode.
#include "rtdevlib-modules.h"

std::vector<cl_event>
getOpenCLHandles(const chipstar::SharedEventVector &ChipEvents) {
  std::vector<cl_event> Result;
  for (std::shared_ptr<chipstar::Event> Event : ChipEvents)
    Result.push_back(std::static_pointer_cast<CHIPEventOpenCL>(Event)->ClEvent);
  return Result;
}

static cl_sampler createSampler(cl_context Ctx, const hipResourceDesc &ResDesc,
                                const hipTextureDesc &TexDesc) {

  bool IsNormalized = TexDesc.normalizedCoords != 0;

  // Identify the address mode
  cl_addressing_mode AddressMode = CL_ADDRESS_NONE;
  if (ResDesc.resType == hipResourceTypeLinear)
    // "This [address mode] is ignored if cudaResourceDesc::resType is
    // cudaResourceTypeLinear." - CUDA 11.6.1/CUDA Runtime API.
    // Effectively out-of-bound references are undefined.
    AddressMode = CL_ADDRESS_NONE;
  else if (TexDesc.addressMode[0] == hipAddressModeClamp)
    AddressMode = CL_ADDRESS_CLAMP_TO_EDGE;
  else if (TexDesc.addressMode[0] == hipAddressModeBorder)
    AddressMode = CL_ADDRESS_CLAMP;
  else if (!IsNormalized)
    // "... if cudaTextureDesc::normalizedCoords is set to zero,
    // cudaAddressModeWrap and cudaAddressModeMirror won't be
    // supported and will be switched to cudaAddressModeClamp."
    // - CUDA 11.6.1/CUDA Runtime API.
    AddressMode = CL_ADDRESS_CLAMP_TO_EDGE;
  else if (TexDesc.addressMode[0] == hipAddressModeWrap)
    AddressMode = CL_ADDRESS_REPEAT;
  else if (TexDesc.addressMode[0] == hipAddressModeMirror)
    AddressMode = CL_ADDRESS_MIRRORED_REPEAT;

  // Identify the filter mode
  cl_filter_mode FilterMode = CL_FILTER_NEAREST;
  if (ResDesc.resType == hipResourceTypeLinear)
    // "This [filter mode] is ignored if cudaResourceDesc::resType is
    // cudaResourceTypeLinear." - CUDA 11.6.1/CUDA Runtime API.
    FilterMode = CL_FILTER_NEAREST;
  else if (TexDesc.filterMode == hipFilterModePoint)
    FilterMode = CL_FILTER_NEAREST;
  else if (TexDesc.filterMode == hipFilterModeLinear)
    FilterMode = CL_FILTER_LINEAR;

  cl_sampler_properties SamplerProps[] = {CL_SAMPLER_NORMALIZED_COORDS,
                                          TexDesc.normalizedCoords != 0,
                                          CL_SAMPLER_ADDRESSING_MODE,
                                          AddressMode,
                                          CL_SAMPLER_FILTER_MODE,
                                          FilterMode,
                                          0};
  auto Sampler = clCreateSamplerWithProperties(Ctx, SamplerProps, &clStatus);
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clCreateSamplerWithProperties);
  return Sampler;
}

static cl_mem_object_type getImageType(unsigned HipTextureID) {
  switch (HipTextureID) {
  default:
  case hipTextureTypeCubemap:
  case hipTextureTypeCubemapLayered:
    break;
  case hipTextureType1D:
    return CL_MEM_OBJECT_IMAGE1D;
  case hipTextureType2D:
    return CL_MEM_OBJECT_IMAGE2D;
  case hipTextureType3D:
    return CL_MEM_OBJECT_IMAGE3D;
  case hipTextureType1DLayered:
    return CL_MEM_OBJECT_IMAGE1D_ARRAY;
  case hipTextureType2DLayered:
    return CL_MEM_OBJECT_IMAGE2D_ARRAY;
  }
  CHIPASSERT(false && "Unknown or unsupported HIP texture type.");
  return CL_MEM_OBJECT_IMAGE2D;
}

static const std::map<std::tuple<int, hipChannelFormatKind, bool>,
                      cl_channel_type>
    IntChannelTypeMap = {
        {{8, hipChannelFormatKindSigned, false}, CL_SIGNED_INT8},
        {{16, hipChannelFormatKindSigned, false}, CL_SIGNED_INT16},
        {{32, hipChannelFormatKindSigned, false}, CL_SIGNED_INT32},
        {{8, hipChannelFormatKindUnsigned, false}, CL_UNSIGNED_INT8},
        {{16, hipChannelFormatKindUnsigned, false}, CL_UNSIGNED_INT16},
        {{32, hipChannelFormatKindUnsigned, false}, CL_UNSIGNED_INT32},
        {{8, hipChannelFormatKindSigned, true}, CL_SNORM_INT8},
        {{16, hipChannelFormatKindSigned, true}, CL_SNORM_INT16},
        {{8, hipChannelFormatKindUnsigned, true}, CL_UNORM_INT8},
        {{16, hipChannelFormatKindUnsigned, true}, CL_UNORM_INT16},
};

static cl_image_format getImageFormat(hipChannelFormatDesc Desc,
                                      bool NormalizedFloat) {

  cl_image_format ImageFormat;
  switch (Desc.f) {
  default:
    CHIPERR_LOG_AND_THROW("Unsupported texel type kind.", hipErrorTbd);
  case hipChannelFormatKindUnsigned:
  case hipChannelFormatKindSigned: {
    if (Desc.x > 16)
      // "Note that this [cudaTextureReadMode] applies only to 8-bit and 16-bit
      // integer formats. 32-bit integer format would not be promoted,
      // regardless of whether or not this cudaTextureDesc::readMode is set
      // cudaReadModeNormalizedFloat is specified."
      //
      // - CUDA 11.6.1/CUDA Runtime API.
      NormalizedFloat = false;

    auto I = IntChannelTypeMap.find(
        std::make_tuple(Desc.x, Desc.f, NormalizedFloat));
    if (I == IntChannelTypeMap.end())
      CHIPERR_LOG_AND_THROW("Unsupported integer texel size.", hipErrorTbd);
    ImageFormat.image_channel_data_type = I->second;
    break;
  }
  case hipChannelFormatKindFloat: {
    if (Desc.x != 32)
      CHIPERR_LOG_AND_THROW("Unsupported float texel size.", hipErrorTbd);
    ImageFormat.image_channel_data_type = CL_FLOAT;
    break;
  }
  }

  // Check the layout is one of: [X, 0, 0, 0], [X, X, 0, 0] or [X, X, X, X].
  if (!((Desc.y == 0 && Desc.z == 0 && Desc.w == 0) ||
        (Desc.y == Desc.x && Desc.z == 0 && Desc.w == 0) ||
        (Desc.y == Desc.x && Desc.z == Desc.x && Desc.w == Desc.x)))
    CHIPERR_LOG_AND_THROW("Unsupported channel layout.", hipErrorTbd);

  unsigned NumChannels = 1u + (Desc.y != 0) + (Desc.z != 0) + (Desc.w != 0);
  constexpr cl_channel_order ChannelOrders[4] = {CL_R, CL_RG, CL_RGB, CL_RGBA};
  ImageFormat.image_channel_order = ChannelOrders[NumChannels - 1];

  return ImageFormat;
}

static cl_image_desc getImageDescription(unsigned HipTextureTypeID,
                                         size_t Width, size_t Height = 0,
                                         size_t Depth = 0) {
  cl_image_desc ImageDesc;
  memset(&ImageDesc, 0, sizeof(cl_image_desc));
  ImageDesc.image_type = getImageType(HipTextureTypeID);
  ImageDesc.image_width = Width;
  ImageDesc.image_height = Height;
  ImageDesc.image_depth = Depth;
  ImageDesc.image_array_size = Depth;
  ImageDesc.image_row_pitch = 0;
  ImageDesc.image_slice_pitch = 0;
  ImageDesc.num_mip_levels = 0;
  ImageDesc.num_samples = 0;
  return ImageDesc;
}

static cl_mem createImage(cl_context Ctx, unsigned int TextureType,
                          hipChannelFormatDesc Format, bool NormalizedFloat,
                          size_t Width, size_t Height = 0, size_t Depth = 0) {

  cl_image_format ImageFormat = getImageFormat(Format, NormalizedFloat);
  cl_image_desc ImageDesc =
      getImageDescription(TextureType, Width, Height, Depth);
  cl_int clStatus;
  // These must be zero when host_ptr argument is NULL.
  CHIPASSERT(ImageDesc.image_row_pitch == 0 &&
             ImageDesc.image_slice_pitch == 0);
  cl_mem Image = clCreateImage(Ctx, CL_MEM_READ_ONLY, &ImageFormat, &ImageDesc,
                               nullptr, &clStatus);
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clCreateImage);

  return Image;
}

static void memCopyToImage(cl_command_queue CmdQ, cl_mem Image,
                           const void *HostSrc,
                           const chipstar::RegionDesc &SrcRegion,
                           bool BlockingCopy = true) {

  size_t InputRowPitch = SrcRegion.isPitched() ? SrcRegion.Pitch[0] : 0;
  size_t InputSlicePitch = 0;
  if (SrcRegion.isPitched() && SrcRegion.getNumDims() > 2)
    // The slice pitch must be zero for non-arrayed 1D and 2D images
    // (OpenCL v2.2/5.3.3).
    InputSlicePitch = SrcRegion.Pitch[1];

  const size_t *DstOrigin = SrcRegion.Offset;
  const size_t *DstRegion = SrcRegion.Size;
  cl_int clStatus = clEnqueueWriteImage(
      CmdQ, Image, BlockingCopy, DstOrigin, DstRegion, InputRowPitch,
      InputSlicePitch, HostSrc, 0, nullptr, nullptr);
  // TODO update last event
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clEnqueueWriteImage);
}

/// Annotate indirect pointers to the OpenCL driver via clSetKernelExecInfo
///
/// This is needed for HIP applications which pass allocations
/// indirectly to kernels (e.g. within an aggregate kernel argument or
/// within another allocation). Without the annotation the allocations
/// may not be properly synchronized.
///
/// Returns the list of annotated pointers.
static std::unique_ptr<std::vector<std::shared_ptr<void>>>
annotateIndirectPointers(const CHIPContextOpenCL &Ctx,
                         const SPVModuleInfo &ModInfo,
                         cl_kernel KernelAPIHandle) {

  // If we have determined that the module does not have indirect
  // global memory accesses (IGBAs; see HipIGBADetectorPass), we may
  // skip the annotation.
  if (ModInfo.HasNoIGBAs)
    return nullptr;

  cl_kernel_exec_info PtrListName;
  switch (Ctx.getAllocStrategy()) {
  default:
    assert(!"Unexpected allocation strategy!");
    return nullptr;
  case AllocationStrategy::CoarseGrainSVM:
  case AllocationStrategy::FineGrainSVM:
    PtrListName = CL_KERNEL_EXEC_INFO_SVM_PTRS;
    break;
  case AllocationStrategy::IntelUSM:
    PtrListName = CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL;
    break;
  case AllocationStrategy::BufferDevAddr:
    PtrListName = CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT;
    break;
  }

  std::unique_ptr<std::vector<std::shared_ptr<void>>> AllocKeepAlives;
  std::vector<void *> AnnotationList;
  LOCK(Ctx.ContextMtx); // CHIPContextOpenCL::MemManager_
  auto NumAllocations = Ctx.getNumAllocations();
  if (NumAllocations) {
    AnnotationList.reserve(NumAllocations);
    AllocKeepAlives.reset(new std::vector<std::shared_ptr<void>>());
    AllocKeepAlives->reserve(NumAllocations);
    for (std::shared_ptr<void> Ptr : Ctx.getAllocPointers()) {
      AnnotationList.push_back(Ptr.get());
      AllocKeepAlives->push_back(Ptr);
    }

    // TODO: Optimization. Don't call this function again if we know the
    //       AnnotationList hasn't changed since the last call.
    clStatus = clSetKernelExecInfo(KernelAPIHandle, PtrListName,
                                   AnnotationList.size() * sizeof(void *),
                                   AnnotationList.data());
    CHIPERR_CHECK_LOG_AND_THROW_TABLE(clSetKernelExecInfo);

    if (Ctx.getAllocStrategy() == AllocationStrategy::IntelUSM) {
      cl_bool param = CL_TRUE;
      if (Ctx.MemManager_.isHostAllocUsed()) {
        clStatus = clSetKernelExecInfo(
            KernelAPIHandle, CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL,
            sizeof(cl_bool), &param);
        CHIPERR_CHECK_LOG_AND_THROW_TABLE(clSetKernelExecInfo);
      }
      if (Ctx.MemManager_.isDeviceAllocUsed()) {
        clStatus = clSetKernelExecInfo(
            KernelAPIHandle, CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
            sizeof(cl_bool), &param);
        CHIPERR_CHECK_LOG_AND_THROW_TABLE(clSetKernelExecInfo);
      }
      if (Ctx.MemManager_.isSharedAllocUsed()) {
        clStatus = clSetKernelExecInfo(
            KernelAPIHandle, CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL,
            sizeof(cl_bool), &param);
        CHIPERR_CHECK_LOG_AND_THROW_TABLE(clSetKernelExecInfo);
      }
    }
  }

  return AllocKeepAlives;
}

struct KernelEventCallbackData {
  std::shared_ptr<chipstar::ArgSpillBuffer> ArgSpillBuffer;
  std::unique_ptr<std::vector<std::shared_ptr<void>>> AllocKeepAlives;
};
static void CL_CALLBACK kernelEventCallback(cl_event Event,
                                            cl_int CommandExecStatus,
                                            void *UserData) {
  delete static_cast<KernelEventCallbackData *>(UserData);
}

// CHIPCallbackDataLevel0
// ************************************************************************

CHIPCallbackDataOpenCL::CHIPCallbackDataOpenCL(hipStreamCallback_t TheCallback,
                                               void *TheCallbackArgs,
                                               chipstar::Queue *ChipQueue)
    : ChipQueue((CHIPQueueOpenCL *)ChipQueue) {
  if (TheCallbackArgs != nullptr)
    CallbackArgs = TheCallbackArgs;
  if (TheCallback == nullptr)
    CHIPERR_LOG_AND_THROW("", hipErrorTbd);
  CallbackF = TheCallback;
}

// EventMonitorOpenCL
// ************************************************************************
EventMonitorOpenCL::EventMonitorOpenCL() : chipstar::EventMonitor() {};

void EventMonitorOpenCL::monitor() {
  logTrace("EventMonitorOpenCL::monitor()");
  chipstar::EventMonitor::monitor();
}

// CHIPDeviceOpenCL
// ************************************************************************

chipstar::Texture *
CHIPDeviceOpenCL::createTexture(const hipResourceDesc *ResDesc,
                                const hipTextureDesc *TexDesc,
                                const struct hipResourceViewDesc *ResViewDesc) {
  logTrace("CHIPDeviceOpenCL::createTexture");

  bool NormalizedFloat = TexDesc->readMode == hipReadModeNormalizedFloat;
  auto *Q = (CHIPQueueOpenCL *)getDefaultQueue();

  cl_context CLCtx = ((CHIPContextOpenCL *)getContext())->get()->get();
  cl_sampler Sampler = createSampler(CLCtx, *ResDesc, *TexDesc);

  std::unique_ptr<CHIPTextureOpenCL> Result;
  chipstar::RegionDesc SrcDesc;
  void *SrcPtr = nullptr;

  if (ResDesc->resType == hipResourceTypeArray) {
    hipArray *Array = ResDesc->res.array.array;
    // Checked in CHIPBindings already.
    CHIPASSERT(Array->data && "Invalid hipArray.");
    CHIPASSERT(!Array->isDrv && "Not supported/implemented yet.");
    size_t Width = Array->width;
    size_t Height = Array->height;
    size_t Depth = Array->depth;

    cl_mem Image = createImage(CLCtx, Array->textureType, Array->desc,
                               NormalizedFloat, Width, Height, Depth);

    Result = std::make_unique<CHIPTextureOpenCL>(*ResDesc, Image, Sampler);
    SrcDesc = chipstar::RegionDesc::from(*Array);
    SrcPtr = Array->data;
  } else if (ResDesc->resType == hipResourceTypeLinear) {
    auto &Res = ResDesc->res.linear;
    auto TexelByteSize = getChannelByteSize(Res.desc);
    size_t Width = Res.sizeInBytes / TexelByteSize;

    cl_mem Image =
        createImage(CLCtx, hipTextureType1D, Res.desc, NormalizedFloat, Width);

    Result = std::make_unique<CHIPTextureOpenCL>(*ResDesc, Image, Sampler);
    SrcDesc = chipstar::RegionDesc::get1DRegion(Width, TexelByteSize);
    SrcPtr = Res.devPtr;
  } else if (ResDesc->resType == hipResourceTypePitch2D) {
    auto &Res = ResDesc->res.pitch2D;
    assert(Res.pitchInBytes >= Res.width); // Checked in CHIPBindings.

    cl_mem Image = createImage(CLCtx, hipTextureType2D, Res.desc,
                               NormalizedFloat, Res.width, Res.height);

    Result = std::make_unique<CHIPTextureOpenCL>(*ResDesc, Image, Sampler);
    SrcDesc = chipstar::RegionDesc::from(*ResDesc);
    SrcPtr = Res.devPtr;
  } else {
    assert(!"Unsupported/unimplemented texture resource type.");
    return nullptr;
  }
  assert(Result && SrcPtr);

  // Copy data to image. For simplicity use an unified but unoptimal
  // way to copy data to image by copying the data first to host
  // (from device) and then to the image.
  auto SrcSize = SrcDesc.getAllocationSize();
  auto HostData = std::unique_ptr<char[]>(new char[SrcSize]);
  Q->memCopyAsyncImpl(HostData.get(), SrcPtr, SrcSize, hipMemcpyDeviceToHost);
  memCopyToImage(Q->get()->get(), Result->getImage(), HostData.get(), SrcDesc);
  clStatus = Q->enqueueDeleteHostArray(HostData.get());
  if (clStatus == CL_SUCCESS)
    HostData.release();

  // The texture might be used in another, non-blocking stream . Avoid
  // data race by flushing the queue now.
  Q->finish();

  logTrace("Created texture: {}", (void *)Result.get());
  return Result.release();
}

static cl_device_fp_atomic_capabilities_ext
getFPAtomicCapabilities(cl::Device Dev, cl_device_info Info) noexcept {
  assert(Info == CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT ||
         Info == CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT);

  cl_int Err;
  auto DevExts = Dev.getInfo<CL_DEVICE_EXTENSIONS>(&Err);
  assert(Err == CL_SUCCESS && "Invalid device?");

  if (DevExts.find("cl_ext_float_atomics") == std::string::npos) {
    logDebug("cl_ext_float_atomics extension is not supported");
    return 0;
  }

  cl_device_fp_atomic_capabilities_ext Capabilities;
  Err = clGetDeviceInfo(Dev.get(), Info,
                        sizeof(cl_device_fp_atomic_capabilities_ext),
                        &Capabilities, nullptr);

  if (Err == CL_SUCCESS)
    return Capabilities;

  logWarn("clGetDeviceInfo returned {} for fp atomic capability query",
          resultToString(Err));
  assert(!"OpenCL API violation?");

  return 0;
}

CHIPDeviceOpenCL::CHIPDeviceOpenCL(CHIPContextOpenCL *ChipCtx,
                                   cl::Device *DevIn, int Idx)
    : Device(ChipCtx, Idx), ClDevice(DevIn), ClContext(ChipCtx->get()) {
  logTrace("CHIPDeviceOpenCL initialized via OpenCL device pointer and context "
           "pointer");

  Fp32AtomicAddCapabilities_ = getFPAtomicCapabilities(
      *DevIn, CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT);
  Fp64AtomicAddCapabilities_ = getFPAtomicCapabilities(
      *DevIn, CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT);

  auto DevExts = DevIn->getInfo<CL_DEVICE_EXTENSIONS>();
  HasSubgroupBallot_ =
      DevExts.find("cl_khr_subgroup_ballot") != std::string::npos;
}

CHIPDeviceOpenCL *CHIPDeviceOpenCL::create(cl::Device *ClDevice,
                                           CHIPContextOpenCL *ChipContext,
                                           int Idx) {
  CHIPDeviceOpenCL *Dev = new CHIPDeviceOpenCL(ChipContext, ClDevice, Idx);
  ChipContext->setDevice(Dev);
  ChipContext->MemManager_.init(ChipContext);
  Dev->init();
  return Dev;
}

void CHIPDeviceOpenCL::populateDevicePropertiesImpl() {
  logTrace("CHIPDeviceOpenCL->populate_device_properties()");
  cl_int Err;
  std::string Temp;

  this->MaxMallocSize_ = ClDevice->getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  assert(ClDevice != nullptr);
  Temp = ClDevice->getInfo<CL_DEVICE_NAME>();
  strncpy(HipDeviceProps_.name, Temp.c_str(), 255);
  HipDeviceProps_.name[255] = 0;

  HipDeviceProps_.totalGlobalMem =
      ClDevice->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&Err);

  HipDeviceProps_.sharedMemPerBlock =
      ClDevice->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&Err);

  HipDeviceProps_.maxThreadsPerBlock =
      ClDevice->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&Err);

  std::vector<size_t> Wi = ClDevice->getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

  HipDeviceProps_.maxThreadsDim[0] = Wi[0];
  HipDeviceProps_.maxThreadsDim[1] = Wi[1];
  HipDeviceProps_.maxThreadsDim[2] = Wi[2];

  // Maximum configured clock frequency of the device in MHz.
  HipDeviceProps_.clockRate =
      1000 * ClDevice->getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

  HipDeviceProps_.multiProcessorCount =
      ClDevice->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  HipDeviceProps_.l2CacheSize =
      ClDevice->getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

  // not actually correct
  HipDeviceProps_.totalConstMem =
      ClDevice->getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();

  // totally made up
  HipDeviceProps_.regsPerBlock = 64;

  HipDeviceProps_.warpSize = CHIP_DEFAULT_WARP_SIZE;
  // Try to check that we support the default warp size.
  std::vector<uint> Sg = ClDevice->getInfo<CL_DEVICE_SUB_GROUP_SIZES_INTEL>();
  if (std::find(Sg.begin(), Sg.end(), CHIP_DEFAULT_WARP_SIZE) == Sg.end()) {
    logWarn(
        "The device might not support subgroup size {}, warp-size sensitive "
        "kernels might not work correctly.",
        CHIP_DEFAULT_WARP_SIZE);
  }

  HipDeviceProps_.maxGridSize[0] = HipDeviceProps_.maxGridSize[1] =
      HipDeviceProps_.maxGridSize[2] = 65536;
  HipDeviceProps_.memoryClockRate = 1000;
  HipDeviceProps_.memoryBusWidth = 256;
  HipDeviceProps_.major = 2;
  HipDeviceProps_.minor = 0;

  HipDeviceProps_.maxThreadsPerMultiProcessor = HipDeviceProps_.maxGridSize[0];

  HipDeviceProps_.computeMode = 0;
  HipDeviceProps_.arch = {};

  Temp = ClDevice->getInfo<CL_DEVICE_EXTENSIONS>();
  if (Temp.find("cl_khr_global_int32_base_atomics") != std::string::npos)
    HipDeviceProps_.arch.hasGlobalInt32Atomics = 1;
  else
    HipDeviceProps_.arch.hasGlobalInt32Atomics = 0;

  if (Temp.find("cl_khr_local_int32_base_atomics") != std::string::npos)
    HipDeviceProps_.arch.hasSharedInt32Atomics = 1;
  else
    HipDeviceProps_.arch.hasSharedInt32Atomics = 0;

  if (Temp.find("cl_khr_int64_base_atomics") != std::string::npos) {
    HipDeviceProps_.arch.hasGlobalInt64Atomics = 1;
    HipDeviceProps_.arch.hasSharedInt64Atomics = 1;
  } else {
    HipDeviceProps_.arch.hasGlobalInt64Atomics = 1;
    HipDeviceProps_.arch.hasSharedInt64Atomics = 1;
  }

  if (Temp.find("cl_khr_fp64") != std::string::npos)
    HipDeviceProps_.arch.hasDoubles = 1;
  else
    HipDeviceProps_.arch.hasDoubles = 0;

  HipDeviceProps_.arch.hasWarpBallot = HasSubgroupBallot_;

  // TODO: OpenCL lacks queries for these. Generate best guesses which are
  // unlikely breaking the program logic.
  HipDeviceProps_.clockInstructionRate = 2465;
  HipDeviceProps_.concurrentKernels = 1;
  HipDeviceProps_.pciDomainID = 0;
  HipDeviceProps_.pciBusID = 0x10;
  HipDeviceProps_.pciDeviceID = 0x40 + getDeviceId();
  HipDeviceProps_.isMultiGpuBoard = 0;
  HipDeviceProps_.canMapHostMemory = 1;
  HipDeviceProps_.integrated = 0;
  HipDeviceProps_.maxSharedMemoryPerMultiProcessor =
      HipDeviceProps_.sharedMemPerBlock * 16;
  HipDeviceProps_.cooperativeLaunch = 0;
  HipDeviceProps_.cooperativeMultiDeviceLaunch = 0;
  HipDeviceProps_.cooperativeMultiDeviceUnmatchedFunc = 0;
  HipDeviceProps_.cooperativeMultiDeviceUnmatchedGridDim = 0;
  HipDeviceProps_.cooperativeMultiDeviceUnmatchedBlockDim = 0;
  HipDeviceProps_.cooperativeMultiDeviceUnmatchedSharedMem = 0;
  HipDeviceProps_.memPitch = 1;
  HipDeviceProps_.textureAlignment = 1;
  HipDeviceProps_.texturePitchAlignment = 1;
  HipDeviceProps_.kernelExecTimeoutEnabled = 0;
  HipDeviceProps_.ECCEnabled = 0;
  HipDeviceProps_.asicRevision = 1;

  cl_device_svm_capabilities SVMCapabilities =
      ClDevice->getInfo<CL_DEVICE_SVM_CAPABILITIES>();

  const bool SupportsFineGrainSVM =
      (SVMCapabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) != 0;
  const bool SupportsSVMAtomics =
      (SVMCapabilities & CL_DEVICE_SVM_ATOMICS) != 0;
  // System atomics are required for CC >= 6.0. We need fine grain
  // SVM + SVM atomics for them to function, thus cap with that feature
  // set.
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
  if (HipDeviceProps_.major > 5 &&
      (!SupportsFineGrainSVM || !SupportsSVMAtomics)) {
    HipDeviceProps_.major = 5;
    HipDeviceProps_.minor = 0;
  }

  // OpenCL 3.0 devices support basic CUDA managed memory via coarse-grain SVM,
  // but some of the functions such as prefetch and advice are unimplemented
  // in chipStar.
  HipDeviceProps_.managedMemory = 0;
  // TODO: Populate these from SVM/USM properties. Advertise the safe
  // defaults for now. Uninitialized properties cause undeterminism.
  HipDeviceProps_.directManagedMemAccessFromHost = 0;
  HipDeviceProps_.concurrentManagedAccess = 0;
  HipDeviceProps_.pageableMemoryAccess = 0;
  HipDeviceProps_.pageableMemoryAccessUsesHostPageTables = 0;

  auto Max1D2DWidth = ClDevice->getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
  auto Max2DHeight = ClDevice->getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();
  auto Max3DWidth = ClDevice->getInfo<CL_DEVICE_IMAGE3D_MAX_WIDTH>();
  auto Max3DHeight = ClDevice->getInfo<CL_DEVICE_IMAGE3D_MAX_HEIGHT>();
  auto Max3DDepth = ClDevice->getInfo<CL_DEVICE_IMAGE3D_MAX_DEPTH>();

  // Clamp texture dimensions to [0, INT_MAX] because the return value
  // of hipDeviceGetAttribute() is int type.
  HipDeviceProps_.maxTexture1DLinear = clampToInt(Max1D2DWidth);
  HipDeviceProps_.maxTexture1D = clampToInt(Max1D2DWidth);
  HipDeviceProps_.maxTexture2D[0] = clampToInt(Max1D2DWidth);
  HipDeviceProps_.maxTexture2D[1] = clampToInt(Max2DHeight);
  HipDeviceProps_.maxTexture3D[0] = clampToInt(Max3DWidth);
  HipDeviceProps_.maxTexture3D[1] = clampToInt(Max3DHeight);
  HipDeviceProps_.maxTexture3D[2] = clampToInt(Max3DDepth);

  constexpr char ArchName[] = "unavailable";
  static_assert(sizeof(ArchName) <= sizeof(HipDeviceProps_.gcnArchName),
                "Buffer overflow!");
  std::strncpy(HipDeviceProps_.gcnArchName, ArchName, sizeof(ArchName));

  if (getContext()->getAllocStrategy() == AllocationStrategy::BufferDevAddr) {
    // Using cl_mem buffers and extension to obtain their fixed device
    // address means that the device addresses may alias with host
    // allocations.
    HipDeviceProps_.unifiedAddressing = false;
  } else {
    HipDeviceProps_.unifiedAddressing = true;
  }
}

void CHIPDeviceOpenCL::resetImpl() { UNIMPLEMENTED(); }
// CHIPEventOpenCL
// ************************************************************************

CHIPEventOpenCL::CHIPEventOpenCL(CHIPContextOpenCL *ChipContext,
                                 cl_event ClEvent, chipstar::EventFlags Flags)
    : chipstar::Event((chipstar::Context *)(ChipContext), Flags),
      ClEvent(ClEvent) {}

CHIPEventOpenCL::CHIPEventOpenCL(CHIPContextOpenCL *ChipContext,
                                 chipstar::EventFlags Flags)
    : CHIPEventOpenCL(ChipContext, nullptr, Flags) {}

uint64_t CHIPEventOpenCL::getFinishTime() {
  int clStatus;
  uint64_t Ret;
  clStatus = clGetEventProfilingInfo(ClEvent, CL_PROFILING_COMMAND_END,
                                     sizeof(Ret), &Ret, NULL);

  if (clStatus != CL_SUCCESS) {
    clStatus = clGetEventInfo(ClEvent, CL_EVENT_COMMAND_EXECUTION_STATUS,
                              sizeof(int), &EventStatus_, NULL);
    CHIPERR_CHECK_LOG_AND_THROW_TABLE(clGetEventInfo);
  }
  return Ret;
}

size_t CHIPEventOpenCL::getRefCount() {
  cl_uint RefCount;
  if (ClEvent == nullptr)
    return 0;
  int clStatus = clGetEventInfo(getNativeRef(), CL_EVENT_REFERENCE_COUNT, 4,
                                &RefCount, NULL);
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clGetEventInfo);
  return RefCount;
}

CHIPEventOpenCL::~CHIPEventOpenCL() {
  this->RecordedEvent = nullptr;
  if (ClEvent)
    clReleaseEvent(ClEvent);
}

std::shared_ptr<chipstar::Event>
CHIPBackendOpenCL::createEventShared(chipstar::Context *ChipCtx,
                                     chipstar::EventFlags Flags) {
  CHIPEventOpenCL *Event =
      new CHIPEventOpenCL((CHIPContextOpenCL *)ChipCtx, nullptr, Flags);

  return std::shared_ptr<chipstar::Event>(Event);
}

chipstar::Event *CHIPBackendOpenCL::createEvent(chipstar::Context *ChipCtx,
                                                chipstar::EventFlags Flags) {
  CHIPEventOpenCL *Event =
      new CHIPEventOpenCL((CHIPContextOpenCL *)ChipCtx, nullptr, Flags);
  Event->setUserEvent(true);
  return Event;
}

void CHIPQueueOpenCL::recordEvent(chipstar::Event *ChipEvent) {
  logTrace("chipstar::Queue::recordEvent({})", (void *)ChipEvent);
  auto ChipEventCL = static_cast<CHIPEventOpenCL *>(ChipEvent);

  // Need profiling command queue for querying timestamps for possible
  // later hipEventElapsedTime() calls.
  switchModeTo(Profiling);

  ChipEventCL->recordEventCopy(enqueueMarker());
  ChipEventCL->setRecording();
}

void CHIPEventOpenCL::recordEventCopy(
    const std::shared_ptr<chipstar::Event> &OtherIn) {
  logTrace("CHIPEventOpenCL::recordEventCopy");
  std::shared_ptr<CHIPEventOpenCL> Other =
      std::static_pointer_cast<CHIPEventOpenCL>(OtherIn);
  this->ClEvent = Other->ClEvent;
  this->RecordedEvent = Other;
  this->Msg = "recordEventCopy: " + Other->Msg;
  this->HostTimeStamp =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

bool CHIPEventOpenCL::wait() {
  logTrace("CHIPEventOpenCL::wait()");

  if (EventStatus_ != EVENT_STATUS_RECORDING) {
    logWarn("Called wait() on an event that isn't active.");
    return false;
  }

  clStatus = clWaitForEvents(1, &ClEvent);

  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clWaitForEvents);
  return true;
}

bool CHIPEventOpenCL::updateFinishStatus(bool ThrowErrorIfNotReady) {
  logTrace("CHIPEventOpenCL::updateFinishStatus()");
  if (ThrowErrorIfNotReady && this->ClEvent == nullptr)
    CHIPERR_LOG_AND_THROW("OpenCL has not been initialized cl_event is null",
                          hipErrorNotReady);

  int UpdatedStatus;
  clStatus = clGetEventInfo(ClEvent, CL_EVENT_COMMAND_EXECUTION_STATUS,
                            sizeof(int), &UpdatedStatus, NULL);
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clGetEventInfo);
  if (ThrowErrorIfNotReady && UpdatedStatus != CL_COMPLETE) {
    CHIPERR_LOG_AND_THROW("chipstar::Event not yet ready", hipErrorNotReady);
  }

  if (UpdatedStatus <= CL_COMPLETE) {
    EventStatus_ = EVENT_STATUS_RECORDED;
    return false;
  } else {
    return true;
  }
}

float CHIPEventOpenCL::getElapsedTime(chipstar::Event *OtherIn) {
  CHIPEventOpenCL *Other = (CHIPEventOpenCL *)OtherIn;

  if (this->getContext() != Other->getContext())
    CHIPERR_LOG_AND_THROW(
        "Attempted to get elapsed time between two events that are not part of "
        "the same context",
        hipErrorTbd);

  if (this->getEventStatus() == EVENT_STATUS_RECORDING)
    this->updateFinishStatus(false);
  if (Other->getEventStatus() == EVENT_STATUS_RECORDING)
    Other->updateFinishStatus(false);

  if (!this->isRecordingOrRecorded() || !Other->isRecordingOrRecorded())
    CHIPERR_LOG_AND_THROW("one of the events isn't/hasn't recorded",
                          hipErrorInvalidHandle);

  if (!this->isFinished() || !Other->isFinished())
    CHIPERR_LOG_AND_THROW("one of the events hasn't finished",
                          hipErrorNotReady);

  uint64_t BeginGPU = this->getFinishTime();
  uint64_t EndGPU = Other->getFinishTime();
  uint64_t BeginCPU = this->HostTimeStamp;
  uint64_t EndCPU = Other->HostTimeStamp;

  bool ReversedEvents = false;
  if (EndCPU < BeginCPU) {
    ReversedEvents = true;
    std::swap(BeginGPU, EndGPU);
  }

  // Handle overflow
  int64_t Elapsed;
  const uint64_t MaxValue = std::numeric_limits<uint64_t>::max();
  if (EndGPU < BeginGPU) {
    logError("Overflow detected in CHIPEventOpenCL::getElapsedTime()");
    logError("BeginGPU: {}, EndGPU: {}", BeginGPU, EndGPU);
    Elapsed =
        (MaxValue - BeginGPU) + EndGPU + 1; // +1 to account for wraparound
  } else {
    Elapsed = EndGPU - BeginGPU;
  }

  const int64_t NANOSECS = 1000000000;
  int64_t MS = (Elapsed / NANOSECS) * 1000;
  int64_t NS = Elapsed % NANOSECS;
  float FractInMS = ((float)NS) / 1000000.0f;
  float Ms = (float)MS + FractInMS;

  Ms = std::abs(Ms);
  if (ReversedEvents)
    Ms = -Ms;

  return Ms;
}

void CHIPEventOpenCL::hostSignal() { UNIMPLEMENTED(); }

// CHIPModuleOpenCL
//*************************************************************************

CHIPModuleOpenCL::CHIPModuleOpenCL(const SPVModule &SrcMod) : Module(SrcMod) {}

cl::Program *CHIPModuleOpenCL::get() { return &Program_; }

/// Prints program log into error stream
static void dumpProgramLog(CHIPDeviceOpenCL &ChipDev, cl::Program Prog) {
  cl_int Err;
  std::string Log =
      Prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*ChipDev.get(), &Err);
  if (Err == CL_SUCCESS)
    logInfo("Program LOG for device #{}:{}:\n{}\n", ChipDev.getDeviceId(),
            ChipDev.getName(), Log);
}

static cl::Program compileIL(cl::Context Ctx, CHIPDeviceOpenCL &ChipDev,
                             const void *IL, size_t Length,
                             const std::string &Options = "") {
  cl_int Err;
  auto start = std::chrono::high_resolution_clock::now();
  cl::Program Prog(clCreateProgramWithIL(Ctx.get(), IL, Length, &Err));
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  logTrace("clCreateProgramWithIL took {} microseconds", duration.count());
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clCreateProgramWithIL);

  cl_device_id DevId = ChipDev.get()->get();
  auto Start = std::chrono::high_resolution_clock::now();
  auto Flags = ChipEnvVars.hasJitOverride() ? ChipEnvVars.getJitFlagsOverride()
                                            : ChipEnvVars.getJitFlags() + " " +
                                                  Backend->getDefaultJitFlags();
  logInfo("JIT flags: {}", Flags);
  Err = clCompileProgram(Prog.get(), 1, &DevId, Flags.c_str(), 0, nullptr,
                         nullptr, nullptr, nullptr);
  auto End = std::chrono::high_resolution_clock::now();
  auto Duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(End - Start);
  logTrace("clCompileProgram took {} ms", Duration.count());
  dumpProgramLog(ChipDev, Prog);
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clCompileProgram);

  return Prog;
}

template <size_t N>
static cl::Program compileIL(cl::Context Ctx, CHIPDeviceOpenCL &ChipDev,
                             std::array<unsigned char, N> IL,
                             const std::string &Options = "") {
  return compileIL(Ctx, ChipDev, IL.data(), IL.size());
}

static void appendRuntimeObjects(cl::Context Ctx, CHIPDeviceOpenCL &ChipDev,
                                 std::vector<cl::Program> &Objects) {

  // TODO: Minor optimization opportunity. Link modules based on
  //       SPIR-V module inspection.

  // TODO: Reuse already compiled modules.

  auto AppendSource = [&](auto &Source) -> void {
    Objects.push_back(compileIL(Ctx, ChipDev, Source));
  };

  if (ChipDev.hasFP32AtomicAdd())
    AppendSource(chipstar::atomicAddFloat_native);
  else
    AppendSource(chipstar::atomicAddFloat_emulation);

  if (ChipDev.hasDoubles()) {
    if (ChipDev.hasFP64AtomicAdd())
      AppendSource(chipstar::atomicAddDouble_native);
    else
      AppendSource(chipstar::atomicAddDouble_emulation);
  }

  if (ChipDev.hasBallot())
    AppendSource(chipstar::ballot_native);

  // No fall-back implementation for ballot - let linker raise an error.
}

static void save(const cl::Program &program, const std::string &cacheName) {
  if (!ChipEnvVars.getModuleCacheDir().has_value()) {
    logTrace("Module caching is disabled");
    return;
  }

  std::string cacheDir = ChipEnvVars.getModuleCacheDir().value();
  // Create the cache directory if it doesn't exist
  std::filesystem::create_directories(cacheDir);
  std::string fullPath = cacheDir + "/" + cacheName;

  // Step 1: Get the sizes of the binaries for each device
  std::vector<size_t> binarySizes;
  program.getInfo(CL_PROGRAM_BINARY_SIZES, &binarySizes);

  size_t numDevices = binarySizes.size();

  if (numDevices == 0) {
    logError("No devices associated with the program.");
    return;
  }

  // Step 2: Allocate memory for each binary
  std::vector<unsigned char *> binaries(numDevices, nullptr);
  for (size_t i = 0; i < numDevices; ++i) {
    if (binarySizes[i] > 0) {
      binaries[i] = new unsigned char[binarySizes[i]];
    } else {
      logError("Binary size for device {} is zero.", i);
      return;
    }
  }

  // Step 3: Retrieve the binaries
  cl_int err = clGetProgramInfo(program(), CL_PROGRAM_BINARIES,
                                numDevices * sizeof(unsigned char *),
                                binaries.data(), nullptr);
  if (err != CL_SUCCESS) {
    logError("clGetProgramInfo(CL_PROGRAM_BINARIES) failed with error {}", err);
    // Clean up allocated memory
    for (auto ptr : binaries) {
      delete[] ptr;
    }
    return;
  }

  // Step 4: Write the binaries to the output file
  std::ofstream outFile(fullPath, std::ios::out | std::ios::binary);
  if (!outFile) {
    logError("Failed to open file for writing kernel binary");
    // Clean up allocated memory
    for (auto ptr : binaries) {
      delete[] ptr;
    }
    return;
  }

  size_t totalBinarySize = 0;
  for (size_t i = 0; i < numDevices; ++i) {
    logTrace("Writing binary for device {}: {} bytes", i, binarySizes[i]);
    outFile.write(reinterpret_cast<const char *>(binaries[i]), binarySizes[i]);
    if (!outFile) {
      logError("Failed to write binary data to file");
      outFile.close();
      // Clean up allocated memory
      for (auto ptr : binaries) {
        delete[] ptr;
      }
      return;
    }
    totalBinarySize += binarySizes[i];
  }

  outFile.close();

  // Step 5: Verify the file size
  std::ifstream inFile(fullPath,
                       std::ios::in | std::ios::binary | std::ios::ate);
  if (!inFile) {
    logError("Failed to open file for reading to verify size");
    // Clean up allocated memory
    for (auto ptr : binaries) {
      delete[] ptr;
    }
    return;
  }
  std::streampos fileSize = inFile.tellg();
  inFile.close();

  if (fileSize != static_cast<std::streampos>(totalBinarySize)) {
    logError("File size mismatch. Expected: {}, Actual: {}", totalBinarySize,
             fileSize);
    // Clean up allocated memory
    for (auto ptr : binaries) {
      delete[] ptr;
    }
    return;
  }

  logTrace("Kernel binary cached as {}", fullPath);
  logTrace("Number of binaries: {}", numDevices);

  // Step 6: Clean up allocated memory
  for (auto ptr : binaries) {
    delete[] ptr;
  }
}

static bool load(cl::Context &context, const std::vector<cl::Device> &devices,
                 const std::string &cacheName, cl::Program &program) {
  if (!ChipEnvVars.getModuleCacheDir().has_value()) {
    return false;
  }

  std::string cacheDir = ChipEnvVars.getModuleCacheDir().value();
  std::string fullPath = cacheDir + "/" + cacheName;

  logTrace("Loading kernel binary from cache at {}", fullPath);

  std::ifstream inFile(fullPath,
                       std::ios::in | std::ios::binary | std::ios::ate);
  if (!inFile) {
    return false;
  }

  size_t size = inFile.tellg();
  logTrace("File size according to tellg(): {}", size);

  inFile.seekg(0, std::ios::beg);

  std::vector<char> binary(size);
  inFile.read(binary.data(), size);

  if (inFile.fail()) {
    logError("Failed to read file. Error: {}", strerror(errno));
    return false;
  }

  size_t bytesRead = inFile.gcount();
  logTrace("Bytes actually read: {}", bytesRead);

  inFile.close();

  try {
    cl::Program::Binaries binaries(
        1, std::vector<unsigned char>(binary.begin(), binary.end()));
    logTrace("loading Binary size: {}", binary.size());

    if (binary.empty()) {
      logError("Binary data is empty. Deleting the cache file.");
      std::remove(fullPath.c_str());
      return false;
    }

    cl_int err;
    program = cl::Program(context, devices, binaries, nullptr, &err);
    assert(err == CL_SUCCESS);
    logTrace("Program created successfully");

    auto buildStart = std::chrono::high_resolution_clock::now();
    err = program.build();
    auto buildEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> buildElapsed = buildEnd - buildStart;
    logInfo("clProgramBuild took {} seconds", buildElapsed.count());
    if (err != CL_SUCCESS) {
      logError("Failed to build program from binary: {}", err);
      return false;
    }

    // Print kernels available in the program
    std::vector<cl::Kernel> kernels;
    err = program.createKernels(&kernels);
    if (err != CL_SUCCESS) {
      logError("Failed to create kernels: {}", err);
    } else {
      logTrace("Kernels available in the program:");
      for (const auto &kernel : kernels) {
        std::string kernelName;
        err = kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &kernelName);
        if (err == CL_SUCCESS) {
          logTrace("  {}", kernelName);
        } else {
          logError("Failed to get kernel name: {}", err);
        }
      }
    }

    if (err != CL_SUCCESS) {
      logError("Failed to create program from binary: {}", err);
      return false;
    }
  } catch (const std::exception &e) {
    logError("OpenCL error creating program from binary: {}", e.what());
    return false;
  }

  logTrace("Kernel binary loaded from cache as {}", fullPath);
  return true;
}

std::string generateCacheName(const std::string &strIn,
                              const std::string &deviceName) {
  std::hash<std::string> hasher;
  std::string combinedStr = strIn + deviceName;
  
  // Include IGC_ environment variables in cache key
  std::string igcVars = collectIGCEnvironmentVariables();
  logDebug("IGC variables for cache key: '{}'", igcVars);
  if (!igcVars.empty()) {
    combinedStr += ";" + igcVars;
  }
  
  logDebug("Combined string for cache key: '{}'", combinedStr.substr(0, 200) + "...");
  size_t hash = hasher(combinedStr);
  std::string cacheKey = std::to_string(hash);
  logDebug("Generated cache key: '{}'", cacheKey);
  return cacheKey;
}

void CHIPModuleOpenCL::compile(chipstar::Device *ChipDev) {
  logTrace("CHIPModuleOpenCL::compile()");
  auto start = std::chrono::high_resolution_clock::now();
  CHIPDeviceOpenCL *ChipDevOcl = (CHIPDeviceOpenCL *)ChipDev;
  CHIPContextOpenCL *ChipCtxOcl =
      (CHIPContextOpenCL *)(ChipDevOcl->getContext());

  int Err;
  auto SrcBin = Src_->getBinary();
  std::string buildOptions =
      Backend->getDefaultJitFlags() + " " + ChipEnvVars.getJitFlags();
  std::string binAsStr = std::string(SrcBin.begin(), SrcBin.end());

  // Include device name in cache key
  std::string deviceName = ChipDevOcl->getName();
  std::string cacheName =
      generateCacheName(binAsStr + buildOptions, deviceName);

  bool cached =
      load(*ChipCtxOcl->get(), {*ChipDevOcl->get()}, cacheName, Program_);

  if (!cached) {
    cl::Program ClMainObj =
        compileIL(*ChipCtxOcl->get(), *ChipDevOcl, SrcBin.data(), SrcBin.size(),
                  buildOptions.c_str());

    std::vector<cl::Program> ClObjects;
    ClObjects.push_back(ClMainObj);
    appendRuntimeObjects(*ChipCtxOcl->get(), *ChipDevOcl, ClObjects);

    auto linkStart = std::chrono::high_resolution_clock::now();

    std::string Flags = "";
    // Check if running on Intel GPU OpenCL driver
    std::string vendor = ChipDevOcl->get()->getInfo<CL_DEVICE_VENDOR>();
    bool isIntelGPU =
        (vendor.find("Intel") != std::string::npos) &&
        (ChipDevOcl->get()->getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_GPU);

    if (isIntelGPU) {
      // Only Intel GPU driver seems to need compile flags at the link step
      Flags = ChipEnvVars.hasJitOverride() ? ChipEnvVars.getJitFlagsOverride()
                                           : ChipEnvVars.getJitFlags() + " " +
                                                 Backend->getDefaultJitFlags();
    }

    logInfo("JIT Link flags: {}", Flags);
    Program_ =
        cl::linkProgram(ClObjects, Flags.c_str(), nullptr, nullptr, &Err);
    auto linkEnd = std::chrono::high_resolution_clock::now();
    auto linkDuration = std::chrono::duration_cast<std::chrono::microseconds>(
        linkEnd - linkStart);
    logTrace("cl::linkProgram took {} microseconds", linkDuration.count());

    if (Err != CL_SUCCESS) {
      dumpProgramLog(*ChipDevOcl, Program_);
      CHIPERR_LOG_AND_THROW("Device library link step failed.",
                            hipErrorInitializationError);
    }

    save(Program_, cacheName);
  }

  std::vector<cl::Kernel> Kernels;
  auto kernelCreationStart = std::chrono::high_resolution_clock::now();
  Err = Program_.createKernels(&Kernels);
  auto kernelCreationEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> kernelCreationDuration =
      kernelCreationEnd - kernelCreationStart;
  logTrace("clCreateKernelsInProgram took {} microseconds",
           kernelCreationDuration.count());
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clCreateKernelsInProgram);

  logTrace("Kernels in CHIPModuleOpenCL: {} \n", Kernels.size());
  for (auto &Krnl : Kernels) {
    std::string HostFName;
    Err = Krnl.getInfo(CL_KERNEL_FUNCTION_NAME, &HostFName);
    CHIPERR_CHECK_LOG_AND_THROW_TABLE(clGetKernelInfo);
    auto *FuncInfo = findFunctionInfo(HostFName);
    if (!FuncInfo) {
      continue;
    }
    CHIPKernelOpenCL *ChipKernel =
        new CHIPKernelOpenCL(Krnl, ChipDevOcl, HostFName, FuncInfo, this);
    addKernel(ChipKernel);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  if (cached)
    logInfo("Loaded from cache, kernel compilation took {} seconds",
            elapsed.count());
  else
    logInfo("Kernel compilation took {} seconds", elapsed.count());
}

chipstar::Queue *CHIPDeviceOpenCL::createQueue(chipstar::QueueFlags Flags,
                                               int Priority) {
  CHIPQueueOpenCL *NewQ = new CHIPQueueOpenCL(this, Priority);
  NewQ->setFlags(Flags);
  return NewQ;
}

chipstar::Queue *CHIPDeviceOpenCL::createQueue(const uintptr_t *NativeHandles,
                                               int NumHandles) {
  cl_command_queue CmdQ = (cl_command_queue)NativeHandles[3];
  CHIPQueueOpenCL *NewQ =
      new CHIPQueueOpenCL(this, OCL_DEFAULT_QUEUE_PRIORITY, CmdQ);
  return NewQ;
}

// CHIPKernelOpenCL
//*************************************************************************

SPVFuncInfo *CHIPKernelOpenCL::getFuncInfo() const { return FuncInfo_; }
std::string CHIPKernelOpenCL::getName() { return Name_; }

hipError_t CHIPKernelOpenCL::getAttributes(hipFuncAttributes *Attr) {

  Attr->binaryVersion = 10;
  Attr->cacheModeCA = 0;

  Attr->constSizeBytes = 0; // TODO
  Attr->localSizeBytes = PrivateSize_;

  Attr->maxThreadsPerBlock = MaxWorkGroupSize_;
  Attr->sharedSizeBytes = StaticLocalSize_;
  Attr->maxDynamicSharedSizeBytes = MaxDynamicLocalSize_;

  Attr->numRegs = 0;
  Attr->preferredShmemCarveout = 0;
  Attr->ptxVersion = 10;

  return hipSuccess;
}

Borrowed<cl::Kernel> CHIPKernelOpenCL::borrowUniqueKernelHandle() {
  LOCK(KernelPoolMutex_); // For KernelPool_.

  auto ReturnToPool = [&](cl::Kernel *k) -> void {
    LOCK(KernelPoolMutex_);
    KernelPool_.emplace(k);
  };

  if (KernelPool_.size()) {
    auto *Kernel = KernelPool_.top().release();
    KernelPool_.pop();
    Kernel->setSVMPointers({}); // Clear CL_KERNEL_EXEC_INFO_SVM_PTRS.
    return Borrowed<cl::Kernel>(Kernel, ReturnToPool);
  }

  // NOTE: clCloneKernel is not used here due to its experience on
  // Intel (GPU) OpenCL which crashed if clSetKernelArgSVMPointer() was
  // called on the original cl_kernel.
  cl_int Err;
  auto *NewK = new cl::Kernel(*Module->get(), Name_.c_str(), &Err);
  if (Err != CL_SUCCESS) {
    delete NewK;
    CHIPERR_CHECK_LOG_AND_THROW_TABLE(clCreateKernel);
  }

  return Borrowed<cl::Kernel>(NewK, ReturnToPool);
}

CHIPKernelOpenCL::CHIPKernelOpenCL(cl::Kernel ClKernel, CHIPDeviceOpenCL *Dev,
                                   std::string HostFName, SPVFuncInfo *FuncInfo,
                                   CHIPModuleOpenCL *Parent)
    : Kernel(HostFName, FuncInfo), Module(Parent), Device(Dev) {

  int Err = 0;
  // TODO attributes
  cl_uint NumArgs = ClKernel.getInfo<CL_KERNEL_NUM_ARGS>(&Err);
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clGetKernelInfo);
  assert(FuncInfo_->getNumKernelArgs() == NumArgs);

  MaxWorkGroupSize_ =
      ClKernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(*Device->get());
  StaticLocalSize_ =
      ClKernel.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(*Device->get());
  MaxDynamicLocalSize_ =
      (size_t)Device->getAttr(hipDeviceAttributeMaxSharedMemoryPerBlock) -
      StaticLocalSize_;
  PrivateSize_ =
      ClKernel.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(*Device->get());

  Name_ = ClKernel.getInfo<CL_KERNEL_FUNCTION_NAME>();

  if (NumArgs > 0) {
    logTrace("Kernel {} numArgs: {} \n", Name_, NumArgs);
    auto ArgVisitor = [&](const SPVFuncInfo::KernelArg &Arg) -> void {
      logTrace("  ARG: SIZE {} SPACE {} KIND {}\n", Arg.Size,
               (unsigned)Arg.StorageClass, (unsigned)Arg.Kind);
    };
    FuncInfo_->visitKernelArgs(ArgVisitor);
  }
}

// CHIPContextOpenCL
//*************************************************************************

bool CHIPContextOpenCL::allDevicesSupportFineGrainSVMorUSM() {
  // TODO ok now, but what if we have more devices per context?
  return MemManager_.getAllocStrategy() == AllocationStrategy::FineGrainSVM ||
         MemManager_.getAllocStrategy() == AllocationStrategy::IntelUSM;
}

void CHIPContextOpenCL::freeImpl(void *Ptr) {
  LOCK(ContextMtx); // CHIPContextOpenCL::MemManager_
  MemManager_.free(Ptr);
}

cl::Context *CHIPContextOpenCL::get() { return &ClContext; }
CHIPContextOpenCL::CHIPContextOpenCL(cl::Context CtxIn, cl::Device Dev,
                                     cl::Platform Plat)
    : Platform_(Plat), ClContext(CtxIn) {
  logTrace("CHIPContextOpenCL Initialized via OpenCL Context pointer.");
}

void *CHIPContextOpenCL::allocateImpl(size_t Size, size_t Alignment,
                                      hipMemoryType MemType,
                                      chipstar::HostAllocFlags Flags) {
  void *Retval;
  LOCK(ContextMtx); // CHIPContextOpenCL::MemManager_

  Retval = MemManager_.allocate(Size, Alignment, MemType);
  return Retval;
}

// CHIPQueueOpenCL
//*************************************************************************
struct HipStreamCallbackData {
  hipStream_t Stream;
  hipError_t clStatus;
  void *UserData;
  hipStreamCallback_t Callback;
  std::shared_ptr<chipstar::Event> CallbackFinishEvent;
  std::shared_ptr<chipstar::Event> CallbackCompleted;
};

void CL_CALLBACK pfn_notify(cl_event Event, cl_int CommandExecStatus,
                            void *UserData) {
  HipStreamCallbackData *Cbo = (HipStreamCallbackData *)(UserData);
  if (Cbo == nullptr)
    return;
  if (Cbo->Callback == nullptr)
    return;
  Cbo->Callback(Cbo->Stream, Cbo->clStatus, Cbo->UserData);
  if (Cbo->CallbackFinishEvent != nullptr) {
    clSetUserEventStatus(
        std::static_pointer_cast<CHIPEventOpenCL>(Cbo->CallbackFinishEvent)
            ->ClEvent,
        CL_COMPLETE);
    CHIPERR_CHECK_LOG_AND_THROW_TABLE(clSetUserEventStatus);
  }
  delete Cbo;
}

void CHIPQueueOpenCL::MemMap(const chipstar::AllocationInfo *AllocInfo,
                             chipstar::Queue::MEM_MAP_TYPE Type) {
  CHIPContextOpenCL *C = static_cast<CHIPContextOpenCL *>(ChipContext_);
  if (C->getAllocStrategy() == AllocationStrategy::BufferDevAddr) {
    CHIPERR_LOG_AND_THROW("Insufficient device memory capabilities for MemMap.",
                          hipErrorTbd);
  } else if (C->allDevicesSupportFineGrainSVMorUSM()) {
    logDebug("Device supports fine grain SVM or USM. Skipping MemMap/Unmap");
    return;
  }

  auto MemMapEvent =
      static_cast<CHIPBackendOpenCL *>(Backend)->createEventShared(
          ChipContext_);
  auto MemMapEventNative =
      std::static_pointer_cast<CHIPEventOpenCL>(MemMapEvent)->getNativePtr();

  auto [EventsToWait, EventLocks] = getSyncQueuesLastEvents(MemMapEvent, false);
  std::vector<cl_event> SyncQueuesEventHandles = getOpenCLHandles(EventsToWait);

  auto QueueHandle = get()->get();
  cl_int clStatus;
  if (Type == chipstar::Queue::MEM_MAP_TYPE::HOST_READ) {
    logDebug("CHIPQueueOpenCL::MemMap HOST_READ");
    clStatus =
        clEnqueueSVMMap(QueueHandle, CL_TRUE, CL_MAP_READ, AllocInfo->HostPtr,
                        AllocInfo->Size, SyncQueuesEventHandles.size(),
                        SyncQueuesEventHandles.data(), MemMapEventNative);
  } else if (Type == chipstar::Queue::MEM_MAP_TYPE::HOST_WRITE) {
    logDebug("CHIPQueueOpenCL::MemMap HOST_WRITE");
    clStatus =
        clEnqueueSVMMap(QueueHandle, CL_TRUE, CL_MAP_WRITE, AllocInfo->HostPtr,
                        AllocInfo->Size, SyncQueuesEventHandles.size(),
                        SyncQueuesEventHandles.data(), MemMapEventNative);
  } else if (Type == chipstar::Queue::MEM_MAP_TYPE::HOST_READ_WRITE) {
    logDebug("CHIPQueueOpenCL::MemMap HOST_READ_WRITE");
    clStatus = clEnqueueSVMMap(
        QueueHandle, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, AllocInfo->HostPtr,
        AllocInfo->Size, SyncQueuesEventHandles.size(),
        SyncQueuesEventHandles.data(), MemMapEventNative);
  } else {
    assert(0 && "Invalid MemMap Type");
  }
  assert(clStatus == CL_SUCCESS);
}

void CHIPQueueOpenCL::MemUnmap(const chipstar::AllocationInfo *AllocInfo) {
  auto MemMapEvent =
      static_cast<CHIPBackendOpenCL *>(Backend)->createEventShared(
          ChipContext_);
  CHIPContextOpenCL *C = static_cast<CHIPContextOpenCL *>(ChipContext_);
  if (C->getAllocStrategy() == AllocationStrategy::BufferDevAddr) {
    CHIPERR_LOG_AND_THROW("Insufficient device memory capabilities for MemMap.",
                          hipErrorTbd);
  } else if (C->allDevicesSupportFineGrainSVMorUSM()) {
    logDebug("Device supports fine grain SVM or USM. Skipping MemMap/Unmap");
    return;
  }
  logDebug("CHIPQueueOpenCL::MemUnmap");
  auto [EventsToWait, EventLocks] = getSyncQueuesLastEvents(MemMapEvent, false);
  std::vector<cl_event> SyncQueuesEventHandles = getOpenCLHandles(EventsToWait);

  clStatus = clEnqueueSVMUnmap(
      get()->get(), AllocInfo->HostPtr, SyncQueuesEventHandles.size(),
      SyncQueuesEventHandles.data(),
      std::static_pointer_cast<CHIPEventOpenCL>(MemMapEvent)->getNativePtr());
  assert(clStatus == CL_SUCCESS);
}

cl::CommandQueue *CHIPQueueOpenCL::get() {
  switch (QueueMode_) {
  default:
    assert(!"Unknown queue mode!");
    // Fallthrough.
  case Profiling:
    return &ClProfilingQueue_;
  case Regular:
    return &ClRegularQueue_;
  }
}

void CHIPQueueOpenCL::addCallback(hipStreamCallback_t Callback,
                                  void *UserData) {
  logTrace("CHIPQueueOpenCL::addCallback()");

  cl::Context *ClContext_ = ((CHIPContextOpenCL *)ChipContext_)->get();
  cl_int Err;

  // Enqueue a barrier used to ensure the callback is not called too early,
  // otherwise it would be (at worst) executed in this host thread when
  // setting it, blocking the execution, while the clients might expect
  // parallel execution.
  std::shared_ptr<chipstar::Event> HoldbackBarrierCompletedEv =
      enqueueBarrier(std::vector<std::shared_ptr<chipstar::Event>>{});

  // OpenCL event callbacks have undefined execution ordering/finishing
  // guarantees. We need to enforce CUDA ordering using user events.

  std::shared_ptr<chipstar::Event> CallbackEvent =
      static_cast<CHIPBackendOpenCL *>(Backend)->createEventShared(
          ChipContext_);

  std::static_pointer_cast<CHIPEventOpenCL>(CallbackEvent)->ClEvent =
      clCreateUserEvent(ClContext_->get(), &Err);
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clCreateUserEvent);

  // Make the succeeding commands wait for the user event which will be
  // set CL_COMPLETE by the callback trampoline function pfn_notify after
  // finishing the user CB's execution.

  HipStreamCallbackData *Cb = new HipStreamCallbackData{
      this, hipSuccess, UserData, Callback, CallbackEvent, nullptr};

  std::vector<std::shared_ptr<chipstar::Event>> WaitForEventsCBB{CallbackEvent};
  Cb->CallbackCompleted = enqueueBarrier(WaitForEventsCBB);

  // We know that the callback won't be yet launched since it's depending
  // on the barrier which waits for the user event.
  clStatus = clSetEventCallback(
      std::static_pointer_cast<CHIPEventOpenCL>(HoldbackBarrierCompletedEv)
          ->ClEvent,
      CL_COMPLETE, pfn_notify, Cb);
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clSetEventCallback);

  updateLastEvent(Cb->CallbackCompleted);

  // Now the CB can start executing in the background:
  clSetUserEventStatus(
      std::static_pointer_cast<CHIPEventOpenCL>(HoldbackBarrierCompletedEv)
          ->ClEvent,
      CL_COMPLETE);
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clSetUserEventStatus);

  return;
}

std::shared_ptr<chipstar::Event> CHIPQueueOpenCL::enqueueMarkerImpl() {
  std::shared_ptr<chipstar::Event> MarkerEvent =
      static_cast<CHIPBackendOpenCL *>(Backend)->createEventShared(
          ChipContext_);

  auto [EventsToWait, EventLocks] = getSyncQueuesLastEvents(MarkerEvent, false);
  std::vector<cl_event> SyncQueuesEventHandles = getOpenCLHandles(EventsToWait);

  clStatus = clEnqueueMarkerWithWaitList(
      this->get()->get(), SyncQueuesEventHandles.size(),
      SyncQueuesEventHandles.data(),
      std::static_pointer_cast<CHIPEventOpenCL>(MarkerEvent)->getNativePtr());
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clEnqueueMarkerWithWaitList);
  MarkerEvent->Msg = "marker";
  updateLastEvent(MarkerEvent);
  return MarkerEvent;
}

std::shared_ptr<chipstar::Event>
CHIPQueueOpenCL::launchImpl(chipstar::ExecItem *ExecItem) {
  logTrace("CHIPQueueOpenCL->launch()");
  auto *OclContext = static_cast<CHIPContextOpenCL *>(ChipContext_);
  std::shared_ptr<chipstar::Event>(LaunchEvent) =
      static_cast<CHIPBackendOpenCL *>(Backend)->createEventShared(OclContext);
  CHIPExecItemOpenCL *ChipOclExecItem = (CHIPExecItemOpenCL *)ExecItem;
  CHIPKernelOpenCL *Kernel = (CHIPKernelOpenCL *)ChipOclExecItem->getKernel();
  cl_kernel KernelHandle = ChipOclExecItem->getKernelHandle();
  assert(Kernel && "Kernel in chipstar::ExecItem is NULL!");
  logTrace("Launching Kernel {}", Kernel->getName());

  ChipOclExecItem->setupAllArgs();

  dim3 GridDim = ChipOclExecItem->getGrid();
  dim3 BlockDim = ChipOclExecItem->getBlock();

  const size_t NumDims = 3;
  const size_t GlobalOffset[NumDims] = {0, 0, 0};
  const size_t Global[NumDims] = {
      GridDim.x * BlockDim.x, GridDim.y * BlockDim.y, GridDim.z * BlockDim.z};
  const size_t Local[NumDims] = {BlockDim.x, BlockDim.y, BlockDim.z};

  logTrace("Launch GLOBAL: {} {} {}", Global[0], Global[1], Global[2]);
  logTrace("Launch LOCAL: {} {} {}", Local[0], Local[1], Local[2]);
  auto AllocationsToKeepAlive = annotateIndirectPointers(
      *OclContext, Kernel->getModule()->getInfo(), KernelHandle);

  auto [EventsToWait, EventLocks] = getSyncQueuesLastEvents(LaunchEvent, false);
  std::vector<cl_event> SyncQueuesEventHandles = getOpenCLHandles(EventsToWait);

  clStatus = clEnqueueNDRangeKernel(
      get()->get(), KernelHandle, NumDims, GlobalOffset, Global, Local,
      SyncQueuesEventHandles.size(), SyncQueuesEventHandles.data(),
      std::static_pointer_cast<CHIPEventOpenCL>(LaunchEvent)->getNativePtr());
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clEnqueueNDRangeKernel);

  std::shared_ptr<chipstar::ArgSpillBuffer> SpillBuf =
      ExecItem->getArgSpillBuffer();

  if (SpillBuf || AllocationsToKeepAlive) {
    // Use an event call back to prolong the lifetimes of the
    // following objects until the kernel terminates.
    //
    // * SpillBuffer holds an allocation referenced by the kernel
    //   shared by exec item, which might get destroyed before the
    //   kernel is launched/completed
    //
    // * Annotated indirect SVM/USM pointers may need to outlive the
    //   kernel execution. The OpenCL spec does not clearly specify
    //   how long the pointers, annotated via clSetKernelExecInfo(),
    //   needs to live.
    auto *CBData = new KernelEventCallbackData;
    CBData->ArgSpillBuffer = SpillBuf;
    CBData->AllocKeepAlives = std::move(AllocationsToKeepAlive);
    clStatus = clSetEventCallback(
        std::static_pointer_cast<CHIPEventOpenCL>(LaunchEvent)->getNativeRef(),
        CL_COMPLETE, kernelEventCallback, CBData);
    if (clStatus != CL_SUCCESS) {
      delete CBData;
      CHIPERR_CHECK_LOG_AND_THROW_TABLE(clSetEventCallback);
    }
  }

  LaunchEvent->Msg = "KernelLaunch";
  updateLastEvent(LaunchEvent);
  return LaunchEvent;
}

CHIPQueueOpenCL::CHIPQueueOpenCL(chipstar::Device *ChipDevice, int Priority,
                                 cl_command_queue QueueForInterop)
    : chipstar::Queue(ChipDevice, chipstar::QueueFlags{}, Priority) {

  cl_queue_priority_khr PrioritySelection;
  switch (Priority_) {
  case 0:
    PrioritySelection = CL_QUEUE_PRIORITY_HIGH_KHR;
    break;
  case 1:
    PrioritySelection = CL_QUEUE_PRIORITY_MED_KHR;
    break;
  case 2:
    PrioritySelection = CL_QUEUE_PRIORITY_LOW_KHR;
    break;
  default:
    CHIPERR_LOG_AND_THROW(
        "Invalid Priority range requested during OpenCL Queue init",
        hipErrorTbd);
  }

  if (PrioritySelection != CL_QUEUE_PRIORITY_MED_KHR)
    logWarn("CHIPQueueOpenCL is ignoring Priority value");

  if (QueueForInterop) {
    auto QFromUser = cl::CommandQueue(QueueForInterop);

    cl_int clStatus;
    cl_command_queue_properties QProps =
        QFromUser.getInfo<CL_QUEUE_PROPERTIES>(&clStatus);
    CHIPERR_CHECK_LOG_AND_THROW_TABLE(clGetCommandQueueInfo);

    bool ProfilingIsEnabled = CL_QUEUE_PROFILING_ENABLE & QProps;
    if (ProfilingIsEnabled) {
      ClProfilingQueue_ = QFromUser;
      QueueMode_ = Profiling;
    } else {
      logWarn("Provided queue has profiling disable. hipEventElapsedTime() "
              "calls will fail.");
      ClRegularQueue_ = QFromUser;
      QueueMode_ = Regular;
    }
    UsedInInterOp = true;
  } else {
    cl::Context &ClContext =
        *static_cast<CHIPContextOpenCL *>(ChipContext_)->get();
    cl::Device &ClDevice = *static_cast<CHIPDeviceOpenCL *>(ChipDevice_)->get();

    cl_queue_properties QPropsForProfiling[] = {CL_QUEUE_PROPERTIES,
                                                CL_QUEUE_PROFILING_ENABLE, 0};

    cl_int clStatus;
    ClRegularQueue_ = cl::CommandQueue(
        clCreateCommandQueueWithProperties(ClContext.get(), ClDevice.get(),
                                           nullptr, &clStatus),
        false);
    CHIPERR_CHECK_LOG_AND_THROW_TABLE(clCreateCommandQueueWithProperties);

    if (!ChipEnvVars.getOCLDisableQueueProfiling()) {
      ClProfilingQueue_ = cl::CommandQueue(
          clCreateCommandQueueWithProperties(ClContext.get(), ClDevice.get(),
                                             QPropsForProfiling, &clStatus),
          false);
      CHIPERR_CHECK_LOG_AND_THROW_TABLE(clCreateCommandQueueWithProperties);
    }

    QueueMode_ = Regular;
  }
}

CHIPQueueOpenCL::~CHIPQueueOpenCL() {
  logTrace("~CHIPQueueOpenCL() {}", (void *)this);
}

std::shared_ptr<chipstar::Event>
CHIPQueueOpenCL::memCopyAsyncImpl(void *Dst, const void *Src, size_t Size,
                                  hipMemcpyKind Kind) {
  std::shared_ptr<chipstar::Event> Event =
      static_cast<CHIPBackendOpenCL *>(Backend)->createEventShared(
          ChipContext_);

  if (Dst == Src) {
    // Although ROCm API ref says that Dst and Src should not overlap,
    // HIP seems to handle Dst == Src as a special (no-operation) case.
    // This is seen in the test unit/memory/hipMemcpyAllApiNegative.

    // Intel GPU OpenCL driver seems to do also so for clEnqueueSVMMemcpy, which
    // makes/ it pass, but Intel CPU OpenCL returns CL_​MEM_​COPY_​OVERLAP
    // like it should. To unify the behavior, let's convert the special case to
    // a maker here, so we can return an event.
    cl::Event MarkerEvent;
    clStatus = clEnqueueMarker(
        get()->get(),
        std::static_pointer_cast<CHIPEventOpenCL>(Event)->getNativePtr());
    CHIPERR_CHECK_LOG_AND_THROW_TABLE(clEnqueueMarker);
  } else {
    auto [EventsToWait, EventLocks] = getSyncQueuesLastEvents(Event, false);
    std::vector<cl_event> SyncQueuesEventHandles =
        getOpenCLHandles(EventsToWait);
    auto *Ctx = getContext();

    switch (Ctx->getAllocStrategy()) {
    default:
      assert(!"Unexpected allocation strategy!");
      CHIPERR_LOG_AND_THROW("Internal error: Unexpected allocation strategy!",
                            hipErrorRuntimeMemory);
    case AllocationStrategy::IntelUSM:
    case AllocationStrategy::CoarseGrainSVM:
    case AllocationStrategy::FineGrainSVM: {
      logTrace("clSVMmemcpy {} -> {} / {} B\n", Src, Dst, Size);
      clStatus = ::clEnqueueSVMMemcpy(
          get()->get(), CL_FALSE, Dst, Src, Size, SyncQueuesEventHandles.size(),
          SyncQueuesEventHandles.data(),
          std::static_pointer_cast<CHIPEventOpenCL>(Event)->getNativePtr());
      CHIPERR_CHECK_LOG_AND_THROW_TABLE(clEnqueueSVMMemcpy);
      break;
    }

    case AllocationStrategy::BufferDevAddr: {
      cl_int clStatus = CL_SUCCESS;
      switch (Kind) {
      default:
      case hipMemcpyHostToHost: // Covered up-front.
        assert(!"Unexpected hipMemcpyKind");

      case hipMemcpyDefault: // Invalid flag under BufferDevAddr (can't
                             // distinguish host and device allocations apart as
                             // they may alias).
        CHIPERR_LOG_AND_THROW("Invalid flag (HipMemcpyDefault)",
                              hipErrorRuntimeMemory);

      case hipMemcpyHostToDevice: {
        auto [DstBuf, DstOffset] = Ctx->translateDevPtrToBuffer(Dst);
        if (!DstBuf)
          CHIPERR_LOG_AND_THROW("Invalid destination pointer.",
                                hipErrorRuntimeMemory);
        logTrace("clEnqueueWriteBuffer {} -> {} / {} B\n", Src, Dst, Size);
        clStatus = ::clEnqueueWriteBuffer(
            get()->get(), DstBuf, CL_FALSE, DstOffset, Size, Src,
            SyncQueuesEventHandles.size(), SyncQueuesEventHandles.data(),
            std::static_pointer_cast<CHIPEventOpenCL>(Event)->getNativePtr());
        break;
      }
      case hipMemcpyDeviceToHost: {
        auto [SrcBuf, SrcOffset] = Ctx->translateDevPtrToBuffer(Src);
        if (!SrcBuf)
          CHIPERR_LOG_AND_THROW("Invalid source pointer.",
                                hipErrorRuntimeMemory);
        logTrace("clEnqueueReadBuffer {} -> {} / {} B\n", Src, Dst, Size);
        clStatus = ::clEnqueueReadBuffer(
            get()->get(), SrcBuf, CL_FALSE, SrcOffset, Size, Dst,
            SyncQueuesEventHandles.size(), SyncQueuesEventHandles.data(),
            std::static_pointer_cast<CHIPEventOpenCL>(Event)->getNativePtr());
        break;
      }
      case hipMemcpyDeviceToDevice: {
        auto [SrcBuf, SrcOffset] = Ctx->translateDevPtrToBuffer(Src);
        auto [DstBuf, DstOffset] = Ctx->translateDevPtrToBuffer(Dst);
        if (!SrcBuf || !DstBuf)
          CHIPERR_LOG_AND_THROW(!SrcBuf ? "Invalid source pointer."
                                        : "Invalid destination pointer.",
                                hipErrorRuntimeMemory);
        logTrace("clEnqueueCopyBuffer {} -> {} / {} B\n", Src, Dst, Size);
        clStatus = ::clEnqueueCopyBuffer(
            get()->get(), SrcBuf, DstBuf, SrcOffset, DstOffset, Size,
            SyncQueuesEventHandles.size(), SyncQueuesEventHandles.data(),
            std::static_pointer_cast<CHIPEventOpenCL>(Event)->getNativePtr());
        break;
      }
      } // switch (Kind)
      CHIPERR_CHECK_LOG_AND_THROW(hipErrorRuntimeMemory);
      break;
    }
    } // switch (Ctx->getAllocStrategy())
  }
  updateLastEvent(Event);
  return Event;
}

void CHIPQueueOpenCL::finish() {
  clStatus = get()->finish();
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clFinish);
  this->LastEvent_ = nullptr;
}

std::shared_ptr<chipstar::Event>
CHIPQueueOpenCL::memFillAsyncImpl(void *Dst, size_t Size, const void *Pattern,
                                  size_t PatternSize) {
  std::shared_ptr<chipstar::Event> Event =
      static_cast<CHIPBackendOpenCL *>(Backend)->createEventShared(
          ChipContext_);

  auto [EventsToWait, EventLocks] = getSyncQueuesLastEvents(Event, false);
  std::vector<cl_event> SyncQueuesEventHandles = getOpenCLHandles(EventsToWait);
  auto *Ctx = getContext();

  if (Ctx->getAllocStrategy() == AllocationStrategy::BufferDevAddr) {
    logTrace("clEnqueueFillBuffer {} / {} B\n", Dst, Size);
    auto [DstBuf, DstOffset] = Ctx->translateDevPtrToBuffer(Dst);
    if (!DstBuf)
      CHIPERR_LOG_AND_THROW("Invalid destination pointer.",
                            hipErrorRuntimeMemory);
    int Retval = ::clEnqueueFillBuffer(
        get()->get(), DstBuf, Pattern, PatternSize, DstOffset, Size,
        SyncQueuesEventHandles.size(), SyncQueuesEventHandles.data(),
        std::static_pointer_cast<CHIPEventOpenCL>(Event)->getNativePtr());
    CHIPERR_CHECK_LOG_AND_THROW_TABLE(clEnqueueFillBuffer);
  } else {
    logTrace("clSVMmemfill {} / {} B\n", Dst, Size);
    int Retval = ::clEnqueueSVMMemFill(
        get()->get(), Dst, Pattern, PatternSize, Size,
        SyncQueuesEventHandles.size(), SyncQueuesEventHandles.data(),
        std::static_pointer_cast<CHIPEventOpenCL>(Event)->getNativePtr());
    CHIPERR_CHECK_LOG_AND_THROW_TABLE(clEnqueueSVMMemFill);
  }

  updateLastEvent(Event);
  return Event;
};

std::shared_ptr<chipstar::Event>
CHIPQueueOpenCL::memCopy2DAsyncImpl(void *Dst, size_t Dpitch, const void *Src,
                                    size_t Spitch, size_t Width, size_t Height,
                                    hipMemcpyKind Kind) {
  UNIMPLEMENTED(nullptr);
  std::shared_ptr<chipstar::Event> ChipEvent;

  for (size_t Offset = 0; Offset < Height; ++Offset) {
    void *DstP = ((unsigned char *)Dst + Offset * Dpitch);
    void *SrcP = ((unsigned char *)Src + Offset * Spitch);
    // capture the event on last iteration
    if (Offset == Height - 1)
      ChipEvent = memCopyAsyncImpl(DstP, SrcP, Width, Kind);
    else
      memCopyAsyncImpl(DstP, SrcP, Width, Kind);
  }

  return ChipEvent;
};

std::shared_ptr<chipstar::Event>
CHIPQueueOpenCL::memCopy3DAsyncImpl(void *Dst, size_t Dpitch, size_t Dspitch,
                                    const void *Src, size_t Spitch,
                                    size_t Sspitch, size_t Width, size_t Height,
                                    size_t Depth, hipMemcpyKind Kind) {
  UNIMPLEMENTED(nullptr);
};

hipError_t CHIPQueueOpenCL::getBackendHandles(uintptr_t *NativeInfo,
                                              int *NumHandles) {
  logTrace("CHIPQueueOpenCL::getBackendHandles");
  if (NumHandles) {
    *NumHandles = 5;
    return hipSuccess;
  }

  // Interop API expects one queue handle. Switch to the profiling queue which
  // works for the interop user and for us in case we need profiling later.
  switchModeTo(Profiling);

  // Get queue handler
  NativeInfo[4] = (uintptr_t)get()->get();

  // Get context handler
  cl::Context *Ctx = ((CHIPContextOpenCL *)ChipContext_)->get();
  NativeInfo[3] = (uintptr_t)Ctx->get();

  // Get device handler
  cl::Device *Dev = ((CHIPDeviceOpenCL *)ChipDevice_)->get();
  NativeInfo[2] = (uintptr_t)Dev->get();

  // Get platform handler
  cl_platform_id Plat = Dev->getInfo<CL_DEVICE_PLATFORM>();
  NativeInfo[1] = (uintptr_t)Plat;

  NativeInfo[0] = (uintptr_t)ChipEnvVars.getBackend().str();
  return hipSuccess;
}

std::shared_ptr<chipstar::Event>
CHIPQueueOpenCL::memPrefetchImpl(const void *Ptr, size_t Count) {
  UNIMPLEMENTED(nullptr);
}

std::shared_ptr<chipstar::Event> CHIPQueueOpenCL::enqueueBarrierImpl(
    const std::vector<std::shared_ptr<chipstar::Event>> &EventsToWaitFor) {
  std::shared_ptr<chipstar::Event> Event =
      static_cast<CHIPBackendOpenCL *>(Backend)->createEventShared(
          this->ChipContext_);
  cl_int RefCount;
  clGetEventInfo(
      std::static_pointer_cast<CHIPEventOpenCL>(Event)->getNativeRef(),
      CL_EVENT_REFERENCE_COUNT, 4, &RefCount, NULL);
  if (EventsToWaitFor.size() > 0) {
    std::vector<cl_event> Events = {};
    for (auto WaitEvent : EventsToWaitFor) {
      Events.push_back(
          std::static_pointer_cast<CHIPEventOpenCL>(WaitEvent)->getNativeRef());
    }
    // clStatus = ClQueue_->enqueueBarrierWithWaitList(&Events, &Barrier);
    auto [EventsToWait, EventLocks] = getSyncQueuesLastEvents(Event, false);
    std::vector<cl_event> SyncQueuesEventHandles =
        getOpenCLHandles(EventsToWait);

    for (auto &Event : Events) {
      SyncQueuesEventHandles.push_back(Event);
    }
    clStatus = clEnqueueBarrierWithWaitList(
        get()->get(), SyncQueuesEventHandles.size(),
        SyncQueuesEventHandles.data(),
        &(std::static_pointer_cast<CHIPEventOpenCL>(Event)->getNativeRef()));
    CHIPERR_CHECK_LOG_AND_THROW_TABLE(clEnqueueBarrierWithWaitList);
  } else {
    // clStatus = ClQueue_->enqueueBarrierWithWaitList(nullptr, &Barrier);
    auto [EventsToWait, EventLocks] = getSyncQueuesLastEvents(Event, false);
    std::vector<cl_event> SyncQueuesEventHandles =
        getOpenCLHandles(EventsToWait);

    clStatus = clEnqueueBarrierWithWaitList(
        get()->get(), SyncQueuesEventHandles.size(),
        SyncQueuesEventHandles.data(),
        std::static_pointer_cast<CHIPEventOpenCL>(Event)->getNativePtr());
    CHIPERR_CHECK_LOG_AND_THROW_TABLE(clEnqueueBarrierWithWaitList);
  }

  clGetEventInfo(
      std::static_pointer_cast<CHIPEventOpenCL>(Event)->getNativeRef(),
      CL_EVENT_REFERENCE_COUNT, 4, &RefCount, NULL);
  updateLastEvent(Event);
  return Event;
}

void CHIPQueueOpenCL::switchModeTo(QueueMode ToMode) {
  if (QueueMode_ == ToMode)
    return;

  if (ToMode == Profiling && ChipEnvVars.getOCLDisableQueueProfiling()) {
    logWarn("Queue profiling is disabled by request. hipEventElapsedTime() "
            "will not work.");
    return;
  }

  // Can't switch mode if the queue is used for interop. Otherwise, we
  // may cause situations where commands enqueued via HIP and other
  // API end up be executed out of order.
  if (UsedInInterOp)
    CHIPERR_LOG_AND_THROW("Can't switch mode for a queue used of interop.",
                          hipErrorTbd);

  logDebug("Queue {}: switching mode to {}", (void *)this,
           ToMode == Profiling ? "profiling" : "regular");

  // Make sure commands are executed in-order across the current and
  // switched-to queue.
  auto &FromQ = *get();
  auto &ToQ = ToMode == Profiling ? ClProfilingQueue_ : ClRegularQueue_;
  assert(FromQ.get());
  assert(ToQ.get());

  cl_event SwitchEv;
  cl_int clStatus;
  clStatus = clEnqueueMarkerWithWaitList(FromQ.get(), 0, nullptr, &SwitchEv);
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clEnqueueMarkerWithWaitList);

  clStatus = clEnqueueBarrierWithWaitList(ToQ.get(), 1, &SwitchEv, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(clEnqueueBarrierWithWaitList);

  auto *ChipEv = new CHIPEventOpenCL(
      static_cast<CHIPContextOpenCL *>(ChipContext_), SwitchEv);
  updateLastEvent(std::shared_ptr<chipstar::Event>(ChipEv));
  QueueMode_ = ToMode;
}

// CHIPExecItemOpenCL
//*************************************************************************

cl_kernel CHIPExecItemOpenCL::getKernelHandle() {
  return ClKernel_.get()->get();
}

void CHIPExecItemOpenCL::setupAllArgs() {
  if (!ArgsSetup) {
    ArgsSetup = true;
  } else {
    return;
  }
  CHIPKernelOpenCL *Kernel = (CHIPKernelOpenCL *)getKernel();
  CHIPContextOpenCL *Ctx = Kernel->getContext();
  SPVFuncInfo *FuncInfo = Kernel->getFuncInfo();
  int Err = 0;

  if (FuncInfo->hasByRefArgs()) {
    ArgSpillBuffer_ =
        std::make_shared<chipstar::ArgSpillBuffer>(ChipQueue_->getContext());
    ArgSpillBuffer_->computeAndReserveSpace(*FuncInfo);
  }

  cl_kernel KernelHandle = ClKernel_.get()->get();

  auto ArgVisitor = [&](const SPVFuncInfo::KernelArg &Arg) -> void {
    switch (Arg.Kind) {
    default:
      CHIPERR_LOG_AND_THROW("Internal chipStar error: Unknown argument kind",
                            hipErrorTbd);
    case SPVTypeKind::Image: {
      auto *TexObj =
          *reinterpret_cast<const CHIPTextureOpenCL *const *>(Arg.Data);
      cl_mem Image = TexObj->getImage();
      logTrace("set image arg {} for tex {}\n", Arg.Index, (void *)TexObj);
      Err = ::clSetKernelArg(KernelHandle, Arg.Index, sizeof(cl_mem), &Image);
      CHIPERR_CHECK_LOG_AND_THROW_TABLE(clSetKernelArg);
      break;
    }
    case SPVTypeKind::Sampler: {
      auto *TexObj =
          *reinterpret_cast<const CHIPTextureOpenCL *const *>(Arg.Data);
      cl_sampler Sampler = TexObj->getSampler();
      logTrace("set sampler arg {} for tex {}\n", Arg.Index, (void *)TexObj);
      Err = ::clSetKernelArg(KernelHandle, Arg.Index, sizeof(cl_sampler),
                             &Sampler);
      CHIPERR_CHECK_LOG_AND_THROW_TABLE(clSetKernelArg);
      break;
    }
    case SPVTypeKind::POD: {
      logTrace("clSetKernelArg {} SIZE {} to {}\n", Arg.Index, Arg.Size,
               Arg.Data);
      Err = ::clSetKernelArg(KernelHandle, Arg.Index, Arg.Size, Arg.Data);
      CHIPERR_CHECK_LOG_AND_THROW_TABLE(clSetKernelArg);
      break;
    }
    case SPVTypeKind::Pointer: {
      CHIPASSERT(Arg.Size == sizeof(void *));

      if (Arg.isWorkgroupPtr()) {
        logTrace("setLocalMemSize to {}\n", SharedMem_);
        Err = ::clSetKernelArg(KernelHandle, Arg.Index, SharedMem_, nullptr);
        break;
      }

      auto *DevPtr = *reinterpret_cast<const void *const *>(Arg.Data);
      if (Ctx->getAllocStrategy() == AllocationStrategy::BufferDevAddr) {
        Err = Ctx->clSetKernelArgDevicePointerEXT(KernelHandle, Arg.Index,
                                                  DevPtr);
        if (Err != CL_SUCCESS) {
          // ROCm seems to allow passing invalid pointers to kernels if they are
          // not derefenced (see test_device_adjacent_difference of rocPRIM).
          // If the setting of the arg fails, let's assume this might be such a
          // case and convert it to a null pointer.
          logWarn("clSetKernelArgDevicePointerEXT {} SIZE {} to {} (value {}) "
                  "returned error, setting the arg to nullptr\n",
                  Arg.Index, Arg.Size, Arg.Data, DevPtr);
          Err = Ctx->clSetKernelArgDevicePointerEXT(KernelHandle, Arg.Index,
                                                    nullptr);
        }
      } else {
        logTrace("clSetKernelArgSVMPointer {} SIZE {} to {} (value {})\n",
                 Arg.Index, Arg.Size, Arg.Data, DevPtr);
        Err = ::clSetKernelArgSVMPointer(
            KernelHandle, Arg.Index,
            // Unlike clSetKernelArg() which takes address to the argument,
            // this function takes the argument value directly.
            DevPtr);
        if (Err != CL_SUCCESS) {
          // Perhaps an invalid pointer - pass nullptr in its place.
          logWarn(
              "clSetKernelArgSVMPointer {} SIZE {} to {} (value {}) returned "
              "error, setting the arg to nullptr\n",
              Arg.Index, Arg.Size, Arg.Data, DevPtr);
          Err = ::clSetKernelArgSVMPointer(KernelHandle, Arg.Index, nullptr);
        }
      }
      CHIPERR_CHECK_LOG_AND_THROW_TABLE(clSetKernelArgSVMPointer);
      break;
    }
    case SPVTypeKind::PODByRef: {
      auto *SpillSlot = ArgSpillBuffer_->allocate(Arg);
      assert(SpillSlot);
      Err = ::clSetKernelArgSVMPointer(KernelHandle, Arg.Index, SpillSlot);
      CHIPERR_CHECK_LOG_AND_THROW_TABLE(clSetKernelArgSVMPointer);
      break;
    }
    }
  };
  FuncInfo->visitKernelArgs(getArgs(), ArgVisitor);

  if (FuncInfo->hasByRefArgs())
    ChipQueue_->memCopyAsync(ArgSpillBuffer_->getDeviceBuffer(),
                             ArgSpillBuffer_->getHostBuffer(),
                             ArgSpillBuffer_->getSize(), hipMemcpyHostToDevice);

  return;
}

void CHIPExecItemOpenCL::setKernel(chipstar::Kernel *Kernel) {
  assert(Kernel && "Kernel is nullptr!");

  ChipKernel_ = static_cast<CHIPKernelOpenCL *>(Kernel);
  ClKernel_ = ChipKernel_->borrowUniqueKernelHandle();

  // "New" kernel handle (could be recycled one) so arguments needs to
  // be set up again.
  ArgsSetup = false;
}

// CHIPBackendOpenCL
//*************************************************************************
chipstar::ExecItem *CHIPBackendOpenCL::createExecItem(dim3 GirdDim,
                                                      dim3 BlockDim,
                                                      size_t SharedMem,
                                                      hipStream_t ChipQueue) {
  CHIPExecItemOpenCL *ExecItem =
      new CHIPExecItemOpenCL(GirdDim, BlockDim, SharedMem, ChipQueue);
  return ExecItem;
};
chipstar::Queue *CHIPBackendOpenCL::createCHIPQueue(chipstar::Device *ChipDev) {
  CHIPDeviceOpenCL *ChipDevCl = (CHIPDeviceOpenCL *)ChipDev;
  return new CHIPQueueOpenCL(ChipDevCl, OCL_DEFAULT_QUEUE_PRIORITY);
}

chipstar::CallbackData *CHIPBackendOpenCL::createCallbackData(
    hipStreamCallback_t Callback, void *UserData, chipstar::Queue *ChipQueue) {
  UNIMPLEMENTED(nullptr);
}

chipstar::EventMonitor *CHIPBackendOpenCL::createEventMonitor_() {
  UNIMPLEMENTED(nullptr);
}

std::string CHIPBackendOpenCL::getDefaultJitFlags() {
  return std::string("-cl-kernel-arg-info -cl-std=CL3.0");
}

void CHIPBackendOpenCL::initializeImpl() {
  logTrace("CHIPBackendOpenCL Initialize");
  MinQueuePriority_ = CL_QUEUE_PRIORITY_MED_KHR;

  // transform device type string into CL
  cl_bitfield SelectedDevType = 0;
  if (ChipEnvVars.getDevice().getType() == DeviceType::CPU)
    SelectedDevType = CL_DEVICE_TYPE_CPU;
  else if (ChipEnvVars.getDevice().getType() == DeviceType::GPU)
    SelectedDevType = CL_DEVICE_TYPE_GPU;
  else if (ChipEnvVars.getDevice().getType() == DeviceType::Accelerator)
    SelectedDevType = CL_DEVICE_TYPE_ACCELERATOR;
  else if (ChipEnvVars.getDevice().getType() == DeviceType::Default)
    SelectedDevType = CL_DEVICE_TYPE_GPU;
  else
    throw InvalidDeviceType("Unknown value provided for CHIP_DEVICE_TYPE\n");
  logTrace("Using Devices of type {}", ChipEnvVars.getDevice().str());

  std::vector<cl::Platform> Platforms;
  clStatus = cl::Platform::get(&Platforms);
  if (clStatus != CL_SUCCESS) {
    logCritical("OpenCL failed to initialize any devices");
    std::exit(1);
  }
  std::stringstream StrStream;
  StrStream << "\nFound " << Platforms.size() << " OpenCL platforms:\n";
  for (int i = 0; i < Platforms.size(); i++) {
    StrStream << i << ". " << Platforms[i].getInfo<CL_PLATFORM_NAME>() << "\n";
  }
  logTrace("{}", StrStream.str());
  int SelectedPlatformIdx = ChipEnvVars.getPlatformIdx();

  cl::Platform SelectedPlatform = Platforms[SelectedPlatformIdx];
  logDebug("CHIP_PLATFORM={} Selected OpenCL platform {}", SelectedPlatformIdx,
           SelectedPlatform.getInfo<CL_PLATFORM_NAME>());

  StrStream.str("");

  StrStream << "OpenCL Devices of type " << ChipEnvVars.getDevice().str()
            << " with SPIR-V_1 support:\n";
  std::vector<cl::Device> SupportedDevices;
  std::vector<cl::Device> Dev;
  clStatus = SelectedPlatform.getDevices(SelectedDevType, &Dev);
  CHIPERR_CHECK_LOG_AND_THROW_TABLE(
      clGetDeviceIDs); // C equivalent of getDevices

  for (auto D : Dev) {

    std::string DeviceName = D.getInfo<CL_DEVICE_NAME>();

    StrStream << DeviceName << " ";
    std::string SPIRVVer = D.getInfo<CL_DEVICE_IL_VERSION>(&clStatus);
    if ((clStatus != CL_SUCCESS) ||
        (SPIRVVer.rfind("SPIR-V_1.", 0) == std::string::npos)) {
      StrStream << " no SPIR-V support.\n";
      continue;
    }

    // We require at least CG SVM, Device USM or cl_ext_buffer_device_address.
    std::string DevExts = D.getInfo<CL_DEVICE_EXTENSIONS>();
    cl_device_svm_capabilities SVMCapabilities =
        D.getInfo<CL_DEVICE_SVM_CAPABILITIES>();

    if ((SVMCapabilities & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) == 0 &&
        DevExts.find("cl_intel_unified_shared_memory") == std::string::npos &&
        DevExts.find("cl_ext_buffer_device_address") == std::string::npos) {
      StrStream << " insufficient device memory capabilities.\n";
      continue;
    }

    StrStream << " is supported.\n";
    SupportedDevices.push_back(D);
  }
  logInfo("{}", StrStream.str());

  if (ChipEnvVars.getDeviceIdx() >= SupportedDevices.size()) {
    logCritical("Selected OpenCL device {} is out of range",
                ChipEnvVars.getDeviceIdx());
    std::exit(1);
  }

  auto Device = SupportedDevices[ChipEnvVars.getDeviceIdx()];
  logDebug("CHIP_DEVICE={} Selected OpenCL device {}",
           ChipEnvVars.getDeviceIdx(), Device.getInfo<CL_DEVICE_NAME>());

  // Create context which has devices
  // Create queues that have devices each of which has an associated context
  cl::Context Ctx(SupportedDevices);
  CHIPContextOpenCL *ChipContext =
      new CHIPContextOpenCL(Ctx, Device, SelectedPlatform);
  ::Backend->addContext(ChipContext);

  // TODO for now only a single device is supported.
  cl::Device *clDev = new cl::Device(Device);
  CHIPDeviceOpenCL::create(clDev, ChipContext, 0);
  logTrace("OpenCL Context Initialized.");
};

void CHIPBackendOpenCL::initializeFromNative(const uintptr_t *NativeHandles,
                                             int NumHandles) {
  logTrace("CHIPBackendOpenCL InitializeNative");
  MinQueuePriority_ = CL_QUEUE_PRIORITY_MED_KHR;
  cl_platform_id PlatId = (cl_platform_id)NativeHandles[0];
  cl_device_id DevId = (cl_device_id)NativeHandles[1];
  cl_context CtxId = (cl_context)NativeHandles[2];

  cl::Context Ctx(CtxId);
  cl::Platform Plat(PlatId);
  cl::Device *Dev = new cl::Device(DevId);

  CHIPContextOpenCL *ChipContext = new CHIPContextOpenCL(Ctx, *Dev, Plat);
  addContext(ChipContext);

  CHIPDeviceOpenCL *ChipDev = CHIPDeviceOpenCL::create(Dev, ChipContext, 0);
  logTrace("CHIPDeviceOpenCL {}", ChipDev->ClDevice->getInfo<CL_DEVICE_NAME>());

  // Add device to context & backend
  ChipContext->setDevice(ChipDev);

  setActiveDevice(ChipDev);

  logTrace("OpenCL Context Initialized.");
}

hipEvent_t CHIPBackendOpenCL::getHipEvent(void *NativeEvent) {
  cl_event E = (cl_event)NativeEvent;
  // this retains cl_event
  CHIPEventOpenCL *NewEvent =
      new CHIPEventOpenCL((CHIPContextOpenCL *)ActiveCtx_, E);
  NewEvent->Msg = "fromHipEvent";
  return NewEvent;
}

void *CHIPBackendOpenCL::getNativeEvent(hipEvent_t HipEvent) {
  CHIPEventOpenCL *E = (CHIPEventOpenCL *)HipEvent;
  if (!E->isRecordingOrRecorded())
    return nullptr;
  return (void *)E->ClEvent;
}

// Other
//*************************************************************************

std::string resultToString(int clStatus) {
  switch (clStatus) {
  case CL_SUCCESS:
    return "CL_SUCCESS";
  case CL_DEVICE_NOT_FOUND:
    return "CL_DEVICE_NOT_FOUND";
  case CL_DEVICE_NOT_AVAILABLE:
    return "CL_DEVICE_NOT_AVAILABLE";
  case CL_COMPILER_NOT_AVAILABLE:
    return "CL_COMPILER_NOT_AVAILABLE";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case CL_OUT_OF_RESOURCES:
    return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY:
    return "CL_OUT_OF_HOST_MEMORY";
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case CL_MEM_COPY_OVERLAP:
    return "CL_MEM_COPY_OVERLAP";
  case CL_IMAGE_FORMAT_MISMATCH:
    return "CL_IMAGE_FORMAT_MISMATCH";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case CL_BUILD_PROGRAM_FAILURE:
    return "CL_BUILD_PROGRAM_FAILURE";
  case CL_MAP_FAILURE:
    return "CL_MAP_FAILURE";
#ifdef CL_VERSION_1_1
  case CL_MISALIGNED_SUB_BUFFER_OFFSET:
    return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
#endif
#ifdef CL_VERSION_1_2
  case CL_COMPILE_PROGRAM_FAILURE:
    return "CL_COMPILE_PROGRAM_FAILURE";
  case CL_LINKER_NOT_AVAILABLE:
    return "CL_LINKER_NOT_AVAILABLE";
  case CL_LINK_PROGRAM_FAILURE:
    return "CL_LINK_PROGRAM_FAILURE";
  case CL_DEVICE_PARTITION_FAILED:
    return "CL_DEVICE_PARTITION_FAILED";
  case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
    return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
#endif
  case (CL_INVALID_VALUE):
    return "CL_INVALID_VALUE";
  case (CL_INVALID_DEVICE_TYPE):
    return "CL_INVALID_DEVICE_TYPE";
  case (CL_INVALID_PLATFORM):
    return "CL_INVALID_PLATFORM";
  case (CL_INVALID_DEVICE):
    return "CL_INVALID_DEVICE";
  case (CL_INVALID_CONTEXT):
    return "CL_INVALID_CONTEXT";
  case (CL_INVALID_QUEUE_PROPERTIES):
    return "CL_INVALID_QUEUE_PROPERTIES";
  case (CL_INVALID_COMMAND_QUEUE):
    return "CL_INVALID_COMMAND_QUEUE";
  case (CL_INVALID_HOST_PTR):
    return "CL_INVALID_HOST_PTR";
  case (CL_INVALID_MEM_OBJECT):
    return "CL_INVALID_MEM_OBJECT";
  case (CL_INVALID_IMAGE_FORMAT_DESCRIPTOR):
    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case (CL_INVALID_IMAGE_SIZE):
    return "CL_INVALID_IMAGE_SIZE";
  case (CL_INVALID_SAMPLER):
    return "CL_INVALID_SAMPLER";
  case (CL_INVALID_BINARY):
    return "CL_INVALID_BINARY";
  case (CL_INVALID_BUILD_OPTIONS):
    return "CL_INVALID_BUILD_OPTIONS";
  case (CL_INVALID_PROGRAM):
    return "CL_INVALID_PROGRAM";
  case (CL_INVALID_PROGRAM_EXECUTABLE):
    return "CL_INVALID_PROGRAM_EXECUTABLE";
  case (CL_INVALID_KERNEL_NAME):
    return "CL_INVALID_KERNEL_NAME";
  case (CL_INVALID_KERNEL_DEFINITION):
    return "CL_INVALID_KERNEL_DEFINITION";
  case (CL_INVALID_KERNEL):
    return "CL_INVALID_KERNEL";
  case (CL_INVALID_ARG_INDEX):
    return "CL_INVALID_ARG_INDEX";
  case (CL_INVALID_ARG_VALUE):
    return "CL_INVALID_ARG_VALUE";
  case (CL_INVALID_ARG_SIZE):
    return "CL_INVALID_ARG_SIZE";
  case (CL_INVALID_KERNEL_ARGS):
    return "CL_INVALID_KERNEL_ARGS";
  case (CL_INVALID_WORK_DIMENSION):
    return "CL_INVALID_WORK_DIMENSION";
  case (CL_INVALID_WORK_GROUP_SIZE):
    return "CL_INVALID_WORK_GROUP_SIZE";
  case (CL_INVALID_WORK_ITEM_SIZE):
    return "CL_INVALID_WORK_ITEM_SIZE";
  case (CL_INVALID_GLOBAL_OFFSET):
    return "CL_INVALID_GLOBAL_OFFSET";
  case (CL_INVALID_EVENT_WAIT_LIST):
    return "CL_INVALID_EVENT_WAIT_LIST";
  case (CL_INVALID_EVENT):
    return "CL_INVALID_EVENT";
  case (CL_INVALID_OPERATION):
    return "CL_INVALID_OPERATION";
  case (CL_INVALID_GL_OBJECT):
    return "CL_INVALID_GL_OBJECT";
  case (CL_INVALID_BUFFER_SIZE):
    return "CL_INVALID_BUFFER_SIZE";
  case (CL_INVALID_MIP_LEVEL):
    return "CL_INVALID_MIP_LEVEL";
  case (CL_INVALID_GLOBAL_WORK_SIZE):
    return "CL_INVALID_GLOBAL_WORK_SIZE";
#ifdef CL_VERSION_1_1
  case (CL_INVALID_PROPERTY):
    return "CL_INVALID_PROPERTY";
#endif
#ifdef CL_VERSION_1_2
  case (CL_INVALID_IMAGE_DESCRIPTOR):
    return "CL_INVALID_IMAGE_DESCRIPTOR";
  case (CL_INVALID_COMPILER_OPTIONS):
    return "CL_INVALID_COMPILER_OPTIONS";
  case (CL_INVALID_LINKER_OPTIONS):
    return "CL_INVALID_LINKER_OPTIONS";
  case (CL_INVALID_DEVICE_PARTITION_COUNT):
    return "CL_INVALID_DEVICE_PARTITION_COUNT";
#endif
#ifdef CL_VERSION_2_0
  case (CL_INVALID_PIPE_SIZE):
    return "CL_INVALID_PIPE_SIZE";
  case (CL_INVALID_DEVICE_QUEUE):
    return "CL_INVALID_DEVICE_QUEUE";
#endif
#ifdef CL_VERSION_2_2
  case (CL_INVALID_SPEC_ID):
    return "CL_INVALID_SPEC_ID";
  case (CL_MAX_SIZE_RESTRICTION_EXCEEDED):
    return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
#endif
  default:
    return "CL_UNKNOWN_ERROR";
  }
}
