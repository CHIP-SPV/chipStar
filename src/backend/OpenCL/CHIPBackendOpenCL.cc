/*
 * Copyright (c) 2021-22 CHIP-SPV developers
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
  cl_int Status = CL_SUCCESS;
  auto Sampler = clCreateSamplerWithProperties(Ctx, SamplerProps, &Status);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
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
  cl_int Status;
  // These must be zero when host_ptr argument is NULL.
  CHIPASSERT(ImageDesc.image_row_pitch == 0);
  CHIPASSERT(ImageDesc.image_slice_pitch == 0);
  cl_mem Image = clCreateImage(Ctx, CL_MEM_READ_ONLY, &ImageFormat, &ImageDesc,
                               nullptr, &Status);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);

  return Image;
}

static void memCopyToImage(cl_command_queue CmdQ, cl_mem Image,
                           const void *HostSrc, const CHIPRegionDesc &SrcRegion,
                           bool BlockingCopy = true) {

  size_t InputRowPitch = SrcRegion.isPitched() ? SrcRegion.Pitch[0] : 0;
  size_t InputSlicePitch = 0;
  if (SrcRegion.isPitched() && SrcRegion.getNumDims() > 2)
    // The slice pitch must be zero for non-arrayed 1D and 2D images
    // (OpenCL v2.2/5.3.3).
    InputSlicePitch = SrcRegion.Pitch[1];

  const size_t *DstOrigin = SrcRegion.Offset;
  const size_t *DstRegion = SrcRegion.Size;
  cl_int Status = clEnqueueWriteImage(CmdQ, Image, BlockingCopy, DstOrigin,
                                      DstRegion, InputRowPitch, InputSlicePitch,
                                      HostSrc, 0, nullptr, nullptr);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
}

/// Annotate SVM pointers to the OpenCL driver via clSetKernelExecInfo
///
/// This is needed for HIP applications which pass allocations
/// indirectly to kernels (e.g. within an aggregate kernel argument or
/// within another allocation). Without the annotation the allocations
/// may not be properly synchronized.
///
/// Returns the list of annotated pointers.
static std::unique_ptr<std::vector<std::shared_ptr<void>>>
annotateSvmPointers(const CHIPContextOpenCL &Ctx, cl_kernel KernelAPIHandle) {
  // By default we pass every allocated SVM pointer at this point to
  // the clSetKernelExecInfo() since any of them could be potentially
  // be accessed indirectly by the kernel.
  //
  // TODO: Optimization. Don't call clSetKernelExecInfo() if the
  //       kernel is known not to access any buffer indirectly
  //       discovered through kernel code inspection.
  std::vector<void *> SvmAnnotationList;
  std::unique_ptr<std::vector<std::shared_ptr<void>>> SvmKeepAlives;
  LOCK(Ctx.ContextMtx); // CHIPContextOpenCL::SvmMemory
  auto NumSvmAllocations = Ctx.SvmMemory.getNumAllocations();
  if (NumSvmAllocations) {
    SvmAnnotationList.reserve(NumSvmAllocations);
    SvmKeepAlives.reset(new std::vector<std::shared_ptr<void>>());
    SvmKeepAlives->reserve(NumSvmAllocations);
    for (std::shared_ptr<void> Ptr : Ctx.SvmMemory.getSvmPointers()) {
      SvmAnnotationList.push_back(Ptr.get());
      SvmKeepAlives->push_back(Ptr);
    }

    // TODO: Optimization. Don't call this function again if we know the
    //       SvmAnnotationList hasn't changed since the last call.
    auto Status = clSetKernelExecInfo(
        KernelAPIHandle, CL_KERNEL_EXEC_INFO_SVM_PTRS,
        SvmAnnotationList.size() * sizeof(void *), SvmAnnotationList.data());
    CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  }

  return SvmKeepAlives;
}

struct KernelEventCallbackData {
  std::shared_ptr<CHIPArgSpillBuffer> ArgSpillBuffer;
  std::unique_ptr<std::vector<std::shared_ptr<void>>> SvmKeepAlives;
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
                                               CHIPQueue *ChipQueue)
    : ChipQueue((CHIPQueueOpenCL *)ChipQueue) {
  if (TheCallbackArgs != nullptr)
    CallbackArgs = TheCallbackArgs;
  if (TheCallback == nullptr)
    CHIPERR_LOG_AND_THROW("", hipErrorTbd);
  CallbackF = TheCallback;
}

// CHIPEventMonitorOpenCL
// ************************************************************************
CHIPEventMonitorOpenCL::CHIPEventMonitorOpenCL() : CHIPEventMonitor(){};

void CHIPEventMonitorOpenCL::monitor() {
  logTrace("CHIPEventMonitorOpenCL::monitor()");
  CHIPEventMonitor::monitor();
}

// CHIPDeviceOpenCL
// ************************************************************************

CHIPTexture *
CHIPDeviceOpenCL::createTexture(const hipResourceDesc *ResDesc,
                                const hipTextureDesc *TexDesc,
                                const struct hipResourceViewDesc *ResViewDesc) {
  logTrace("CHIPDeviceOpenCL::createTexture");

  bool NormalizedFloat = TexDesc->readMode == hipReadModeNormalizedFloat;
  auto *Q = (CHIPQueueOpenCL *)getDefaultQueue();

  cl_context CLCtx = ((CHIPContextOpenCL *)getContext())->get()->get();
  cl_sampler Sampler = createSampler(CLCtx, *ResDesc, *TexDesc);

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

    auto Tex = std::make_unique<CHIPTextureOpenCL>(*ResDesc, Image, Sampler);
    logTrace("Created texture: {}", (void *)Tex.get());

    CHIPRegionDesc SrcRegion = CHIPRegionDesc::from(*Array);
    memCopyToImage(Q->get()->get(), Image, Array->data, SrcRegion);

    return Tex.release();
  }

  if (ResDesc->resType == hipResourceTypeLinear) {
    auto &Res = ResDesc->res.linear;
    auto TexelByteSize = getChannelByteSize(Res.desc);
    size_t Width = Res.sizeInBytes / TexelByteSize;

    cl_mem Image =
        createImage(CLCtx, hipTextureType1D, Res.desc, NormalizedFloat, Width);

    auto Tex = std::make_unique<CHIPTextureOpenCL>(*ResDesc, Image, Sampler);
    logTrace("Created texture: {}", (void *)Tex.get());

    // Copy data to image.
    auto SrcDesc = CHIPRegionDesc::get1DRegion(Width, TexelByteSize);
    memCopyToImage(Q->get()->get(), Image, Res.devPtr, SrcDesc);

    return Tex.release();
  }

  if (ResDesc->resType == hipResourceTypePitch2D) {
    auto &Res = ResDesc->res.pitch2D;
    assert(Res.pitchInBytes >= Res.width); // Checked in CHIPBindings.

    cl_mem Image = createImage(CLCtx, hipTextureType2D, Res.desc,
                               NormalizedFloat, Res.width, Res.height);

    auto Tex = std::make_unique<CHIPTextureOpenCL>(*ResDesc, Image, Sampler);
    logTrace("Created texture: {}", (void *)Tex.get());

    // Copy data to image.
    auto SrcDesc = CHIPRegionDesc::from(*ResDesc);
    memCopyToImage(Q->get()->get(), Image, Res.devPtr, SrcDesc);

    return Tex.release();
  }

  CHIPASSERT(false && "Unsupported/unimplemented texture resource type.");
  return nullptr;
}

CHIPDeviceOpenCL::CHIPDeviceOpenCL(CHIPContextOpenCL *ChipCtx,
                                   cl::Device *DevIn, int Idx)
    : CHIPDevice(ChipCtx, Idx), ClDevice(DevIn), ClContext(ChipCtx->get()) {
  logTrace("CHIPDeviceOpenCL initialized via OpenCL device pointer and context "
           "pointer");
  cl_device_svm_capabilities DeviceSVMCapabilities;
  auto Status =
      DevIn->getInfo(CL_DEVICE_SVM_CAPABILITIES, &DeviceSVMCapabilities);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  this->SupportsFineGrainSVM =
      DeviceSVMCapabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER;
  if (this->SupportsFineGrainSVM) {
    logTrace("Device supports fine grain SVM");
  } else {
    logTrace("Device does not support fine grain SVM");
  }
}

CHIPDeviceOpenCL *CHIPDeviceOpenCL::create(cl::Device *ClDevice,
                                           CHIPContextOpenCL *ChipContext,
                                           int Idx) {
  CHIPDeviceOpenCL *Dev = new CHIPDeviceOpenCL(ChipContext, ClDevice, Idx);
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

  // TODO: OpenCL lacks queries for these. Generate best guesses which are
  // unlikely breaking the program logic.
  HipDeviceProps_.clockInstructionRate = 2465;
  HipDeviceProps_.concurrentKernels = 1;
  HipDeviceProps_.pciDomainID = 0;
  HipDeviceProps_.pciBusID = 0x10;
  HipDeviceProps_.pciDeviceID = 0x40 + getDeviceId();
  HipDeviceProps_.isMultiGpuBoard = 0;
  HipDeviceProps_.canMapHostMemory = 1;
  HipDeviceProps_.gcnArch = 0;
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

  // OpenCL 3.0 devices support basic CUDA managed memory via coarse-grain SVM,
  // but some of the functions such as prefetch and advice are unimplemented
  // in CHIP-SPV.
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
}

void CHIPDeviceOpenCL::resetImpl() { UNIMPLEMENTED(); }
// CHIPEventOpenCL
// ************************************************************************

CHIPEventOpenCL::CHIPEventOpenCL(CHIPContextOpenCL *ChipContext,
                                 cl_event ClEvent, CHIPEventFlags Flags,
                                 bool UserEvent)
    : CHIPEvent((CHIPContext *)(ChipContext), Flags), ClEvent(ClEvent) {
  UserEvent_ = UserEvent;
}

CHIPEventOpenCL::CHIPEventOpenCL(CHIPContextOpenCL *ChipContext,
                                 CHIPEventFlags Flags)
    : CHIPEventOpenCL(ChipContext, nullptr, Flags, false) {}

uint64_t CHIPEventOpenCL::getFinishTime() {
  int Status;
  uint64_t Ret;
  Status = clGetEventProfilingInfo(ClEvent, CL_PROFILING_COMMAND_END,
                                   sizeof(Ret), &Ret, NULL);

  if (Status != CL_SUCCESS) {
    auto Status = clGetEventInfo(ClEvent, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                 sizeof(int), &EventStatus_, NULL);
    CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  }
  // CHIPERR_CHECK_LOG_AND_THROW(status, CL_SUCCESS, hipErrorTbd,
  //                             "Failed to query event for profiling info.");
  return Ret;
}

size_t CHIPEventOpenCL::getRefCount() {
  cl_uint RefCount;
  if (ClEvent == nullptr)
    return 0;
  int Status = clGetEventInfo(getNativeRef(), CL_EVENT_REFERENCE_COUNT, 4,
                              &RefCount, NULL);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  return RefCount;
}

CHIPEventOpenCL::~CHIPEventOpenCL() { ClEvent = nullptr; }

CHIPEventOpenCL *CHIPBackendOpenCL::createCHIPEvent(CHIPContext *ChipCtx,
                                                    CHIPEventFlags Flags,
                                                    bool UserEvent) {
  CHIPEventOpenCL *Event = new CHIPEventOpenCL((CHIPContextOpenCL *)ChipCtx,
                                               nullptr, Flags, UserEvent);

  return Event;
}

void CHIPEventOpenCL::recordStream(CHIPQueue *ChipQueue) {
  logTrace("CHIPEvent::recordStream()");
  auto MarkerEvent = ChipQueue->enqueueMarker();
  this->takeOver(MarkerEvent);

  this->EventStatus_ = EVENT_STATUS_RECORDING;
  return;
}

void CHIPEventOpenCL::takeOver(CHIPEvent *OtherIn) {
  logTrace("CHIPEventOpenCL::takeOver");
  decreaseRefCount("takeOver");
  {
    auto *Other = (CHIPEventOpenCL *)OtherIn;
    LOCK(EventMtx); // CHIPEvent::Refc_
    this->ClEvent = Other->ClEvent;
    this->Refc_ = Other->Refc_;
    this->Msg = Other->Msg;
  }
  increaseRefCount("takeOver");
}

bool CHIPEventOpenCL::wait() {
  logTrace("CHIPEventOpenCL::wait()");

  if (EventStatus_ != EVENT_STATUS_RECORDING) {
    logWarn("Called wait() on an event that isn't active.");
    return false;
  }

  auto Status = clWaitForEvents(1, &ClEvent);

  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  return true;
}

bool CHIPEventOpenCL::updateFinishStatus(bool ThrowErrorIfNotReady) {
  logTrace("CHIPEventOpenCL::updateFinishStatus()");
  if (ThrowErrorIfNotReady && this->ClEvent == nullptr)
    CHIPERR_LOG_AND_THROW("OpenCL has not been initialized cl_event is null",
                          hipErrorNotReady);

  int UpdatedStatus;
  auto Status = clGetEventInfo(ClEvent, CL_EVENT_COMMAND_EXECUTION_STATUS,
                               sizeof(int), &UpdatedStatus, NULL);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  if (ThrowErrorIfNotReady && UpdatedStatus != CL_COMPLETE) {
    CHIPERR_LOG_AND_THROW("Event not yet ready", hipErrorNotReady);
  }

  if (UpdatedStatus <= CL_COMPLETE) {
    EventStatus_ = EVENT_STATUS_RECORDED;
    return false;
  } else {
    return true;
  }
}

float CHIPEventOpenCL::getElapsedTime(CHIPEvent *OtherIn) {
  // Why do I need to lock the context mutex?
  // Can I lock the mutex of this and the other event?
  //

  CHIPEventOpenCL *Other = (CHIPEventOpenCL *)OtherIn;

  if (this->getContext() != Other->getContext())
    CHIPERR_LOG_AND_THROW(
        "Attempted to get elapsed time between two events that are not part of "
        "the same context",
        hipErrorTbd);

  this->updateFinishStatus(true);
  Other->updateFinishStatus(true);

  if (!this->isRecordingOrRecorded() || !Other->isRecordingOrRecorded())
    CHIPERR_LOG_AND_THROW("one of the events isn't/hasn't recorded",
                          hipErrorInvalidHandle);

  if (!this->isFinished() || !Other->isFinished())
    CHIPERR_LOG_AND_THROW("one of the events hasn't finished",
                          hipErrorNotReady);

  uint64_t Started = this->getFinishTime();
  uint64_t Finished = Other->getFinishTime();

  logTrace("EventElapsedTime: STARTED {} / {} FINISHED {} / {} \n",
           (void *)this, Started, (void *)Other, Finished);

  // apparently fails for Intel NEO, god knows why
  // assert(Finished >= Started);
  int64_t Elapsed;
  const int64_t NANOSECS = 1000000000;
  if (Finished < Started)
    logWarn("Finished < Started\n");
  Elapsed = Finished - Started;
  int64_t MS = (Elapsed / NANOSECS) * 1000;
  int64_t NS = Elapsed % NANOSECS;
  float FractInMS = ((float)NS) / 1000000.0f;
  return (float)MS + FractInMS;
}

void CHIPEventOpenCL::hostSignal() { UNIMPLEMENTED(); }

size_t CHIPEventOpenCL::increaseRefCount(std::string Reason) {
  LOCK(EventMtx); // CHIPEvent::Refc_
  auto status = clRetainEvent(this->ClEvent);
  if (!UserEvent_)
    assert(status == 0);
  // logDebug("CHIPEventOpenCL::increaseRefCount() {} {} refc {}->{} REASON:
  // {}",
  //          (void *)this, Msg.c_str(), *Refc_, *Refc_ + 1, Reason);
  (*Refc_)++;
  assert(*Refc_ = getRefCount() - 1);
  // logDebug("CHIPEventOpenCL::increaseRefCount() {} OpenCL RefCount: {}",
  //          (void *)this, getRefCount());
  return *Refc_;
}

size_t CHIPEventOpenCL::decreaseRefCount(std::string Reason) {
  LOCK(EventMtx); // CHIPEvent::Refc_
  // logDebug("CHIPEventOpenCL::decreaseRefCount() {} OpenCL RefCount: {}",
  //          (void *)this, getRefCount());
  // logDebug("CHIPEventOpenCL::decreaseRefCount() {} {} refc {}->{} REASON:
  // {}",
  //          (void *)this, Msg.c_str(), *Refc_, *Refc_ - 1, Reason);
  if (*Refc_ > 0) {
    (*Refc_)--;
  } else {
    logError("CHIPEvent::decreaseRefCount() called when refc == 0");
  }
  clReleaseEvent(this->ClEvent);
  return *Refc_;
}

// CHIPModuleOpenCL
//*************************************************************************

CHIPModuleOpenCL::CHIPModuleOpenCL(const SPVModule &SrcMod)
    : CHIPModule(SrcMod) {}

cl::Program *CHIPModuleOpenCL::get() { return &Program_; }

void CHIPModuleOpenCL::compile(CHIPDevice *ChipDev) {

  // TODO make compile_ which calls consumeSPIRV()
  logTrace("CHIPModuleOpenCL::compile()");
  consumeSPIRV();
  CHIPDeviceOpenCL *ChipDevOcl = (CHIPDeviceOpenCL *)ChipDev;
  CHIPContextOpenCL *ChipCtxOcl =
      (CHIPContextOpenCL *)(ChipDevOcl->getContext());

  int Err;
  auto SrcBin = Src_->getBinary();
  std::vector<char> BinaryVec(SrcBin.begin(), SrcBin.end());
  auto Program = cl::Program(*(ChipCtxOcl->get()), BinaryVec, false, &Err);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError);

  //   for (CHIPDevice *chip_dev : chip_devices) {
  std::string Name = ChipDevOcl->getName();
  Err = Program.build(Backend->getJitFlags().c_str());
  auto ErrBuild = Err;

  std::string Log =
      Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(*ChipDevOcl->ClDevice, &Err);
  if (ErrBuild != CL_SUCCESS)
    logError("Program BUILD LOG for device #{}:{}:\n{}\n",
             ChipDevOcl->getDeviceId(), Name, Log);

  CHIPERR_CHECK_LOG_AND_THROW(ErrBuild, CL_SUCCESS,
                              hipErrorInitializationError);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError);

  logTrace("Program BUILD LOG for device #{}:{}:\n{}\n",
           ChipDevOcl->getDeviceId(), Name, Log);

  std::vector<cl::Kernel> Kernels;
  Err = Program.createKernels(&Kernels);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError);

  logTrace("Kernels in CHIPModuleOpenCL: {} \n", Kernels.size());
  for (int KernelIdx = 0; KernelIdx < Kernels.size(); KernelIdx++) {
    auto Kernel = Kernels[KernelIdx];
    std::string HostFName = Kernel.getInfo<CL_KERNEL_FUNCTION_NAME>(&Err);
    CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError,
                                "Failed to fetch OpenCL kernel name");
    auto *FuncInfo = findFunctionInfo(HostFName);
    if (!FuncInfo) {
      continue; // TODO
      // CHIPERR_LOG_AND_THROW("Failed to find kernel in
      // OpenCLFunctionInfoMap",
      //                      hipErrorInitializationError);
    }
    CHIPKernelOpenCL *ChipKernel =
        new CHIPKernelOpenCL(Kernel, ChipDevOcl, HostFName, FuncInfo, this);
    addKernel(ChipKernel);
  }

  Program_ = Program;
}

CHIPQueue *CHIPDeviceOpenCL::createQueue(CHIPQueueFlags Flags, int Priority) {
  CHIPQueueOpenCL *NewQ = new CHIPQueueOpenCL(this, Priority);
  NewQ->setFlags(Flags);
  return NewQ;
}

CHIPQueue *CHIPDeviceOpenCL::createQueue(const uintptr_t *NativeHandles,
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
cl::Kernel *CHIPKernelOpenCL::get() { return &OclKernel_; }

/// Clones the instance but with separate cl_kernel handle.
CHIPKernelOpenCL *CHIPKernelOpenCL::clone() {
  cl_int Err;
  // NOTE: clCloneKernel is not used here due to its experience on
  // Intel (GPU) OpenCL which crashed if clSetKernelArgSVMPointer() was
  // called on the original cl_kernel.
  auto Cloned = clCreateKernel(Module->get()->get(), Name_.c_str(), &Err);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd);
  return new CHIPKernelOpenCL(cl::Kernel(Cloned, false), Device, Name_,
                              getFuncInfo(), Module);
}

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

CHIPKernelOpenCL::CHIPKernelOpenCL(cl::Kernel ClKernel, CHIPDeviceOpenCL *Dev,
                                   std::string HostFName, SPVFuncInfo *FuncInfo,
                                   CHIPModuleOpenCL *Parent)
    : CHIPKernel(HostFName, FuncInfo), Module(Parent), Device(Dev) {

  OclKernel_ = ClKernel;
  int Err = 0;
  // TODO attributes
  cl_uint NumArgs = OclKernel_.getInfo<CL_KERNEL_NUM_ARGS>(&Err);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                              "Failed to get num args for kernel");
  assert(FuncInfo_->getNumKernelArgs() == NumArgs);

  MaxWorkGroupSize_ =
      OclKernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(*Device->get());
  StaticLocalSize_ =
      OclKernel_.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(*Device->get());
  MaxDynamicLocalSize_ =
      (size_t)Device->getAttr(hipDeviceAttributeMaxSharedMemoryPerBlock) -
      StaticLocalSize_;
  PrivateSize_ =
      OclKernel_.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(*Device->get());

  Name_ = OclKernel_.getInfo<CL_KERNEL_FUNCTION_NAME>();

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

bool CHIPContextOpenCL::allDevicesSupportFineGrainSVM() {
  bool allFineGrainSVM = true;
  if (!static_cast<CHIPDeviceOpenCL *>(this->ChipDevice_)
           ->supportsFineGrainSVM()) {
    allFineGrainSVM = false;
  }
  return allFineGrainSVM;
}

void CHIPContextOpenCL::freeImpl(void *Ptr) {
  LOCK(ContextMtx); // CHIPContextOpenCL::SvmMemory
  SvmMemory.free(Ptr);
}

cl::Context *CHIPContextOpenCL::get() { return ClContext; }
CHIPContextOpenCL::CHIPContextOpenCL(cl::Context *CtxIn) {
  logTrace("CHIPContextOpenCL Initialized via OpenCL Context pointer.");
  ClContext = CtxIn;
  SvmMemory.init(*CtxIn);
}

void *CHIPContextOpenCL::allocateImpl(size_t Size, size_t Alignment,
                                      hipMemoryType MemType,
                                      CHIPHostAllocFlags Flags) {
  void *Retval;
  LOCK(ContextMtx); // CHIPContextOpenCL::SvmMemory

  Retval = SvmMemory.allocate(Size);
  return Retval;
}

// CHIPQueueOpenCL
//*************************************************************************
struct HipStreamCallbackData {
  hipStream_t Stream;
  hipError_t Status;
  void *UserData;
  hipStreamCallback_t Callback;
  CHIPEventOpenCL *CallbackFinishEvent;
};

void CL_CALLBACK pfn_notify(cl_event Event, cl_int CommandExecStatus,
                            void *UserData) {
  HipStreamCallbackData *Cbo = (HipStreamCallbackData *)(UserData);
  if (Cbo == nullptr)
    return;
  if (Cbo->Callback == nullptr)
    return;
  Cbo->Callback(Cbo->Stream, Cbo->Status, Cbo->UserData);
  if (Cbo->CallbackFinishEvent != nullptr) {
    clSetUserEventStatus(Cbo->CallbackFinishEvent->ClEvent, CL_COMPLETE);
    Cbo->CallbackFinishEvent->decreaseRefCount("Notified finished.");
  }
  delete Cbo;
}

void CHIPQueueOpenCL::MemMap(const AllocationInfo *AllocInfo,
                             CHIPQueue::MEM_MAP_TYPE Type) {
  if (static_cast<CHIPDeviceOpenCL *>(this->getDevice())
          ->supportsFineGrainSVM()) {
    logDebug("Device supports fine grain SVM. Skipping MemMap/Unmap");
  }
  cl_int Status;
  if (Type == CHIPQueue::MEM_MAP_TYPE::HOST_READ) {
    logDebug("CHIPQueueOpenCL::MemMap HOST_READ");
    Status =
        clEnqueueSVMMap(ClQueue_->get(), CL_TRUE, CL_MAP_READ,
                        AllocInfo->HostPtr, AllocInfo->Size, 0, NULL, NULL);
  } else if (Type == CHIPQueue::MEM_MAP_TYPE::HOST_WRITE) {
    logDebug("CHIPQueueOpenCL::MemMap HOST_WRITE");
    Status =
        clEnqueueSVMMap(ClQueue_->get(), CL_TRUE, CL_MAP_WRITE,
                        AllocInfo->HostPtr, AllocInfo->Size, 0, NULL, NULL);
  } else if (Type == CHIPQueue::MEM_MAP_TYPE::HOST_READ_WRITE) {
    logDebug("CHIPQueueOpenCL::MemMap HOST_READ_WRITE");
    Status =
        clEnqueueSVMMap(ClQueue_->get(), CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                        AllocInfo->HostPtr, AllocInfo->Size, 0, NULL, NULL);
  } else {
    assert(0 && "Invalid MemMap Type");
  }
  assert(Status == CL_SUCCESS);
}

void CHIPQueueOpenCL::MemUnmap(const AllocationInfo *AllocInfo) {
  if (static_cast<CHIPDeviceOpenCL *>(this->getDevice())
          ->supportsFineGrainSVM()) {
    logDebug("Device supports fine grain SVM. Skipping MemMap/Unmap");
  }
  logDebug("CHIPQueueOpenCL::MemUnmap");

  auto Status =
      clEnqueueSVMUnmap(ClQueue_->get(), AllocInfo->HostPtr, 0, NULL, NULL);
  assert(Status == CL_SUCCESS);
}

cl::CommandQueue *CHIPQueueOpenCL::get() { return ClQueue_; }

void CHIPQueueOpenCL::addCallback(hipStreamCallback_t Callback,
                                  void *UserData) {
  logTrace("CHIPQueueOpenCL::addCallback()");

  cl::Context *ClContext_ = ((CHIPContextOpenCL *)ChipContext_)->get();
  cl_int Err;

  CHIPEventOpenCL *HoldBackEvent =
      (CHIPEventOpenCL *)Backend->createCHIPEvent(ChipContext_);

  HoldBackEvent->ClEvent = clCreateUserEvent(ClContext_->get(), &Err);

  std::vector<CHIPEvent *> WaitForEvents{HoldBackEvent};
  auto LastEvent = getLastEvent();
  if (LastEvent != nullptr)
    WaitForEvents.push_back(LastEvent);

  // Enqueue a barrier used to ensure the callback is not called too early,
  // otherwise it would be (at worst) executed in this host thread when
  // setting it, blocking the execution, while the clients might expect
  // parallel execution.
  auto HoldbackBarrierCompletedEv =
      (CHIPEventOpenCL *)enqueueBarrier(&WaitForEvents);

  // OpenCL event callbacks have undefined execution ordering/finishing
  // guarantees. We need to enforce CUDA ordering using user events.

  CHIPEventOpenCL *CallbackEvent =
      (CHIPEventOpenCL *)Backend->createCHIPEvent(ChipContext_);

  CallbackEvent->ClEvent = clCreateUserEvent(ClContext_->get(), &Err);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd);

  // Make the succeeding commands wait for the user event which will be
  // set CL_COMPLETE by the callback trampoline function pfn_notify after
  // finishing the user CB's execution.

  HipStreamCallbackData *Cb = new HipStreamCallbackData{
      this, hipSuccess, UserData, Callback, CallbackEvent};

  std::vector<CHIPEvent *> WaitForEventsCBB{CallbackEvent};
  auto CallbackCompleted = enqueueBarrier(&WaitForEventsCBB);

  // We know that the callback won't be yet launched since it's depending
  // on the barrier which waits for the user event.
  auto Status = clSetEventCallback(HoldbackBarrierCompletedEv->ClEvent,
                                   CL_COMPLETE, pfn_notify, Cb);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);

  updateLastEvent(CallbackCompleted);
  ClQueue_->flush();

  // Now the CB can start executing in the background:
  clSetUserEventStatus(HoldBackEvent->ClEvent, CL_COMPLETE);
  HoldBackEvent->decreaseRefCount("Notified finished.");

  return;
};

CHIPEvent *CHIPQueueOpenCL::enqueueMarkerImpl() {
  CHIPEventOpenCL *MarkerEvent =
      (CHIPEventOpenCL *)Backend->createCHIPEvent(ChipContext_);
  auto Status =
      clEnqueueMarker(this->get()->get(), MarkerEvent->getNativePtr());
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  MarkerEvent->Msg = "marker";
  return MarkerEvent;
}

CHIPEventOpenCL *CHIPQueueOpenCL::getLastEvent() {
  LOCK(LastEventMtx); // CHIPQueue::LastEvent_
  // TODO: shouldn't we increment the ref count here, assuming it will be
  // needed to be kept alive for the client?
  return (CHIPEventOpenCL *)LastEvent_;
}

CHIPEvent *CHIPQueueOpenCL::launchImpl(CHIPExecItem *ExecItem) {
  logTrace("CHIPQueueOpenCL->launch()");
  auto *OclContext = static_cast<CHIPContextOpenCL *>(ChipContext_);
  CHIPEventOpenCL *LaunchEvent =
      (CHIPEventOpenCL *)Backend->createCHIPEvent(OclContext);
  CHIPExecItemOpenCL *ChipOclExecItem = (CHIPExecItemOpenCL *)ExecItem;
  CHIPKernelOpenCL *Kernel = (CHIPKernelOpenCL *)ChipOclExecItem->getKernel();
  assert(Kernel && "Kernel in ExecItem is NULL!");
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
#ifdef DUBIOUS_LOCKS
  LOCK(Backend->DubiousLockOpenCL);
#endif

  auto SvmAllocationsToKeepAlive =
      annotateSvmPointers(*OclContext, Kernel->get()->get());

  auto Status = clEnqueueNDRangeKernel(ClQueue_->get(), Kernel->get()->get(),
                                       NumDims, GlobalOffset, Global, Local, 0,
                                       nullptr, LaunchEvent->getNativePtr());
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);

  std::shared_ptr<CHIPArgSpillBuffer> SpillBuf = ExecItem->getArgSpillBuffer();

  if (SpillBuf || SvmAllocationsToKeepAlive) {
    // Use an event call back to prolong the lifetimes of the
    // following objects until the kernel terminates.
    //
    // * SpillBuffer holds an allocation referenced by the kernel
    //   shared by exec item, which might get destroyed before the
    //   kernel is launched/completed
    //
    // * Annotated SVM pointers may need to outlive the kernel
    //   execution. The OpenCL spec does not clearly specify how long
    //   the pointers, annotated via clSetKernelExecInfo(), needs to
    //   live.
    auto *CBData = new KernelEventCallbackData;
    CBData->ArgSpillBuffer = SpillBuf;
    CBData->SvmKeepAlives = std::move(SvmAllocationsToKeepAlive);
    Status = clSetEventCallback(LaunchEvent->getNativeRef(), CL_COMPLETE,
                                kernelEventCallback, CBData);
    if (Status != CL_SUCCESS) {
      delete CBData;
      CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
    }
  }

  LaunchEvent->Msg = "KernelLaunch";
  return LaunchEvent;
}

CHIPQueueOpenCL::CHIPQueueOpenCL(CHIPDevice *ChipDevice, int Priority,
                                 cl_command_queue Queue)
    : CHIPQueue(ChipDevice, CHIPQueueFlags{}, Priority) {

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

  if (Queue)
    ClQueue_ = new cl::CommandQueue(Queue);
  else {
    cl::Context *ClContext_ = ((CHIPContextOpenCL *)ChipContext_)->get();
    cl::Device *ClDevice_ = ((CHIPDeviceOpenCL *)ChipDevice_)->get();
    cl_int Status;
    // Adding priority breaks correctness?
    // cl_queue_properties QueueProperties[] = {
    //     CL_QUEUE_PRIORITY_KHR, PrioritySelection, CL_QUEUE_PROPERTIES,
    //     CL_QUEUE_PROFILING_ENABLE, 0};
    cl_queue_properties QueueProperties[] = {CL_QUEUE_PROPERTIES,
                                             CL_QUEUE_PROFILING_ENABLE, 0};

    const cl_command_queue Q = clCreateCommandQueueWithProperties(
        ClContext_->get(), ClDevice_->get(), QueueProperties, &Status);
    ClQueue_ = new cl::CommandQueue(Q);

    CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS,
                                hipErrorInitializationError);
  }
}

CHIPQueueOpenCL::~CHIPQueueOpenCL() {
  logTrace("~CHIPQueueOpenCL() {}", (void *)this);
}

CHIPEvent *CHIPQueueOpenCL::memCopyAsyncImpl(void *Dst, const void *Src,
                                             size_t Size) {
  CHIPEventOpenCL *Event =
      (CHIPEventOpenCL *)Backend->createCHIPEvent(ChipContext_);
  logTrace("clSVMmemcpy {} -> {} / {} B\n", Src, Dst, Size);
  if (Dst == Src) {
    // Although ROCm API ref says that Dst and Src should not overlap,
    // HIP seems to handle Dst == Src as a special (no-operation) case.
    // This is seen in the test unit/memory/hipMemcpyAllApiNegative.

    // Intel GPU OpenCL driver seems to do also so for clEnqueueSVMMemcpy, which
    // makes/ it pass, but Intel CPU OpenCL returns CL_​MEM_​COPY_​OVERLAP
    // like it should. To unify the behavior, let's convert the special case to
    // a maker here, so we can return an event.
    cl::Event MarkerEvent;
    auto Status = clEnqueueMarker(ClQueue_->get(), Event->getNativePtr());
    CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  } else {
#ifdef DUBIOUS_LOCKS
    LOCK(Backend->DubiousLockOpenCL)
#endif
    auto Status = ::clEnqueueSVMMemcpy(ClQueue_->get(), CL_FALSE, Dst, Src,
                                       Size, 0, nullptr, Event->getNativePtr());
    CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorRuntimeMemory);
  }
  return Event;
}

void CHIPQueueOpenCL::finish() {
#ifdef DUBIOUS_LOCKS
  LOCK(Backend->DubiousLockOpenCL)
#endif
  auto Status = ClQueue_->finish();
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
}

CHIPEvent *CHIPQueueOpenCL::memFillAsyncImpl(void *Dst, size_t Size,
                                             const void *Pattern,
                                             size_t PatternSize) {
  CHIPEventOpenCL *Event =
      (CHIPEventOpenCL *)Backend->createCHIPEvent(ChipContext_);
  logTrace("clSVMmemfill {} / {} B\n", Dst, Size);
  cl_event Ev = nullptr;
  int Retval = ::clEnqueueSVMMemFill(ClQueue_->get(), Dst, Pattern, PatternSize,
                                     Size, 0, nullptr, Event->getNativePtr());
  CHIPERR_CHECK_LOG_AND_THROW(Retval, CL_SUCCESS, hipErrorRuntimeMemory);
  return Event;
};

CHIPEvent *CHIPQueueOpenCL::memCopy2DAsyncImpl(void *Dst, size_t Dpitch,
                                               const void *Src, size_t Spitch,
                                               size_t Width, size_t Height) {
  UNIMPLEMENTED(nullptr);
};

CHIPEvent *CHIPQueueOpenCL::memCopy3DAsyncImpl(void *Dst, size_t Dpitch,
                                               size_t Dspitch, const void *Src,
                                               size_t Spitch, size_t Sspitch,
                                               size_t Width, size_t Height,
                                               size_t Depth) {
  UNIMPLEMENTED(nullptr);
};

hipError_t CHIPQueueOpenCL::getBackendHandles(uintptr_t *NativeInfo,
                                              int *NumHandles) {
  logTrace("CHIPQueueOpenCL::getBackendHandles");
  if (*NumHandles < 4) {
    logError("getBackendHandles requires space for 4 handles");
    return hipErrorInvalidValue;
  }
  *NumHandles = 4;

  // Get queue handler
  NativeInfo[3] = (uintptr_t)ClQueue_->get();

  // Get context handler
  cl::Context *Ctx = ((CHIPContextOpenCL *)ChipContext_)->get();
  NativeInfo[2] = (uintptr_t)Ctx->get();

  // Get device handler
  cl::Device *Dev = ((CHIPDeviceOpenCL *)ChipDevice_)->get();
  NativeInfo[1] = (uintptr_t)Dev->get();

  // Get platform handler
  cl_platform_id Plat = Dev->getInfo<CL_DEVICE_PLATFORM>();
  NativeInfo[0] = (uintptr_t)Plat;
  return hipSuccess;
}

CHIPEvent *CHIPQueueOpenCL::memPrefetchImpl(const void *Ptr, size_t Count) {
  UNIMPLEMENTED(nullptr);
}

CHIPEvent *
CHIPQueueOpenCL::enqueueBarrierImpl(std::vector<CHIPEvent *> *EventsToWaitFor) {
#ifdef DUBIOUS_LOCKS
  LOCK(Backend->DubiousLockOpenCL)
#endif
  CHIPEventOpenCL *Event =
      (CHIPEventOpenCL *)Backend->createCHIPEvent(this->ChipContext_);
  cl_int RefCount;
  int Status;
  Status = clGetEventInfo(Event->getNativeRef(), CL_EVENT_REFERENCE_COUNT, 4,
                          &RefCount, NULL);
  if (EventsToWaitFor && EventsToWaitFor->size() > 0) {
    std::vector<cl_event> Events = {};
    for (auto E : *EventsToWaitFor) {
      auto Ee = (CHIPEventOpenCL *)E;
      // assert(Ee->getRefCount() > 0);
      Events.push_back(Ee->getNativeRef());
    }
    // auto Status = ClQueue_->enqueueBarrierWithWaitList(&Events, &Barrier);
    auto Status =
        clEnqueueBarrierWithWaitList(ClQueue_->get(), Events.size(),
                                     Events.data(), &(Event->getNativeRef()));
    CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  } else {
    // auto Status = ClQueue_->enqueueBarrierWithWaitList(nullptr, &Barrier);
    auto Status = clEnqueueBarrierWithWaitList(ClQueue_->get(), 0, nullptr,
                                               Event->getNativePtr());
    CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  }

  Status = clGetEventInfo(Event->getNativeRef(), CL_EVENT_REFERENCE_COUNT, 4,
                          &RefCount, NULL);
  return Event;
}

// CHIPExecItemOpenCL
//*************************************************************************

cl::Kernel *CHIPExecItemOpenCL::get() { return ClKernel_; }

void CHIPExecItemOpenCL::setupAllArgs() {
  if (!ArgsSetup) {
    ArgsSetup = true;
  } else {
    return;
  }
  CHIPKernelOpenCL *Kernel = (CHIPKernelOpenCL *)getKernel();
  SPVFuncInfo *FuncInfo = Kernel->getFuncInfo();
  int Err = 0;

  if (FuncInfo->hasByRefArgs()) {
    ArgSpillBuffer_ =
        std::make_shared<CHIPArgSpillBuffer>(ChipQueue_->getContext());
    ArgSpillBuffer_->computeAndReserveSpace(*FuncInfo);
  }

  auto ArgVisitor = [&](const SPVFuncInfo::KernelArg &Arg) -> void {
    switch (Arg.Kind) {
    default:
      CHIPERR_LOG_AND_THROW("Internal CHIP-SPV error: Unknown argument kind",
                            hipErrorTbd);
    case SPVTypeKind::Image: {
      auto *TexObj =
          *reinterpret_cast<const CHIPTextureOpenCL *const *>(Arg.Data);
      cl_mem Image = TexObj->getImage();
      logTrace("set image arg {} for tex {}\n", Arg.Index, (void *)TexObj);
      Err = ::clSetKernelArg(Kernel->get()->get(), Arg.Index, sizeof(cl_mem),
                             &Image);
      CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                                  "clSetKernelArg failed for image argument.");
      break;
    }
    case SPVTypeKind::Sampler: {
      auto *TexObj =
          *reinterpret_cast<const CHIPTextureOpenCL *const *>(Arg.Data);
      cl_sampler Sampler = TexObj->getSampler();
      logTrace("set sampler arg {} for tex {}\n", Arg.Index, (void *)TexObj);
      Err = ::clSetKernelArg(Kernel->get()->get(), Arg.Index,
                             sizeof(cl_sampler), &Sampler);
      CHIPERR_CHECK_LOG_AND_THROW(
          Err, CL_SUCCESS, hipErrorTbd,
          "clSetKernelArg failed for sampler argument.");
      break;
    }
    case SPVTypeKind::POD: {
      logTrace("clSetKernelArg {} SIZE {} to {}\n", Arg.Index, Arg.Size,
               Arg.Data);
      Err =
          ::clSetKernelArg(Kernel->get()->get(), Arg.Index, Arg.Size, Arg.Data);
      CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                                  "clSetKernelArg failed");
      break;
    }
    case SPVTypeKind::Pointer: {
      CHIPASSERT(Arg.Size == sizeof(void *));
      if (Arg.isWorkgroupPtr()) {
        logTrace("setLocalMemSize to {}\n", SharedMem_);
        Err = ::clSetKernelArg(Kernel->get()->get(), Arg.Index, SharedMem_,
                               nullptr);
      } else {
        logTrace("clSetKernelArgSVMPointer {} SIZE {} to {} (value {})\n",
                 Arg.Index, Arg.Size, Arg.Data, *(const void **)Arg.Data);
        Err = ::clSetKernelArgSVMPointer(
            Kernel->get()->get(), Arg.Index,
            // Unlike clSetKernelArg() which takes address to the argument,
            // this function takes the argument value directly.
            *(const void **)Arg.Data);
        if (Err != CL_SUCCESS) {
          // ROCm seems to allow passing invalid pointers to kernels if they are
          // not derefenced (see test_device_adjacent_difference of rocPRIM).
          // If the setting of the arg fails, let's assume this might be such a
          // case and convert it to a null pointer.
          logWarn(
              "clSetKernelArgSVMPointer {} SIZE {} to {} (value {}) returned "
              "error, setting the arg to nullptr\n",
              Arg.Index, Arg.Size, Arg.Data, *(const void **)Arg.Data);
          Err = ::clSetKernelArgSVMPointer(Kernel->get()->get(), Arg.Index,
                                           nullptr);
        }
      }
      CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                                  "clSetKernelArgSVMPointer failed");
      break;
    }
    case SPVTypeKind::PODByRef: {
      auto *SpillSlot = ArgSpillBuffer_->allocate(Arg);
      assert(SpillSlot);
      Err = ::clSetKernelArgSVMPointer(Kernel->get()->get(), Arg.Index,
                                       SpillSlot);
      CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                                  "clSetKernelArgSVMPointer failed");
      break;
    }
    }
  };
  FuncInfo->visitKernelArgs(getArgs(), ArgVisitor);

  if (FuncInfo->hasByRefArgs())
    ChipQueue_->memCopyAsync(ArgSpillBuffer_->getDeviceBuffer(),
                             ArgSpillBuffer_->getHostBuffer(),
                             ArgSpillBuffer_->getSize());

  return;
}

void CHIPExecItemOpenCL::setKernel(CHIPKernel *Kernel) {
  assert(Kernel && "Kernel is nullptr!");
  // Make a clone of the kernel so the its cl_kernel object is not
  // shared among other threads (sharing cl_kernel is discouraged by
  // the OpenCL spec).
  auto *Clone = static_cast<CHIPKernelOpenCL *>(Kernel)->clone();
  ChipKernel_.reset(Clone);

  // Arguments set on the original cl_kernel are not copied.
  ArgsSetup = false;
}

// CHIPBackendOpenCL
//*************************************************************************
CHIPExecItem *CHIPBackendOpenCL::createCHIPExecItem(dim3 GirdDim, dim3 BlockDim,
                                                    size_t SharedMem,
                                                    hipStream_t ChipQueue) {
  CHIPExecItemOpenCL *ExecItem =
      new CHIPExecItemOpenCL(GirdDim, BlockDim, SharedMem, ChipQueue);
  return ExecItem;
};
CHIPQueue *CHIPBackendOpenCL::createCHIPQueue(CHIPDevice *ChipDev) {
  CHIPDeviceOpenCL *ChipDevCl = (CHIPDeviceOpenCL *)ChipDev;
  return new CHIPQueueOpenCL(ChipDevCl, OCL_DEFAULT_QUEUE_PRIORITY);
}

CHIPCallbackData *
CHIPBackendOpenCL::createCallbackData(hipStreamCallback_t Callback,
                                      void *UserData, CHIPQueue *ChipQueue) {
  UNIMPLEMENTED(nullptr);
}

CHIPEventMonitor *CHIPBackendOpenCL::createCallbackEventMonitor_() {
  auto Evm = new CHIPEventMonitorOpenCL();
  Evm->start();
  return Evm;
}

CHIPEventMonitor *CHIPBackendOpenCL::createStaleEventMonitor_() {
  UNIMPLEMENTED(nullptr);
}

std::string CHIPBackendOpenCL::getDefaultJitFlags() {
  return std::string("-x spir -cl-kernel-arg-info");
}

void CHIPBackendOpenCL::initializeImpl(std::string CHIPPlatformStr,
                                       std::string CHIPDeviceTypeStr,
                                       std::string CHIPDeviceStr) {
  logTrace("CHIPBackendOpenCL Initialize");
  MinQueuePriority_ = CL_QUEUE_PRIORITY_MED_KHR;

  // transform device type string into CL
  cl_bitfield SelectedDevType = 0;
  if (CHIPDeviceTypeStr == "all")
    SelectedDevType = CL_DEVICE_TYPE_ALL;
  else if (CHIPDeviceTypeStr == "cpu")
    SelectedDevType = CL_DEVICE_TYPE_CPU;
  else if (CHIPDeviceTypeStr == "gpu")
    SelectedDevType = CL_DEVICE_TYPE_GPU;
  else if (CHIPDeviceTypeStr == "default")
    SelectedDevType = CL_DEVICE_TYPE_DEFAULT;
  else if (CHIPDeviceTypeStr == "accel")
    SelectedDevType = CL_DEVICE_TYPE_ACCELERATOR;
  else
    throw InvalidDeviceType("Unknown value provided for CHIP_DEVICE_TYPE\n");
  logTrace("Using Devices of type {}", CHIPDeviceTypeStr);

  std::vector<cl::Platform> Platforms;
  cl_int Err = cl::Platform::get(&Platforms);
  if (Err != CL_SUCCESS) {
    logCritical("OpenCL failed to initialize any devices");
    std::exit(1);
  }
  std::stringstream StrStream;
  StrStream << "\nFound " << Platforms.size() << " OpenCL platforms:\n";
  for (int i = 0; i < Platforms.size(); i++) {
    StrStream << i << ". " << Platforms[i].getInfo<CL_PLATFORM_NAME>() << "\n";
  }
  logTrace("{}", StrStream.str());
  int SelectedPlatformIdx = atoi(CHIPPlatformStr.c_str());
  if (SelectedPlatformIdx >= Platforms.size()) {
    logCritical("Selected OpenCL platform {} is out of range",
                SelectedPlatformIdx);
    std::exit(1);
  }

  cl::Platform SelectedPlatform = Platforms[SelectedPlatformIdx];
  logDebug("CHIP_PLATFORM={} Selected OpenCL platform {}", SelectedPlatformIdx,
           SelectedPlatform.getInfo<CL_PLATFORM_NAME>());

  StrStream.str("");

  StrStream << "OpenCL Devices of type " << CHIPDeviceTypeStr
            << " with SPIR-V_1 support:\n";
  std::vector<cl::Device> SpirvDevices;
  std::vector<cl::Device> Dev;
  Err = SelectedPlatform.getDevices(SelectedDevType, &Dev);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError);
  for (auto D : Dev) {
    std::string Ver = D.getInfo<CL_DEVICE_IL_VERSION>(&Err);
    if ((Err == CL_SUCCESS) && (Ver.rfind("SPIR-V_1.", 0) == 0)) {
      std::string DeviceName = D.getInfo<CL_DEVICE_NAME>();
      StrStream << DeviceName << "\n";
      SpirvDevices.push_back(D);
    }
  }
  logTrace("{}", StrStream.str());

  int SelectedDeviceIdx = atoi(CHIPDeviceStr.c_str());
  if (SelectedDeviceIdx >= SpirvDevices.size()) {
    logCritical("Selected OpenCL device {} is out of range", SelectedDeviceIdx);
    std::exit(1);
  }

  auto Device = SpirvDevices[SelectedDeviceIdx];
  logDebug("CHIP_DEVICE={} Selected OpenCL device {}", SelectedDeviceIdx,
           Device.getInfo<CL_DEVICE_NAME>());

  // Create context which has devices
  // Create queues that have devices each of which has an associated context
  // TODO Change this to spirv_enabled_devices
  cl::Context *Ctx = new cl::Context(SpirvDevices);
  CHIPContextOpenCL *ChipContext = new CHIPContextOpenCL(Ctx);
  Backend->addContext(ChipContext);

  // TODO for now only a single device is supported.
  cl::Device *clDev = new cl::Device(Device);
  CHIPDeviceOpenCL *ChipDev = CHIPDeviceOpenCL::create(clDev, ChipContext, 0);

  // Add device to context & backend
  ChipContext->setDevice(ChipDev);
  logTrace("OpenCL Context Initialized.");
};

void CHIPBackendOpenCL::initializeFromNative(const uintptr_t *NativeHandles,
                                             int NumHandles) {
  logTrace("CHIPBackendOpenCL InitializeNative");
  MinQueuePriority_ = CL_QUEUE_PRIORITY_MED_KHR;
  // cl_platform_id PlatId = (cl_platform_id)NativeHandles[0];
  cl_device_id DevId = (cl_device_id)NativeHandles[1];
  cl_context CtxId = (cl_context)NativeHandles[2];

  cl::Context *Ctx = new cl::Context(CtxId);
  CHIPContextOpenCL *ChipContext = new CHIPContextOpenCL(Ctx);
  addContext(ChipContext);

  cl::Device *Dev = new cl::Device(DevId);
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

std::string resultToString(int Status) {
  switch (Status) {
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
