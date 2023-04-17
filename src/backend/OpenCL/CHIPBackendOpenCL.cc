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
  auto NumSvmAllocations = Ctx.getRegion().getNumAllocations();
  if (NumSvmAllocations) {
    SvmAnnotationList.reserve(NumSvmAllocations);
    SvmKeepAlives.reset(new std::vector<std::shared_ptr<void>>());
    SvmKeepAlives->reserve(NumSvmAllocations);
    for (std::shared_ptr<void> Ptr : Ctx.getRegion().getSvmPointers()) {
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
}

// CHIPDeviceOpenCL
// ************************************************************************

CHIPTexture *
CHIPDeviceOpenCL::createTexture(const hipResourceDesc *ResDesc,
                                const hipTextureDesc *TexDesc,
                                const struct hipResourceViewDesc *ResViewDesc) {
  logTrace("CHIPDeviceOpenCL::createTexture");

  bool NormalizedFloat = TexDesc->readMode == hipReadModeNormalizedFloat;
  cl::CommandQueue &ClCmdQ = ((CHIPQueueOpenCL *)getDefaultQueue())->get();

  cl::Context &ClContext = ((CHIPContextOpenCL *)getContext())->get();
  cl_sampler Sampler = createSampler(ClContext.get(), *ResDesc, *TexDesc);

  if (ResDesc->resType == hipResourceTypeArray) {
    hipArray *Array = ResDesc->res.array.array;
    // Checked in CHIPBindings already.
    CHIPASSERT(Array->data && "Invalid hipArray.");
    CHIPASSERT(!Array->isDrv && "Not supported/implemented yet.");
    size_t Width = Array->width;
    size_t Height = Array->height;
    size_t Depth = Array->depth;

    cl_mem Image = createImage(ClContext.get(), Array->textureType, Array->desc,
                               NormalizedFloat, Width, Height, Depth);

    auto Tex = std::make_unique<CHIPTextureOpenCL>(*ResDesc, Image, Sampler);
    logTrace("Created texture: {}", (void *)Tex.get());

    CHIPRegionDesc SrcRegion = CHIPRegionDesc::from(*Array);
    memCopyToImage(ClCmdQ.get(), Image, Array->data, SrcRegion);

    return Tex.release();
  }

  if (ResDesc->resType == hipResourceTypeLinear) {
    auto &Res = ResDesc->res.linear;
    auto TexelByteSize = getChannelByteSize(Res.desc);
    size_t Width = Res.sizeInBytes / TexelByteSize;

    cl_mem Image = createImage(ClContext.get(), hipTextureType1D, Res.desc,
                               NormalizedFloat, Width);

    auto Tex = std::make_unique<CHIPTextureOpenCL>(*ResDesc, Image, Sampler);
    logTrace("Created texture: {}", (void *)Tex.get());

    // Copy data to image.
    auto SrcDesc = CHIPRegionDesc::get1DRegion(Width, TexelByteSize);
    memCopyToImage(ClCmdQ.get(), Image, Res.devPtr, SrcDesc);

    return Tex.release();
  }

  if (ResDesc->resType == hipResourceTypePitch2D) {
    auto &Res = ResDesc->res.pitch2D;
    assert(Res.pitchInBytes >= Res.width); // Checked in CHIPBindings.

    cl_mem Image = createImage(ClContext.get(), hipTextureType2D, Res.desc,
                               NormalizedFloat, Res.width, Res.height);

    auto Tex = std::make_unique<CHIPTextureOpenCL>(*ResDesc, Image, Sampler);
    logTrace("Created texture: {}", (void *)Tex.get());

    // Copy data to image.
    auto SrcDesc = CHIPRegionDesc::from(*ResDesc);
    memCopyToImage(ClCmdQ.get(), Image, Res.devPtr, SrcDesc);

    return Tex.release();
  }

  CHIPASSERT(false && "Unsupported/unimplemented texture resource type.");
  return nullptr;
}

CHIPDeviceOpenCL::CHIPDeviceOpenCL(CHIPContextOpenCL *ChipCtx, cl::Device DevIn,
                                   int Idx)
    : CHIPDevice(ChipCtx, Idx), ClDevice(DevIn) {
  logTrace("CHIPDeviceOpenCL initialized via OpenCL device pointer and context "
           "pointer");
  cl_device_svm_capabilities DeviceSVMCapabilities;
  auto Status =
      DevIn.getInfo(CL_DEVICE_SVM_CAPABILITIES, &DeviceSVMCapabilities);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  this->SupportsFineGrainSVM =
      DeviceSVMCapabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER;
  if (this->SupportsFineGrainSVM) {
    logTrace("Device supports fine grain SVM");
  } else {
    logTrace("Device does not support fine grain SVM");
  }
}

CHIPDeviceOpenCL *CHIPDeviceOpenCL::create(cl::Device ClDevice,
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

  this->MaxMallocSize_ = ClDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  Temp = ClDevice.getInfo<CL_DEVICE_NAME>();
  strncpy(HipDeviceProps_.name, Temp.c_str(), 255);
  HipDeviceProps_.name[255] = 0;

  HipDeviceProps_.totalGlobalMem =
      ClDevice.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&Err);

  HipDeviceProps_.sharedMemPerBlock =
      ClDevice.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&Err);

  HipDeviceProps_.maxThreadsPerBlock =
      ClDevice.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&Err);

  std::vector<size_t> Wi = ClDevice.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

  HipDeviceProps_.maxThreadsDim[0] = Wi[0];
  HipDeviceProps_.maxThreadsDim[1] = Wi[1];
  HipDeviceProps_.maxThreadsDim[2] = Wi[2];

  // Maximum configured clock frequency of the device in MHz.
  HipDeviceProps_.clockRate =
      1000 * ClDevice.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

  HipDeviceProps_.multiProcessorCount =
      ClDevice.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  HipDeviceProps_.l2CacheSize =
      ClDevice.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();

  // not actually correct
  HipDeviceProps_.totalConstMem =
      ClDevice.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();

  // totally made up
  HipDeviceProps_.regsPerBlock = 64;

  HipDeviceProps_.warpSize = CHIP_DEFAULT_WARP_SIZE;
  // Try to check that we support the default warp size.
  std::vector<size_t> Sg = ClDevice.getInfo<CL_DEVICE_SUB_GROUP_SIZES_INTEL>();
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

  Temp = ClDevice.getInfo<CL_DEVICE_EXTENSIONS>();
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

  auto Max1D2DWidth = ClDevice.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
  auto Max2DHeight = ClDevice.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();
  auto Max3DWidth = ClDevice.getInfo<CL_DEVICE_IMAGE3D_MAX_WIDTH>();
  auto Max3DHeight = ClDevice.getInfo<CL_DEVICE_IMAGE3D_MAX_HEIGHT>();
  auto Max3DDepth = ClDevice.getInfo<CL_DEVICE_IMAGE3D_MAX_DEPTH>();

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
    : CHIPEventOpenCL(ChipContext, nullptr, Flags) {}

uint64_t CHIPEventOpenCL::getFinishTime() {
  int Status = CL_SUCCESS;
  uint64_t Ret;
  Ret = ClEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>(&Status);

  if (Status != CL_SUCCESS) {
    logError("Failed to query event for profiling info.");
    return 0;
  }

  return Ret;
}

CHIPEventOpenCL *CHIPBackendOpenCL::createCHIPEvent(CHIPContext *ChipCtx,
                                                    CHIPEventFlags Flags,
                                                    bool UserEvent) {
  CHIPEventOpenCL *Event = new CHIPEventOpenCL((CHIPContextOpenCL *)ChipCtx,
                                               nullptr, Flags, UserEvent);

  return Event;
}

void CHIPEventOpenCL::recordStream(CHIPQueue *ChipQueue) {
  LOCK(Backend->EventsMtx); // trackImpl CHIPBackend::Events
  LOCK(EventMtx);           // changing this event's fields
  logTrace("CHIPEvent::recordStream()");

  CHIPEventOpenCL *Marker = (CHIPEventOpenCL *)ChipQueue->enqueueMarkerImpl();
  // see operator=() on cl::Event
  // should automatically release ClEvent if it already contains valid handle
  ClEvent = Marker->ClEvent;
  Msg = "recordStreamMarker";
  EventStatus_ = EVENT_STATUS_RECORDING;
  delete Marker;

  ChipQueue->updateLastEvent(this);
  // can't use this->track() because it calls locks
  if (!TrackCalled_) {
    Backend->Events.push_back(this);
    TrackCalled_ = true;
  }

  return;
}

size_t CHIPEventOpenCL::getCHIPRefc() {
  int Err = CL_SUCCESS;
  size_t RefC = ClEvent.getInfo<CL_EVENT_REFERENCE_COUNT>(&Err);
  if (Err != CL_SUCCESS) {
    logError("failed to get Reference count from OpenCL event");
    return 0;
  } else {
    return RefC;
  }
}

bool CHIPEventOpenCL::wait() {
  logTrace("CHIPEventOpenCL::wait()");

  if (EventStatus_ != EVENT_STATUS_RECORDING) {
    logWarn("Called wait() on an event that isn't active.");
    return false;
  }

  auto Status = ClEvent.wait();
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  return true;
}

bool CHIPEventOpenCL::updateFinishStatus(bool ThrowErrorIfNotReady) {
  logTrace("CHIPEventOpenCL::updateFinishStatus()");
  if (ThrowErrorIfNotReady && ClEvent.get() == nullptr)
    CHIPERR_LOG_AND_THROW("OpenCL has not been initialized cl_event is null",
                          hipErrorNotReady);

  int Status = CL_SUCCESS;
  int UpdatedStatus =
      ClEvent.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>(&Status);
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

// CHIPModuleOpenCL
//*************************************************************************

CHIPModuleOpenCL::CHIPModuleOpenCL(const SPVModule &SrcMod)
    : CHIPModule(SrcMod) {}

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
  auto Program = cl::Program(ChipCtxOcl->get(), BinaryVec, false, &Err);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorInitializationError);

  //   for (CHIPDevice *chip_dev : chip_devices) {
  std::string Name = ChipDevOcl->getName();
  Err = Program.build(Backend->getJitFlags().c_str());
  auto ErrBuild = Err;

  std::string Log =
      Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ChipDevOcl->get(), &Err);
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
  for (size_t KernelIdx = 0; KernelIdx < Kernels.size(); KernelIdx++) {
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


/// Clones the instance but with separate cl_kernel handle.
CHIPKernelOpenCL *CHIPKernelOpenCL::clone() {
  cl_int Err;
  // NOTE: clCloneKernel is not used here due to its experience on
  // Intel (GPU) OpenCL which crashed if clSetKernelArgSVMPointer() was
  // called on the original cl_kernel.
  auto Kernel = cl::Kernel(Module->get(), Name_.c_str(), &Err);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd);
  return new CHIPKernelOpenCL(Kernel, Device, Name_, getFuncInfo(), Module);
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
      OclKernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(Device->get());
  StaticLocalSize_ =
      OclKernel_.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(Device->get());
  MaxDynamicLocalSize_ =
      (size_t)Device->getAttr(hipDeviceAttributeMaxSharedMemoryPerBlock) -
      StaticLocalSize_;
  PrivateSize_ =
      OclKernel_.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(Device->get());

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

CHIPContextOpenCL::CHIPContextOpenCL(cl::Context CtxIn, cl::Device Dev,
                                     cl::Platform Plat) {

  logTrace("CHIPContextOpenCL Initialized via OpenCL Context pointer.");
  std::string DevExts = Dev.getInfo<CL_DEVICE_EXTENSIONS>();
  std::memset(&Exts, 0, sizeof(Exts));
  SupportsCommandBuffers =
      DevExts.find("cl_khr_command_buffer") != std::string::npos;
  if (SupportsCommandBuffers) {
    logDebug("Device supports cl_khr_command_buffer");
    Exts.clCreateCommandBufferKHR =
        (clCreateCommandBufferKHR_fn)::clGetExtensionFunctionAddressForPlatform(
            Plat(), "clCreateCommandBufferKHR");
    Exts.clCommandCopyBufferKHR =
        (clCommandCopyBufferKHR_fn)::clGetExtensionFunctionAddressForPlatform(
            Plat(), "clCommandCopyBufferKHR");
    Exts.clCommandCopyBufferRectKHR = (clCommandCopyBufferRectKHR_fn)::
        clGetExtensionFunctionAddressForPlatform(Plat(),
                                                 "clCommandCopyBufferRectKHR");
    Exts.clCommandFillBufferKHR =
        (clCommandFillBufferKHR_fn)::clGetExtensionFunctionAddressForPlatform(
            Plat(), "clCommandFillBufferKHR");
    Exts.clCommandNDRangeKernelKHR = (clCommandNDRangeKernelKHR_fn)::
        clGetExtensionFunctionAddressForPlatform(Plat(),
                                                 "clCommandNDRangeKernelKHR");
    Exts.clCommandBarrierWithWaitListKHR =
        (clCommandBarrierWithWaitListKHR_fn)::
            clGetExtensionFunctionAddressForPlatform(
                Plat(), "clCommandBarrierWithWaitListKHR");
    Exts.clFinalizeCommandBufferKHR = (clFinalizeCommandBufferKHR_fn)::
        clGetExtensionFunctionAddressForPlatform(Plat(),
                                                 "clFinalizeCommandBufferKHR");
    Exts.clEnqueueCommandBufferKHR = (clEnqueueCommandBufferKHR_fn)::
        clGetExtensionFunctionAddressForPlatform(Plat(),
                                                 "clEnqueueCommandBufferKHR");
    Exts.clReleaseCommandBufferKHR = (clReleaseCommandBufferKHR_fn)::
        clGetExtensionFunctionAddressForPlatform(Plat(),
                                                 "clReleaseCommandBufferKHR");
    Exts.clGetCommandBufferInfoKHR = (clGetCommandBufferInfoKHR_fn)::
        clGetExtensionFunctionAddressForPlatform(Plat(),
                                                 "clGetCommandBufferInfoKHR");
  }
#ifdef cl_pocl_command_buffer_svm
  SupportsCommandBuffersSVM =
      DevExts.find("cl_pocl_command_buffer_svm") != std::string::npos;
  if (SupportsCommandBuffersSVM) {
    logDebug("Device supports cl_pocl_command_buffer_svm");
    Exts.clCommandSVMMemcpyPOCL =
        (clCommandSVMMemcpyPOCL_fn)::clGetExtensionFunctionAddressForPlatform(
            Plat(), "clCommandSVMMemcpyPOCL");
    Exts.clCommandSVMMemcpyRectPOCL = (clCommandSVMMemcpyRectPOCL_fn)::
        clGetExtensionFunctionAddressForPlatform(Plat(),
                                                 "clCommandSVMMemcpyRectPOCL");
    Exts.clCommandSVMMemfillPOCL =
        (clCommandSVMMemfillPOCL_fn)::clGetExtensionFunctionAddressForPlatform(
            Plat(), "clCommandSVMMemfillPOCL");
    Exts.clCommandSVMMemfillRectPOCL = (clCommandSVMMemfillRectPOCL_fn)::
        clGetExtensionFunctionAddressForPlatform(Plat(),
                                                 "clCommandSVMMemfillRectPOCL");
  }
#endif
#ifdef cl_pocl_command_buffer_host_exec
  SupportsCommandBuffersHost =
      DevExts.find("cl_pocl_command_buffer_host_exec") != std::string::npos;
  if (SupportsCommandBuffersHost) {
    logDebug("Device supports cl_pocl_command_buffer_host_exec");
    Exts.clCommandHostFuncPOCL =
        (clCommandHostFuncPOCL_fn)::clGetExtensionFunctionAddressForPlatform(
            Plat(), "clCommandHostFuncPOCL");
    Exts.clCommandWaitForEventPOCL = (clCommandWaitForEventPOCL_fn)::
        clGetExtensionFunctionAddressForPlatform(Plat(),
                                                 "clCommandWaitForEventPOCL");
    Exts.clCommandSignalEventPOCL =
        (clCommandSignalEventPOCL_fn)::clGetExtensionFunctionAddressForPlatform(
            Plat(), "clCommandSignalEventPOCL");
  }
#endif

  ClContext = CtxIn;
  SvmMemory.init(CtxIn);
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
    static_cast<cl::UserEvent &>(Cbo->CallbackFinishEvent->get()).setStatus(CL_COMPLETE);
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
  // TODO why does this code use blocking = true ??
  if (Type == CHIPQueue::MEM_MAP_TYPE::HOST_READ) {
    logDebug("CHIPQueueOpenCL::MemMap HOST_READ");
    Status = ClQueue.enqueueMapSVM(AllocInfo->HostPtr, CL_TRUE, CL_MAP_READ,
                                   AllocInfo->Size);
  } else if (Type == CHIPQueue::MEM_MAP_TYPE::HOST_WRITE) {
    logDebug("CHIPQueueOpenCL::MemMap HOST_WRITE");
    Status = ClQueue.enqueueMapSVM(AllocInfo->HostPtr, CL_TRUE, CL_MAP_WRITE,
                                   AllocInfo->Size);
  } else if (Type == CHIPQueue::MEM_MAP_TYPE::HOST_READ_WRITE) {
    logDebug("CHIPQueueOpenCL::MemMap HOST_READ_WRITE");
    Status = ClQueue.enqueueMapSVM(AllocInfo->HostPtr, CL_TRUE,
                                   CL_MAP_READ | CL_MAP_WRITE,
                                   AllocInfo->Size);
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

  auto Status = ClQueue.enqueueUnmapSVM(AllocInfo->HostPtr);
  assert(Status == CL_SUCCESS);
}


void CHIPQueueOpenCL::addCallback(hipStreamCallback_t Callback,
                                  void *UserData) {
  logTrace("CHIPQueueOpenCL::addCallback()");

  cl::Context &ClContext_ = ((CHIPContextOpenCL *)ChipContext_)->get();
  cl_int Err;

  CHIPEventOpenCL *HoldBackEvent =
      (CHIPEventOpenCL *)Backend->createCHIPEvent(ChipContext_);
  cl::UserEvent HoldBackClEvent(ClContext_, &Err);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd);
  HoldBackEvent->reset(HoldBackClEvent());

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
  cl::UserEvent CallbackClEvent(ClContext_, &Err);
  CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd);
  CallbackEvent->reset(CallbackClEvent());

  // Make the succeeding commands wait for the user event which will be
  // set CL_COMPLETE by the callback trampoline function pfn_notify after
  // finishing the user CB's execution.

  HipStreamCallbackData *Cb = new HipStreamCallbackData{
      this, hipSuccess, UserData, Callback, CallbackEvent};

  std::vector<CHIPEvent *> WaitForEventsCBB{CallbackEvent};
  auto CallbackCompleted = enqueueBarrier(&WaitForEventsCBB);

  // We know that the callback won't be yet launched since it's depending
  // on the barrier which waits for the user event.
  auto Status = HoldbackBarrierCompletedEv->get().setCallback(CL_COMPLETE, pfn_notify, Cb);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);

  updateLastEvent(CallbackCompleted);
  ClQueue.flush();

  // Now the CB can start executing in the background:
  HoldBackClEvent.setStatus(CL_COMPLETE);

  return;
};

CHIPEvent *CHIPQueueOpenCL::enqueueMarkerImpl() {
  CHIPEventOpenCL *MarkerEvent =
      (CHIPEventOpenCL *)Backend->createCHIPEvent(ChipContext_);
  cl::Event RetEv;
  auto Status = ClQueue.enqueueMarkerWithWaitList(nullptr, &RetEv);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  MarkerEvent->reset(std::move(RetEv));
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

  const cl::NDRange GlobalOffset{0, 0, 0};
  const cl::NDRange Global{GridDim.x * BlockDim.x, GridDim.y * BlockDim.y,
                           GridDim.z * BlockDim.z};
  const cl::NDRange Local{BlockDim.x, BlockDim.y, BlockDim.z};

  logTrace("Launch GLOBAL: {} {} {}", Global[0], Global[1], Global[2]);

  logTrace("Launch LOCAL: {} {} {}", Local[0], Local[1], Local[2]);
#ifdef DUBIOUS_LOCKS
  LOCK(Backend->DubiousLockOpenCL);
#endif

  auto SvmAllocationsToKeepAlive =
      annotateSvmPointers(*OclContext, Kernel->get().get());

  cl::Event RetEv;
  auto Status = ClQueue.enqueueNDRangeKernel(Kernel->get(), GlobalOffset,
                                             Global, Local, nullptr, &RetEv);

  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  LaunchEvent->reset(std::move(RetEv));

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
    Status = LaunchEvent->get().setCallback(CL_COMPLETE, kernelEventCallback,
                                            CBData);

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
    ClQueue = cl::CommandQueue(Queue);
  else {
    cl::Context &ClContext_ = ((CHIPContextOpenCL *)ChipContext_)->get();
    cl::Device &ClDevice_ = ((CHIPDeviceOpenCL *)ChipDevice_)->get();
    cl_int Status;
    // Adding priority breaks correctness?
    // cl_queue_properties QueueProperties[] = {
    //     CL_QUEUE_PRIORITY_KHR, PrioritySelection, CL_QUEUE_PROPERTIES,
    //     CL_QUEUE_PROFILING_ENABLE, 0};
    cl_queue_properties QueueProperties[] = {CL_QUEUE_PROPERTIES,
                                             CL_QUEUE_PROFILING_ENABLE, 0};

    const cl_command_queue Q = clCreateCommandQueueWithProperties(
        ClContext_.get(), ClDevice_.get(), QueueProperties, &Status);
    ClQueue = cl::CommandQueue(Q);

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
  cl::Event RetEv;
  if (Dst == Src) {
    // Although ROCm API ref says that Dst and Src should not overlap,
    // HIP seems to handle Dst == Src as a special (no-operation) case.
    // This is seen in the test unit/memory/hipMemcpyAllApiNegative.

    // Intel GPU OpenCL driver seems to do also so for clEnqueueSVMMemcpy, which
    // makes/ it pass, but Intel CPU OpenCL returns CL_​MEM_​COPY_​OVERLAP
    // like it should. To unify the behavior, let's convert the special case to
    // a maker here, so we can return an event.
    auto Status = ClQueue.enqueueMarkerWithWaitList(nullptr, &RetEv);
    CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  } else {
#ifdef DUBIOUS_LOCKS
    LOCK(Backend->DubiousLockOpenCL)
#endif
    cl_event E = nullptr;
    auto Status = ::clEnqueueSVMMemcpy(ClQueue.get(), CL_FALSE, Dst, Src, Size,
                                       0, nullptr, &E);
    CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorRuntimeMemory);
    RetEv = E;
  }
  Event->reset(std::move(RetEv));
  return Event;
}

void CHIPQueueOpenCL::finish() {
#ifdef DUBIOUS_LOCKS
  LOCK(Backend->DubiousLockOpenCL)
#endif
  auto Status = ClQueue.finish();
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
}

CHIPEvent *CHIPQueueOpenCL::memFillAsyncImpl(void *Dst, size_t Size,
                                             const void *Pattern,
                                             size_t PatternSize) {
  CHIPEventOpenCL *Event =
      (CHIPEventOpenCL *)Backend->createCHIPEvent(ChipContext_);
  logTrace("clSVMmemfill {} / {} B\n", Dst, Size);
  cl_event Ev = nullptr;
  auto Status = ::clEnqueueSVMMemFill(ClQueue.get(), Dst, Pattern, PatternSize,
                                      Size, 0, nullptr, &Ev);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorRuntimeMemory);
  cl::Event RetEv(Ev);
  Event->reset(std::move(RetEv));
  return Event;
};

CHIPEvent *CHIPQueueOpenCL::memCopy2DAsyncImpl(void *Dst, size_t Dpitch,
                                               const void *Src, size_t Spitch,
                                               size_t Width, size_t Height) {
  // TODO
  UNIMPLEMENTED(nullptr);
};

CHIPEvent *CHIPQueueOpenCL::memCopy3DAsyncImpl(void *Dst, size_t Dpitch,
                                               size_t Dspitch, const void *Src,
                                               size_t Spitch, size_t Sspitch,
                                               size_t Width, size_t Height,
                                               size_t Depth) {
  // TODO
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
  NativeInfo[3] = (uintptr_t)ClQueue.get();

  // Get context handler
  cl::Context &Ctx = ((CHIPContextOpenCL *)ChipContext_)->get();
  NativeInfo[2] = (uintptr_t)Ctx.get();

  // Get device handler
  cl::Device &Dev = ((CHIPDeviceOpenCL *)ChipDevice_)->get();
  NativeInfo[1] = (uintptr_t)Dev.get();

  // Get platform handler
  cl_platform_id Plat = Dev.getInfo<CL_DEVICE_PLATFORM>();
  NativeInfo[0] = (uintptr_t)Plat;
  return hipSuccess;
}

CHIPEvent *CHIPQueueOpenCL::memPrefetchImpl(const void *Ptr, size_t Count) {
  // TODO
  UNIMPLEMENTED(nullptr);
}

CHIPEvent *
CHIPQueueOpenCL::enqueueBarrierImpl(std::vector<CHIPEvent *> *EventsToWaitFor) {
#ifdef DUBIOUS_LOCKS
  LOCK(Backend->DubiousLockOpenCL)
#endif
  CHIPEventOpenCL *Event =
      (CHIPEventOpenCL *)Backend->createCHIPEvent(this->ChipContext_);

  int Status;
  cl::Event RetEv;

  if (EventsToWaitFor && EventsToWaitFor->size() > 0) {
    std::vector<cl::Event> Events = {};
    for (auto E : *EventsToWaitFor) {
      auto Ee = (CHIPEventOpenCL *)E;
      Events.push_back(Ee->get());
    }
    Status = ClQueue.enqueueBarrierWithWaitList(&Events, &RetEv);
  } else {
    Status = ClQueue.enqueueBarrierWithWaitList(nullptr, &RetEv);
  }

  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  Event->reset(std::move(RetEv));
  return Event;
}

/*****************************************************************************/

CHIPGraphNative *CHIPQueueOpenCL::createNativeGraph() {
  // should not raise an error if we fail to create a graph,
  // because there is a fallback solution
  CHIPContextOpenCL *Ctx = (CHIPContextOpenCL *)ChipContext_;
  if (!Ctx->supportsCommandBuffers())
    return nullptr;

  cl_command_queue CQ = ClQueue.get();
  int err = CL_SUCCESS;
  cl_command_buffer_khr Res =
      Ctx->exts()->clCreateCommandBufferKHR(1, &CQ, 0, &err);
  if (Res == nullptr || err != CL_SUCCESS) {
    logError("clCreateCommandBufferKHR FAILED with status {}",
             resultToString(err));
    return nullptr;
  }

  return new CHIPGraphNativeOpenCL(Res, CQ, Ctx->exts());
}

CHIPEvent *CHIPQueueOpenCL::enqueueNativeGraph(CHIPGraphNative *NativeGraph) {
  CHIPEventOpenCL *Event =
      (CHIPEventOpenCL *)Backend->createCHIPEvent(ChipContext_);

  CHIPContextOpenCL *Ctx = (CHIPContextOpenCL *)ChipContext_;
  CHIPGraphNativeOpenCL *G = (CHIPGraphNativeOpenCL *)NativeGraph;
  if (!Ctx->supportsCommandBuffers())
    return nullptr;
  if (NativeGraph == nullptr)
    return nullptr;
  cl_command_queue CQ = ClQueue.get();
  cl_event TmpEv = nullptr;
  int Status = Ctx->exts()->clEnqueueCommandBufferKHR(1, &CQ, G->get(), 0,
                                                      nullptr, &TmpEv);
  CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
  Event->reset(TmpEv);
  return Event;
}

void CHIPQueueOpenCL::destroyNativeGraph(CHIPGraphNative *NativeGraph) {
  if (NativeGraph == nullptr)
    return;
  CHIPGraphNativeOpenCL *G = (CHIPGraphNativeOpenCL *)NativeGraph;
  delete G;
}

bool CHIPGraphNativeOpenCL::addNode(CHIPGraphNode *NewNode) {
  cl_sync_point_khr NewSyncPoint = -1;

  // map the dependent CHIPGraphNodes to OpenCL syncpoints
  const std::vector<CHIPGraphNode *> &Dependencies = NewNode->getDependencies();
  std::vector<cl_sync_point_khr> SyncPointDeps;
  for (auto Node : Dependencies) {
    auto Iter = SyncPointMap.find(Node);
    if (Iter == SyncPointMap.end()) {
      logError("Can't find SyncPoint for Node");
      return false;
    }
    SyncPointDeps.push_back(Iter->second);
  }

  hipGraphNodeType NodeType = NewNode->getType();
  bool Res;
  switch (NodeType) {
  case hipGraphNodeTypeKernel:
    Res = addKernelNode((CHIPGraphNodeKernel *)NewNode, SyncPointDeps,
                        &NewSyncPoint);
    break;
  case hipGraphNodeTypeEmpty:
    assert(0 && "Empty node should be removed earlier");

#ifdef cl_pocl_command_buffer_svm
  case hipGraphNodeTypeMemcpy:
    Res = addMemcpyNode((CHIPGraphNodeMemcpy *)NewNode, SyncPointDeps,
                        &NewSyncPoint);
    break;
  case hipGraphNodeTypeMemset:
    Res = addMemsetNode((CHIPGraphNodeMemset *)NewNode, SyncPointDeps,
                        &NewSyncPoint);
    break;
  case hipGraphNodeTypeMemcpyFromSymbol:
    Res = addMemcpyNode((CHIPGraphNodeMemcpyFromSymbol *)NewNode, SyncPointDeps,
                        &NewSyncPoint);
    break;
  case hipGraphNodeTypeMemcpyToSymbol:
    Res = addMemcpyNode((CHIPGraphNodeMemcpyToSymbol *)NewNode, SyncPointDeps,
                        &NewSyncPoint);
    break;
#endif

#ifdef cl_pocl_command_buffer_host_exec
  case hipGraphNodeTypeWaitEvent:
    Res = addEventWaitNode((CHIPGraphNodeWaitEvent *)NewNode, SyncPointDeps,
                           &NewSyncPoint);
    break;
  case hipGraphNodeTypeEventRecord:
    Res = addEventRecordNode((CHIPGraphNodeEventRecord *)NewNode, SyncPointDeps,
                             &NewSyncPoint);
    break;
  case hipGraphNodeTypeHost:
    Res =
        addHostNode((CHIPGraphNodeHost *)NewNode, SyncPointDeps, &NewSyncPoint);
    break;
#endif

  default:
    Res = false;
  }
  if (!Res)
    return false;

  SyncPointMap.insert(std::make_pair(NewNode, NewSyncPoint));
  return true;
}

bool CHIPGraphNativeOpenCL::finalize() {
  int Status = Exts->clFinalizeCommandBufferKHR(Handle);
  if (Status == CL_SUCCESS) {
    Finalized = true;
    return true;
  } else {
    logError("clFinalizeCommandBufferKHR FAILED with status {}",
             resultToString(Status));
    return false;
  }
}

CHIPGraphNativeOpenCL::~CHIPGraphNativeOpenCL() {
  if (Handle == nullptr)
    return;
  int Err = Exts->clReleaseCommandBufferKHR(Handle);
  logError("clReleaseCommandBufferKHR FAILED with status {}",
           resultToString(Err));
  assert(Err == CL_SUCCESS);
}

// TODO finish
bool CHIPGraphNativeOpenCL::addKernelNode(
    CHIPGraphNodeKernel *Node, std::vector<cl_sync_point_khr> &SyncPointDeps,
    cl_sync_point_khr *SyncPoint) {

  int Status;
  // possibly use: CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR
  cl_ndrange_kernel_command_properties_khr Properties[] = {0, 0};

  // TODO: we should add what CHIPQueue::launch does with Registered (Global)
  // Vars
  // TODO also look at SpillBuffer handling in:
  // CHIPEvent *CHIPQueueOpenCL::launchImpl(CHIPExecItem *ExecItem) {

  // setup the kernel arguments before calling clCommandNDRange
  Node->setupKernelArgs();

  CHIPKernel *K = Node->getKernel();
  CHIPKernelOpenCL *CLK = static_cast<CHIPKernelOpenCL *>(K);

  hipKernelNodeParams Params = Node->getParams();
  size_t LWSize[3] = {Params.blockDim.x, Params.blockDim.y, Params.blockDim.z};
  size_t GWSize[3] = {Params.blockDim.x * Params.gridDim.x,
                      Params.blockDim.y * Params.gridDim.y,
                      Params.blockDim.z * Params.gridDim.z};
  uint WorkDim = 3;

  assert(Exts->clCommandNDRangeKernelKHR);
  Status = Exts->clCommandNDRangeKernelKHR(
      Handle, nullptr, Properties,
      CLK->get().get(), // cl_kernel
      WorkDim,          // cl_uint work_dim
      nullptr,          // const size_t* global_work_offset,
      GWSize,           // const size_t* global_work_size,
      LWSize,           // const size_t* local_work_size,
      SyncPointDeps.size(), SyncPointDeps.data(), SyncPoint, nullptr);
  return Status == CL_SUCCESS;
}

#ifdef cl_pocl_command_buffer_svm

// TODO finish Arrays
bool CHIPGraphNativeOpenCL::addMemcpyNode(
    CHIPGraphNodeMemcpy *Node, std::vector<cl_sync_point_khr> &SyncPointDeps,
    cl_sync_point_khr *SyncPoint) {
  int Status;
  void *Dst;
  const void *Src;
  size_t Size;
  hipMemcpyKind Kind;
  hipMemcpy3DParms Params;

  // Although ROCm API ref says that Dst and Src should not overlap,
  // HIP seems to handle Dst == Src as a special (no-operation) case.
  // This is seen in the test unit/memory/hipMemcpyAllApiNegative.
  // Intel GPU OpenCL driver seems to do also so for clEnqueueSVMMemcpy, which
  // makes/ it pass, but Intel CPU OpenCL returns CL_​MEM_​COPY_​OVERLAP
  // like it should. To unify the behavior, let's convert the special case to
  // a marker here, so we can return an event.

  Node->getParams(Dst, Src, Size, Kind);
  Params = Node->getParams();
  if (Dst == nullptr || Src == nullptr) {
    if (!Exts->clCommandSVMMemcpyRectPOCL)
      return false;
    // 3D copy
    // TODO handle arrays
    assert(Params.dstArray == nullptr && "Arrays not supported yet");
    assert(Params.srcArray == nullptr && "Arrays not supported yet");

    /*
     * The struct passed to cudaMemcpy3D() must specify one of srcArray or
     * srcPtr and one of dstArray or dstPtr. Passing more than one non-zero
     * source or destination will cause cudaMemcpy3D() to return an error. The
     * srcPos and dstPos fields are optional offsets into the source and
     * destination objects and are defined in units of each object's elements.
     * The element for a host or device pointer is assumed to be unsigned char.
     * The extent field defines the dimensions of the transferred area in
     * elements. If a CUDA array is participating in the copy, the extent is
     * defined in terms of that array's elements. If no CUDA array is
     * participating in the copy then the extents are defined in elements of
     * unsigned char.
     */

    // TODO: HANDLE FOR ARRAYS:
    // The srcPos and dstPos fields are optional offsets into the source &
    // destination objects and are defined in units of each object's elements
    // ... The element for a host or device pointer is assumed to be unsigned
    // char.
    size_t src_origin[3] = {Params.srcPos.x, Params.srcPos.y, Params.srcPos.z};
    size_t dst_origin[3] = {Params.dstPos.x, Params.dstPos.y, Params.dstPos.z};
    // If no CUDA array is participating in the copy then the extents
    // are defined in elements of unsigned char.
    size_t region[3] = {Params.extent.width, Params.extent.height,
                        Params.extent.depth};

    // TODO this might be wrong.
    size_t src_row_pitch = Params.srcPtr.pitch;
    size_t src_slice_pitch = src_row_pitch * Params.srcPtr.ysize;
    size_t dst_row_pitch = Params.dstPtr.pitch;
    size_t dst_slice_pitch = dst_row_pitch * Params.dstPtr.ysize;

    Status = Exts->clCommandSVMMemcpyRectPOCL(
        Handle, nullptr, Dst, Src, dst_origin, src_origin, region,
        dst_row_pitch, dst_slice_pitch, src_row_pitch, src_slice_pitch,
        SyncPointDeps.size(), SyncPointDeps.data(), SyncPoint, nullptr);
  } else {
    // 1D copy
    if (!Exts->clCommandSVMMemcpyPOCL)
      return false;
    if (Dst == Src) {
      Status = Exts->clCommandBarrierWithWaitListKHR(
          Handle, nullptr, SyncPointDeps.size(), SyncPointDeps.data(),
          SyncPoint, nullptr);
      CHIPERR_CHECK_LOG_AND_THROW(Status, CL_SUCCESS, hipErrorTbd);
    } else {
      Status = Exts->clCommandSVMMemcpyPOCL(
          Handle, nullptr, Dst, Src, Size, SyncPointDeps.size(),
          SyncPointDeps.data(), SyncPoint, nullptr);
    }
  }

  return Status == CL_SUCCESS;
}

// DONE
bool CHIPGraphNativeOpenCL::addMemcpyNode(
    CHIPGraphNodeMemcpyFromSymbol *Node,
    std::vector<cl_sync_point_khr> &SyncPointDeps,
    cl_sync_point_khr *SyncPoint) {

  if (!Exts->clCommandSVMMemcpyPOCL)
    return false;

  void *Dst = nullptr;
  void *Src = nullptr;
  const void *Symbol;
  size_t SizeBytes;
  size_t Offset;
  hipMemcpyKind Kind;
  Node->getParams(Dst, Symbol, SizeBytes, Offset, Kind);

  hipError_t Err = hipGetSymbolAddress(&Src, Symbol);
  if (Err != HIP_SUCCESS)
    return false;

  int Status = Exts->clCommandSVMMemcpyPOCL(
      Handle, nullptr, Dst, (const char *)Src + Offset, SizeBytes,
      SyncPointDeps.size(), SyncPointDeps.data(), SyncPoint, nullptr);

  return Status == CL_SUCCESS;
}

// DONE
bool CHIPGraphNativeOpenCL::addMemcpyNode(
    CHIPGraphNodeMemcpyToSymbol *Node,
    std::vector<cl_sync_point_khr> &SyncPointDeps,
    cl_sync_point_khr *SyncPoint) {
  if (!Exts->clCommandSVMMemcpyPOCL)
    return false;

  void *Dst = nullptr;
  void *Src = nullptr;
  const void *Symbol;
  size_t SizeBytes;
  size_t Offset;
  hipMemcpyKind Kind;
  Node->getParams(Src, Symbol, SizeBytes, Offset, Kind);

  hipError_t Err = hipGetSymbolAddress(&Dst, Symbol);
  if (Err != HIP_SUCCESS)
    return false;

  int Status = Exts->clCommandSVMMemcpyPOCL(
      Handle, nullptr, (char *)Dst + Offset, Src, SizeBytes,
      SyncPointDeps.size(), SyncPointDeps.data(), SyncPoint, nullptr);
  return Status == CL_SUCCESS;
}

// DONE
bool CHIPGraphNativeOpenCL::addMemsetNode(
    CHIPGraphNodeMemset *Node, std::vector<cl_sync_point_khr> &SyncPointDeps,
    cl_sync_point_khr *SyncPoint) {
  if (!Exts->clCommandSVMMemfillRectPOCL)
    return false;

  hipMemsetParams Params = Node->getParams();

  int Status;
  size_t Region[3] = {Params.width, Params.height, 1};
  Status = Exts->clCommandSVMMemfillRectPOCL(
      Handle, nullptr, Params.dst,
      nullptr,      // origin
      Region,       // region
      Params.pitch, // row pitch
      0,            // slice pitch
      (const void *)&Params.value, Params.elementSize, SyncPointDeps.size(),
      SyncPointDeps.data(), SyncPoint, nullptr);
  return Status == CL_SUCCESS;
}
#endif

#ifdef cl_pocl_command_buffer_host_exec

// DONE
bool CHIPGraphNativeOpenCL::addHostNode(
    CHIPGraphNodeHost *Node, std::vector<cl_sync_point_khr> &SyncPointDeps,
    cl_sync_point_khr *SyncPoint) {
  if (!Exts->clCommandHostFuncPOCL)
    return false;

  hipHostNodeParams Params = Node->getParams();

  int Status;
  Status = Exts->clCommandHostFuncPOCL(
      Handle, nullptr, Params.fn, Params.userData, SyncPointDeps.size(),
      SyncPointDeps.data(), SyncPoint, nullptr);
  return Status == CL_SUCCESS;
}

// TODO output cl_event
bool CHIPGraphNativeOpenCL::addEventRecordNode(
    CHIPGraphNodeEventRecord *Node,
    std::vector<cl_sync_point_khr> &SyncPointDeps,
    cl_sync_point_khr *SyncPoint) {
  if (!Exts->clCommandSignalEventPOCL)
    return false;

  CHIPEvent *E = Node->getEvent();
  CHIPEventOpenCL *CLE = static_cast<CHIPEventOpenCL *>(E);

  int Status;
  // TODO BROKEN
  Status = Exts->clCommandSignalEventPOCL(Handle, nullptr, nullptr, SyncPoint,
                                          nullptr);
  return Status == CL_SUCCESS;
}

// DONE
bool CHIPGraphNativeOpenCL::addEventWaitNode(
    CHIPGraphNodeWaitEvent *Node, std::vector<cl_sync_point_khr> &SyncPointDeps,
    cl_sync_point_khr *SyncPoint) {
  if (!Exts->clCommandWaitForEventPOCL)
    return false;

  CHIPEvent *E = Node->getEvent();
  CHIPEventOpenCL *CLE = static_cast<CHIPEventOpenCL *>(E);

  int Status;
  Status = Exts->clCommandWaitForEventPOCL(Handle, nullptr, CLE->get().get(),
                                           SyncPoint, nullptr);
  return Status == CL_SUCCESS;
}
#endif

// CHIPExecItemOpenCL
//*************************************************************************

void CHIPExecItemOpenCL::setupAllArgs() {
  if (!ArgsSetup) {
    ArgsSetup = true;
  } else {
    return;
  }
  CHIPKernelOpenCL *Kernel = (CHIPKernelOpenCL *)getKernel();
  cl::Kernel &K = Kernel->get();
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
      Err = K.setArg(Arg.Index, sizeof(cl_mem), &Image);
      CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                                  "clSetKernelArg failed for image argument.");
      break;
    }
    case SPVTypeKind::Sampler: {
      auto *TexObj =
          *reinterpret_cast<const CHIPTextureOpenCL *const *>(Arg.Data);
      cl_sampler Sampler = TexObj->getSampler();
      logTrace("set sampler arg {} for tex {}\n", Arg.Index, (void *)TexObj);
      K.setArg(Arg.Index, sizeof(cl_sampler), &Sampler);
      CHIPERR_CHECK_LOG_AND_THROW(
          Err, CL_SUCCESS, hipErrorTbd,
          "clSetKernelArg failed for sampler argument.");
      break;
    }
    case SPVTypeKind::POD: {
      logTrace("clSetKernelArg {} SIZE {} to {}\n", Arg.Index, Arg.Size,
               Arg.Data);
      Err = K.setArg(Arg.Index, Arg.Size, Arg.Data);
      CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                                  "clSetKernelArg failed");
      break;
    }
    case SPVTypeKind::Pointer: {
      CHIPASSERT(Arg.Size == sizeof(void *));
      if (Arg.isWorkgroupPtr()) {
        logTrace("setLocalMemSize to {}\n", SharedMem_);
        Err = K.setArg(Arg.Index, SharedMem_, nullptr);
      } else {
        const void *Ptr = *(const void **)Arg.Data;
        logTrace("clSetKernelArgSVMPointer {} SIZE {} to {} (value {})\n",
                 Arg.Index, Arg.Size, Arg.Data, Ptr);

        // Unlike clSetKernelArg() which takes address to the argument,
        // this function takes the argument value directly.
        Err = K.setArg(Arg.Index, Ptr);

        if (Err != CL_SUCCESS) {
          // ROCm seems to allow passing invalid pointers to kernels if they are
          // not derefenced (see test_device_adjacent_difference of rocPRIM).
          // If the setting of the arg fails, let's assume this might be such a
          // case and convert it to a null pointer.
          logWarn(
              "clSetKernelArgSVMPointer {} SIZE {} to {} (value {}) returned "
              "error, setting the arg to nullptr\n",
              Arg.Index, Arg.Size, Arg.Data, Ptr);
          Err = K.setArg(Arg.Index, nullptr);
        }
      }
      CHIPERR_CHECK_LOG_AND_THROW(Err, CL_SUCCESS, hipErrorTbd,
                                  "clSetKernelArgSVMPointer failed");
      break;
    }
    case SPVTypeKind::PODByRef: {
      void *SpillSlot = ArgSpillBuffer_->allocate(Arg);
      assert(SpillSlot);
      Err = K.setArg(Arg.Index, SpillSlot);
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
  for (size_t i = 0; i < Platforms.size(); i++) {
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

  cl::Device Device = SpirvDevices[SelectedDeviceIdx];
  logDebug("CHIP_DEVICE={} Selected OpenCL device {}", SelectedDeviceIdx,
           Device.getInfo<CL_DEVICE_NAME>());

  // Create context which has devices
  // Create queues that have devices each of which has an associated context
  // TODO Change this to spirv_enabled_devices
  CHIPContextOpenCL *ChipContext = new CHIPContextOpenCL(
      cl::Context(SpirvDevices), Device, SelectedPlatform);
  Backend->addContext(ChipContext);

  // TODO for now only a single device is supported.
  CHIPDeviceOpenCL *ChipDev = CHIPDeviceOpenCL::create(Device, ChipContext, 0);

  // Add device to context & backend
  ChipContext->setDevice(ChipDev);
  logTrace("OpenCL Context Initialized.");
};

void CHIPBackendOpenCL::initializeFromNative(const uintptr_t *NativeHandles,
                                             int NumHandles) {
  logTrace("CHIPBackendOpenCL InitializeNative");
  MinQueuePriority_ = CL_QUEUE_PRIORITY_MED_KHR;
  cl_platform_id PlatId = (cl_platform_id)NativeHandles[0];
  cl_device_id DevId = (cl_device_id)NativeHandles[1];
  cl_context CtxId = (cl_context)NativeHandles[2];

  // Platform can also be get from this: Dev.getInfo<CL_DEVICE_PLATFORM>()
  cl::Platform Plat(PlatId);
  cl::Device Dev(DevId);
  cl::Context Ctx(CtxId);
  CHIPContextOpenCL *ChipContext = new CHIPContextOpenCL(Ctx, Dev, Plat);
  addContext(ChipContext);

  CHIPDeviceOpenCL *ChipDev = CHIPDeviceOpenCL::create(Dev, ChipContext, 0);
  logTrace("CHIPDeviceOpenCL {}", Dev.getInfo<CL_DEVICE_NAME>());

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
  return (void *)E->get().get();
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
