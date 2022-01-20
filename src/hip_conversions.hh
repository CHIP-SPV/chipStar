/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <hip/driver_types.h>
#include <hip/texture_types.h>

inline cl_channel_type getCLChannelType(const hipArray_Format hipFormat,
                                        const hipTextureReadMode hipReadMode) {
  if (hipReadMode == hipReadModeElementType) {
    switch (hipFormat) {
    case HIP_AD_FORMAT_UNSIGNED_INT8:
      return CL_UNSIGNED_INT8;
    case HIP_AD_FORMAT_SIGNED_INT8:
      return CL_SIGNED_INT8;
    case HIP_AD_FORMAT_UNSIGNED_INT16:
      return CL_UNSIGNED_INT16;
    case HIP_AD_FORMAT_SIGNED_INT16:
      return CL_SIGNED_INT16;
    case HIP_AD_FORMAT_UNSIGNED_INT32:
      return CL_UNSIGNED_INT32;
    case HIP_AD_FORMAT_SIGNED_INT32:
      return CL_SIGNED_INT32;
    case HIP_AD_FORMAT_HALF:
      return CL_HALF_FLOAT;
    case HIP_AD_FORMAT_FLOAT:
      return CL_FLOAT;
    }
  } else if (hipReadMode == hipReadModeNormalizedFloat) {
    switch (hipFormat) {
    case HIP_AD_FORMAT_UNSIGNED_INT8:
      return CL_UNORM_INT8;
    case HIP_AD_FORMAT_SIGNED_INT8:
      return CL_SNORM_INT8;
    case HIP_AD_FORMAT_UNSIGNED_INT16:
      return CL_UNORM_INT16;
    case HIP_AD_FORMAT_SIGNED_INT16:
      return CL_SNORM_INT16;
    case HIP_AD_FORMAT_UNSIGNED_INT32:
      return CL_UNSIGNED_INT32;
    case HIP_AD_FORMAT_SIGNED_INT32:
      return CL_SIGNED_INT32;
    case HIP_AD_FORMAT_HALF:
      return CL_HALF_FLOAT;
    case HIP_AD_FORMAT_FLOAT:
      return CL_FLOAT;
    }
  }

  // ShouldNotReachHere();();

  return {};
}

inline cl_channel_order getCLChannelOrder(const unsigned int hipNumChannels,
                                          const int sRGB) {
  switch (hipNumChannels) {
  case 1:
    return CL_R;
  case 2:
    return CL_RG;
  case 4:
    return (sRGB == 1) ? CL_sRGBA : CL_RGBA;
  default:
    break;
  }

  // ShouldNotReachHere();();

  return {};
}

inline cl_mem_object_type getCLMemObjectType(const unsigned int hipWidth,
                                             const unsigned int hipHeight,
                                             const unsigned int hipDepth,
                                             const unsigned int flags) {
  if (flags == hipArrayDefault) {
    if ((hipWidth != 0) && (hipHeight == 0) && (hipDepth == 0)) {
      return CL_MEM_OBJECT_IMAGE1D;
    } else if ((hipWidth != 0) && (hipHeight != 0) && (hipDepth == 0)) {
      return CL_MEM_OBJECT_IMAGE2D;
    } else if ((hipWidth != 0) && (hipHeight != 0) && (hipDepth != 0)) {
      return CL_MEM_OBJECT_IMAGE3D;
    }
  } else if (flags == hipArrayLayered) {
    if ((hipWidth != 0) && (hipHeight == 0) && (hipDepth != 0)) {
      return CL_MEM_OBJECT_IMAGE1D_ARRAY;
    } else if ((hipWidth != 0) && (hipHeight != 0) && (hipDepth != 0)) {
      return CL_MEM_OBJECT_IMAGE2D_ARRAY;
    }
  }

  // ShouldNotReachHere();();

  return {};
}

inline cl_addressing_mode
getCLAddressingMode(const hipTextureAddressMode hipAddressMode) {
  switch (hipAddressMode) {
  case hipAddressModeWrap:
    return CL_ADDRESS_REPEAT;
  case hipAddressModeClamp:
    return CL_ADDRESS_CLAMP_TO_EDGE;
  case hipAddressModeMirror:
    return CL_ADDRESS_MIRRORED_REPEAT;
  case hipAddressModeBorder:
    return CL_ADDRESS_CLAMP;
  }

  // ShouldNotReachHere();();

  return {};
}

inline cl_filter_mode
getCLFilterMode(const hipTextureFilterMode hipFilterMode) {
  switch (hipFilterMode) {
  case hipFilterModePoint:
    return CL_FILTER_NEAREST;
  case hipFilterModeLinear:
    return CL_FILTER_LINEAR;
  }

  // ShouldNotReachHere();();

  return {};
}

inline cl_mem_object_type getCLMemObjectType(const hipResourceType hipResType) {
  switch (hipResType) {
  case hipResourceTypeLinear:
    return CL_MEM_OBJECT_IMAGE1D_BUFFER;
  case hipResourceTypePitch2D:
    return CL_MEM_OBJECT_IMAGE2D;
  default:
    break;
  }

  // ShouldNotReachHere();();

  return {};
}

inline size_t getElementSize(const hipArray_const_t array) {
  switch (array->Format) {
  case HIP_AD_FORMAT_UNSIGNED_INT8:
  case HIP_AD_FORMAT_SIGNED_INT8:
    return 1 * array->NumChannels;
  case HIP_AD_FORMAT_UNSIGNED_INT16:
  case HIP_AD_FORMAT_SIGNED_INT16:
  case HIP_AD_FORMAT_HALF:
    return 2 * array->NumChannels;
  case HIP_AD_FORMAT_UNSIGNED_INT32:
  case HIP_AD_FORMAT_SIGNED_INT32:
  case HIP_AD_FORMAT_FLOAT:
    return 4 * array->NumChannels;
  }

  // ShouldNotReachHere();();

  return {};
}

inline hipChannelFormatDesc getChannelFormatDesc(int NumChannels,
                                                 hipArray_Format ArrayFormat) {
  switch (ArrayFormat) {
  case HIP_AD_FORMAT_UNSIGNED_INT8:
    switch (NumChannels) {
    case 1:
      return {8, 0, 0, 0, hipChannelFormatKindUnsigned};
    case 2:
      return {8, 8, 0, 0, hipChannelFormatKindUnsigned};
    case 4:
      return {8, 8, 8, 8, hipChannelFormatKindUnsigned};
    }
  case HIP_AD_FORMAT_SIGNED_INT8:
    switch (NumChannels) {
    case 1:
      return {8, 0, 0, 0, hipChannelFormatKindSigned};
    case 2:
      return {8, 8, 0, 0, hipChannelFormatKindSigned};
    case 4:
      return {8, 8, 8, 8, hipChannelFormatKindSigned};
    }
  case HIP_AD_FORMAT_UNSIGNED_INT16:
    switch (NumChannels) {
    case 1:
      return {16, 0, 0, 0, hipChannelFormatKindUnsigned};
    case 2:
      return {16, 16, 0, 0, hipChannelFormatKindUnsigned};
    case 4:
      return {16, 16, 16, 16, hipChannelFormatKindUnsigned};
    }
  case HIP_AD_FORMAT_SIGNED_INT16:
    switch (NumChannels) {
    case 1:
      return {16, 0, 0, 0, hipChannelFormatKindSigned};
    case 2:
      return {16, 16, 0, 0, hipChannelFormatKindSigned};
    case 4:
      return {16, 16, 16, 16, hipChannelFormatKindSigned};
    }
  case HIP_AD_FORMAT_UNSIGNED_INT32:
    switch (NumChannels) {
    case 1:
      return {32, 0, 0, 0, hipChannelFormatKindUnsigned};
    case 2:
      return {32, 32, 0, 0, hipChannelFormatKindUnsigned};
    case 4:
      return {32, 32, 32, 32, hipChannelFormatKindUnsigned};
    }
  case HIP_AD_FORMAT_SIGNED_INT32:
    switch (NumChannels) {
    case 1:
      return {32, 0, 0, 0, hipChannelFormatKindSigned};
    case 2:
      return {32, 32, 0, 0, hipChannelFormatKindSigned};
    case 4:
      return {32, 32, 32, 32, hipChannelFormatKindSigned};
    }
  case HIP_AD_FORMAT_HALF:
    switch (NumChannels) {
    case 1:
      return {16, 0, 0, 0, hipChannelFormatKindFloat};
    case 2:
      return {16, 16, 0, 0, hipChannelFormatKindFloat};
    case 4:
      return {16, 16, 16, 16, hipChannelFormatKindFloat};
    }
  case HIP_AD_FORMAT_FLOAT:
    switch (NumChannels) {
    case 1:
      return {32, 0, 0, 0, hipChannelFormatKindFloat};
    case 2:
      return {32, 32, 0, 0, hipChannelFormatKindFloat};
    case 4:
      return {32, 32, 32, 32, hipChannelFormatKindFloat};
    }
  }

  // ShouldNotReachHere();();

  return {};
}

inline unsigned int getNumChannels(const hipChannelFormatDesc& Desc) {
  return ((Desc.x != 0) + (Desc.y != 0) + (Desc.z != 0) + (Desc.w != 0));
}

inline hipArray_Format getArrayFormat(const hipChannelFormatDesc& Desc) {
  switch (Desc.f) {
  case hipChannelFormatKindUnsigned:
    switch (Desc.x) {
    case 8:
      return HIP_AD_FORMAT_UNSIGNED_INT8;
    case 16:
      return HIP_AD_FORMAT_UNSIGNED_INT16;
    case 32:
      return HIP_AD_FORMAT_UNSIGNED_INT32;
    }
  case hipChannelFormatKindSigned:
    switch (Desc.x) {
    case 8:
      return HIP_AD_FORMAT_SIGNED_INT8;
    case 16:
      return HIP_AD_FORMAT_SIGNED_INT16;
    case 32:
      return HIP_AD_FORMAT_SIGNED_INT32;
    }
  case hipChannelFormatKindFloat:
    switch (Desc.x) {
    case 16:
      return HIP_AD_FORMAT_HALF;
    case 32:
      return HIP_AD_FORMAT_FLOAT;
    }
  default:
    break;
  }

  // ShouldNotReachHere();();

  return {};
}

inline int getNumChannels(const hipResourceViewFormat HipFormat) {
  switch (HipFormat) {
  case hipResViewFormatUnsignedChar1:
  case hipResViewFormatSignedChar1:
  case hipResViewFormatUnsignedShort1:
  case hipResViewFormatSignedShort1:
  case hipResViewFormatUnsignedInt1:
  case hipResViewFormatSignedInt1:
  case hipResViewFormatHalf1:
  case hipResViewFormatFloat1:
    return 1;
  case hipResViewFormatUnsignedChar2:
  case hipResViewFormatSignedChar2:
  case hipResViewFormatUnsignedShort2:
  case hipResViewFormatSignedShort2:
  case hipResViewFormatUnsignedInt2:
  case hipResViewFormatSignedInt2:
  case hipResViewFormatHalf2:
  case hipResViewFormatFloat2:
    return 2;
  case hipResViewFormatUnsignedChar4:
  case hipResViewFormatSignedChar4:
  case hipResViewFormatUnsignedShort4:
  case hipResViewFormatSignedShort4:
  case hipResViewFormatUnsignedInt4:
  case hipResViewFormatSignedInt4:
  case hipResViewFormatHalf4:
  case hipResViewFormatFloat4:
    return 4;
  default:
    break;
  }

  // ShouldNotReachHere();();

  return {};
}

inline hipArray_Format getArrayFormat(const hipResourceViewFormat HipFormat) {
  switch (HipFormat) {
  case hipResViewFormatUnsignedChar1:
  case hipResViewFormatUnsignedChar2:
  case hipResViewFormatUnsignedChar4:
    return HIP_AD_FORMAT_UNSIGNED_INT8;
  case hipResViewFormatSignedChar1:
  case hipResViewFormatSignedChar2:
  case hipResViewFormatSignedChar4:
    return HIP_AD_FORMAT_SIGNED_INT8;
  case hipResViewFormatUnsignedShort1:
  case hipResViewFormatUnsignedShort2:
  case hipResViewFormatUnsignedShort4:
    return HIP_AD_FORMAT_UNSIGNED_INT16;
  case hipResViewFormatSignedShort1:
  case hipResViewFormatSignedShort2:
  case hipResViewFormatSignedShort4:
    return HIP_AD_FORMAT_SIGNED_INT16;
  case hipResViewFormatUnsignedInt1:
  case hipResViewFormatUnsignedInt2:
  case hipResViewFormatUnsignedInt4:
    return HIP_AD_FORMAT_UNSIGNED_INT32;
  case hipResViewFormatSignedInt1:
  case hipResViewFormatSignedInt2:
  case hipResViewFormatSignedInt4:
    return HIP_AD_FORMAT_SIGNED_INT32;
  case hipResViewFormatHalf1:
  case hipResViewFormatHalf2:
  case hipResViewFormatHalf4:
    return HIP_AD_FORMAT_HALF;
  case hipResViewFormatFloat1:
  case hipResViewFormatFloat2:
  case hipResViewFormatFloat4:
    return HIP_AD_FORMAT_FLOAT;
  default:
    break;
  }

  // ShouldNotReachHere();();

  return {};
}

inline hipResourceViewFormat
getResourceViewFormat(const hipChannelFormatDesc& Desc) {
  switch (Desc.f) {
  case hipChannelFormatKindUnsigned:
    switch (getNumChannels(Desc)) {
    case 1:
      switch (Desc.x) {
      case 8:
        return hipResViewFormatUnsignedChar1;
      case 16:
        return hipResViewFormatUnsignedShort1;
      case 32:
        return hipResViewFormatUnsignedInt1;
      }
    case 2:
      switch (Desc.x) {
      case 8:
        return hipResViewFormatUnsignedChar2;
      case 16:
        return hipResViewFormatUnsignedShort2;
      case 32:
        return hipResViewFormatUnsignedInt2;
      }
    case 4:
      switch (Desc.x) {
      case 8:
        return hipResViewFormatUnsignedChar4;
      case 16:
        return hipResViewFormatUnsignedShort4;
      case 32:
        return hipResViewFormatUnsignedInt4;
      }
    }
  case hipChannelFormatKindSigned:
    switch (getNumChannels(Desc)) {
    case 1:
      switch (Desc.x) {
      case 8:
        return hipResViewFormatSignedChar1;
      case 16:
        return hipResViewFormatSignedShort1;
      case 32:
        return hipResViewFormatSignedInt1;
      }
    case 2:
      switch (Desc.x) {
      case 8:
        return hipResViewFormatSignedChar2;
      case 16:
        return hipResViewFormatSignedShort2;
      case 32:
        return hipResViewFormatSignedInt2;
      }
    case 4:
      switch (Desc.x) {
      case 8:
        return hipResViewFormatSignedChar4;
      case 16:
        return hipResViewFormatSignedShort4;
      case 32:
        return hipResViewFormatSignedInt4;
      }
    }
  case hipChannelFormatKindFloat:
    switch (getNumChannels(Desc)) {
    case 1:
      switch (Desc.x) {
      case 16:
        return hipResViewFormatHalf1;
      case 32:
        return hipResViewFormatFloat1;
      }
    case 2:
      switch (Desc.x) {
      case 16:
        return hipResViewFormatHalf2;
      case 32:
        return hipResViewFormatFloat2;
      }
    case 4:
      switch (Desc.x) {
      case 16:
        return hipResViewFormatHalf4;
      case 32:
        return hipResViewFormatFloat4;
      }
    }
  default:
    break;
  }

  // ShouldNotReachHere();();

  return {};
}

inline hipTextureDesc getTextureDesc(const textureReference* TexRef) {
  hipTextureDesc TexDesc = {};
  std::memcpy(TexDesc.addressMode, TexRef->addressMode,
              sizeof(TexDesc.addressMode));
  TexDesc.filterMode = TexRef->filterMode;
  TexDesc.readMode = TexRef->readMode;
  TexDesc.sRGB = TexRef->sRGB;
  TexDesc.normalizedCoords = TexRef->normalized;
  TexDesc.maxAnisotropy = TexRef->maxAnisotropy;
  TexDesc.mipmapFilterMode = TexRef->mipmapFilterMode;
  TexDesc.mipmapLevelBias = TexRef->mipmapLevelBias;
  TexDesc.minMipmapLevelClamp = TexRef->minMipmapLevelClamp;
  TexDesc.maxMipmapLevelClamp = TexRef->maxMipmapLevelClamp;

  return TexDesc;
}

inline hipResourceViewDesc
getResourceViewDesc(hipArray_const_t Array,
                    const hipResourceViewFormat Format) {
  hipResourceViewDesc ResViewDesc = {};
  ResViewDesc.format = Format;
  ResViewDesc.width = Array->width;
  ResViewDesc.height = Array->height;
  ResViewDesc.depth = Array->depth;
  ResViewDesc.firstMipmapLevel = 0;
  ResViewDesc.lastMipmapLevel = 0;
  ResViewDesc.firstLayer = 0;
  ResViewDesc.lastLayer = 0; /* TODO add hipArray::numLayers */

  return ResViewDesc;
}

inline hipResourceViewDesc
getResourceViewDesc(hipMipmappedArray_const_t Array,
                    const hipResourceViewFormat Format) {
  hipResourceViewDesc ResViewDesc = {};
  ResViewDesc.format = Format;
  ResViewDesc.width = Array->width;
  ResViewDesc.height = Array->height;
  ResViewDesc.depth = Array->depth;
  ResViewDesc.firstMipmapLevel = 0;
  ResViewDesc.lastMipmapLevel =
      0; /* TODO add hipMipmappedArray::numMipLevels */
  ResViewDesc.firstLayer = 0;
  ResViewDesc.lastLayer = 0; /* TODO add hipArray::numLayers */

  return ResViewDesc;
}

inline std::pair<hipMemoryType, hipMemoryType>
getMemoryType(const hipMemcpyKind kind) {
  switch (kind) {
  case hipMemcpyHostToHost:
    return {hipMemoryTypeHost, hipMemoryTypeHost};
  case hipMemcpyHostToDevice:
    return {hipMemoryTypeHost, hipMemoryTypeDevice};
  case hipMemcpyDeviceToHost:
    return {hipMemoryTypeDevice, hipMemoryTypeHost};
  case hipMemcpyDeviceToDevice:
    return {hipMemoryTypeDevice, hipMemoryTypeDevice};
  case hipMemcpyDefault:
    return {hipMemoryTypeUnified, hipMemoryTypeUnified};
  }

  // ShouldNotReachHere();();

  return {};
}

inline HIP_MEMCPY3D getDrvMemcpy3DDesc(const hip_Memcpy2D& Desc2D) {
  HIP_MEMCPY3D Desc3D = {};

  Desc3D.srcXInBytes = Desc2D.srcXInBytes;
  Desc3D.srcY = Desc2D.srcY;
  Desc3D.srcZ = 0;
  Desc3D.srcLOD = 0;
  Desc3D.srcMemoryType = Desc2D.srcMemoryType;
  Desc3D.srcHost = Desc2D.srcHost;
  Desc3D.srcDevice = Desc2D.srcDevice;
  Desc3D.srcArray = Desc2D.srcArray;
  Desc3D.srcPitch = Desc2D.srcPitch;
  Desc3D.srcHeight = 0;

  Desc3D.dstXInBytes = Desc2D.dstXInBytes;
  Desc3D.dstY = Desc2D.dstY;
  Desc3D.dstZ = 0;
  Desc3D.dstLOD = 0;
  Desc3D.dstMemoryType = Desc2D.dstMemoryType;
  Desc3D.dstHost = Desc2D.dstHost;
  Desc3D.dstDevice = Desc2D.dstDevice;
  Desc3D.dstArray = Desc2D.dstArray;
  Desc3D.dstPitch = Desc2D.dstPitch;
  Desc3D.dstHeight = 0;

  Desc3D.WidthInBytes = Desc2D.WidthInBytes;
  Desc3D.Height = Desc2D.Height;
  Desc3D.Depth = 1;

  return Desc3D;
}

inline HIP_MEMCPY3D getDrvMemcpy3DDesc(const hipMemcpy3DParms& Desc) {
  HIP_MEMCPY3D DescDrv = {};

  DescDrv.WidthInBytes = Desc.extent.width;
  DescDrv.Height = Desc.extent.height;
  DescDrv.Depth = Desc.extent.depth;

  DescDrv.srcXInBytes = Desc.srcPos.x;
  DescDrv.srcY = Desc.srcPos.y;
  DescDrv.srcZ = Desc.srcPos.z;
  DescDrv.srcLOD = 0;

  DescDrv.dstXInBytes = Desc.dstPos.x;
  DescDrv.dstY = Desc.dstPos.y;
  DescDrv.dstZ = Desc.dstPos.z;
  DescDrv.dstLOD = 0;

  if (Desc.srcArray != nullptr) {
    DescDrv.srcMemoryType = hipMemoryTypeArray;
    DescDrv.srcArray = Desc.srcArray;
    // When reffering to array memory, hipPos::x is in elements.
    DescDrv.srcXInBytes *= getElementSize(Desc.srcArray);
  }

  if (Desc.srcPtr.ptr != nullptr) {
    DescDrv.srcMemoryType = std::get<0>(getMemoryType(Desc.kind));
    DescDrv.srcHost = Desc.srcPtr.ptr;
    DescDrv.srcDevice = Desc.srcPtr.ptr;
    DescDrv.srcPitch = Desc.srcPtr.pitch;
    DescDrv.srcHeight = Desc.srcPtr.ysize;
  }

  if (Desc.dstArray != nullptr) {
    DescDrv.dstMemoryType = hipMemoryTypeArray;
    DescDrv.dstArray = Desc.dstArray;
    // When reffering to array memory, hipPos::x is in elements.
    DescDrv.dstXInBytes *= getElementSize(Desc.dstArray);
  }

  if (Desc.dstPtr.ptr != nullptr) {
    DescDrv.dstMemoryType = std::get<1>(getMemoryType(Desc.kind));
    DescDrv.dstHost = Desc.dstPtr.ptr;
    DescDrv.dstDevice = Desc.dstPtr.ptr;
    DescDrv.dstPitch = Desc.dstPtr.pitch;
    DescDrv.dstHeight = Desc.dstPtr.ysize;
  }

  // If a HIP array is participating in the copy, the extent is defined in terms
  // of that array's elements.
  if ((Desc.srcArray != nullptr) && (Desc.dstArray == nullptr)) {
    DescDrv.WidthInBytes *= getElementSize(Desc.srcArray);
  } else if ((Desc.srcArray == nullptr) && (Desc.dstArray != nullptr)) {
    DescDrv.WidthInBytes *= getElementSize(Desc.dstArray);
  } else if ((Desc.srcArray != nullptr) && (Desc.dstArray != nullptr)) {
    DescDrv.WidthInBytes *= getElementSize(Desc.dstArray);
  }

  return DescDrv;
}

inline hipResourceType getResourceType(const HIPresourcetype ResType) {
  // These two enums should be isomorphic.
  return static_cast<hipResourceType>(ResType);
}

inline HIPresourcetype getResourceType(const hipResourceType ResType) {
  // These two enums should be isomorphic.
  return static_cast<HIPresourcetype>(ResType);
}

inline hipResourceDesc getResourceDesc(const HIP_RESOURCE_DESC& ResDesc) {
  hipResourceDesc Desc;

  Desc.resType = getResourceType(ResDesc.resType);
  switch (Desc.resType) {
  case hipResourceTypeArray:
    Desc.res.array.array = ResDesc.res.array.hArray;
    break;
  case hipResourceTypeMipmappedArray:
    Desc.res.mipmap.mipmap = ResDesc.res.mipmap.hMipmappedArray;
    break;
  case hipResourceTypeLinear:
    Desc.res.linear.devPtr = ResDesc.res.linear.devPtr;
    Desc.res.linear.desc = getChannelFormatDesc(ResDesc.res.linear.numChannels,
                                                ResDesc.res.linear.format);
    Desc.res.linear.sizeInBytes = ResDesc.res.linear.sizeInBytes;
    break;
  case hipResourceTypePitch2D:
    Desc.res.pitch2D.devPtr = ResDesc.res.pitch2D.devPtr;
    Desc.res.pitch2D.desc = getChannelFormatDesc(
        ResDesc.res.pitch2D.numChannels, ResDesc.res.pitch2D.format);
    Desc.res.pitch2D.width = ResDesc.res.pitch2D.width;
    Desc.res.pitch2D.height = ResDesc.res.pitch2D.height;
    Desc.res.pitch2D.pitchInBytes = ResDesc.res.pitch2D.pitchInBytes;
    break;
  default:
    break;
  }

  return Desc;
}

inline HIP_RESOURCE_DESC getResourceDesc(const hipResourceDesc& ResDesc) {
  HIP_RESOURCE_DESC Desc;

  Desc.resType = getResourceType(ResDesc.resType);
  switch (Desc.resType) {
  case HIP_RESOURCE_TYPE_ARRAY:
    Desc.res.array.hArray = ResDesc.res.array.array;
    break;
  case HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY:
    Desc.res.mipmap.hMipmappedArray = ResDesc.res.mipmap.mipmap;
    break;
  case HIP_RESOURCE_TYPE_LINEAR:
    Desc.res.linear.devPtr = ResDesc.res.linear.devPtr;
    Desc.res.linear.numChannels = getNumChannels(ResDesc.res.linear.desc);
    Desc.res.linear.format = getArrayFormat(ResDesc.res.linear.desc);
    Desc.res.linear.sizeInBytes = ResDesc.res.linear.sizeInBytes;
    break;
  case HIP_RESOURCE_TYPE_PITCH2D:
    Desc.res.pitch2D.devPtr = ResDesc.res.pitch2D.devPtr;
    Desc.res.pitch2D.numChannels = getNumChannels(ResDesc.res.pitch2D.desc);
    Desc.res.pitch2D.format = getArrayFormat(ResDesc.res.pitch2D.desc);
    Desc.res.pitch2D.width = ResDesc.res.pitch2D.width;
    Desc.res.pitch2D.height = ResDesc.res.pitch2D.height;
    Desc.res.pitch2D.pitchInBytes = ResDesc.res.pitch2D.pitchInBytes;
    break;
  default:
    break;
  }

  return Desc;
}

inline hipTextureAddressMode getAddressMode(const HIPaddress_mode Mode) {
  // These two enums should be isomorphic.
  return static_cast<hipTextureAddressMode>(Mode);
}

inline HIPaddress_mode getAddressMode(const hipTextureAddressMode Mode) {
  // These two enums should be isomorphic.
  return static_cast<HIPaddress_mode>(Mode);
}

inline hipTextureFilterMode getFilterMode(const HIPfilter_mode Mode) {
  // These two enums should be isomorphic.
  return static_cast<hipTextureFilterMode>(Mode);
}

inline HIPfilter_mode getFilterMode(const hipTextureFilterMode Mode) {
  // These two enums should be isomorphic.
  return static_cast<HIPfilter_mode>(Mode);
}

inline hipTextureReadMode getReadMode(const unsigned int Flags) {
  if (Flags & HIP_TRSF_READ_AS_INTEGER) {
    return hipReadModeElementType;
  } else {
    return hipReadModeNormalizedFloat;
  }
}

inline unsigned int getReadMode(const hipTextureReadMode Mode) {
  if (Mode == hipReadModeElementType) {
    return HIP_TRSF_READ_AS_INTEGER;
  } else {
    return 0;
  }
}

inline int getsRGB(const unsigned int Flags) {
  if (Flags & HIP_TRSF_SRGB) {
    return 1;
  } else {
    return 0;
  }
}

inline unsigned int getsRGB(const int Srgb) {
  if (Srgb == 1) {
    return HIP_TRSF_SRGB;
  } else {
    return 0;
  }
}

inline int getNormalizedCoords(const unsigned int Flags) {
  if (Flags & HIP_TRSF_NORMALIZED_COORDINATES) {
    return 1;
  } else {
    return 0;
  }
}

inline unsigned int getNormalizedCoords(const int NormalizedCoords) {
  if (NormalizedCoords == 1) {
    return HIP_TRSF_NORMALIZED_COORDINATES;
  } else {
    return 0;
  }
}

inline hipTextureDesc getTextureDesc(const HIP_TEXTURE_DESC& TexDesc) {
  hipTextureDesc Desc;

  Desc.addressMode[0] = getAddressMode(TexDesc.addressMode[0]);
  Desc.addressMode[1] = getAddressMode(TexDesc.addressMode[1]);
  Desc.addressMode[2] = getAddressMode(TexDesc.addressMode[2]);
  Desc.filterMode = getFilterMode(TexDesc.filterMode);
  Desc.readMode = getReadMode(TexDesc.flags);
  Desc.sRGB = getsRGB(TexDesc.flags);
  std::memcpy(Desc.borderColor, TexDesc.borderColor, sizeof(Desc.borderColor));
  Desc.normalizedCoords = getNormalizedCoords(TexDesc.flags);
  Desc.maxAnisotropy = TexDesc.maxAnisotropy;
  Desc.mipmapFilterMode = getFilterMode(TexDesc.mipmapFilterMode);
  Desc.mipmapLevelBias = TexDesc.mipmapLevelBias;
  Desc.minMipmapLevelClamp = TexDesc.minMipmapLevelClamp;
  Desc.maxMipmapLevelClamp = TexDesc.maxMipmapLevelClamp;

  return Desc;
}

inline HIP_TEXTURE_DESC getTextureDesc(const hipTextureDesc& TexDesc) {
  HIP_TEXTURE_DESC Desc;

  Desc.addressMode[0] = getAddressMode(TexDesc.addressMode[0]);
  Desc.addressMode[1] = getAddressMode(TexDesc.addressMode[1]);
  Desc.addressMode[2] = getAddressMode(TexDesc.addressMode[2]);
  Desc.filterMode = getFilterMode(TexDesc.filterMode);
  Desc.flags = 0;
  Desc.flags |= getReadMode(TexDesc.readMode);
  Desc.flags |= getsRGB(TexDesc.sRGB);
  Desc.flags |= getNormalizedCoords(TexDesc.normalizedCoords);
  Desc.maxAnisotropy = TexDesc.maxAnisotropy;
  Desc.mipmapFilterMode = getFilterMode(TexDesc.mipmapFilterMode);
  Desc.mipmapLevelBias = TexDesc.mipmapLevelBias;
  Desc.minMipmapLevelClamp = TexDesc.minMipmapLevelClamp;
  Desc.maxMipmapLevelClamp = TexDesc.maxMipmapLevelClamp;
  std::memcpy(Desc.borderColor, TexDesc.borderColor, sizeof(Desc.borderColor));

  return Desc;
}

inline hipResourceViewFormat
getResourceViewFormat(const HIPresourceViewFormat Format) {
  // These two enums should be isomorphic.
  return static_cast<hipResourceViewFormat>(Format);
}

inline HIPresourceViewFormat
getResourceViewFormat(const hipResourceViewFormat Format) {
  // These two enums should be isomorphic.
  return static_cast<HIPresourceViewFormat>(Format);
}

inline hipResourceViewDesc
getResourceViewDesc(const HIP_RESOURCE_VIEW_DESC& ResViewDesc) {
  hipResourceViewDesc Desc;

  Desc.format = getResourceViewFormat(ResViewDesc.format);
  Desc.width = ResViewDesc.width;
  Desc.height = ResViewDesc.height;
  Desc.depth = ResViewDesc.depth;
  Desc.firstMipmapLevel = ResViewDesc.firstMipmapLevel;
  Desc.lastMipmapLevel = ResViewDesc.lastMipmapLevel;
  Desc.firstLayer = ResViewDesc.firstLayer;
  Desc.lastLayer = ResViewDesc.lastLayer;

  return Desc;
}

inline HIP_RESOURCE_VIEW_DESC
getResourceViewDesc(const hipResourceViewDesc& ResViewDesc) {
  HIP_RESOURCE_VIEW_DESC Desc;

  Desc.format = getResourceViewFormat(ResViewDesc.format);
  Desc.width = ResViewDesc.width;
  Desc.height = ResViewDesc.height;
  Desc.depth = ResViewDesc.depth;
  Desc.firstMipmapLevel = ResViewDesc.firstMipmapLevel;
  Desc.lastMipmapLevel = ResViewDesc.lastMipmapLevel;
  Desc.firstLayer = ResViewDesc.firstLayer;
  Desc.lastLayer = ResViewDesc.lastLayer;

  return Desc;
}

inline size_t getElementSize(const hipChannelFormatDesc& desc) {
  return (desc.x / 8) * getNumChannels(desc);
}
