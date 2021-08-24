/**
 * @file Backend.hh
 * @author Paulius Velesko (pvelesko@gmail.com)
 * @brief OpenCL backend for HIPxx. HIPxxBackendOpenCL class definition with
 * inheritance from HIPxxBackend. Subsequent virtual function overrides.
 * @version 0.1
 * @date 2021-08-19
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef HIPXX_BACKEND_OPENCL_H
#define HIPXX_BACKEND_OPENCL_H

#define CL_TARGET_OPENCL_VERSION 210
#define CL_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 210
#define CL_HPP_MINIMUM_OPENCL_VERSION 200

#include <CL/cl_ext_intel.h>

#include <CL/opencl.hpp>

#include "../../HIPxxBackend.hh"
#include "exceptions.hh"

class HIPxxContextOpenCL : public HIPxxContext {
 protected:
  cl::Context *ctx;

 public:
  HIPxxContextOpenCL(cl::Context *ctx_in) {
    std::cout << "HIPxxContextOpenCL Initialized via OpenCL Context pointer.\n";
    ctx = ctx_in;
  };
};

class HIPxxBackendOpenCL : public HIPxxBackend {
 public:
  void initialize(std::string HIPxxPlatformStr, std::string HIPxxDeviceTypeStr,
                  std::string HIPxxDeviceStr) override {
    std::cout << "HIPxxBackendOpenCL Initialize\n";
    std::vector<cl::Platform> Platforms;
    cl_int err = cl::Platform::get(&Platforms);
    if (err != CL_SUCCESS) return;

    std::vector<cl::Device> Devices;
    int selected_platform;
    int selected_device;
    cl_bitfield selected_dev_type = 0;

    try {
      selected_platform = std::stoi(HIPxxPlatformStr);
      if (!HIPxxDeviceStr.compare("all")) {  // Use all devices that match type
        selected_device = -1;
      } else {
        selected_device = std::stoi(HIPxxDeviceStr);
      }

      // Platform index in range?
      if ((selected_platform < 0) || (selected_platform >= Platforms.size()))
        throw InvalidPlatformOrDeviceNumber(
            "HIPXX_PLATFORM: platform number out of range");
      // Device  index in range?
      if (selected_device >= Devices.size())
        throw InvalidPlatformOrDeviceNumber(
            "HIPXX_DEVICE: device number out of range");

      if (err != CL_SUCCESS)
        throw InvalidPlatformOrDeviceNumber(
            "HIPXX_DEVICE: can't get devices for platform");

      if (HIPxxDeviceTypeStr == "all")
        selected_dev_type = CL_DEVICE_TYPE_ALL;
      else if (HIPxxDeviceTypeStr == "cpu")
        selected_dev_type = CL_DEVICE_TYPE_CPU;
      else if (HIPxxDeviceTypeStr == "gpu")
        selected_dev_type = CL_DEVICE_TYPE_GPU;
      else if (HIPxxDeviceTypeStr == "default")
        selected_dev_type = CL_DEVICE_TYPE_DEFAULT;
      else if (HIPxxDeviceTypeStr == "accel")
        selected_dev_type = CL_DEVICE_TYPE_ACCELERATOR;
      else
        throw InvalidDeviceType(
            "Unknown value provided for HIPXX_DEVICE_TYPE\n");
    } catch (const InvalidDeviceType &e) {
      // logCritical("{}\n", e.what());
      return;
    } catch (const InvalidPlatformOrDeviceNumber &e) {
      // logCritical("{}\n", e.what());
      return;
    } catch (const std::invalid_argument &e) {
      // logCritical(
      //    "Could not convert HIPXX_PLATFORM or HIPXX_DEVICES to a number\n");
      return;
    } catch (const std::out_of_range &e) {
      // logCritical("HIPXX_PLATFORM or HIPXX_DEVICES is out of range\n");
      return;
    }

    // Get All the devices on the selected platform of selected type
    err = Platforms[selected_platform].getDevices(selected_dev_type, &Devices);

    std::vector<cl::Device> spirv_enabled_devices;
    for (cl::Platform &platform : Platforms) {
      for (cl::Device &dev : Devices) {
        std::string ver = dev.getInfo<CL_DEVICE_IL_VERSION>(&err);
        if ((err == CL_SUCCESS) && (ver.rfind("SPIR-V_1.", 0) == 0)) {
          spirv_enabled_devices.push_back(dev);
        }
      }
    }

    // 1. Create a context for this device
    cl::Context *ctx = new cl::Context(spirv_enabled_devices);
    HIPxxContextOpenCL *HIPxxCtx = new HIPxxContextOpenCL(ctx);
    Backend->add_context(HIPxxCtx);
  };
};

#endif