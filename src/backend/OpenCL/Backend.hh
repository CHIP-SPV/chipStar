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

class InvalidDeviceType : public std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

class InvalidPlatformOrDeviceNumber : public std::out_of_range {
  using std::out_of_range::out_of_range;
};

class HIPxxBackendOpenCL : public HIPxxBackend {
 public:
  void initialize(std::string HIPxxPlatformStr, std::string HIPxxDeviceTypeStr,
                  std::string HIPxxDeviceStr) override {
    std::cout << "HIPxxBackendOpenCL Initialize\n";
    std::vector<cl::Platform> Platforms;
    cl_int err = cl::Platform::get(&Platforms);

    std::string ver;
    if (err != CL_SUCCESS) return;

    size_t NumDevices = 0;
    std::vector<cl::Device> Devices;
    int selected_platform = -1;
    int selected_device = -1;
    cl_bitfield selected_dev_type = 0;
    try {
      if (!HIPxxPlatformStr.compare("")) {
        selected_platform = std::stoi(HIPxxPlatformStr);
        if ((selected_platform < 0) || (selected_platform >= Platforms.size()))
          throw InvalidPlatformOrDeviceNumber(
              "HIPLZ_PLATFORM: platform number out of range");
      }

      if (!HIPxxDeviceStr.compare("")) {
        selected_device = std::stoi(HIPxxDeviceStr);
        Devices.clear();
        if (selected_platform < 0) selected_platform = 0;
        err = Platforms[selected_platform].getDevices(CL_DEVICE_TYPE_ALL,
                                                      &Devices);
        if (err != CL_SUCCESS)
          throw InvalidPlatformOrDeviceNumber(
              "HIPLZ_DEVICE: can't get devices for platform");
        if ((selected_device < 0) || (selected_device >= Devices.size()))
          throw InvalidPlatformOrDeviceNumber(
              "HIPLZ_DEVICE: device number out of range");
      }

      if (!HIPxxDeviceStr.compare("")) {
        std::string s(HIPxxDeviceStr);
        if (s == "all")
          selected_dev_type = CL_DEVICE_TYPE_ALL;
        else if (s == "cpu")
          selected_dev_type = CL_DEVICE_TYPE_CPU;
        else if (s == "gpu")
          selected_dev_type = CL_DEVICE_TYPE_GPU;
        else if (s == "default")
          selected_dev_type = CL_DEVICE_TYPE_DEFAULT;
        else if (s == "accel")
          selected_dev_type = CL_DEVICE_TYPE_ACCELERATOR;
        else
          throw InvalidDeviceType(
              "Unknown value provided for HIPLZ_DEVICE_TYPE\n");
      }
    } catch (const InvalidDeviceType &e) {
      // logCritical("{}\n", e.what());
      return;
    } catch (const InvalidPlatformOrDeviceNumber &e) {
      // logCritical("{}\n", e.what());
      return;
    } catch (const std::invalid_argument &e) {
      // logCritical(
      //    "Could not convert HIPLZ_PLATFORM or HIPLZ_DEVICES to a number\n");
      return;
    } catch (const std::out_of_range &e) {
      // logCritical("HIPLZ_PLATFORM or HIPLZ_DEVICES is out of range\n");
      return;
    }

    if (selected_dev_type == 0) selected_dev_type = CL_DEVICE_TYPE_ALL;
    for (auto Platform : Platforms) {
      Devices.clear();
      err = Platform.getDevices(selected_dev_type, &Devices);
      if (err != CL_SUCCESS) continue;
      if (Devices.size() == 0) continue;
      if (selected_platform >= 0 && (Platforms[selected_platform] != Platform))
        continue;

      for (cl::Device &Dev : Devices) {
        ver.clear();
        if (selected_device >= 0 && (Devices[selected_device] != Dev)) continue;
        ver = Dev.getInfo<CL_DEVICE_IL_VERSION>(&err);
        if ((err == CL_SUCCESS) && (ver.rfind("SPIR-V_1.", 0) == 0)) {
          // ClDevice *temp = new ClDevice(Dev, Platform, NumDevices);
          // temp->setPrimaryCtx();
          // OpenCLDevices.emplace_back(temp);
          ++NumDevices;
        }
      }
    }

    logDebug("DEVICES {}", NumDevices);
    // assert(NumDevices == OpenCLDevices.size());
  };
};

#endif