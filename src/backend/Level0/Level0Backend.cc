#include "Level0Backend.hh"

HIPxxQueueLevel0::HIPxxQueueLevel0(HIPxxContextLevel0* _hipxx_ctx,
                                   HIPxxDeviceLevel0* _hipxx_dev) {
  hipxx_device = _hipxx_dev;
  hipxx_context = _hipxx_ctx;
  logTrace(
      "HIPxxQueueLevel0 constructor called via HIPxxContextLevel0 and "
      "HIPxxDeviceLevel0");

  HIPxxQueueLevel0(_hipxx_ctx->get(), _hipxx_dev->get());
}