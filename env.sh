source /opt/intel/oneapi/setvars.sh
export OPENCL_LAYERS=/home/pvelesko/OpenCL-Layers-Tutorial-Folder/OpenCL-Layers/build/object-lifetime/libCLObjectLifetimeLayer.so
export LD_LIBRARY_PATH=/home/pvelesko/OpenCL-Layers-Tutorial-Folder/OpenCL-ICD-Loader/build:$LD_LIBRARY_PATH
export CHIP_BE=opencl
export CHIP_PLATFORM=1
export CHIP_DEVICE_TYPE=cpu