// Second translation unit of TestFixFunctionLocalStaticDeviceVar: a second
// device module carrying the same unregistered comdat device variables.
#include "FunctionLocalStaticDeviceVar.hh"

__global__ void readShapesTU2(int *Out) { readShapes(Out); }
__global__ void writeShapesTU2(int V) { writeShapes(V); }

void launchReadShapesTU2(int *Out) { readShapesTU2<<<1, 1>>>(Out); }
void launchWriteShapesTU2(int V) { writeShapesTU2<<<1, 1>>>(V); }
const void *readShapesTU2HostPtr() {
  return reinterpret_cast<const void *>(readShapesTU2);
}
