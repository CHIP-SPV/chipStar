// The __HIPCC_RTC__ branch of spirv_hip_vector_types.h must compile (#1899).
#include <stddef.h>
#include <hip/spirv_hip_vector_types.h>

int main() {
  int2 A = make_int2(1, 2);
  float4 F = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
  int3 B = -make_int3(1, 2, 3);
  return (-A).x + (~A).y + (int)F.w + B.z;
}
