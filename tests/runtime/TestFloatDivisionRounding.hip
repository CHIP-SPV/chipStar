// float and double division are correctly rounded, including huge, tiny and
// subnormal operands, zeros of either sign, infinities and NaN.
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      printf("FAIL: %s returned %s (line %d)\n", #Call, hipGetErrorName(Err),  \
             __LINE__);                                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

template <typename T>
__global__ void divide(const T *X, const T *Y, T *Out, int N) {
  int I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N)
    Out[I] = X[I] / Y[I];
}

template <typename T> int check(std::vector<T> Special) {
  std::vector<T> X, Y;
  for (T A : Special)
    for (T B : Special) {
      X.push_back(A);
      Y.push_back(B);
    }
  std::mt19937 G(1);
  std::uniform_real_distribution<T> U(-1e3, 1e3);
  while (X.size() < 1 << 14) {
    X.push_back(U(G));
    Y.push_back(U(G));
  }
  int N = X.size();
  std::vector<T> Out(N);
  T *DX, *DY, *DO;
  CHECK(hipMalloc(&DX, N * sizeof(T)));
  CHECK(hipMalloc(&DY, N * sizeof(T)));
  CHECK(hipMalloc(&DO, N * sizeof(T)));
  CHECK(hipMemcpy(DX, X.data(), N * sizeof(T), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(DY, Y.data(), N * sizeof(T), hipMemcpyHostToDevice));
  divide<T><<<(N + 255) / 256, 256>>>(DX, DY, DO, N);
  CHECK(hipGetLastError());
  CHECK(hipMemcpy(Out.data(), DO, N * sizeof(T), hipMemcpyDeviceToHost));
  int Bad = 0;
  for (int I = 0; I < N; ++I) {
    T H = X[I] / Y[I];
    bool Ok = (std::isnan(H) && std::isnan(Out[I])) ||
              (H == Out[I] && std::signbit(H) == std::signbit(Out[I]));
    if (!Ok && Bad++ < 4)
      printf("%a / %a = %a, expected %a\n", (double)X[I], (double)Y[I],
             (double)Out[I], (double)H);
  }
  return Bad;
}

typedef float Float2 __attribute__((ext_vector_type(2)));
__global__ void divideVec(const Float2 *X, const Float2 *Y, Float2 *Out) {
  Out[0] = X[0] / Y[0];
}

int main() {
  int Bad = check<float>({0.f, -0.f, 1.f, -3.f, INFINITY, -INFINITY, NAN,
                          0x1p127f, 0x1p-126f, 0x1p-149f, 0x1.fffffep127f,
                          3.0e-39f});
  Bad += check<double>({0.0, -0.0, 1.0, -3.0, INFINITY, -INFINITY, NAN,
                        0x1p1023, 0x1p-1022, 0x1p-1074, 0x1.fffffffffffffp1023,
                        3.0e-310});
  Float2 VX = {0x1p127f, 1.0f}, VY = {0x1p127f, 3.0f}, VO, *D;
  CHECK(hipMalloc(&D, 3 * sizeof(Float2)));
  CHECK(hipMemcpy(D, &VX, sizeof(Float2), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(D + 1, &VY, sizeof(Float2), hipMemcpyHostToDevice));
  divideVec<<<1, 1>>>(D, D + 1, D + 2);
  CHECK(hipGetLastError());
  CHECK(hipMemcpy(&VO, D + 2, sizeof(Float2), hipMemcpyDeviceToHost));
  if (VO.x != 1.0f || VO.y != 1.0f / 3.0f) {
    printf("vector: %a %a\n", VO.x, VO.y);
    ++Bad;
  }
  printf("%s\n", Bad ? "FAILED" : "PASSED");
  return Bad != 0;
}
