#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <hip/hip_runtime.h>

struct FooT {
  char a;
  struct {
    long b;
    short c;
  } bar;
  int d;
};

__device__ FooT Foo{1, {2, 3}, 4};

__global__ void readFoo(int* Out) {
  Out[0] = Foo.a;
  Out[1] = Foo.bar.b;
  Out[2] = Foo.bar.c;
  Out[3] = Foo.d;
}

int main() {
  constexpr unsigned NumElts = 4;
  int *OutD, OutH[4];
  (void)hipMalloc(&OutD, sizeof(int) * NumElts);
  readFoo<<<1, 1>>>(OutD);
  (void)hipMemcpy(&OutH, OutD, sizeof(int) * NumElts, hipMemcpyDeviceToHost);
  for (unsigned i = 0; i < NumElts; i++)
    assert(OutH[i] == i + 1);
  return 0;
}
