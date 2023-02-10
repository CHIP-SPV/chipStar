#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>

int main(int argc, char ** argv) {
    hipDoubleComplex x = make_hipDoubleComplex(2.0, 3.0); 
    hipDoubleComplex y = make_hipDoubleComplex(4.0, 2.0);

    y = y * x;

    printf("%f%+fi\n", hipCreal(y), hipCimag(y));

    return 0;
}