#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <cmath>
#include <cassert>

// GPU kernel to compute math functions
__global__ void mathFunctionsKernel(double* results) {
    results[0] = cospi(0.5);
    results[1] = cospif(0.25f);
    results[2] = erfcinv(0.95f + 0.95f);
    results[3] = erfcinvf(0.2f);
    results[4] = erfcx(0.3);
    results[5] = erfcxf(0.4f);
    results[6] = erfinv(0.5);
    results[7] = erfinvf(0.6f);
    results[8] = normcdf(0.7);
    results[9] = normcdff(0.8f);
    results[10] = normcdfinv(0.95f);
    results[11] = normcdfinvf(0.95f);
    results[12] = rcbrt(8.0); // causes 128bit error
    results[13] = rcbrtf(27.0f);
    double sincos_result_cos;
    double sincos_result_sin;
    sincospi(0.25, &sincos_result_sin, &sincos_result_cos);
    results[14] = sincos_result_sin;
    results[15] = sincos_result_cos;
    
    float sincosf_result_cos;
    float sincosf_result_sin;
    sincospif(0.75f, &sincosf_result_sin, &sincosf_result_cos);
    results[16] = sincosf_result_sin;
    results[17] = sincosf_result_cos;
    
    results[18] = sinpi(0.5);
    results[19] = sinpif(0.25f);
    results[20] = llmax(10LL, 20LL);
    results[21] = llmin(30LL, 40LL);
    results[22] = ullmax(50ULL, 60ULL);
    results[23] = ullmin(70ULL, 80ULL);
    results[24] = umax(90U, 100U);
    results[25] = umin(110U, 120U);
    results[26] = signbit(-1.5);
    results[27] = rsqrt(4.0);
    results[28] = rsqrtf(9.0f);
    results[29] = erfc(0.5);
    results[30] = erfcf(0.5);
    results[31] = log(2.0);
    results[32] = logf(2.0f);
}

int main() {    
    const int numResults = 33;
    double hostResults[numResults];
    double* deviceResults;
    
    // Allocate device memory
    hipMalloc(&deviceResults, numResults * sizeof(double));
    
    // Launch kernel
    hipLaunchKernelGGL(mathFunctionsKernel, dim3(1), dim3(1), 0, 0, deviceResults);
    
    // Copy results back to host
    hipMemcpy(hostResults, deviceResults, numResults * sizeof(double), hipMemcpyDeviceToHost);
    
    // Compute host results
    double host_cospi_result = cospi(0.5);
    float host_cospif_result = cospif(0.25f);
    double host_erfcinv_result = erfcinv(0.95f + 0.95f); // segfaults
    float host_erfcinvf_result = erfcinvf(0.2f);
    double host_erfcx_result = erfcx(0.3);
    float host_erfcxf_result = erfcxf(0.4f);
    double host_erfinv_result = erfinv(0.5);
    float host_erfinvf_result = erfinvf(0.6f);
    double host_normcdf_result = normcdf(0.7);
    float host_normcdff_result = normcdff(0.8f);
    double host_normcdfinv_result = normcdfinv(0.95f);
    float host_normcdfinvf_result = normcdfinvf(0.95f);
    double host_rcbrt_result = rcbrt(8.0);
    float host_rcbrtf_result = rcbrtf(27.0f);
    double host_sincos_result_sin, host_sincos_result_cos;
    sincospi(0.25, &host_sincos_result_sin, &host_sincos_result_cos);
    float host_sincosf_result_sin, host_sincosf_result_cos;
    sincospif(0.75f, &host_sincosf_result_sin, &host_sincosf_result_cos);
    double host_sinpi_result = sinpi(0.5);
    float host_sinpif_result = sinpif(0.25f);
    long long host_llmax_result = llmax(10LL, 20LL);
    long long host_llmin_result = llmin(30LL, 40LL);
    unsigned long long host_ullmax_result = ullmax(50ULL, 60ULL);
    unsigned long long host_ullmin_result = ullmin(70ULL, 80ULL);
    unsigned int host_umax_result = umax(90U, 100U);
    unsigned int host_umin_result = umin(110U, 120U);
    int host_signbit_result = signbit(-1.5);
    double host_rsqrt_result = rsqrt(4.0);
    float host_rsqrtf_result = rsqrtf(9.0f);
    double host_erfc_result = erfc(0.5);
    float host_erfcf_result = erfcf(0.5);
    double host_log_result = log(2.0);
    float host_logf_result = logf(2.0f);
    
    // Compare results
    const double epsilon = 1e-6;
    bool all_passed = true;
    
    if (std::abs(host_cospi_result - hostResults[0]) >= epsilon) {
        std::cout << "cospi test failed - Host: " << host_cospi_result << ", Device: " << hostResults[0] << ", Diff: " << std::abs(host_cospi_result - hostResults[0]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_cospif_result - hostResults[1]) >= epsilon) {
        std::cout << "cospif test failed - Host: " << host_cospif_result << ", Device: " << hostResults[1] << ", Diff: " << std::abs(host_cospif_result - hostResults[1]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_erfcinv_result - hostResults[2]) >= epsilon) {
        std::cout << "erfcinv test failed - Host: " << host_erfcinv_result << ", Device: " << hostResults[2] << ", Diff: " << std::abs(host_erfcinv_result - hostResults[2]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_erfcinvf_result - hostResults[3]) >= epsilon) {
        std::cout << "erfcinvf test failed - Host: " << host_erfcinvf_result << ", Device: " << hostResults[3] << ", Diff: " << std::abs(host_erfcinvf_result - hostResults[3]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_erfcx_result - hostResults[4]) >= epsilon) {
        std::cout << "erfcx test failed - Host: " << host_erfcx_result << ", Device: " << hostResults[4] << ", Diff: " << std::abs(host_erfcx_result - hostResults[4]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_erfcxf_result - hostResults[5]) >= epsilon) {
        std::cout << "erfcxf test failed - Host: " << host_erfcxf_result << ", Device: " << hostResults[5] << ", Diff: " << std::abs(host_erfcxf_result - hostResults[5]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_erfinv_result - hostResults[6]) >= epsilon) {
        std::cout << "erfinv test failed - Host: " << host_erfinv_result << ", Device: " << hostResults[6] << ", Diff: " << std::abs(host_erfinv_result - hostResults[6]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_erfinvf_result - hostResults[7]) >= epsilon) {
        std::cout << "erfinvf test failed - Host: " << host_erfinvf_result << ", Device: " << hostResults[7] << ", Diff: " << std::abs(host_erfinvf_result - hostResults[7]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_normcdf_result - hostResults[8]) >= epsilon) {
        std::cout << "normcdf test failed - Host: " << host_normcdf_result << ", Device: " << hostResults[8] << ", Diff: " << std::abs(host_normcdf_result - hostResults[8]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_normcdff_result - hostResults[9]) >= epsilon) {
        std::cout << "normcdff test failed - Host: " << host_normcdff_result << ", Device: " << hostResults[9] << ", Diff: " << std::abs(host_normcdff_result - hostResults[9]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_normcdfinv_result - hostResults[10]) >= epsilon) {
        std::cout << "normcdfinv test failed - Host: " << host_normcdfinv_result << ", Device: " << hostResults[10] << ", Diff: " << std::abs(host_normcdfinv_result - hostResults[10]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_normcdfinvf_result - hostResults[11]) >= epsilon) {
        std::cout << "normcdfinvf test failed - Host: " << host_normcdfinvf_result << ", Device: " << hostResults[11] << ", Diff: " << std::abs(host_normcdfinvf_result - hostResults[11]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_rcbrt_result - hostResults[12]) >= epsilon) {
        std::cout << "rcbrt test failed - Host: " << host_rcbrt_result << ", Device: " << hostResults[12] << ", Diff: " << std::abs(host_rcbrt_result - hostResults[12]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_rcbrtf_result - hostResults[13]) >= epsilon) {
        std::cout << "rcbrtf test failed - Host: " << host_rcbrtf_result << ", Device: " << hostResults[13] << ", Diff: " << std::abs(host_rcbrtf_result - hostResults[13]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_sincos_result_sin - hostResults[14]) >= epsilon) {
        std::cout << "sincospi (sin) test failed - Host: " << host_sincos_result_sin << ", Device: " << hostResults[14] << ", Diff: " << std::abs(host_sincos_result_sin - hostResults[14]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_sincos_result_cos - hostResults[15]) >= epsilon) {
        std::cout << "sincospi (cos) test failed - Host: " << host_sincos_result_cos << ", Device: " << hostResults[15] << ", Diff: " << std::abs(host_sincos_result_cos - hostResults[15]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_sincosf_result_sin - hostResults[16]) >= epsilon) {
        std::cout << "sincospif (sin) test failed - Host: " << host_sincosf_result_sin << ", Device: " << hostResults[16] << ", Diff: " << std::abs(host_sincosf_result_sin - hostResults[16]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_sincosf_result_cos - hostResults[17]) >= epsilon) {
        std::cout << "sincospif (cos) test failed - Host: " << host_sincosf_result_cos << ", Device: " << hostResults[17] << ", Diff: " << std::abs(host_sincosf_result_cos - hostResults[17]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_sinpi_result - hostResults[18]) >= epsilon) {
        std::cout << "sinpi test failed - Host: " << host_sinpi_result << ", Device: " << hostResults[18] << ", Diff: " << std::abs(host_sinpi_result - hostResults[18]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_sinpif_result - hostResults[19]) >= epsilon) {
        std::cout << "sinpif test failed - Host: " << host_sinpif_result << ", Device: " << hostResults[19] << ", Diff: " << std::abs(host_sinpif_result - hostResults[19]) << std::endl;
        all_passed = false;
    }
    if (host_llmax_result != static_cast<long long>(hostResults[20])) {
        std::cout << "llmax test failed - Host: " << host_llmax_result << ", Device: " << static_cast<long long>(hostResults[20]) << ", Diff: " << std::abs(host_llmax_result - static_cast<long long>(hostResults[20])) << std::endl;
        all_passed = false;
    }
    if (host_llmin_result != static_cast<long long>(hostResults[21])) {
        std::cout << "llmin test failed - Host: " << host_llmin_result << ", Device: " << static_cast<long long>(hostResults[21]) << ", Diff: " << std::abs(host_llmin_result - static_cast<long long>(hostResults[21])) << std::endl;
        all_passed = false;
    }
    if (host_ullmax_result != static_cast<unsigned long long>(hostResults[22])) {
        std::cout << "ullmax test failed - Host: " << host_ullmax_result << ", Device: " << static_cast<unsigned long long>(hostResults[22]) << ", Diff: " << std::abs(static_cast<long long>(host_ullmax_result - static_cast<unsigned long long>(hostResults[22]))) << std::endl;
        all_passed = false;
    }
    if (host_ullmin_result != static_cast<unsigned long long>(hostResults[23])) {
        std::cout << "ullmin test failed - Host: " << host_ullmin_result << ", Device: " << static_cast<unsigned long long>(hostResults[23]) << ", Diff: " << std::abs(static_cast<long long>(host_ullmin_result - static_cast<unsigned long long>(hostResults[23]))) << std::endl;
        all_passed = false;
    }
    if (host_umax_result != static_cast<unsigned int>(hostResults[24])) {
        std::cout << "umax test failed - Host: " << host_umax_result << ", Device: " << static_cast<unsigned int>(hostResults[24]) << ", Diff: " << std::abs(static_cast<int>(host_umax_result - static_cast<unsigned int>(hostResults[24]))) << std::endl;
        all_passed = false;
    }
    if (host_umin_result != static_cast<unsigned int>(hostResults[25])) {
        std::cout << "umin test failed - Host: " << host_umin_result << ", Device: " << static_cast<unsigned int>(hostResults[25]) << ", Diff: " << std::abs(static_cast<int>(host_umin_result - static_cast<unsigned int>(hostResults[25]))) << std::endl;
        all_passed = false;
    }
    if (host_signbit_result != static_cast<int>(hostResults[26])) {
        std::cout << "signbit test failed - Host: " << host_signbit_result << ", Device: " << static_cast<int>(hostResults[26]) << ", Diff: " << std::abs(host_signbit_result - static_cast<int>(hostResults[26])) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_rsqrt_result - hostResults[27]) >= epsilon) {
        std::cout << "rsqrt test failed - Host: " << host_rsqrt_result << ", Device: " << hostResults[27] << ", Diff: " << std::abs(host_rsqrt_result - hostResults[27]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_rsqrtf_result - hostResults[28]) >= epsilon) {
        std::cout << "rsqrtf test failed - Host: " << host_rsqrtf_result << ", Device: " << hostResults[28] << ", Diff: " << std::abs(host_rsqrtf_result - hostResults[28]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_erfc_result - hostResults[29]) >= epsilon) {
        std::cout << "erfc test failed - Host: " << host_erfc_result << ", Device: " << hostResults[29] << ", Diff: " << std::abs(host_erfc_result - hostResults[29]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_erfcf_result - hostResults[30]) >= epsilon) {
        std::cout << "erfcf test failed - Host: " << host_erfcf_result << ", Device: " << hostResults[30] << ", Diff: " << std::abs(host_erfcf_result - hostResults[30]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_log_result - hostResults[31]) >= epsilon) {
        std::cout << "log test failed - Host: " << host_log_result << ", Device: " << hostResults[31] << ", Diff: " << std::abs(host_log_result - hostResults[31]) << std::endl;
        all_passed = false;
    }
    if (std::abs(host_logf_result - hostResults[32]) >= epsilon) {
        std::cout << "logf test failed - Host: " << host_logf_result << ", Device: " << hostResults[32] << ", Diff: " << std::abs(host_logf_result - hostResults[32]) << std::endl;
        all_passed = false;
    }
    if (all_passed) {
        std::cout << "PASS: All host and device results match within epsilon!" << std::endl;
    } else {
        std::cout << "FAIL: Some tests failed. Check the output above for details." << std::endl;
    }
    
    // Free device memory
    hipFree(deviceResults);
    
    return 0;
}
