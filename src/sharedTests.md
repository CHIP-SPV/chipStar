# Testing /gpu/hip/shared using CHIP-SPV

## t310-basis 
The tests hangs. 

The line that causes the hang is commented out here:
```
//------------------------------------------------------------------------------
// Interp kernel by dim
//------------------------------------------------------------------------------
extern "C" __launch_bounds__(BASIS_INTERP_BLOCK_SIZE) __global__
    void Interp(const CeedInt num_elem, const CeedScalar *d_interp_1d, const CeedScalar *__restrict__ d_U, CeedScalar *__restrict__ d_V) {
  extern __shared__ CeedScalar slice[];
  // load interp_1d into shared memory
  __shared__ CeedScalar s_B[BASIS_P_1D * BASIS_Q_1D];
  loadMatrix<BASIS_P_1D * BASIS_Q_1D>(d_interp_1d, s_B);
  __syncthreads();

  SharedData_Hip data;
  data.t_id_x = threadIdx.x;
  data.t_id_y = threadIdx.y;
  data.t_id_z = threadIdx.z;
  data.t_id   = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
  data.slice  = slice + data.t_id_z * T_1D * (BASIS_DIM > 1 ? T_1D : 1);

  CeedScalar r_U[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_P_1D : 1)];
  CeedScalar r_V[BASIS_NUM_COMP * (BASIS_DIM > 2 ? BASIS_Q_1D : 1)];

  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < num_elem; elem += gridDim.x * blockDim.z) {
    if (BASIS_DIM == 1) {
      ReadElementStrided1d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * num_elem, BASIS_P_1D, d_U, r_U);
      Interp1d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, s_B, r_V);
      WriteElementStrided1d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * num_elem, BASIS_Q_1D, r_V, d_V);
    } else if (BASIS_DIM == 2) {
      ReadElementStrided2d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * num_elem, BASIS_P_1D * BASIS_P_1D, d_U, r_U);
      // InterpTensor2d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, s_B, r_V);
      WriteElementStrided2d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * BASIS_Q_1D * num_elem, BASIS_Q_1D * BASIS_Q_1D, r_V, d_V);
    } else if (BASIS_DIM == 3) {
      ReadElementStrided3d<BASIS_NUM_COMP, BASIS_P_1D>(data, elem, 1, BASIS_P_1D * BASIS_P_1D * BASIS_P_1D * num_elem,
                                                       BASIS_P_1D * BASIS_P_1D * BASIS_P_1D, d_U, r_U);
      InterpTensor3d<BASIS_NUM_COMP, BASIS_P_1D, BASIS_Q_1D>(data, r_U, s_B, r_V);
      WriteElementStrided3d<BASIS_NUM_COMP, BASIS_Q_1D>(data, elem, 1, BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D * num_elem,
                                                        BASIS_Q_1D * BASIS_Q_1D * BASIS_Q_1D, r_V, d_V);
    }
  }
}
```

where:
```
//------------------------------------------------------------------------------
// 2D interpolate to quadrature points
//------------------------------------------------------------------------------
template <int NUM_COMP, int P_1D, int Q_1D>
inline __device__ void InterpTensor2d(SharedData_Hip &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, CeedScalar *__restrict__ r_V) {
  CeedScalar r_t[1];
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    ContractX2d<NUM_COMP, P_1D, Q_1D>(data, r_U + comp, c_B, r_t);
    ContractY2d<NUM_COMP, P_1D, Q_1D>(data, r_t, c_B, r_V + comp);
  }
}
```