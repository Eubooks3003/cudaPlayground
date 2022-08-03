#include <cassert>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <helper_cuda.h>

#include <getopt.h>


//template <typename scalar_t, typename accscalar_t, typename index_t>
extern "C" __global__ void nll_loss_forward_reduce_cuda_kernel_2d(float* output,
    float* total_weight,
    float* input,
    long* target,
    float* weights,
    bool size_average,
    int nframe,
    int ndim,
    int n_classes,
    int64_t ignore_index);

#define BENCHMARK(i) \
void benchmark_##i(float *const X, float *const W, float *const Y, \
                   const int B, const int I, const int H, \
                   const bool is_nvprof_enabled, long *const Z, float *const A, bool const R, int const L, int const J, int const K, int64_t const P)

BENCHMARK(0);
BENCHMARK(1);
BENCHMARK(2);


int main(int argc, char *argv[]) {
  int i = -1;  // benchmark to run
  int T = -1;
  int I = -1;
  int H = -1;
  bool is_nvprof_enabled = false;
  int opt;

  while ((opt = getopt(argc, argv, "i:T:I:H:p")) != -1) {
    switch (opt) {
    case 'i':
      std::cout << "Benchmark #" << optarg << std::endl;
      i = std::atoi(optarg);
      continue;
    case 'T':
      T = std::atoi(optarg);
      continue;
    case 'I':
      I = std::atoi(optarg);
      continue;
    case 'H':
      H = std::atoi(optarg);
      continue;
    case 'p':
      std::cout << "Enabling NVProf" << std::endl;
      is_nvprof_enabled = true;
      continue;
    default:
      exit(EXIT_FAILURE);
    }
  }

  assert(i != -1 && "The benchmark index MUST be provided");

  if (i == 0 || i == 2) {
    T = 128;
    I = 768;
    H = 2304;
  } else if (i == 1) {
    I = 768;
    H = 2304;
  }

  const int B = 16 * T;
  const int MaxB = 16 * 128;
  assert(argc != 1 && "Size of argument must be equal to 1");
  const int MaxI = 3072;
  const int MaxH = 3072;
  std::cout << "T=" << T << std::endl;
  std::cout << "I=" << I << std::endl;
  std::cout << "H=" << H << std::endl;
  assert(T > 0 && I > 0 && H > 0 && "The parameters must be provided");
  assert(B <= MaxB && I <= MaxI && H <= MaxH &&
         "The parameters must be smaller than the allowed maximum value");
    
    float *X, *W, *Y;
    long *Z;
    
    float *A;
    bool R;
    int L, K, J;
    int64_t P;
    
 

  cudaMalloc(&X, sizeof(float) * MaxB * MaxI);
  cudaMalloc(&W, sizeof(float) * MaxH * MaxI);
  cudaMalloc(&Y, sizeof(float) * MaxB * MaxH);
    
  cudaMalloc(&Z, sizeof(long) * MaxB * MaxH);
  
  cudaMalloc(&A, sizeof(float) * MaxI * MaxH);
    
  
  R = true;
    
  L = MaxB * MaxI;
  K =  MaxB * MaxH;
  J = MaxH * MaxI;
  
    
  P = static_cast<int64_t>(MaxB * MaxI);
  

#define CALL_BENCHMARK(num)                                                     \
if (i == num) benchmark_##num(X, W, Y, B, I, H, is_nvprof_enabled, Z, A,R, L, K, J, P)

  CALL_BENCHMARK(0);  // ./main.exe -i 0
  CALL_BENCHMARK(1);  // ./main.exe -i 1 -T 128
  CALL_BENCHMARK(2);  // ./main.exe -i 2
  return 0;
}


inline int floordiv(int a, int b) {
  return a / b;
}


#define TIMER_BEGIN(BlockName)                                                  \
float elapsedTime##BlockName = 0.;                                              \
{                                                                               \
  auto tic = std::chrono::system_clock::now();

#define TIMER_END(BlockName)                                                    \
  auto toc = std::chrono::system_clock::now();                                  \
  elapsedTime##BlockName =                                                      \
      std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count()  \
      * 1.0;                                                                    \
  std::cout << "ElapsedTime (us)=" << elapsedTime##BlockName << std::endl;      \
}


class CUDAFunctionWrapper {
private:
  const std::function<void(void)> f;
  const float FLOPs;
  const bool is_nvprof_enabled;
public:
  CUDAFunctionWrapper(const std::function<void(void)> f, const float FLOPs,
                      const bool is_nvprof_enabled)
      : f(f), FLOPs(FLOPs), is_nvprof_enabled(is_nvprof_enabled) {}
  void operator()() const {
    if (is_nvprof_enabled) {
      f();
      return;
    }
    for (int i = 0; i < 1000; ++i) {  // warmup run
      f();
    }
    checkCudaErrors(cudaDeviceSynchronize());
    TIMER_BEGIN();
    for (int i = 0; i < 1000; ++i) {
      f();
    }
    checkCudaErrors(cudaDeviceSynchronize());
    TIMER_END();
    std::cout << "TFLOPS=" << FLOPs / elapsedTime / 1e3 << std::endl;
  }
};


BENCHMARK(0) {
  size_t grid_size = B * H / 128;
  std::cout << "pading<<<" << grid_size << ", 128>>>"
            << std::endl;
  auto f = [&]() {
             cudaMemcpy(X, Y, sizeof(float) * B * H, cudaMemcpyDeviceToDevice);
           };
  CUDAFunctionWrapper wrapper(f, B * H, is_nvprof_enabled);
  wrapper();
}


BENCHMARK(1) {
    
  size_t grid_size = B * H / 128 / 64;
  std::cout << "nll_loss_forward<<<" << grid_size << ", 64>>>"
            << std::endl;
  auto f = [&]() {
             nll_loss_forward_reduce_cuda_kernel_2d<<<grid_size, 64>>>(X, W, Y, Z, A, R, L, K, J, P);
           };
  CUDAFunctionWrapper wrapper(f, 2. * B * I * H, is_nvprof_enabled);
  wrapper();
}


BENCHMARK(2) {
  {
    auto f = [&]() {
               nll_loss_forward_reduce_cuda_kernel_2d<<<576, 64>>>(X, W, Y, Z, A, R, L, K, J, P);
             };
    CUDAFunctionWrapper wrapper(f, 2. * 16 * 128 * I * H, is_nvprof_enabled);
    wrapper();
  }
  {
    auto f = [&]() {
               nll_loss_forward_reduce_cuda_kernel_2d<<<540, 64>>>(X, W, Y, Z, A, R, L, K, J, P);
             };
    CUDAFunctionWrapper wrapper(f, 2. * 16 * 120 * I * H, is_nvprof_enabled);
    wrapper();
  }
  {
    auto f = [&]() {
               nll_loss_forward_reduce_cuda_kernel_2d<<<576, 64>>>(X, W, Y, Z, A, R, L, K, J, P);
             };
    CUDAFunctionWrapper wrapper(f, 2. * 16 * 128 * I * H, is_nvprof_enabled);
    wrapper();
  }
}
