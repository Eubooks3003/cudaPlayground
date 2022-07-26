constexpr int NLL_LOSS_THREADS = 32;

#define AT_DISPATCH_NLL_LOSS_INDEX_TYPES(TYPE, NAME, ...)                   \
  [&] {                                                                     \
    at::ScalarType _it = TYPE;                                              \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _it)                                 \
    switch (_it) {                                                          \
      AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, at::ScalarType::Byte, uint8_t, index_t, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, at::ScalarType::Long, int64_t, index_t, __VA_ARGS__)\
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_it), "'");      \
    }                                                                       \
  }()

//template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void nll_loss_forward_reduce_cuda_kernel_2d(
    float* output,
    float* total_weight,
    float* input,
    long* target,
    float* weights,
    bool size_average,
    int nframe,
    int ndim,
    int n_classes,
    int64_t ignore_index) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  __shared__ float sh_inputs[NLL_LOSS_THREADS],
      acc_weight[NLL_LOSS_THREADS];

  sh_inputs[threadIdx.x] = static_cast<float>(0);
  acc_weight[threadIdx.x] = static_cast<float>(0);
  for (int i = threadIdx.x; i < nframe; i += NLL_LOSS_THREADS) {
    int t = target[i];
    if (t != static_cast<int>(ignore_index)) {
      CUDA_KERNEL_ASSERT(t >= 0 && t < n_classes);
      scalar_t cur_weight =
          weights != nullptr ? weights[t] : static_cast<float>(1);
      sh_inputs[threadIdx.x] -= input[i * ndim + t] * cur_weight;
      acc_weight[threadIdx.x] += cur_weight;
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    accscalar_t output_acc = 0;
    accscalar_t total_weight_acc = 0;
    for (int i = 0; i < NLL_LOSS_THREADS; ++i) {
      output_acc += sh_inputs[i];
      total_weight_acc += acc_weight[i];
    }
    *total_weight = static_cast<float>(total_weight_acc);
    if (size_average) {
      *output = static_cast<float>(output_acc / total_weight_acc);
    } else {
      *output = static_cast<float>(output_acc);
    }
  }
}
