#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <iostream>
#include <string>


__global__ void cityblock_kernel(const float* xa, const float* xb, float* xc, const int32_t dim){

    // get cuda core index information 
    auto n_rows = gridDim.x; 
    auto n_cols = blockDim.x;
    auto row = blockIdx.x;
    auto col = threadIdx.x;
    
    const float* a = &xa[row*dim];
    const float* b = &xb[col*dim];

    for(uint32_t idx = 0 ; idx < dim; ++idx){
        xc[row * n_cols + col] += fabsf(a[idx] - b[idx]);
    }
}

__global__ void euclidean_kernel(const float* xa, const float* xb, float* xc, const int64_t dim){

    // get cuda core index information 
    auto n_rows = gridDim.x; 
    auto n_cols = blockDim.x;
    auto row = blockIdx.x;
    auto col = threadIdx.x;

    const float* a = &xa[row*dim];
    const float* b = &xb[col*dim];

    for(auto d = 0; d < dim; ++d){
        xc[row * n_cols + col] += (a[d] - b[d]) * (a[d] - b[d]);
    }

    xc[row * n_cols + col] = sqrtf(xc[row * n_cols + col]);
}

__global__ void cosine_kernel(const float* __restrict__ xa, const float* __restrict__ xb, float* __restrict__ xc, const int64_t dim){

    // get cuda core index information 
    auto n_rows = gridDim.x; 
    auto n_cols = blockDim.x;
    
    auto row = blockIdx.x;
    auto col = threadIdx.x;

    const float* a = &xa[row*dim];
    const float* b = &xb[col*dim];

    float sum_aa = 0.f;
    float sum_bb = 0.f;
    float sum_ab = 0.f;

    for(auto d = 0; d < dim; d++){
        // compute sum(a*b)
        sum_ab += a[d] * b[d];
        // compute sum(a*a)
        sum_aa += a[d] * a[d];
        // compute sum(b*b)
        sum_bb += b[d] * b[d];
    }
    xc[row * n_rows + col] = 1. - (sum_ab / (sqrtf(sum_aa) * sqrtf(sum_bb) + 1e-12));
}
 
torch::Tensor cdist_cuda(torch::Tensor& XA, torch::Tensor& XB, std::string metric){

    torch::Tensor XC = torch::zeros({XA.size(0), XB.size(0)}, torch::dtype(torch::kFloat).requires_grad(false).device(torch::kCUDA));

    auto n_threads = XC.size(1);
    auto n_blocks = XC.size(0);
    auto dim = XA.size(1);

    std::cout << "starting cuda kernel with n_threads: " << n_threads << " n_blocks: " << n_blocks << " dim: " << dim << std::endl;
    std::cout << "Data type: " << XA.type() << std::endl;

    if(metric == "cityblock"){
        cityblock_kernel<<<n_blocks, n_threads>>>((float*)XA.data_ptr(), (float*)XB.data_ptr(), (float*)XC.data_ptr(), dim);
    }
    else if(metric == "euclidean"){
        euclidean_kernel<<<n_blocks, n_threads>>>((float*)XA.data_ptr(), (float*)XB.data_ptr(), (float*)XC.data_ptr(), dim);
    }
    else if(metric == "cosine"){
        cosine_kernel<<<n_blocks, n_threads>>>(XA.data_ptr<float>(), XB.data_ptr<float>(), XC.data_ptr<float>(), dim);
    }
    cudaDeviceSynchronize();

    return XC;
}