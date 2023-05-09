#include <torch/extension.h>
#include <cmath>
#include <omp.h>

static inline void cityblock(torch::Tensor& XA, torch::Tensor& XB, torch::Tensor& XC){

    uint32_t n_rows = XC.size(0);
    uint32_t n_cols = XC.size(1);
    uint32_t n_elem = XA.size(1);

    float* xa = (float*) XA.data_ptr();
    float* xb = (float*) XB.data_ptr();
    float* xc = (float*) XC.data_ptr();

    #pragma omp parallel for 
    for(uint32_t row = 0; row < n_rows; ++row){
        for(uint32_t col = 0; col < n_cols; ++col){
            #pragma omp simd
            for(uint32_t e = 0; e < n_elem; ++e){
                xc[row * n_cols + col] += std::fabs( xa[row * n_elem + e] - xb[col * n_elem + e] );
            }
        }
    }
}

static inline void euclidean(torch::Tensor& XA, torch::Tensor& XB, torch::Tensor& XC){

    uint32_t n_rows = XC.size(0);
    uint32_t n_cols = XC.size(1);
    uint32_t n_elem = XA.size(1);

    float* xa = (float*) XA.data_ptr();
    float* xb = (float*) XB.data_ptr();
    float* xc = (float*) XC.data_ptr();

    #pragma omp parallel for 
    for(uint32_t row = 0; row < n_rows; ++row){
        for(uint32_t col = 0; col < n_cols; ++col){
            #pragma omp simd
            for(uint32_t e = 0; e < n_elem; ++e){
                xc[row * n_cols + col] += (xa[row * n_elem + e] - xb[col * n_elem + e]) * (xa[row * n_elem + e] - xb[col * n_elem + e]);
            }
            xc[row * n_cols + col] = std::sqrt(xc[row * n_cols + col]);
        }
    }
}

static inline void cosine(torch::Tensor& XA, torch::Tensor& XB, torch::Tensor& XC){

    uint32_t n_rows = XC.size(0);
    uint32_t n_cols = XC.size(1);
    uint32_t n_elem = XA.size(1);

    float* xa = (float*) XA.data_ptr();
    float* xb = (float*) XB.data_ptr();
    float* xc = (float*) XC.data_ptr();

    float sum_ab = 0.; 
    float sum_aa = 0.;
    float sum_bb = 0.;

    // computing cityblock distance - |XA - XB| 
    #pragma omp parallel for
    for(uint32_t row = 0; row < n_rows; row++){
        for(uint32_t col = 0; col < n_cols; col++){
            sum_ab = 0.0;
            sum_aa = 0.0;
            sum_bb = 0.0;
            #pragma omp simd
            for(uint32_t e = 0; e < n_elem; e++){
                // compute sum(a*b)
                sum_ab += xa[row * n_elem + e] * xb[col * n_elem + e];
                // compute sum(a*a)
                sum_aa += xa[row * n_elem + e] * xa[row * n_elem + e];
                // compute sum(b*b)
                sum_bb += xb[col * n_elem + e] * xb[col * n_elem + e];
            }
            xc[row * n_rows + col] = 1. - (sum_ab / (std::sqrt(sum_aa) * std::sqrt(sum_bb) + 1e-12));
        }
    }
}

torch::Tensor cdist(torch::Tensor XA, torch::Tensor XB, std::string metric, uint32_t n_threads){

    torch::Tensor XC = torch::zeros({XA.size(0), XB.size(0)}, torch::dtype(XA.scalar_type()).requires_grad(false).device(torch::kCPU));

    omp_set_num_threads(n_threads);

    if(metric == "cityblock"){
        cityblock(XA, XB, XC);
    }
    else if(metric == "euclidean"){
        euclidean(XA, XB, XC);
    }
    else if(metric == "cosine"){
        cosine(XA, XB, XC);
    }

    return XC;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("cdist", &cdist, "Cityblock distance");
}
