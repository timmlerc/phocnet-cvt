#include <iostream>
#include <torch/extension.h>

// forward decalration
torch::Tensor cdist_cuda(torch::Tensor& XA, torch::Tensor& XB, std::string metric);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("cdist", &cdist_cuda, "Cityblock distance (CUDA)");
}