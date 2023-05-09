#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <math.h>
#include <iostream>
#include <omp.h>
#include <vector>

static inline void cityblock(
  npy_float32* XA, // pointer to matrix XA (2d float array)
  npy_float32* XB, // pointer to matrix XB (2d float array)
  npy_float32* XC, // pointer to output matrix XC (2d float array)
  npy_intp& n_rows, // reference to integer number of rows in matrices
  npy_intp& n_cols, // reference to integer number of columns in matrices
  npy_intp& n_dims // reference to vector dimensionality in matrix XA and XB
){
  #pragma omp parallel for simd
  for(npy_intp row = 0; row < n_rows; row++){
    for(npy_intp col = 0; col < n_cols; col++){
      for(npy_intp d = 0; d < n_dims; d++){
        // precompute index of element in output matrix XC
        npy_intp idx = row * n_cols + col;
        // compute difference between value XA[row][col] and XB[col][row]
        XC[idx] += fabsf( XA[row * n_dims + d] - XB[col * n_dims + d] );
      }
    }
  }
}

static inline void euclidean(
  npy_float32* XA, // pointer to matrix XA (2d float array)
  npy_float32* XB, // pointer to matrix XB (2d float array)
  npy_float32* XC, // pointer to output matrix XC (2d float array)
  npy_intp& n_rows, // reference to integer number of rows in matrices
  npy_intp& n_cols, // reference to integer number of columns in matrices
  npy_intp& n_elements // reference to vector dimensionality in matrix XA and XB
){
  #pragma omp parallel for simd
  for(npy_intp row = 0; row < n_rows; row++){
    for(npy_intp col = 0; col < n_cols; col++){
      // precompute index of element in output matrix XC
      npy_intp idx = row * n_cols + col;
      for(npy_intp e = 0; e < n_elements; e++){
        // compute difference between value XA[row][col] and XB[col][row]
        float difference = XA[row * n_elements + e] - XB[col * n_elements + e];
        // compute square difference -> (XA-XB)^2
        XC[idx] += difference * difference;
      }
      // compute square root of sum of squared differences with epsilon = 1e-24
      XC[idx] = sqrtf(XC[idx] + 1e-24);
    }
  }
}

static inline void cosine(
  npy_float32* XA, // pointer to matrix XA (2d float array)
  npy_float32* XB, // pointer to matrix XB (2d float array)
  npy_float32* XC, // pointer to output matrix XC (2d float array)
  npy_intp& n_rows, // reference to integer number of rows in matrices
  npy_intp& n_cols, // reference to integer number of columns in matrices
  npy_intp& n_elements // reference to vector dimensionality in matrix XA and XB
){
  #pragma omp parallel for simd
  for(npy_intp row = 0; row < n_rows; row++){
    for(npy_intp col = 0; col < n_cols; col++){
      float prod_ab = 1e-24;
      float prod_aa = 1e-24;
      float prod_bb = 1e-24;
      npy_intp row_i = row * n_elements;
      npy_intp col_i = col * n_elements;
      for(npy_intp e = 0; e < n_elements; e++){
        npy_intp row_e = row_i + e;
        npy_intp col_e = col_i + e;
        // compute sum(a*b)
        prod_ab += XA[row_e] * XB[col_e];
        // compute sum(a*a)
        prod_aa += XA[row_e] * XA[row_e];
        // compute sum(b*b)
        prod_bb += XB[col_e] * XB[col_e];
      }
      XC[row * n_cols + col] = 1.f - (prod_ab / (sqrtf(prod_aa) * sqrtf(prod_bb)));
    }
  }
}

static inline void crossentropy(
  npy_float32* XA, // pointer to matrix XA (2d float array)
  npy_float32* XB, // pointer to matrix XB (2d float array)
  npy_float32* XC, // pointer to output matrix XC (2d float array)
  npy_intp& n_rows, // reference to integer number of rows in matrices
  npy_intp& n_cols, // reference to integer number of columns in matrices
  npy_intp& n_elements // reference to vector dimensionality in matrix XA and XB
){
  npy_float32 scale = -1. / (npy_float32)n_elements;

  // computing cityblock distance - |XA - XB|
  #pragma omp parallel for simd
  for(unsigned int row = 0; row < n_rows; row++){
    for(unsigned int col = 0; col < n_cols; col++){
      for(unsigned int e = 0; e < n_elements; e++){
        XC[row * n_cols + col] += XA[row * n_elements + e] * log(XB[col * n_elements + e] + 1e-24);
      }
      XC[row * n_cols + col] *= scale;
    }
  }
}

static inline void binarycrossentropy(
  npy_float32* XA, // pointer to matrix XA (2d float array)
  npy_float32* XB, // pointer to matrix XB (2d float array)
  npy_float32* XC, // pointer to output matrix XC (2d float array)
  npy_intp& n_rows, // reference to integer number of rows in matrices
  npy_intp& n_cols, // reference to integer number of columns in matrices
  npy_intp& n_elements // reference to vector dimensionality in matrix XA and XB
){
  npy_float32 scale = -1. / (npy_float32)n_elements;
  // computing cityblock distance - |XA - XB|
  #pragma omp parallel for simd
  for(unsigned int row = 0; row < n_rows; row++){
    for(unsigned int col = 0; col < n_cols; col++){
      for(unsigned int e = 0; e < n_elements; e++){
        XC[row * n_cols + col] += XA[row * n_elements + e] * log(XB[col * n_elements + e]);
        XC[row * n_cols + col] += (1. - XA[row * n_elements + e]) * log((1. - XB[col * n_elements + e]) + 1e-24);
      }
      XC[row * n_cols + col] *= (-1. / (float)n_elements);
    }
  }
}

static PyObject* cdist(PyObject *self, PyObject *args) {

  npy_int64 threads = 8;
  const char* py_metric = nullptr;
  PyArrayObject_fields *py_XA, *py_XB, *py_XC, *py_XW, *py_XY;

  //std::cout << "checking arguments" << std::endl;
  // check arguments
  if(!PyArg_ParseTuple(args, "lsOOO",
                       &threads,
                       &py_metric,
                       &py_XA,
                       &py_XB,
                       &py_XC)){
    return nullptr;
  }

  //std::cout << "setting number of threads to " << threads << std::endl;
  omp_set_num_threads(threads);
  //std::cout << "n_threads: " << threads << std::endl;

  //std::cout << "checking vector dimensions" << std::endl;
  // check if vectors XA and XB are of same dimensionality
  if(PyArray_DIM(py_XA, 1) != PyArray_DIM(py_XB, 1)){
    std::cout << "XA and XB are not of same dimensionality XA.shape[1] != XB.shape[1]" << std::endl;
    return nullptr;
  }

  //std::cout << "assigning numpy arrays to c++ variables" << std::endl;
  // assign numpy arrays to c variables
  npy_float32 *XA = (npy_float32*)PyArray_DATA(py_XA);
  npy_float32 *XB = (npy_float32*)PyArray_DATA(py_XB);
  npy_float32 *XC = (npy_float32*)PyArray_DATA(py_XC);

  // const npy_float32 *XW = (npy_float32*)PyArray_DATA(py_XW);
  // const npy_float32 *XY = (npy_float32*)PyArray_DATA(py_XY);

  //std::cout << "getting rows, columns and vector dimensions" << std::endl;
  npy_intp n_rows = PyArray_DIM(py_XC, 0);
  npy_intp n_cols = PyArray_DIM(py_XC, 1);
  npy_intp n_elements = PyArray_DIM(py_XA, 1);
  //std::cout << "n_rows: " << n_rows << std::endl;
  //std::cout << "n_cols: " << n_cols << std::endl;
  //std::cout << "n_elements: " << n_elements << std::endl;


  // std::cout << "XA.shape " << PyArray_DIM(py_XA, 0) << "," << PyArray_DIM(py_XA, 1) << std::endl;
  // std::cout << "XB.shape " << PyArray_DIM(py_XB, 0) << "," << PyArray_DIM(py_XB, 1) << std::endl;
  // std::cout << "XC.shape " << PyArray_DIM(py_XC, 0) << "," << PyArray_DIM(py_XC, 1) << std::endl;

  std::string metric(py_metric);

  if(metric == "cityblock")
  {
    cityblock(XA, XB, XC, n_rows, n_cols, n_elements);
  }
  else if(metric == "euclidean")
  {
    euclidean(XA, XB, XC, n_rows, n_cols, n_elements);
  }
  else if(metric == "cosine")
  {
    cosine(XA, XB, XC, n_rows, n_cols, n_elements);
  }
  else if(metric == "crossentropy")
  {
    crossentropy(XA, XB, XC, n_rows, n_cols, n_elements);
  }
  else if(metric == "binarycrossentropy")
  {
    binarycrossentropy(XA, XB, XC, n_rows, n_cols, n_elements);
  }
  else
  {
    std::cout << "Given metric " << metric << " is not supported!\n"
              << "Possible metrics are cityblock euclidean cosine crossentropy binarycrossentropy!"
              << std::endl;
    return nullptr;
  }

  Py_RETURN_NONE;
}

// Boilerplate: function list.
static PyMethodDef distance_methods[] = {
  { "cdist", cdist, METH_VARARGS, "Doc string."},
  { NULL, NULL, 0, NULL } /* Sentinel */
};

static struct PyModuleDef distance_definition = {
  PyModuleDef_HEAD_INIT,
  "distances",
  "computing cross distances",
  -1,
  distance_methods
};

// Boilerplate: Module initialization.
PyMODINIT_FUNC PyInit_distances(void) {
  Py_Initialize();
  return PyModule_Create(&distance_definition);
}
