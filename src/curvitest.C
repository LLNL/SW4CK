//////////////////////////////////////////////////////////////////////////////
//// Copyright (c) 2021, Lawrence Livermore National Security, LLC and SW4CK
//// project contributors. See the COPYRIGHT file for details.
////
//// SPDX-License-Identifier: GPL-2.0-only
////////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <ios>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <tuple>
#include <chrono>
#include <limits>
#define float_sw4 double
#include "SW4CKConfig.h"
#include "foralls.h"
#ifndef NO_RAJA
#include "RAJA/RAJA.hpp"
#endif
#ifdef ENABLE_CUDA
#include <cuda_profiler_api.h>
#endif
#ifdef ENABLE_CUDA
void CheckError(cudaError_t const err, const char *file, char const *const fun,
                const int line);
#elif ENABLE_HIP
void CheckError(hipError_t const err, const char *file, char const *const fun,
                const int line);
#endif


#define CheckDeviceError(err) \
  CheckError(err, __FILE__, __FUNCTION__, __LINE__)








class Sarray {
 public:
  Sarray() {}
  Sarray(int nc, int ibeg, int iend, int jbeg, int jend, int kbeg, int kend);
  std::string fill(std::istringstream& iss);
  void init();
  void init2();
  double norm();
  std::tuple<double,double> minmax();
  int m_nc, m_ni, m_nj, m_nk;
  int m_ib, m_ie, m_jb, m_je, m_kb, m_ke;
  ssize_t m_base;
  size_t m_offi, m_offj, m_offk, m_offc, m_npts;
  double* m_data;
  size_t size;
  int g;
  
};

std::string Sarray::fill(std::istringstream& iss) {
  std::string name;
  if (!(iss >> name >> g >> m_nc >> m_ni >> m_nj >> m_nk >> m_ib >> m_ie >>
        m_jb >> m_je >> m_kb >> m_ke >> m_base >> m_offi >> m_offj >> m_offk >>
        m_offc >> m_npts))
    return "Break";
#ifdef VERBOSE
  std::cout << name << " " << m_npts << "\n";
#endif
  void* ptr;
  size = m_nc * m_ni * m_nj * m_nk * sizeof(double);

#ifdef ENABLE_CUDA
  if (cudaMallocManaged(&ptr, size) != cudaSuccess) {
    std::cerr << "cudaMallocManaged failed for size " << size << " bytes\n";
    abort();
  }
#endif
#ifdef ENABLE_HIP
  if (hipMalloc(&ptr, size) != hipSuccess) {
    std::cerr << "hipMallocManaged failed for size " << size << " bytes\n";
    abort();
  }
#endif

#ifdef VERBOSE
  std::cout << "Allocated " << m_nc * m_ni * m_nj * m_nk * sizeof(double)
            << " bytes for array " << name << "[" << g << "]\n";
#endif
  m_data = (double*)ptr;
  return name;
}

void Sarray::init() {
  double* lm_data = m_data;
  forallasync(0, size / 8,
              [=] __device__(int i) { lm_data[i] = sin(double(i)); });
}

void Sarray::init2() {

  Range<64> I(0,m_ni);
  Range<2> J(0,m_nj);
  Range<2> K(0,m_nk); 

  double dx = 0.001;
    int nc = m_nc;
  int offi = nc;
  int offj = nc*m_ni;
  int offk = nc*m_ni*m_nj;

  double *data = m_data;
  
 
  forall3asyncnotimer(
	       I, J, K, [=] __device__(int i, int j, int k) {
		 for (int c=0;c<nc;c++){
		   
		   int indx = c + i * offi + j * offj +
		     k * offk;
		   double x = i*dx;
		   double y = j*dx;
		   double z = k*dx;
		   double f = sin(x)*sin(y)*sin(z);
		   data[indx]=f;
	       }
	       });
  
}
double Sarray::norm() {
  double ret = 0.0;
  for (size_t i = 0; i < size / 8; i++) ret += m_data[i] * m_data[i];
  return ret;
}
std::tuple<double,double> Sarray::minmax(){
  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::min();
  for (size_t i = 0; i < size / 8; i++) {
    min=std::min(min,m_data[i]);
    max=std::max(max,m_data[i]);
  }
  return std::make_tuple(min,max);
}

void curvilinear4sg_ci(
    int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
    float_sw4* __restrict__ a_u1, float_sw4* __restrict__ a_u2, float_sw4* __restrict__ a_u3, 
    float_sw4* __restrict__ a_mu,
    float_sw4* __restrict__ a_lambda, float_sw4* __restrict__ a_met,
    float_sw4* __restrict__ a_jac, float_sw4* __restrict__ a_lu, int* onesided,
    float_sw4* __restrict__ a_acof, float_sw4* __restrict__ a_bope,
    float_sw4* __restrict__ a_ghcof, float_sw4* __restrict__ a_acof_no_gp,
    float_sw4* __restrict__ a_ghcof_no_gp, float_sw4* __restrict__ a_strx,
    float_sw4* __restrict__ a_stry, int nk, char op);

int main(int argc, char* argv[]) {
  std::ifstream iff;
  iff.open(argv[1]);
  promo_version();
  std::map<std::string, Sarray*> arrays[10];
  std::vector<int*> onesided;
  std::string line;
  int lc = 0;
  std::cout << "Reading from file " << argv[1] << "\n";
  while (std::getline(iff, line)) {
    std::istringstream iss(line);
    int* optr = new int[14];
    const int N = 16;
    if ((lc % N) == 0) {
      if (!(iss >> optr[0] >> optr[1] >> optr[2] >> optr[3] >> optr[4] >>
            optr[5] >> optr[6] >> optr[7] >> optr[8] >> optr[9] >> optr[10] >>
            optr[11] >> optr[12] >> optr[13])) {
        std::cerr << "ERROR READING data on line " << lc + 1 << "\n";
        break;
      }

      onesided.push_back(optr);
    } else {
      Sarray* s = new Sarray();
      auto name = s->fill(iss);
      if (name == "Break") {
        std::cerr << "Error reading Sarray data on line " << lc + 1 << "\n";
        break;
      } else {
        arrays[lc / N][name] = s;
      }
    }
    lc++;
  }
#ifdef VERBOSE
  std::cout << "\nCurrent state of map array\n";
#endif
  for (int i = 0; i < 2; i++)
    for (auto const& x : arrays[i]) {
#ifdef VERBOSE
      std::cout << x.first << " " << x.second->g << " " << x.second->m_npts
                << "\n";
#endif
      x.second->init2();
    }


#ifdef ENABLE_CUDA
  cudaStreamSynchronize(0);
 #endif
#ifdef ENABLE_HIP
  hipStreamSynchronize(0);
#endif

#ifdef VERBOSE
  std::cout << "Done with map array output\n";
#endif

  void* ptr;
#ifdef ENABLE_CUDA
  if (cudaMallocManaged(&ptr, (6 + 384 + 24 + 48 + 6 + 384 + 6 + 6) *
                                  sizeof(double)) != cudaSuccess) {
    std::cerr << "cudaMallocManaged failed for cofs\n";
    abort();
  }
#endif
#ifdef ENABLE_HIP
  if (hipMalloc(&ptr, (6 + 384 + 24 + 48 + 6 + 384 + 6 + 6) * sizeof(double)) !=
      hipSuccess) {
    std::cerr << "hipMallocManaged failed for cofs\n";
    abort();
  }
#endif

  double *m_sbop, *m_acof, *m_bop, *m_bope, *m_ghcof, *m_acof_no_gp,
      *m_ghcof_no_gp;
  double* tmpa = (double*)ptr;
  m_sbop = tmpa;
  m_acof = m_sbop + 6;
  m_bop = m_acof + 384;
  m_bope = m_bop + 24;
  m_ghcof = m_bope + 48;
  m_acof_no_gp = m_ghcof + 6;
  m_ghcof_no_gp = m_acof_no_gp + 384;

  // std::cout << "Init the cof arrays\n";
  forallasync(0, (6 + 384 + 24 + 48 + 6 + 384 + 6 + 6),
              [=] __device__(int i) { m_sbop[i] = i / 1000.0; });
  // std::cout << "Done\n";

  for (int i = 1; i < 2; i++) {  // 0 has the smaller datatset
    int* optr = onesided[i];
    double* alpha_ptr = arrays[i]["a_AlphaVE_0"]->m_data;
    double* mua_ptr = arrays[i]["mMuVE_0"]->m_data;
    double* lambdaa_ptr = arrays[i]["mLambdaVE_0"]->m_data;
    double* met_ptr = arrays[i]["mMetric"]->m_data;
    double* jac_ptr = arrays[i]["mJ"]->m_data;
    double* uacc_ptr = arrays[i]["a_Uacc"]->m_data;
    int* onesided_ptr = optr;
    int nkg = optr[12];
    //     int m_number_mechanisms=optr[13];
    char op = '-';
    void* ptr;
    int size = optr[7] - optr[6] + optr[9] - optr[8] + 2;
#ifdef ENABLE_CUDA
    cudaMallocManaged(&ptr, size * sizeof(double));
#endif
#ifdef ENABLE_HIP
    hipMalloc(&ptr, size * sizeof(double));
#endif

    double* m_sg_str_x = (double*)ptr;
    double* m_sg_str_y = m_sg_str_x + optr[7] - optr[6] + 1;
    forallasync(0, size, [=] __device__(int i) { m_sg_str_x[i] = i / 1000.0; });
    // std::cout << "Done initilizing m_sg_str_x and y\n" << std::flush;
#ifdef ENABLE_CUDA
    cudaProfilerStart();
#endif
    std::cout << "Launching sw4 kernels\n\n" << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
   
    const int ifirst = optr[6];
    const int ilast = optr[7];
    const int jfirst = optr[8];
    const int jlast = optr[9];
    const int kfirst = optr[10];
    const int klast = optr[11];
    
    const int ni = ilast - ifirst + 1;
    const int nij = ni * (jlast - jfirst + 1);
    const int nijk = nij * (klast - kfirst + 1);
    const int base = -(ifirst + ni * jfirst + nij * kfirst);
    const int base3 = base - nijk;
    //const int base4 = base - nijk;
    //const int ifirst0 = ifirst;
    //const int jfirst0 = jfirst;
    for (int p = 0; p < 1; p++)
      curvilinear4sg_ci(optr[6], optr[7], optr[8], optr[9], optr[10], optr[11],
                        alpha_ptr+base3+nijk,alpha_ptr+base3+2*nijk,alpha_ptr+base3+3*nijk,
			mua_ptr, lambdaa_ptr, met_ptr, jac_ptr,
                        uacc_ptr, onesided_ptr, m_acof_no_gp, m_bope,
                        m_ghcof_no_gp, m_acof_no_gp, m_ghcof_no_gp, m_sg_str_x,
                        m_sg_str_y, nkg, op);
#ifdef ENABLE_CUDA
CheckDeviceError(cudaStreamSynchronize(0));
    cudaProfilerStop();
    CheckDeviceError(cudaPeekAtLastError());
    CheckDeviceError(cudaStreamSynchronize(0));
    cudaFree(ptr);
#endif
#ifdef ENABLE_HIP
    CheckDeviceError(hipStreamSynchronize(0));
    CheckDeviceError(hipPeekAtLastError());
    CheckDeviceError(hipStreamSynchronize(0));
    // cudaProfilerStop();
    hipFree(ptr);
#endif
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout<<"\nTotal kernel runtime = "<<std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count()<<" milliseconds("<<std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count()<<" us )\n\n";
    auto  minmax =arrays[i]["a_Uacc"]->minmax();
    std::cout << "MIN = " << std::defaultfloat << std::setprecision(20)
              << std::get<0>(minmax)<<"\nMAX = "<<std::get<1>(minmax)  << "\n\n";
    double norm=arrays[i]["a_Uacc"]->norm();
    std::cout << "Norm of output " << std::hexfloat
              << norm  << "\n";
    std::cout << "Norm of output " << std::defaultfloat << std::setprecision(20)
              << norm  << "\n";
    //const double exact_norm = 9.86238393426104e+17;
    const double exact_norm = 202.0512747393526638; // for init2
    double err = (norm - exact_norm) / exact_norm * 100;
    std::cout << "Error = " << std::setprecision(2) << err << " %\n";
  }
}
void promo_version(){
  std::stringstream s;
#ifdef ENABLE_HIP
  s<<"HIP("<<HIP_VERSION_MAJOR<<"."<<HIP_VERSION_MINOR<<"."<<HIP_VERSION_PATCH<<")\n";
#elif ENABLE_CUDA
  s<<"CUDA("<<CUDA_VERSION<<")\n";
#else
  s<<"Unknown programming model\n";
#endif
#ifndef NO_RAJA
s<<"RAJA("<<RAJA_VERSION_MAJOR<<"."<<RAJA_VERSION_MINOR<<"."<<RAJA_VERSION_PATCHLEVEL<<")\n";
#endif
std::cout<<s.str();
  
}

#ifdef ENABLE_CUDA
void CheckError(cudaError_t const err, const char *file, char const *const fun,
                const int line) {
  if (err) {
    std::cerr << "CUDA Error Code[" << err << "]: " << cudaGetErrorString(err)
              << " " << file << " " << fun << " Line number:  " << line << "\n";
    abort();
  }
}
#elif ENABLE_HIP
void CheckError(hipError_t const err, const char *file, char const *const fun,
                const int line) {
  if (err) {
    std::cerr << "HIP Error Code[" << err << "]: " << hipGetErrorString(err)
              << " " << file << " " << fun << " Line number:  " << line << "\n";
    abort();
  }
}
#endif
