#ifdef ENABLE_HIP
#include "hip/hip_ext.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

hipEvent_t newEvent() {
  hipEvent_t event;
  hipEventCreate(&event);
  return event;
}

void insertEvent(hipEvent_t &event) { hipEventRecord(event); }

float timeEvent(hipEvent_t &start, hipEvent_t &stop) {
  hipEventSynchronize(stop);
  float ms = 0;
  hipEventElapsedTime(&ms, start, stop);
  return ms;
}
#endif
#ifdef ENABLE_CUDA

cudaEvent_t newEvent() {
  cudaEvent_t event;
  cudaEventCreate(&event);
  return event;
}

void insertEvent(cudaEvent_t &event) { cudaEventRecord(event); }

float timeEvent(cudaEvent_t &start, cudaEvent_t &stop) {
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  return ms;
}
#endif
