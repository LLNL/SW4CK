set(ENABLE_HIP ON CACHE BOOL "")
set(HIPCC_VERSION "rocm-4.2.0" CACHE STRING "")

set(CMAKE_CXX_COMPILER "$ENV{ROCM_PATH}/bin/hipcc" CACHE PATH "")
#set(GPU_TARGETS "gfx908" CACHE STRING "GPU targets to compile for")
set(SW4_HIP_FLAGS -DENABLE_HIP=1 -isystem $ENV{ROCM_PATH}/hsa/include -isystem $ENV{ROCM_PATH}/hip/include -D__HIP_ARCH_GFX908__=1 -O3 -x hip -Wno-inconsistent-missing-override --amdgpu-target=gfx908  CACHE STRING "")

