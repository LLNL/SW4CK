set(ENABLE_CUDA ON CACHE BOOL "")

set(GCC_VERSION "gcc-8.3.1" CACHE STRING "")
set(GCC_HOME "/usr/tce/packages/gcc/${GCC_VERSION}")

set(CMAKE_C_COMPILER   "${GCC_HOME}/bin/gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${GCC_HOME}/bin/g++" CACHE PATH "")
set(BLT_CXX_STD "c++14" CACHE STRING "")

#------------------------------------------------------------------------------
# CUDA support
#------------------------------------------------------------------------------


set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_HOME}" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE PATH "")

#set(_cuda_arch "sm_70")
set(CMAKE_CUDA_FLAGS "-restrict --expt-extended-lambda " CACHE STRING "")

set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" )

# nvcc does not like gtest's 'pthreads' flag
set(gtest_disable_pthreads ON CACHE BOOL "")

