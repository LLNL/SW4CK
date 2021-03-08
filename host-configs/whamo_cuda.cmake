set(ENABLE_CUDA ON CACHE BOOL "")
# Use Clang compilers for C/C++
set(GCC_VERSION "gcc-9.2.1" CACHE STRING "")
set(GCC_HOME "/opt/rh/gcc-toolset-9/root/usr")

set(CMAKE_C_COMPILER   "${GCC_HOME}/bin/gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${GCC_HOME}/bin/g++" CACHE PATH "")
set(BLT_CXX_STD "c++11" CACHE STRING "")

#------------------------------------------------------------------------------
# CUDA support
#------------------------------------------------------------------------------


set(CUDA_TOOLKIT_ROOT_DIR "/opt/toss/cudatoolkit/11.1" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE PATH "")


set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" )

# nvcc does not like gtest's 'pthreads' flag
set(gtest_disable_pthreads ON CACHE BOOL "")

#set(SW4CK_ENABLE_RAJA ${ENABLE_RAJA})

