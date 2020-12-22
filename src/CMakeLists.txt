cmake_minimum_required(VERSION 3.14)


project(SW4CK VERSION 1.0)

set(BLT_CXX_STD "c++11" CACHE STRING "")
include(blt/SetupBLT.cmake)

configure_file(SW4CKConfig.h.in SW4CKConfig.h)


# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++11" CACHE STRING "")
endif()

#add_executable(sw4ck curvitest.C)
if (ENABLE_CUDA)
blt_add_executable(NAME sw4ck 
  SOURCES curvitest.C curvilinear4sgc.C Events.C 
  DEPENDS_ON cuda )
else()
blt_add_executable(NAME sw4ck
  SOURCES curvitest.C curvilinear4sgc.C Events.C)
endif()

target_include_directories(sw4ck PUBLIC
                          "${PROJECT_BINARY_DIR}"
                         )
option(ENABLE_RAJA "ENABLE_RAJA" OFF)
if (ENABLE_RAJA)
    add_compile_options(-DENABLE_RAJA=1)
else()
    blt_add_target_compile_flags(TO sw4ck FLAGS -DNO_RAJA=1)
endif()

if (ENABLE_CUDA)
    blt_add_target_compile_flags(TO sw4ck FLAGS -DENABLE_CUDA=1)
endif()

if (ENABLE_HIP)
    blt_add_target_compile_flags(TO sw4ck FLAGS -DENABLE_HIP=1)
    blt_add_target_compile_flags(TO sw4ck FLAGS -isystem /opt/rocm-3.9.0/hsa/include -isystem /opt/rocm-3.9.0/hip/include -D__HIP_ARCH_GFX906__=1 --cuda-gpu-arch=gfx906 -O3 -x hip -Wno-inconsistent-missing-override --amdgpu-target=gfx906 --save-temps -fgpu-rdc )
    set(CMAKE_EXE_LINKER_FLAGS  "--amdgpu-target=gfx906 -fgpu-rdc" )
endif()

