 
#add_executable(sw4ck curvitest.C)
if (ENABLE_CUDA)
set(_cuda_arch "sm_70")
set(CMAKE_CUDA_FLAGS "-restrict  -arch=${_cuda_arch} -std=c++11 --expt-extended-lambda " CACHE STRING "" FORCE)
blt_add_executable(NAME sw4ck 
  SOURCES curvitest.C curvilinear4sgc.C Events.C 
  DEPENDS_ON cuda)
else()
blt_add_executable(NAME sw4ck
  SOURCES curvitest.C curvilinear4sgc.C Events.C)
endif()

target_include_directories(sw4ck PUBLIC
                          "${PROJECT_BINARY_DIR}"
                         )

#target_include_directories(sw4ck PUBLIC
#                           "${PROJECT_BINARY_DIR}"
#                           ${EXTRA_INCLUDES}
#                           )

if (ENABLE_RAJA)
    add_compile_options(-DENABLE_RAJA=1)
    target_link_libraries(sw4ck PUBLIC RAJA)
else()
    blt_add_target_compile_flags(TO sw4ck FLAGS -DNO_RAJA=1)
endif()

if (ENABLE_CUDA)
    blt_add_target_compile_flags(TO sw4ck FLAGS -DENABLE_CUDA=1)
    set_property(TARGET sw4ck PROPERTY CUDA_ARCHITECTURES 70)
    if (ENABLE_SPILL_WARNS)
    	blt_add_target_compile_flags(TO sw4ck FLAGS -Xptxas -v,--warn-on-spills)
    endif()
endif()

if (ENABLE_HIP)
    blt_add_target_compile_flags(TO sw4ck FLAGS -DENABLE_HIP=1)
    blt_add_target_compile_flags(TO sw4ck FLAGS -isystem /opt/rocm-4.0.1/hsa/include -isystem /opt/rocm-4.0.1/hip/include -D__HIP_ARCH_GFX906__=1 --cuda-gpu-arch=gfx906 -O3 -x hip -Wno-inconsistent-missing-override --amdgpu-target=gfx906 )
    if (ENABLE_SPILL_WARNS)
    	blt_add_target_compile_flags(TO sw4ck FLAGS --save-temps -fno-gpu-rdc)
    else()
    	blt_add_target_compile_flags(TO sw4ck FLAGS -fgpu-rdc)
    endif()
#set(CMAKE_HIP_FLAGS "-isystem /opt/rocm-4.0.1/hsa/include -isystem /opt/rocm-4.0.1/hip/include -D__HIP_ARCH_GFX906__=1 --cuda-gpu-arch=gfx906 -O3 -x hip -Wno-inconsistent-missing-override --amdgpu-target=gfx906 --save-temps -fgpu-rdc" CACHE STRING "" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS  "--amdgpu-target=gfx906 -fgpu-rdc" )
endif()

