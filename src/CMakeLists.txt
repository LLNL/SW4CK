 
if (ENABLE_CUDA)
blt_add_executable(NAME sw4ck 
  SOURCES curvitest.C curvilinear4sgc.C Events.C 
  DEPENDS_ON cuda)
    blt_add_target_compile_flags(TO sw4ck FLAGS -DENABLE_CUDA=1)
    set_property(TARGET sw4ck PROPERTY CUDA_ARCHITECTURES 70)
    if (ENABLE_SPILL_WARNS)
    	blt_add_target_compile_flags(TO sw4ck FLAGS -Xptxas -v,--warn-on-spills)
    endif()
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


if (ENABLE_HIP)
	blt_add_target_compile_flags(TO sw4ck FLAGS ${SW4_HIP_FLAGS})
    if (ENABLE_SPILL_WARNS)
    	blt_add_target_compile_flags(TO sw4ck FLAGS --save-temps -g -ggdb -fno-gpu-rdc)
    else()
    	blt_add_target_compile_flags(TO sw4ck FLAGS -fgpu-rdc)
    endif()
    set(CMAKE_EXE_LINKER_FLAGS  "--amdgpu-target=gfx908 -fgpu-rdc" )
endif()

