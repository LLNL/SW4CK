#use make -f Makefile.hipcc
LINKER = hipcc
CXX = $(PREP) hipcc
RAJA_LOCATION = /home/users/coe0145/RAJA/Dev2/raja/install_908

HIP_ROOT_DIR = $(HIP_PATH)

HSA_ROOT_DIR = ${ROCM_PATH}/hsa

#GCC_LINK_LINE = -L $(HIP_ROOT_DIR)/lib -L $(HSA_ROOT_DIR)/lib -lhsa-runtime64  -lm -lstdc++ -ldl


MORE_FLAGS = -I${ROCM_PATH}/include -isystem ${ROCM_PATH}/llvm/lib/clang/12.0.0/include/.. -isystem ${ROCM_PATH}/hsa/include -isystem ${ROCM_PATH}/hip/include -D__HIP_ARCH_GFX908__=1 --cuda-gpu-arch=gfx908 -O3 -x hip -Winconsistent-missing-override --amdgpu-target=gfx908 --save-temps -fgpu-rdc -DNO_RAJA=1 -Wall -Wno-unknown-pragmas -I${ROCM_PATH}/roctracer/include
LINKFLAGS =  -std=c++11  --amdgpu-target=gfx908 -fgpu-rdc
EXTRA_CXX_FLAGS =  -O3  -x hip  -std=c++11 -I$(RAJA_LOCATION)/include  -DENABLE_HIP $(MORE_FLAGS) -Winconsistent-missing-override 

EXTRA_CXX_FLAGS2 =  -O3  -std=c++11 -I $(CUDA_HOME)/include -I$(RAJA_LOCATION)/include  -DRAJA_USE_RESTRICT_PTR -DCUDA_CODE -DENABLE_HIP   $(MORE_FLAGS)

EXTRA_LINK_FLAGS = -L $(RAJA_LOCATION)/lib $(GCC_LINK_LINE) -L ${ROCM_PATH}roctracer/lib -lroctracer64
