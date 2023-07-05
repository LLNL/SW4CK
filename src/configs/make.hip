#use make -f Makefile.hipcc

LINKER = hipcc
CXX = $(PREP) hipcc
RAJA_LOCATION = /g/g17/vargas45/mySW4CK/RAJA/install_lc_toss4-hipcc-5.4.0-gfx90a

HIP_ROOT_DIR = $(HIP_PATH)

HSA_ROOT_DIR = ${ROCM_PATH}/hsa

GCC_LINK_LINE = -L $(HIP_ROOT_DIR)/lib -L $(HSA_ROOT_DIR)/lib -lhsa-runtime64  -lm -lstdc++ -ldl


MORE_FLAGS = -I${ROCM_PATH}/include -O3 -x hip -Winconsistent-missing-override --offload-arch=gfx90a -DNO_RAJA=0 -Wall -Wno-unknown-pragmas -Rpass-analysis=kernel-resource-usage


LINKFLAGS =  -std=c++14  --offload-arch=gfx90a  -fgpu-rdc -Xoffload-linker --whole-archive -lRAJA -lcamp
EXTRA_CXX_FLAGS =  -O3  -x hip  -std=c++14 -I$(RAJA_LOCATION)/include  -DENABLE_HIP $(MORE_FLAGS) -Winconsistent-missing-override 

EXTRA_CXX_FLAGS2 =  -O3  -std=c++14 -I $(CUDA_HOME)/include -I$(RAJA_LOCATION)/include  -DRAJA_USE_RESTRICT_PTR -DCUDA_CODE -DENABLE_HIP   $(MORE_FLAGS)

EXTRA_LINK_FLAGS = -L $(RAJA_LOCATION)/lib $(GCC_LINK_LINE) 

ifeq ($(best),yes)
amd_unroll_fix=yes
magic_sync=yes
endif

ifeq ($(amd_unroll_fix),yes)
MORE_FLAGS+=  -DAMD_UNROLL_FIX=1
endif

ifeq ($(magic_sync),yes)
MORE_FLAGS+=  -DMAGIC_SYNC=1
endif

ifeq ($(register_count),yes)
MORE_FLAGS+=  --save-temps
else
MORE_FLAGS+= -fgpu-rdc
endif

