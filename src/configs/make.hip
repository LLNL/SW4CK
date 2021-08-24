#use make -f Makefile.hipcc
LINKER = hipcc
CXX = $(PREP) hipcc
RAJA_LOCATION = /home/users/coe0145/RAJA/Dev2/raja/install_908

HIP_ROOT_DIR = $(HIP_PATH)

HSA_ROOT_DIR = ${ROCM_PATH}/hsa



MORE_FLAGS =  -O3 -x hip -Winconsistent-missing-override --amdgpu-target=gfx908 --save-temps -fgpu-rdc -DNO_RAJA=1 -Wall -Wno-unknown-pragmas -I${ROCM_PATH}/roctracer/include
LINKFLAGS =  -std=c++11  --amdgpu-target=gfx908 -fgpu-rdc
EXTRA_CXX_FLAGS =  -O3  -x hip  -std=c++11 -I$(RAJA_LOCATION)/include  -DENABLE_HIP $(MORE_FLAGS) -Winconsistent-missing-override 

EXTRA_CXX_FLAGS2 =  -O3  -std=c++11 -I$(RAJA_LOCATION)/include  -DRAJA_USE_RESTRICT_PTR  -DENABLE_HIP   $(MORE_FLAGS)

EXTRA_LINK_FLAGS = -L $(RAJA_LOCATION)/lib -lroctracer64
