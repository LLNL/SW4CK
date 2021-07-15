#use make -f Makefile.CC
LINKER = /opt/cray/pe/craype/2.7.8/bin/CC
CXX = $(PREP) /opt/cray/pe/craype/2.7.8/bin/CC
RAJA_LOCATION = /home/users/coe0145/RAJA/Dev2/raja/install_906

HIP_ROOT_DIR = $(HIP_PATH)

HSA_ROOT_DIR = ${ROCM_PATH}/hsa

#GCC_LINK_LINE = -L $(HIP_ROOT_DIR)/lib -L $(HSA_ROOT_DIR)/lib -lhsa-runtime64  -lm -lstdc++ -ldl


# --save-temps aborts compilation. -fno-cray fixes this. Incorrect and slow results with CC. Correct slower results wth CC -fno-cray
# Add -fsave-loopmark for optimization report
MORE_FLAGS = -I${ROCM_PATH}/include -Ofast  -x hip -Winconsistent-missing-override  -Wall -Wno-unknown-pragmas --rocm-path=${ROCM_PATH} --offload-arch=gfx908 -fsave-loopmark -fno-cray

LINKFLAGS =  -std=c++11 

EXTRA_CXX_FLAGS =  -I$(RAJA_LOCATION)/include  -DENABLE_HIP $(MORE_FLAGS) -DNO_RAJA=1


EXTRA_LINK_FLAGS = -L $(RAJA_LOCATION)/lib -lamdhip64
