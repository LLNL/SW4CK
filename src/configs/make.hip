# modules loaded should look like
#1) texlive/2016   2) StdEnv (S)   3) opt   4) rocm/3.10.0   5) openmpi-gnu/4.0   6) gnu/8.2.1
#use make -f Makefile.hipcc
proj = no
FC = gfortran
LINKER = hipcc
CXX = $(PREP) hipcc
ckernel=yes 
openmp=no
raja_cuda=yes
RAJA_LOCATION = /home/users/coe0145/RAJA/Dev2/raja/install_rocm3.9
RAJA_LOCATION = /home/users/coe0145/RAJA/Dev2/raja/install_906

HIP_ROOT_DIR = $(HIP_PATH)

HSA_ROOT_DIR = /opt/rocm-3.10.0/hsa

GCC_LINK_LINE = -L $(HIP_ROOT_DIR)/lib -L $(HSA_ROOT_DIR)/lib -lhsa-runtime64  -lm -lstdc++ -ldl

MPI_HOME =  /opt/cray/pe/cray-mvapich2/2.3.4/infiniband/cray/10.0
MPI_HOME = /cm/shared/apps/mpich/ge/gcc/64/3.3.2
MPI_HOME = /opt/cray/pe/cray-mvapich2_nogpu/2.3.4/infiniband/cray/10.0
MPI_HOME = /opt/cray/pe/cray-mvapich2_nogpu/2.3.3/infiniband/cray/10.0

MORE_FLAGS = -I$(MPI_HOME)/include -I${ROCM_PATH}/include -isystem /opt/rocm-3.10.0/llvm/lib/clang/11.0.0/include/.. -isystem /opt/rocm-3.10.0/hsa/include -isystem /opt/rocm-3.10.0/hip/include -D__HIP_ARCH_GFX906__=1 --cuda-gpu-arch=gfx906 -O3 -x hip -Winconsistent-missing-override --amdgpu-target=gfx906 --save-temps -fgpu-rdc -DNO_RAJA=1 -Wall -Wno-unknown-pragmas
LINKFLAGS =  -std=c++11  --amdgpu-target=gfx906 -fgpu-rdc
EXTRA_CXX_FLAGS =  -O3  -x hip  -std=c++11 -I$(RAJA_LOCATION)/include  -DENABLE_HIP $(MORE_FLAGS) -Winconsistent-missing-override 

EXTRA_CXX_FLAGS2 =  -O3  -std=c++11 -I $(CUDA_HOME)/include -I$(UMPIRE_HOME)/include -I$(RAJA_LOCATION)/include  -DUSE_NVTX -DRAJA_USE_CUDA -DSW4_CROUTINES -DRAJA_USE_RESTRICT_PTR -DCUDA_CODE -DENABLE_HIP   $(MORE_FLAGS)

EXTRA_LINK_FLAGS = -L $(RAJA_LOCATION)/lib $(GCC_LINK_LINE) 
