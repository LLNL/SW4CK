#-----------------------------------------------------------------------
# Usage:
# make sw4 [debug=yes/no] [prec=single/double] [fortran=yes/no] [openmp=yes/no]
#   Default is: debug=no prec=double fortran=yes openmp=yes
# This Makefile asumes that the following environmental variables have been assigned:
# etree = [yes/no]
# proj = [yes/no]
# CXX = C++ compiler
# FC  = Fortran-77 compiler
# SW4ROOT = path to third party libraries (used when etree=yes). 
#
# Note: third party libraries should have include files in $(SW4ROOT)/include, libraries in $(SW4ROOT)/lib
#
# The following environmental variables are optional:
# EXTRA_CXX_FLAGS  = additional c++ compiler flags
# EXTRA_FORT_FLAGS = additional fortran compiler flags
# EXTRA_LINK_FLAGS = additional arguments to the linker
#
# There a three ways of assigning the environmental variables:
# 1) Set them in your .cshrc (or similar) file
# 2) Set them in the configs/make.inc file
# 3) Set them on the command line before running make
#
#-----------------------------------------------------------------------
# Do not make changes below this line (don't blame us if you do!)
#-----------------------------------------------------------------------

ifeq ($(debug),yes)
   optlevel = DEBUG
else
   debug := "no"
   optlevel = OPTIMIZE
endif

ifeq ($(optlevel),DEBUG)
   FFLAGS    = -g -O2
   CXXFLAGS  = -g -I../src -DBZ_DEBUG
   CFLAGS    = -g
else
   FFLAGS   = -O2
# AP (160419) Note that cmake uses -O3 instead of -O for CXX and C
   CXXFLAGS =  -I../src
   CXXFLAGS2 =  -I../src	
   CFLAGS   = -O3 
endif

fullpath := $(shell pwd)

HOSTNAME := $(shell hostname)
UNAME := $(shell uname)

debugdir := debug
optdir := ./

SW4INC    = $(SW4ROOT)/include
SW4LIB    = $(SW4ROOT)/lib

emptystring := ""
foundincfile := $(emptystring)

# check if the file configs/make.inc exists?
USERMAKE := $(shell if test -r configs/make.inc; then echo "configs/make.inc"; fi)

ifeq ($(USERMAKE),configs/make.inc)
  include configs/make.inc
  foundincfile := "configs/make.inc"
else

# if configs/make.inc does not exist
  ifeq ($(UNAME),Darwin)
  # for Anders' old laptop
    ifeq ($(findstring yorkville,$(HOSTNAME)),yorkville)
      include configs/make.yorkville
      foundincfile := "configs/make.yorkville"
  # for Anders' new laptop
    else ifeq ($(findstring fourier,$(HOSTNAME)),fourier)
      include configs/make.fourier
      foundincfile := "configs/make.fourier"
    endif
  endif
  
  # put the variables in the configs/make.xyz file
  ifeq ($(UNAME),Linux)
  # For Cab at LC
    ifeq ($(findstring cab,$(HOSTNAME)),cab)
      include configs/make.cab
      foundincfile := "configs/make.cab"
# object code goes in machine specific directory on LC
      debugdir := debug_cab
      optdir := optimize_cab
# For Quartz at LC
    else ifeq ($(findstring quartz,$(HOSTNAME)),quartz)
      include configs/make.quartz
      foundincfile := "configs/make.quartz"
# Cori @ NERSC
    else ifeq ($(findstring cori,$(HOSTNAME)),cori)
      include configs/make.cori
      foundincfile := "configs/make.cori"
  # for Bjorn's tux box
    else ifeq ($(findstring tux337,$(HOSTNAME)),tux337)
      include configs/make.tux337
      foundincfile := "configs/make.tux337"
  # for Anders' tux box
    else ifeq ($(findstring tux355,$(HOSTNAME)),tux355)
      include configs/make.tux355
      foundincfile := "configs/make.tux355"
  # For Edison at NERSC
    else ifeq ($(findstring edison,$(HOSTNAME)),edison)
      include configs/make.edison
      foundincfile := "configs/make.edison"
  # For Vulcan at LC
    else ifeq ($(findstring vulcan,$(HOSTNAME)),vulcan)
      include configs/make.bgq
      foundincfile := "configs/make.bgq"
# object code goes in machine specific directory on LC
      debugdir := debug_vulcan
      optdir := optimize_vulcan
    endif
  endif

endif

ifdef EXTRA_CXX_FLAGS
   CXXFLAGS2 += $(EXTRA_CXX_FLAGS2)
   CXXFLAGS += $(EXTRA_CXX_FLAGS)
endif

ifdef EXTRA_FORT_FLAGS
   FFLAGS += $(EXTRA_FORT_FLAGS)
endif

ifeq ($(etree),yes)
   CXXFLAGS += -DENABLE_ETREE -DENABLE_PROJ4 -I$(SW4INC)
   linklibs += -L$(SW4LIB) -lcencalvm -lproj
else ifeq ($(proj),yes)
   CXXFLAGS += -DENABLE_PROJ4 -I$(SW4INC)
   linklibs += -L$(SW4LIB) -lproj
   etree := "no"
else
   etree := "no"
   proj  := "no"
endif

# openmp=yes is default
ifeq ($(openmp),no)
   CXXFLAGS += 
else
   debugdir := $(debugdir)_mp
   optdir   := $(optdir)_mp
   CXXFLAGS += -fopenmp
   FFLAGS   += -fopenmp
endif

# fortran=no is default
ifeq ($(fortran),yes)
   debugdir := $(debugdir)_fort
   optdir   := $(optdir)_fort
   CXXFLAGS += -DSW4_NOC
else
   fortran := "no"
endif

ifeq ($(prec),single)
   debugdir := $(debugdir)_sp
   optdir   := $(optdir)_sp
   CXXFLAGS += -I../src/float
else
   CXXFLAGS += -I../src/double
   CXXFLAGS2 += -I../src/double	
endif

ifdef EXTRA_LINK_FLAGS
   linklibs += $(EXTRA_LINK_FLAGS)
endif

ifeq ($(optlevel),DEBUG)
   builddir = $(debugdir)
else
   builddir = $(optdir)
endif

# sw4 main program (kept separate)
OBJSW4 = 

OBJ  = curvitest.o curvilinear4sgc.o Events.o


# OpenMP & C-version of the F-77 routine curvilinear4sg() is in rhs4sgcurv.o

# prefix object files with build directory
FSW4 = $(addprefix $(builddir)/,$(OBJSW4))
FOBJ = $(addprefix $(builddir)/,$(OBJ)) $(addprefix $(builddir)/,$(QUADPACK))

# prefix
ifeq ($(raja_cuda),yes)
x: $(FSW4) $(FOBJ)
	@echo "*** Configuration file: '" $(foundincfile) "' ***"
	@echo "********* User configuration variables **************"
	@echo "debug=" $(debug) " proj=" $(proj) " etree=" $(etree) " SW4ROOT"= $(SW4ROOT) 
	@echo "CXX=" $(CXX) "EXTRA_CXX_FLAGS"= $(EXTRA_CXX_FLAGS)
	@echo "FC=" $(FC) " EXTRA_FORT_FLAGS=" $(EXTRA_FORT_FLAGS)
	@echo "EXTRA_LINK_FLAGS"= $(EXTRA_LINK_FLAGS)
	@echo "******************************************************"
	cd $(builddir); $(LINKER) $(LINKFLAGS) -o $@ $(OBJ) $(linklibs)
# test: linking with openmp for the routine rhs4sgcurv.o
#	cd $(builddir); $(CXX) $(CXXFLAGS) -qopenmp -o $@ main.o $(OBJ) $(QUADPACK) $(linklibs)
	@echo "*** Build directory: " $(builddir) " ***"
else
sw4: $(FSW4) $(FOBJ)
	@echo "*** Configuration file: '" $(foundincfile) "' ***"
	@echo "********* User configuration variables **************"
	@echo "debug=" $(debug) " proj=" $(proj) " etree=" $(etree) " SW4ROOT"= $(SW4ROOT) 
	@echo "CXX=" $(CXX) "EXTRA_CXX_FLAGS"= $(EXTRA_CXX_FLAGS)
	@echo "FC=" $(FC) " EXTRA_FORT_FLAGS=" $(EXTRA_FORT_FLAGS)
	@echo "EXTRA_LINK_FLAGS"= $(EXTRA_LINK_FLAGS)
	@echo "******************************************************"
	cd $(builddir); $(LINKER) $(LINKFLAGS) -o $@ main.o $(OBJ) $(QUADPACK) $(linklibs)
# test: linking with openmp for the routine rhs4sgcurv.o
#	cd $(builddir); $(CXX) $(CXXFLAGS) -qopenmp -o $@ main.o $(OBJ) $(QUADPACK) $(linklibs)
	@echo "*** Build directory: " $(builddir) " ***"
endif
# tarball
sw4-v1.1.tgz:  $(FSW4) $(FOBJ)
	rm -rf sw4-v1.1
	mkdir sw4-v1.1
	cp -r src configs tools examples doc Makefile wave.txt CMakeLists.txt INSTALL.txt LICENSE.txt README.txt sw4-v1.1
	tar czf $@ sw4-v1.1
	rm -rf sw4-v1.1 

# test
$(builddir)/rhs4sgcurv.o:src/rhs4sgcurv.C
	cd $(builddir); $(CXX) $(CXXFLAGS) -c ../$<
#	cd $(builddir); $(CXX) $(CXXFLAGS) -qopenmp -c ../$<


$(builddir)/version.o:src/version.C .FORCE
	cd $(builddir); $(CXX) $(CXXFLAGS) -DEW_MADEBY=\"$(USER)\"  -DEW_OPT_LEVEL=\"$(optlevel)\" -DEW_COMPILER=\""$(shell which $(CXX))"\" -DEW_LIBDIR=\"${SW4LIB}\" -DEW_INCDIR=\"${SW4INC}\" -DEW_HOSTNAME=\""$(shell hostname)"\" -DEW_WHEN=\""$(shell date)"\" -c ../$<

# having version.o depend on .FORCE has the effect of always building version.o
.FORCE:

$(builddir)/%.o:src/%.f
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(FC) $(FC_FIXED_FORMAT) $(FFLAGS) -c ../$<

$(builddir)/%.o:src/%.f90
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(FC) $(FFLAGS) -c ../$<

$(builddir)/%.o:src/quadpack/%.f
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(FC) $(FC_FIXED_FORMAT) $(FFLAGS) -c ../$<

$(builddir)/%.o:./%.C
	/bin/mkdir -p $(builddir)
	 cd $(builddir); $(CXX) $(CXXFLAGS) -c $< 

clean:
	/bin/mkdir -p $(optdir)
	/bin/mkdir -p $(debugdir)
	cd $(optdir); /bin/rm -f sw4 *.o; cd ../$(debugdir); /bin/rm -f sw4 *.o

# Special rule for the target test
test:
	echo "Running tests..."
	/opt/local/bin/ctest --force-new-ctest-process $(ARGS)
format:
	clang-format -style Google -i src/*.C
	clang-format -style Google -i src/*.h
ptest: 
	cd $(builddir); $(CXX) -O3 -std=c++11 --expt-extended-lambda -arch=sm_70 -I$(RAJA_LOCATION)/include -x cu -c -dc ../src/Policies.C
	cd $(builddir); nvcc -O3 -arch=sm_70 -dlink -o file_link.o Policies.o
	cd $(builddir); nvcc -O3 -arch=sm_70 -o p file_link.o Policies.o -L $(RAJA_LOCATION)/lib -lRAJA
tags:
	etags -o src/TAGS src/*.C src/*.h 
