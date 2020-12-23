include configs/make.hip
builddir=./
ifdef EXTRA_CXX_FLAGS
   CXXFLAGS2 += $(EXTRA_CXX_FLAGS2)
   CXXFLAGS += $(EXTRA_CXX_FLAGS)
endif

ifdef EXTRA_LINK_FLAGS
   linklibs += $(EXTRA_LINK_FLAGS)
endif

OBJSW4 = 

OBJ  = curvitest.o curvilinear4sgc.o Events.o


# prefix object files with build directory
FSW4 = $(addprefix $(builddir)/,$(OBJSW4))
FOBJ = $(addprefix $(builddir)/,$(OBJ)) $(addprefix $(builddir)/,$(QUADPACK))

# prefix
sw4ck: $(FSW4) $(FOBJ)
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
 
$(builddir)/%.o:./%.C
	 cd $(builddir); $(CXX) $(CXXFLAGS) -c $< 

clean:
	rm sw4ck *.o *.bc *.s *.cui

format:
	clang-format -style Google -i src/*.C
	clang-format -style Google -i src/*.h

tags:
	etags -o src/TAGS src/*.C src/*.h 
