#set(CMAKE_VERBOSE_MAKEFILE ON)
cmake_minimum_required(VERSION 3.18)
project(SW4CK VERSION 1.0)

set(BLT_CXX_STD "c++14" CACHE STRING "")
include(blt/SetupBLT.cmake)

configure_file(SW4CKConfig.h.in SW4CKConfig.h)


# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++14" CACHE STRING "")
endif()

option(ENABLE_RAJA "ENABLE_RAJA" OFF)
if (ENABLE_RAJA)
	set (ENABLE_TESTS Off CACHE BOOL "" FORCE)
	set (ENABLE_EXAMPLES Off CACHE BOOL "" FORCE)
	set (ENABLE_REPRODUCERS Off CACHE BOOL "" FORCE)
	set (ENABLE_EXERCISES Off CACHE BOOL "" FORCE)
	set (ENABLE_DOCUMENTATION Off CACHE BOOL "" FORCE)
	set (ENABLE_BENCHMARKS Off CACHE BOOL "" FORCE)
if (DEFINED RAJA_DIR)
  find_package(RAJA REQUIRED)
  blt_print_target_properties(TARGET RAJA)
else ()
  add_subdirectory(RAJA)
endif ()
   #add_subdirectory(raja)
   #list(APPEND EXTRA_LIBS raja)
   #list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/raja")
   #find_package(RAJA)
endif()
add_subdirectory(src)
