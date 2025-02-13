cmake_minimum_required(VERSION 3.16.3)

project(CIAO-ML-MODULE VERSION 1.0 LANGUAGES Fortran C CXX)

# CMP0074: find_package() uses <PackageName>_ROOT variables.
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
#add_compile_options("-fsanitize=address")
#add_link_options("-fsanitize=address")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(WITH_AIX "Enable AIxeleratorService as inference strategy." OFF)
if(WITH_AIX)
  add_compile_options("-DWITH_AIX")
  set(AIX_BUILD_DIR ${CMAKE_SOURCE_DIR}/extern/aixeleratorservice/BUILD)
  set(PHYDLL_BUILD_DIR ${CMAKE_SOURCE_DIR}/extern/phydll/BUILD)
endif()

option(WITH_PHYDLL "Enable PhyDLL as inference strategy." OFF)
if(WITH_PHYDLL)
  add_compile_options("-DWITH_PHYDLL")
endif()


if(NOT ${WITH_AIX} AND NOT ${WITH_PHYDLL})
  message(FATAL_ERROR "No inference strategy chosen! At least one inference strategy is required. Possible choices: WITH_AIX=ON or WITH_PHYDLL=ON")
endif()

find_package(MPI REQUIRED)
find_package(h5fortran REQUIRED)

option(WITH_SCOREP "Build with Score-P instrumentation for profiling/tracing" OFF)
if(WITH_SCOREP)
  find_package(Scorep REQUIRED)
  add_compile_options("-DSCOREP")
  message(STATUS "Fortran ML Interface will be build with Score-P intrumentation!")
  set(AIX_BUILD_DIR ${CMAKE_SOURCE_DIR}/extern/aixeleratorservice/BUILD-SCOREP)
  set(PHYDLL_BUILD_DIR ${CMAKE_SOURCE_DIR}/extern/phydll/BUILD-SCOREP)
endif()

option(WITH_SCOREP_MANUAL "Build with Score-P instrumentation for profiling/tracing" OFF)
if(WITH_SCOREP_MANUAL)
  message(STATUS "Fortran ML Interface will be build with Score-P intrumentation!")
  set(AIX_BUILD_DIR ${CMAKE_SOURCE_DIR}/extern/aixeleratorservice/BUILD-SCOREP)
  set(PHYDLL_BUILD_DIR ${CMAKE_SOURCE_DIR}/extern/phydll/BUILD-SCOREP)
endif()

# find package TensorFlow
message(STATUS "Tensorflow_DIR = ${Tensorflow_DIR}")
set(TENSORFLOW_PROTOBUF_INCLUDE ${Tensorflow_Python_DIR}/include/)
set(TENSORFLOW_INCLUDE_DIRS ${Tensorflow_DIR}/include ${TENSORFLOW_PROTOBUF_INCLUDE})
message(STATUS "TENSORFLOW_INCLUDE_DIRS = ${TENSORFLOW_INCLUDE_DIRS}")
set(TENSORFLOW_LIB_DIR ${Tensorflow_DIR}/lib)
message(STATUS "TENSORFLOW_LIB_DIR = ${TENSORFLOW_LIB_DIR}")
set(TENSORFLOW_LIBRARIES ${TENSORFLOW_LIB_DIR}/libtensorflow.so ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so)
#/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/protobuf/3.19.4-GCCcore-11.3.0/lib64/libprotobuf.so
#set(TENSORFLOW_LIBRARIES /home/rwth0792/AI-Frameworks/libtensorflow-cpu-linux-x86_64-2.10.0-CUSTOM/lib/libtensorflow_cc.so ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so)

message(STATUS "TENSORFLOW_LIBRARIES = ${TENSORFLOW_LIBRARIES}")

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod)

add_compile_options("-free")
add_compile_options("-g")

add_subdirectory(src)
add_subdirectory(test)