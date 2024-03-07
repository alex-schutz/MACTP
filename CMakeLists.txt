cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
project(MACTP)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")  
#   workaround for changed linker
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-ld_classic")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fPIC")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/MCJESP/Include)

add_subdirectory(src/mactp)
add_subdirectory(src/examples)
add_subdirectory(MCJESP)

set(MCJESP_SOURCE 
MCJESP/src/ParserDecPOMDPSparse.cpp
MCJESP/src/ParserPOMDPSparse.cpp
MCJESP/src/SimModel.cpp
MCJESP/src/ExtendedGenerativeModel.cpp
MCJESP/src/BeliefParticles.cpp
MCJESP/src/Planner.cpp
MCJESP/src/MCJESP.cpp
MCJESP/src/FSC.cpp
MCJESP/src/TreeNode.cpp
MCJESP/src/Utils.cpp
)
add_library(mcjesp_library SHARED ${MCJESP_SOURCE})
