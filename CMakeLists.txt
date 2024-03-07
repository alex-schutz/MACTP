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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src/mactp)
add_subdirectory(src/examples)
