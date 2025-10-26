cmake_minimum_required(VERSION 3.28)
project(amx_project CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE MODULE_FILES "${CMAKE_SOURCE_DIR}/modules/*.cppm")

add_executable(amx_app)
target_sources(amx_app PRIVATE main.cpp)

if(MODULE_FILES)
    target_sources(amx_app PUBLIC FILE_SET CXX_MODULES FILES ${MODULE_FILES})
endif()

target_compile_options(amx_app PRIVATE -stdlib=libc++ -mamx-tile -mamx-int8 -mamx-bf16 -march=native)
target_link_options(amx_app PRIVATE -stdlib=libc++ -lc++abi)
target_link_libraries(amx_app PRIVATE numa)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(amx_app PRIVATE -O3)
endif()

# NUMA test executable
add_executable(numatest)
target_sources(numatest PRIVATE numatest.cpp)

if(MODULE_FILES)
    target_sources(numatest PUBLIC FILE_SET CXX_MODULES FILES ${MODULE_FILES})
endif()

target_compile_options(numatest PRIVATE -stdlib=libc++ -mamx-tile -mamx-int8 -mamx-bf16 -march=native)
target_link_options(numatest PRIVATE -stdlib=libc++ -lc++abi)
target_link_libraries(numatest PRIVATE numa)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(numatest PRIVATE -O3)
endif()
