cmake_minimum_required(VERSION 3.28)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")
set(CMAKE_CXX_MODULE_STD 1)
project(amx_project CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)  # Try with ON

file(GLOB_RECURSE MODULE_FILES "${CMAKE_SOURCE_DIR}/modules/*.cppm")
add_executable(amx_app)
target_sources(amx_app PRIVATE main.cpp)

if(MODULE_FILES)
    target_sources(amx_app PUBLIC FILE_SET CXX_MODULES FILES ${MODULE_FILES})
endif()

target_compile_options(amx_app PRIVATE -stdlib=libc++ -mamx-tile -mamx-int8 -mamx-bf16 -march=native)
target_link_options(amx_app PRIVATE -stdlib=libc++ -lc++abi)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(amx_app PRIVATE -O3)
endif()
