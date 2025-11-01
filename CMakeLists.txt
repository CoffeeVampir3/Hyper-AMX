cmake_minimum_required(VERSION 3.28)
project(amx_project CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE MODULE_FILES "${CMAKE_SOURCE_DIR}/modules/*.cppm")
file(GLOB_RECURSE TEST_FILES "${CMAKE_SOURCE_DIR}/tests/*.cppm")

add_executable(amx_app)
target_sources(amx_app PRIVATE main.cpp)

set(ALL_MODULES ${MODULE_FILES} ${TEST_FILES} ${BENCHMARK_FILES})
if(ALL_MODULES)
    target_sources(amx_app PUBLIC FILE_SET CXX_MODULES FILES ${ALL_MODULES})
endif()

target_compile_options(amx_app PRIVATE
    -stdlib=libc++
    -mamx-tile
    -mamx-int8
    -mamx-bf16
    -march=native
    -fsanitize=address
    -fno-omit-frame-pointer
)
target_link_options(amx_app PRIVATE -stdlib=libc++ -lc++abi -fsanitize=address)
target_link_libraries(amx_app PRIVATE numa)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(amx_app PRIVATE -O3)
endif()
