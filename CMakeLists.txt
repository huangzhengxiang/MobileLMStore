cmake_minimum_required(VERSION 3.19)
project(mobile_lm_store)

set(CMAKE_CXX_STANDARD 17)

option(PYBIND "Need to pybind a module for testing" OFF)

add_compile_options(-O3)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "Enabling AVX2")
    add_definitions(-D__AVX2__)
    add_compile_options(-mavx2)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    message(STATUS "Enabling NEON")
    add_definitions(-D__NEON__)
    add_compile_options(-mfpu=neon)
endif()

message(STATUS "Check sqlite...")
execute_process(
    COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/vendor.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)

include_directories(
                    # sqlite include
                    ${CMAKE_CURRENT_LIST_DIR}/vendor/
                    ${CMAKE_CURRENT_LIST_DIR}/include/
                    )

FILE(GLOB LMStore_Srcs "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp"
                       "${CMAKE_CURRENT_LIST_DIR}/src/ops/*.cpp")
add_library(LMStore SHARED
            ${CMAKE_CURRENT_LIST_DIR}/vendor/sqlite3.c
            ${LMStore_Srcs})
target_compile_definitions(LMStore PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
add_definitions(-DSQLITE_CORE)

if (PYBIND) 
    execute_process(
        COMMAND python -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE Torch_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    list(APPEND CMAKE_PREFIX_PATH ${Torch_DIR})
    find_package(Torch REQUIRED)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
    find_package(pybind11 REQUIRED)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)

    add_definitions(-DNEED_PYBIND)
    if(pybind11_FOUND)
        pybind11_add_module(pylmstore ${CMAKE_CURRENT_LIST_DIR}/src/pybind/pybind.cpp)
        target_link_libraries(pylmstore PRIVATE LMStore ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
        target_include_directories(pylmstore PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include/
                                                     ${TORCH_INCLUDE_DIRS})
        target_compile_definitions(pylmstore PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    endif()
endif()