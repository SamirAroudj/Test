#
# Copyright (C) 2020 by Author: Aroudj, Samir
# TU Darmstadt - Graphics, Capture and Massively Parallel Computing
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD 3-Clause license. See the License.txt file for details.
#

# project Reconstruction
# main CMakeLists.txt file
cmake_minimum_required(VERSION 2.8)

project(Test)

# Test executable
set(componentName Test)
set(appName TestApp)
set(componentPath ${PROJECT_SOURCE_DIR})
option(CUDA "Make use of CUDA." OFF)


# required C++ standard
# for gcc & nvcc
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!! CAUTION - DON'T FORGET TO UPDATE MSVC SETTINGS BELOW !!!!!!!!!!!!!!!!!!!!!!!!
set(CPP_STANDARD "-std=c++17")

# for gcc via CMake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CPP_STANDARD}")

# for MSVC
if (MSVC_VERSION GREATER_EQUAL "1900")
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("/std:c++latest" _cpp_latest_flag_supported)
    if (_cpp_latest_flag_supported)
        #add_compile_options("/std:c++latest")
        add_compile_options("/std:c++17")
    endif()
endif()

# common preprocessor flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DTEST_DEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DTEST_DEBUG")

if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fopenmp -Wall -Wextra -Wreorder -Wunused-variable -Wunused-but-set-variable -Wmaybe-uninitialized -Wreturn-type")
    set(QMAKE_CXXFLAGS "${QMAKE_CXXFLAGS} -fopenmp")
endif()
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

if (CUDA)
    include(Cuda.cmake.txt)
endif()

# CMake files
set(cmakeFiles
    ${componentPath}/CMakeLists.txt
)

# find source code
set(testPath ${componentPath})
file(GLOB headerFiles "${testPath}/*.h")
file(GLOB testCPPFiles "${testPath}/*.cpp")
file(GLOB testCudaFiles "${testPath}/*.cu")
file(GLOB sourceFiles ${testCPPFiles} ${testCudaFiles})

# get all file groups together
set(sourceCode
    ${cmakeFiles}
    ${generatedHeaderFiles}
    ${generatedSourceFiles}
    ${headerFiles}
    ${sourceFiles}
)

# define executable
if (CUDA)
    cuda_add_executable(${appName} ${sourceCode})
else()
    add_executable(${appName} ${sourceCode})
endif()

# required libs
set(requiredLibs
    ${requiredLibs}
#    ${baseProjectLib}
)

# link the whole thing
target_link_libraries(${appName} ${requiredLibs})

# define source groups for file management within IDE
source_group("CMake Files" FILES ${cmakeFiles})
source_group("Test" FILES ${headerFiles} ${sourceFiles})
