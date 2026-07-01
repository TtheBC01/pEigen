cmake_minimum_required(VERSION 3.18...4.0)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

option(PEIGEN_USE_OPENMP "Enable OpenMP when available" OFF)
option(PEIGEN_USE_BLAS "Enable BLAS backend for Eigen where available" ON)
option(PEIGEN_USE_LAPACK "Enable LAPACK backend for factorizations where available" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(_core_module MODULE WITH_SOABI src/bindings/module.cpp)
target_link_libraries(_core_module PRIVATE pybind11::headers)
target_include_directories(_core_module PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/eigen)
target_compile_definitions(_core_module PRIVATE
  EIGEN_MPL2_ONLY
  "PEIGEN_PROJECT_VERSION=\"${PROJECT_VERSION}\""
  "PEIGEN_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\""
)

set(PEIGEN_BLAS_BACKEND "none")
set(PEIGEN_LAPACK_BACKEND "none")
set(PEIGEN_OPENMP_ENABLED 0)

if(MSVC)
  target_compile_options(_core_module PRIVATE /O2)
else()
  target_compile_options(_core_module PRIVATE -O3)
endif()

if(PEIGEN_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(_core_module PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_core_module PRIVATE PEIGEN_USE_OPENMP=1)
    set(PEIGEN_OPENMP_ENABLED 1)
  endif()
endif()

if(PEIGEN_USE_BLAS OR PEIGEN_USE_LAPACK)
  if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
      target_link_libraries(_core_module PRIVATE ${ACCELERATE_FRAMEWORK})
      if(PEIGEN_USE_BLAS)
        target_compile_definitions(_core_module PRIVATE EIGEN_USE_BLAS)
        set(PEIGEN_BLAS_BACKEND "accelerate")
      endif()
      if(PEIGEN_USE_LAPACK)
        target_compile_definitions(_core_module PRIVATE PEIGEN_LAPACK_ENABLED=1)
        set(PEIGEN_LAPACK_BACKEND "accelerate")
      endif()
      message(STATUS "pEigen: using Apple Accelerate for BLAS/LAPACK")
    else()
      message(STATUS "pEigen: Accelerate not found; BLAS/LAPACK disabled")
    endif()
  else()
    find_package(BLAS)
    find_package(LAPACK)
    if(BLAS_FOUND)
      if(TARGET BLAS::BLAS)
        target_link_libraries(_core_module PRIVATE BLAS::BLAS)
      else()
        target_link_libraries(_core_module PRIVATE ${BLAS_LIBRARIES})
      endif()
      if(PEIGEN_USE_BLAS)
        target_compile_definitions(_core_module PRIVATE EIGEN_USE_BLAS)
        set(PEIGEN_BLAS_BACKEND "system-blas")
      endif()
    endif()
    if(LAPACK_FOUND AND PEIGEN_USE_LAPACK)
      if(TARGET LAPACK::LAPACK)
        target_link_libraries(_core_module PRIVATE LAPACK::LAPACK)
      else()
        target_link_libraries(_core_module PRIVATE ${LAPACK_LIBRARIES})
      endif()
      target_compile_definitions(_core_module PRIVATE PEIGEN_LAPACK_ENABLED=1)
      set(PEIGEN_LAPACK_BACKEND "system-lapack")
      message(STATUS "pEigen: using system BLAS/LAPACK")
    elseif(BLAS_FOUND)
      message(STATUS "pEigen: using system BLAS (LAPACK not found)")
    else()
      message(STATUS "pEigen: BLAS/LAPACK not found; using Eigen kernels only")
    endif()
  endif()
endif()

target_compile_definitions(_core_module PRIVATE
  "PEIGEN_BLAS_BACKEND=\"${PEIGEN_BLAS_BACKEND}\""
  "PEIGEN_LAPACK_BACKEND=\"${PEIGEN_LAPACK_BACKEND}\""
  PEIGEN_OPENMP_ENABLED=${PEIGEN_OPENMP_ENABLED}
)

set_target_properties(_core_module PROPERTIES OUTPUT_NAME _core)
install(TARGETS _core_module DESTINATION peigen)
