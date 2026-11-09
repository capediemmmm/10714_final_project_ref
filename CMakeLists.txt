cmake_minimum_required(VERSION 3.19)
project(needle_sparse LANGUAGES CXX)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(ndarray_backend_cpu src/ndarray_backend_cpu.cc)
target_compile_features(ndarray_backend_cpu PRIVATE cxx_std_11)
set_target_properties(ndarray_backend_cpu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/needle/backend_ndarray")
