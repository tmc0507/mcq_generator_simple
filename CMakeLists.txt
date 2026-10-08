cmake_minimum_required(VERSION 3.16)
project(mcq_generator_simple LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(doc_processor src/doc_processor.cpp)

target_compile_options(doc_processor PRIVATE -Wall -Wextra -Wpedantic)
