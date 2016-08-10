cmake_minimum_required(VERSION 3.4)
project(predictors CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
file(GLOB SOURCES "src/*.cpp" "src/*.h")
add_executable(predictors ${SOURCES})
target_compile_features(predictors PRIVATE cxx_range_for)

