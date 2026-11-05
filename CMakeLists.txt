cmake_minimum_required(VERSION 3.22)
project(tachyon_pi CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O3 -fno-plt)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  add_compile_options(-march=native -mtune=native -fexcess-precision=standard)
endif()

find_library(GMP_LIB gmp REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP)

# CUDA backend is optional; engine falls back to CPU when absent.
option(TACHYON_ENABLE_CUDA "Build the CUDA exact-NTT multiplication backend" ON)
set(TACHYON_HAVE_CUDA FALSE)
if(TACHYON_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "120")
    set(TACHYON_HAVE_CUDA TRUE)
  endif()
endif()

add_library(tachyon_core STATIC
  src/pi_chudnovsky.cpp
  src/convert.cpp
  src/verify.cpp
  src/engine.cpp
  src/pmulp.cpp
  src/jobs.cpp
  src/http_server.cpp)
target_link_libraries(tachyon_core PUBLIC ${GMP_LIB} Threads::Threads)
if(TACHYON_HAVE_CUDA)
  target_sources(tachyon_core PRIVATE src/cuda/cuda_backend.cu)
  target_compile_definitions(tachyon_core PUBLIC TACHYON_HAVE_CUDA=1)
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(tachyon_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(tachyon-pi src/main.cpp)
target_link_libraries(tachyon-pi PRIVATE tachyon_core)

add_executable(machin-check tests/machin_check.cpp)
target_link_libraries(machin-check PRIVATE ${GMP_LIB})

add_executable(gpu-mul-test tests/gpu_mul_test.cpp)
target_link_libraries(gpu-mul-test PRIVATE tachyon_core)

add_executable(gpu-bench bench/gpu_bench.cpp)
target_link_libraries(gpu-bench PRIVATE tachyon_core)

add_executable(convert-test tests/convert_test.cpp src/convert.cpp)
target_link_libraries(convert-test PRIVATE ${GMP_LIB} Threads::Threads)

add_executable(mul-test tests/mul_test.cpp src/pmulp.cpp)
target_link_libraries(mul-test PRIVATE ${GMP_LIB} Threads::Threads)

add_executable(mul-bench bench/mul_bench.cpp src/pmulp.cpp)
target_link_libraries(mul-bench PRIVATE ${GMP_LIB} Threads::Threads)

add_executable(tail-test tests/tail_test.cpp)
target_link_libraries(tail-test PRIVATE tachyon_core)

enable_testing()
add_test(NAME tail-correctness COMMAND tail-test 8)
add_test(NAME smoke COMMAND tachyon-pi --digits 101 --verify)
add_test(NAME digits10k-bbp COMMAND tachyon-pi --digits 10000 --verify)
add_test(NAME gpu-mul-correctness COMMAND gpu-mul-test)
set_tests_properties(gpu-mul-correctness PROPERTIES SKIP_RETURN_CODE 77)
