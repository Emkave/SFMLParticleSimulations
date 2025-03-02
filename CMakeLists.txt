cmake_minimum_required(VERSION 3.29)
project(TheBigBangSFML CUDA)

set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

add_executable(TheBigBangSFML main.cu
        particle.h
        thread_pool.h
        thread_safe_queue.h
        registers.h
        simulations.h
        simulation1.cu
)

set_target_properties(TheBigBangSFML PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set(SFML_STATIC_LIBRARIES TRUE)
set(SFML_DIR D:/SFML-2.6.1/lib/cmake/SFML)
find_package(SFML COMPONENTS system window graphics audio network REQUIRED)
include_directories(D:/SFML-2.6.1/include/SFML)

set(GLFW_LIB_DIR ${CMAKE_SOURCE_DIR}/lib-vc2019)
link_directories(${GLFW_LIB_DIR})

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/glad)
include_directories(${CMAKE_SOURCE_DIR}/include/KHR)
include_directories(${CMAKE_SOURCE_DIR})

set(VLD_DIR "D:/Visual Leak Detector")
include_directories(${VLD_DIR}/include)
link_directories(${VLD_DIR}/lib/Win64)
add_definitions(-DVLDBUILD)

target_link_libraries(TheBigBangSFML PRIVATE glfw3 opengl32 sfml-system sfml-window sfml-graphics sfml-audio vld)
target_include_directories(TheBigBangSFML PRIVATE ${CMAKE_SOURCE_DIR}/glad ${CMAKE_SOURCE_DIR}/include ${GLFW_LIB_DIR})