cmake_minimum_required(VERSION 3.26)
project(raytracer LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(raytracer 
    main.cpp 
    kernel.cu
    "lib/glad/src/glad.c"
)

target_include_directories(raytracer PRIVATE 
    "lib/glad/include"
    "lib/GLFW/include"
    "lib/OpenGL"
)

target_link_libraries(raytracer PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/GLFW/lib-vc2019/glfw3.lib"
    OpenGL::GL
)

target_compile_definitions(raytracer PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
    _CONSOLE
)

set_target_properties(raytracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)