# Raytracer

This project is a simple ray tracer written in C++ and CUDA, using OpenGL to render the output to the screen. It supports dynamic window resizing.

## Building and Running

### Prerequisites

*   NVIDIA GPU with CUDA support
*   CUDA Toolkit
*   CMake
*   A C++ compiler that supports C++17 (e.g., MSVC, GCC, Clang)

### Build Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

3.  **Configure the project with CMake:**
    ```bash
    cmake ..
    ```

4.  **Build the project:**
    *   **On Windows (with Visual Studio):**
        ```bash
        cmake --build . --config Release
        ```
    *   **On Linux/macOS:**
        ```bash
        make
        ```

### Running the Application

After a successful build, the executable will be located in the `build/Release` directory (on Windows) or `build` directory (on Linux/macOS).

```bash
./raytracer
```
