#include <stdio.h>
#include <iostream>
#include "glad/glad.h"
#include <array>
#include "GLFW/glfw3.h"
#include <glm/vec3.hpp>					 // glm::vec3
#include <glm/vec4.hpp>					 // glm::vec4
#include <glm/mat4x4.hpp>				 // glm::mat4
#include <glm/ext/matrix_transform.hpp>	 // glm::translate, glm::rotate, glm::scale
#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace std;
using namespace glm;

uint WINDOW_WIDTH = 640;
uint WINDOW_HEIGHT = 480;

void setupCudaGL(int width, int height); // Forward declaration

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
	glViewport(0, 0, width, height);
	WINDOW_WIDTH = width;
	WINDOW_HEIGHT = height;
	setupCudaGL(width, height);
}

// Declare the CUDA rendering kernel
extern "C" void renderCudaKernel(uchar4 *outputData, int width, int height);

// CUDA Graphics Resource
cudaGraphicsResource_t cuda_pbo_resource;
uint pbo = 0;	  // Initialize to 0 to indicate no resource is allocated
uint texture = 0; // Initialize to 0

// Function to check for CUDA errors
static void CheckCudaError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define CUDA_CHECK(err) CheckCudaError(err, __FILE__, __LINE__)

// Function to print CUDA device information
void printCudaDeviceInfo()
{
	int deviceCount;
	CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

	if (deviceCount == 0)
	{
		cout << "No CUDA devices found." << endl;
		return;
	}

	cout << "CUDA Device Info:" << endl;
	for (int i = 0; i < deviceCount; ++i)
	{
		cudaDeviceProp deviceProp;
		CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, i));

		cout << "  Device " << i << ": " << deviceProp.name << endl;
		cout << "    Compute Capability: " << deviceProp.major << "." << deviceProp.minor << endl;
		cout << "    Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << endl;
		cout << "    Multiprocessors: " << deviceProp.multiProcessorCount << endl;
		cout << "    CUDA Cores: " << deviceProp.multiProcessorCount * ((deviceProp.major == 9) ? 128 : ((deviceProp.major >= 3) ? 192 : 0)) << " \n"; // Simplified estimation
		cout << "    Clock Rate: " << deviceProp.clockRate / 1000 << " MHz" << endl;
		cout << "    Memory Clock Rate: " << deviceProp.memoryClockRate / 1000 << " MHz" << endl;
		cout << "    ECC Enabled: " << (deviceProp.ECCEnabled ? "Yes" : "No") << endl;
	}
}

#include <fstream>
#include <sstream>

static string ReadFile(const string &filepath)
{
	ifstream stream(filepath);
	stringstream ss;
	ss << stream.rdbuf();
	return ss.str();
}

static uint CompileShader(uint type, const string &source)
{
	uint id = glCreateShader(type);
	const char *src = source.c_str();
	glShaderSource(id, 1, &src, nullptr);
	glCompileShader(id);

	int result;
	glGetShaderiv(id, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE)
	{
		int length;
		glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
		auto *message = (char *)_malloca(length * sizeof(char));
		glGetShaderInfoLog(id, length, &length, message);
		std::cout << "Failed to compile " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader!" << std::endl;
		std::cout << message << std::endl;
		glDeleteShader(id);
		return 0;
	}

	return id;
}

static uint CreateShader(const string &vertexShader, const string &fragmentShader)
{
	uint program = glCreateProgram();
	uint vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
	uint fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);

	glAttachShader(program, vs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	glValidateProgram(program);

	glDeleteShader(vs);
	glDeleteShader(fs);

	return program;
}

void setupCudaGL(int width, int height)
{
	/* Clean up existing resources if they were already created */
	if (pbo != 0)
	{
		CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_pbo_resource));
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &texture);
	}

	/* Create OpenGL texture */
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(uchar4), nullptr, GL_STREAM_DRAW);

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

	/* Register PBO with CUDA */
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));
}

int main()
{
	GLFWwindow *window;

	/* Initialize the library */
	if (!glfwInit())
	{
		return -1;
	}

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Raytracer", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	/* Load all OpenGL functions using the glfw loader function */
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		printf("Failed to initialize OpenGL context\n");
		return -1;
	}

	/* Assign rectangle vertex positions and texture coordinates */
	/* Vertices for a rectangle (two triangles) */
	/* Positions (x, y), Texture Coordinates (u, v) */
	std::array<float, 24> vertices = {
		// First triangle
		-1.0f, -1.0f, 0.0f, 0.0f, // Bottom-left
		1.0f, -1.0f, 1.0f, 0.0f,  // Bottom-right
		1.0f, 1.0f, 1.0f, 1.0f,	  // Top-right
		// Second triangle
		-1.0f, -1.0f, 0.0f, 0.0f, // Bottom-left
		1.0f, 1.0f, 1.0f, 1.0f,	  // Top-right
		-1.0f, 1.0f, 0.0f, 1.0f	  // Top-left
	};

	/* Generate an OpenGL buffer */
	uint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

	/* Set vertex layout and attributes */
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));

	string vertexShader = ReadFile("shaders/shader.vert");

	string fragmentShader = ReadFile("shaders/shader.frag");

	uint shader = CreateShader(vertexShader, fragmentShader);
	glUseProgram(shader);

	/* Get the location of the uniform sampler */
	GLint texLocation = glGetUniformLocation(shader, "u_texture");
	glUniform1i(texLocation, 0);

	/* Get device info */
	printCudaDeviceInfo();

	setupCudaGL(WINDOW_WIDTH, WINDOW_HEIGHT);

	/* Loop until the user closes the window */
	while (!glfwWindowShouldClose(window))
	{
		/* Render here */
		glClear(GL_COLOR_BUFFER_BIT);

		/* Map the PBO to CUDA */
		uchar4 *d_ptr;
		size_t num_bytes;
		CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&d_ptr, &num_bytes, cuda_pbo_resource));

		/* Launch CUDA kernel */
		renderCudaKernel(d_ptr, WINDOW_WIDTH, WINDOW_HEIGHT);

		/* Unmap the PBO */
		CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

		/* Bind texture and draw */
		glBindTexture(GL_TEXTURE_2D, texture);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		glDrawArrays(GL_TRIANGLES, 0, 6);

		/* Swap front and back buffers */
		glfwSwapBuffers(window);

		/* Poll for and process events */
		glfwPollEvents();
	}

	/* Cleanup */
	CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_pbo_resource));
	glDeleteProgram(shader);
	glDeleteBuffers(1, &vbo);

	glfwTerminate();
	return 0;
}
