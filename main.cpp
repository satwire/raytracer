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

using namespace std;
using namespace glm;

const uint imageWidth = 640;
const uint imageHeight = 480;

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
		std::cout << "Failed to compile" << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << "shader!" << std::endl;
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

int main()
{
	GLFWwindow *window;

	/* Initialize the library */
	if (!glfwInit())
	{
		return -1;
	}

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(imageWidth, imageHeight, "Hello World", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	/* Load all OpenGL functions using the glfw loader function */
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		printf("Failed to initialize OpenGL context\n");
		return -1;
	}

	/* Assign triangle vertex positions */
	std::array<float, 6> positions{
		-1.0f, -1.0f,
		1.0f, -1.0f,
		0.0f, 1.0f};

	/* Generate an OpenGL buffer */
	uint buffer;
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(float), &positions[0], GL_STATIC_DRAW);

	/* Set vertex layout and attributes */
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);

	string vertexShader =
		"#version 330 core\n"
		"\n"
		"layout(location = 0) in vec4 position;\n"
		"\n"
		"void main()\n"
		"{\n"
		"\tgl_Position = position;\n"
		"}\n";

	string fragmentShader =
		"#version 330 core\n"
		"\n"
		"layout(location = 0) out vec4 color;\n"
		"\n"
		"void main()\n"
		"{\n"
		"\tcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
		"}\n";

	uint shader = CreateShader(vertexShader, fragmentShader);
	glUseProgram(shader);

	/* Get device info */
	printCudaDeviceInfo();

	/* Loop until the user closes the window */
	while (!glfwWindowShouldClose(window))
	{
		/* Render here */
		glClear(GL_COLOR_BUFFER_BIT);

		/* Draw the currently bound buffer */
		glDrawArrays(GL_TRIANGLES, 0, 3);

		/* Swap front and back buffers */
		glfwSwapBuffers(window);

		/* Poll for and process events */
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}
