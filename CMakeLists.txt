cmake_minimum_required(VERSION 4.0)
project(depth_demo CUDA CXX C)

set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
include(cmake/cross_test.cmake)


find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED CONFIG)
find_package(Torch REQUIRED CONFIG)

# PicRoulette-specific dependencies
find_package(SDL2 REQUIRED CONFIG)
find_package(SDL2_ttf REQUIRED CONFIG)
find_package(SDL2_image REQUIRED CONFIG)
find_package(opengl_system REQUIRED CONFIG)
find_package(nlohmann_json REQUIRED CONFIG)

set(DEPENDENCIES 
	"LINKER:--no-as-needed"
	CUDA::cudart
	/home/admin/.conan2/p/b/cuda-614561622e30e/p/lib64/stubs/libcuda.so
	#CUDA::cuda_driver
	#"LINKER:--as-needed"
	CUDA::nvrtc
	#"LINKER:--no-as-needed"
	CUDA::cuFile
	CUDA::cufft_static
	CUDA::curand
	CUDA::cublas
	CUDA::cublasLt
	CUDA::cusparse

	CUDA::cufft
	CUDA::cusolver

	#"LINKER:--as-needed"

	libtorch::c10
	libtorch::clog
	libtorch::caffe2_nvrtc
	libtorch::pytorch_qnnpack
	libtorch::torch_cpu

	"LINKER:--no-as-needed"
	libtorch::c10_cuda
	"LINKER:--as-needed"

	"LINKER:--no-as-needed"
	libtorch::torch_cuda
	"LINKER:--as-needed"


	libtorch::torch

	${OpenCV_LIBRARIES}
	opencv::opencv
)

add_executable(depth_demo depth_demo/depth_estimation.cpp)
target_link_libraries(depth_demo
	${DEPENDENCIES}
) 

add_executable(pose_demo pose_demo/pose_estimation.cpp)
target_link_libraries(pose_demo
	${DEPENDENCIES}
)

add_executable(picroulette_demo
	picroulette/src/main.cpp picroulette/src/OpenGLRenderer.cpp
	picroulette/src/PicRoulette.cpp picroulette/src/utils.hpp
	picroulette/src/QuickDrawRecognizer.cpp picroulette/src/VideoInput.cpp)
set_target_properties(picroulette_demo PROPERTIES
	CXX_STANDARD 20
	CXX_STANDARD_REQUIRED ON)
add_definitions(-DASSETS_DIR="./assets")
target_link_libraries(picroulette_demo
	${DEPENDENCIES}
	SDL2::SDL2main
	SDL2::SDL2
	opengl::opengl
	sdl_ttf::sdl_ttf
	SDL2_image::SDL2_image
	nlohmann_json::nlohmann_json
)


include(CTest)

if (BUILD_TESTING)
	enable_testing()

	# Device query app and test
	add_executable(device_query device_query/device_query.cpp)
	target_link_libraries(device_query PRIVATE CUDA::cudart fmt::fmt)
	target_include_directories(device_query PRIVATE device_query/include)

	add_test(NAME test_device_query COMMAND device_query)
	set_tests_properties(test_device_query PROPERTIES
		PASS_REGULAR_EXPRESSION "Device 0: \"Orin\"")

endif() # BUILD_TESTING
