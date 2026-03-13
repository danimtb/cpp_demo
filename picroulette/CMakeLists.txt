cmake_minimum_required(VERSION 3.26)
project(picroulette)

find_package(OpenCV REQUIRED CONFIG)
find_package(SDL2 REQUIRED CONFIG)
find_package(SDL2_ttf REQUIRED CONFIG)
find_package(SDL2_image REQUIRED CONFIG)
find_package(opengl_system REQUIRED CONFIG)
find_package(Torch REQUIRED CONFIG)
find_package(nlohmann_json REQUIRED CONFIG)

add_executable(
  ${PROJECT_NAME} src/main.cpp src/OpenGLRenderer.cpp src/PicRoulette.cpp
  src/utils.hpp src/QuickDrawRecognizer.cpp src/VideoInput.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON)
add_definitions(-DASSETS_DIR="${CMAKE_SOURCE_DIR}/assets")

target_link_libraries(
  ${PROJECT_NAME}
  opencv::opencv
  SDL2::SDL2main
  SDL2::SDL2-static
  opengl::opengl
  sdl_ttf::sdl_ttf
  SDL2_image::SDL2_image
  libtorch::libtorch
  nlohmann_json::nlohmann_json)
