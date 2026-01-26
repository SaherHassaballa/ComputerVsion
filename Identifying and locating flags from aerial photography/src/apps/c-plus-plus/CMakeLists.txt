cmake_minimum_required(VERSION 3.10)
project(yolo_onnx_demo)

find_package(OpenCV REQUIRED)
add_executable(yolo_onnx_demo yolo_onnx_demo.cpp)
target_include_directories(yolo_onnx_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo_onnx_demo PRIVATE ${OpenCV_LIBS})
