cmake_minimum_required(VERSION 3.10)
project(openvino_samples)

message("--> Project: " ${PROJECT_NAME})

set(EXE_NAME "${PROJECT_NAME}")
# set(ov_link_libraries openvino::runtime)
# add_definitions(-DUSE_OPENCV)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(ngraph REQUIRED)
find_package(Threads REQUIRED)
find_package(InferenceEngine REQUIRED)

# file (GLOB MAIN_SRC
#         ${CMAKE_CURRENT_SOURCE_DIR}/format_reader/*.cpp
#         )

# file (GLOB LIBRARY_HEADERS
#         ${CMAKE_CURRENT_SOURCE_DIR}/format_reader/*.h
#         )

include_directories(${InferenceEngine_INCLUDE_DIRS})

add_executable(${EXE_NAME} main.cpp )

set_target_properties(${EXE_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES LINKER_LANGUAGE CXX)

# target_sources(${EXE_NAME} PRIVATE ${MAIN_SRC} ${LIBRARY_HEADERS})

target_link_libraries(${EXE_NAME} ${NGRAPH_LIBRARIES} ${InferenceEngine_LIBRARIES} Threads::Threads ${OpenCV_LIBS})

message("Inference Engine dirs: ${InferenceEngine_INCLUDE_DIRS}")

# get_target_property(LIBS ${EXE_NAME} LINK_LIBRARIES)
# message("###### ${LIBS}")

# message("*** ${InferenceEngine_VERSION}")
# message("*** ${OpenVINO_VERSION}")

