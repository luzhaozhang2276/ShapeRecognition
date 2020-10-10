cmake_minimum_required(VERSION 2.8)
project(ShapeRecognition)

set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(OpenCV 3 REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

#add_library(${PROJECT_NAME} SHARED
#
#)

#target_link_libraries(${PROJECT_NAME}
#        ${OpenCV_LIBS}
#)

add_executable(geometry UseGeometry.cpp)
target_link_libraries(geometry ${OpenCV_LIBS})

add_executable(svm UseSVM.cpp)
target_link_libraries(svm ${OpenCV_LIBS})