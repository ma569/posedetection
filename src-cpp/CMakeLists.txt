cmake_minimum_required(VERSION 3.5) 

project(poseDetection LANGUAGES CXX VERSION 1.0.0)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_C_COMPILER icc)
#set(CMAKE_CXX_COMPILER icpc)


find_package(OpenCV REQUIRED)  
find_package(OpenMP REQUIRED)

set(WORKSPACE_DIR "/home/musabbir/workspace/poseDetection")
set(CMAKE_SOURCE_DIR "${WORKSPACE_DIR}/src-cpp")
set(DARKNET_DIR "${WORKSPACE_DIR}/darknet")
set(DARKNET_INC "${DARKNET_DIR}/include")
set(SRC_FILE ${CMAKE_SOURCE_DIR}/main.cpp)


get_filename_component(SOURCE_NAME ${SRC_FILE} NAME_WE)
set(EXE_NAME "${SOURCE_NAME}.bin")

add_executable(${EXE_NAME} 
               ${SRC_FILE}
               ${CMAKE_SOURCE_DIR}/detct_objDetect.cpp)

find_library(CAFFE caffe PATHS "/home/musabbir/lib")
find_library(DARKNET darknet PATHS "/home/musabbir/lib")
find_library(PTHREAD pthread)
find_library(OPENCV_HIGHGUI opencv_highgui)
find_library(OPENCV_VIDEOIO opencv_videoio)
find_library(OPENCV_IMGPROC opencv_imgproc)
find_library(OPENCV_CORE opencv_core)
find_library(GFLAGS gflags PATHS "/usr/lib/x86_64-linux-gnu" NO_DEFAULT_PATH)
#find_library(OPENMP iomp5 PATHS "/opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/lib/intel64_lin" NO_DEFAULT_PATH)

target_link_libraries(${EXE_NAME}
#                      ${OPENMP}
                      ${CAFFE}
                      ${DARKNET}
                      openpose
                      ${GFLAGS}
                      ${OPENCV_HIGHGUI}
                      ${OPENCV_CORE}
                      )

target_include_directories(${EXE_NAME} PUBLIC 
                           ${DARKNET_INC}
                           ${CMAKE_SOURCE_DIR})
