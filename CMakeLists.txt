cmake_minimum_required (VERSION 2.8.11)
project (caffe_tensorRT)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED)
find_package(Caffe REQUIRED)

SET(CAFFE_HOME /home/jungmo/mylibs/caffe)

INCLUDE_DIRECTORIES(
    /usr/local/cuda-8.0/include
    ${CAFFE_HOME}/distribute/include
    )

cuda_add_executable (${PROJECT_NAME}_GIE src/withGIE.cpp)
#add_executable (${PROJECT_NAME}_GIE src/withGIE.cpp)
add_executable (${PROJECT_NAME}_NO_GIE src/noGIE.cpp)

set(BOOST_LIB boost_system boost_filesystem boost_thread)
#set(Caffe_LIBRARIES "${CAFFE_HOME}/distribute/lib/libcaffe.so")

target_link_libraries (${PROJECT_NAME}_GIE
#      ${Caffe_LIBRARIES}
      ${OpenCV_LIBS}
      ${BOOST_LIB}
      caffe
      cuda
      nvcaffe_parser nvinfer
      glog gflags protobuf m hdf5_hl hdf5
      cblas atlas
      stdc++
      )
target_link_libraries (${PROJECT_NAME}_NO_GIE
#      ${Caffe_LIBRARIES}
      ${OpenCV_LIBS}
      ${BOOST_LIB}
      caffe
      cuda
      nvcaffe_parser nvinfer
      glog gflags protobuf m hdf5_hl hdf5
      cblas atlas
      stdc++
      )
