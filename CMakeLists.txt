project(bmp_converter)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/Modules)

find_package(Boost 1.46 COMPONENTS filesystem system date_time thread REQUIRED)
find_package(LevelDB REQUIRED)
find_package(LMDB REQUIRED)
find_package(GFlags REQUIRED)
find_package(Glog REQUIRED)
find_package(Protobuf REQUIRED)
find_package(HDF5 COMPONENTS HL REQUIRED)
find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)

set(Boost_USE_MULTITHREADED ON)
#includes
include_directories(bmp_converter ${Boost_INCLUDE_DIR} ${LMDB_INCLUDE_DIR} ${LevelDB_INCLUDE}
                    ${GLOG_INCLUDE_DIRS} ${GFLAGS_INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIR}
                    ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SRC_LIST})

message(${OpenCV_INCLUDE_DIRS})
#Add linking libraries
set( CAFFE_LIBRARY "/home/victor/Programming/caffe-master/build/lib/libcaffe.so"
                    "/home/victor/anaconda/lib/libhdf5_hl.so.9")

target_link_libraries(bmp_converter ${Boost_LIBRARIES} ${GLOG_LIBRARIES} ${LevelDB_LIBRARY}
                    ${LMDB_LIBRARIES} ${GFLAGS_LIBRARIES} ${PROTOBUF_LIBRARIES} ${CAFFE_LIBRARY}
                    ${HDF5_LIBRARIES} ${OpenCV_LIBS})
