cmake_minimum_required(VERSION 3.10)

SET(MSVC 0)

if (MSVC)
    SET(CMAKE_C_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.12.25827/bin/Hostx64/x64/cl.exe")
    SET(CMAKE_CXX_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.12.25827/bin/Hostx64/x64/cl.exe")
else (MSVC) # Expecting mingw gcc here
    SET(CMAKE_C_COMPILER "c:/Program Files/mingw-w64/x86_64-8.1.0-win32-seh-rt_v6-rev0/mingw64/bin/x86_64-w64-mingw32-gcc.exe")
    SET(CMAKE_CXX_COMPILER "c:/Program Files/mingw-w64/x86_64-8.1.0-win32-seh-rt_v6-rev0/mingw64/bin/x86_64-w64-mingw32-g++.exe")
endif (MSVC)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

if (MSVC)
    
    # Include dirs
    list(APPEND INCL_DIRS "C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.12.25827/include")
    list(APPEND INCL_DIRS "C:/Program Files (x86)/Windows Kits/10/Include/10.0.10240.0/ucrt")
    list(APPEND INCL_DIRS "E:/Programming/libraries/cpp/opencv/build/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/json11")
    set(ENV{INCLUDE} "${INCL_DIRS}")
    
    # Lib dirs
    list(APPEND LINK_DIRS "C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.12.25827/lib/x64")
    list(APPEND LINK_DIRS "C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.12.25827/atlmfc/lib/x64")
    list(APPEND LINK_DIRS "C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/VS/UnitTest/lib")
    list(APPEND LINK_DIRS "C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/VS/lib/x64")
    list(APPEND LINK_DIRS "C:/Program Files (x86)/Windows Kits/10/lib/10.0.16299.0/ucrt/x64")
    list(APPEND LINK_DIRS "C:/Program Files (x86)/Windows Kits/10/lib/10.0.16299.0/um/x64")
    list(APPEND LINK_DIRS "e:/Programming/libraries/cpp/opencv/build/x64/vc15/lib")
    set(ENV{LIB} "${LINK_DIRS}")

    # Activate openmp
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")

else (MSVC) # Expecting mingw gcc here

    # Include dirs
    list(APPEND INCL_DIRS "c:/Program Files/mingw-w64/x86_64-8.1.0-win32-seh-rt_v6-rev0/mingw64/x86_64-w64-mingw32/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/json11")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/build_mingw")
    list(APPEND INCL_DIRS "E:/Programming/libraries/cpp/opencv/build_mingw/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/world/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/viz/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/videostab/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/videoio/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/video/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/ts/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/superres/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/stitching/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/shape/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/python/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/photo/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/objdetect/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/ml/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/js/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/java/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/imgproc/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/imgcodecs/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/highgui/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/flann/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/features2d/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/dnn/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/cudev/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/cudawarping/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/cudastereo/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/cudaoptflow/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/cudaobjdetect/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/cudalegacy/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/cudaimgproc/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/cudafilters/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/cudafeatures2d/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/cudacodec/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/cudabgsegm/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/cudaarithm/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/core/include")
    list(APPEND INCL_DIRS "e:/Programming/libraries/cpp/opencv/sources/modules/calib3d/include")

    # Lib dirs
    list(APPEND LINK_DIRS "e:/Programming/libraries/cpp/opencv/build_mingw/lib")
    list(APPEND LINK_DIRS "c:/Program Files/mingw-w64/x86_64-8.1.0-win32-seh-rt_v6-rev0/mingw64/x86_64-w64-mingw32/lib")

    # Activate openmp
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    
    # Mingw make path
    set (ENV{PATH} "$ENV{PATH};c:/Program Files/mingw-w64/x86_64-8.1.0-win32-seh-rt_v6-rev0/mingw64/bin")
    
endif(MSVC)

# Project
project(${EXEFILE})
include_directories(${INCL_DIRS})
link_directories(${LINK_DIRS})
add_executable(${EXEFILE} ${CPPFILE})
if (MSVC)
    target_link_libraries(${EXEFILE} "opencv_world340")
else (MSVC)
    target_link_libraries(${EXEFILE} "libopencv_imgcodecs340" "libopencv_imgproc340" "libopencv_core340")
endif(MSVC)
