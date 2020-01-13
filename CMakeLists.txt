cmake_minimum_required (VERSION 3.1)
project(location_matcher)

################# G-TEST Build #####################
# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()
################ G-TEST end ######################

find_package(OpenCV REQUIRED)

include_directories(
  .
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_library(lm_utils src/utils.cpp)
target_link_libraries(lm_utils
  ${OpenCV_LIBS}
)

add_library(line_detector src/line_detector.cpp)
target_link_libraries(line_detector
    ${OpenCV_LIBS}
)

add_library(line_filter src/line_filter.cpp)
target_link_libraries(line_filter
  ${OpenCV_LIBS}
)

add_library(segment src/segment.cpp)
target_link_libraries(segment
  ${OpenCV_LIBS}
)

## === Tests === 
add_library(test_datasets test/test_datasets.cpp)
target_link_libraries(test_datasets
  lm_utils
  gtest_main
  ${OpenCV_LIBS}
)

add_executable(line_detector_test test/line_detector_test.cpp)
target_link_libraries(line_detector_test
  test_datasets
  line_detector
  lm_utils
  gtest_main
)
add_test(NAME lineDetectorTest COMMAND line_detector_test)

add_executable(line_filter_test test/line_filter_test.cpp)
target_link_libraries(line_filter_test
  line_filter
  lm_utils
  gtest_main
)
add_test(NAME lineFilterTest COMMAND line_filter_test)

add_executable(segment_test test/segment_test.cpp)
target_link_libraries(segment_test
  segment
  test_datasets
  gtest_main
)