cmake_minimum_required (VERSION 3.1)
project(location_matcher)

find_package(OpenCV REQUIRED)

include_directories(
  .
  include/opencv2/
  ${OpenCV_INCLUDE_DIRS}
)

set(LINE_DESCRIPTOR_CPP
src/binary_descriptor_matcher.cpp
src/binary_descriptor.cpp
src/draw.cpp
src/LSDDetector.cpp)

add_library(custom_line_descriptor ${LINE_DESCRIPTOR_CPP})
target_link_libraries(custom_line_descriptor
${OpenCV_LIBS}
)