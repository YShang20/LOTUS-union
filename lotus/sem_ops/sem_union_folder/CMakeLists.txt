cmake_minimum_required(VERSION 3.14)
project(tfidf_nimble)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Arrow
find_package(Arrow REQUIRED)

# Eigen headers (header-only)
find_path(EIGEN3_INCLUDE_DIR Eigen/Dense PATHS /opt/homebrew/include/eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

# Nimble headers and libraries (adjust as needed)
include_directories(/Users/yolandazhou/Documents/untitled_folder/CSE_584/lotus/nimble)
include_directories(/opt/homebrew/opt/folly/include)

include_directories(/Users/yolandazhou/Documents/untitled_folder/CSE_584/lotus/nimble/velox)
link_directories(/Users/yolandazhou/Documents/untitled_folder/CSE_584/lotus/nimble/build)
include_directories(/opt/homebrew/include)
link_directories(/opt/homebrew/lib)

add_executable(write_nimble write_nimble.cpp)

# Link with Arrow and Nimble (Eigen is header-only so no link required)
target_link_libraries(write_nimble
  velox_exec
  velox_vector
  velox_parse
  nimble
  folly
  glog
)