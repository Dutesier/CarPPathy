#######################################
#######################################
#############  DUTESIER  ##############
#######################################
#######################################

cmake_minimum_required(VERSION 3.14)

# Define the project name and the programming language
project(CarPPathy LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set compiler flags
add_compile_options(-Wall -Werror -Wextra)

# Define source files
set(SRC_FILES 
    src/Value.cpp
    )

# Add the library target
add_library(CarPPathy ${SRC_FILES})
target_include_directories(CarPPathy PUBLIC include)

# Add target for manually testing
add_executable(CarPPathyMainTest ${SRC_FILES} main.cpp)
target_link_libraries(CarPPathyMainTest PRIVATE CarPPathy)

# Add Google Test
find_package(GTest)

# Enable testing
enable_testing()

# Add the test executable
add_executable(CarPPathyTest 
    tests/Value_test.cpp
    )


# Link the test executable with the library and Google Test
target_link_libraries(CarPPathyTest PRIVATE CarPPathy GTest::gtest_main)

# Add tests
include(GoogleTest)
gtest_discover_tests(CarPPathyTest)