# Define version
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Define project name
project(State-Estimation CXX)

# Define the VS startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT State-Estimation)

# Group the cmake projects into a folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Define C++ Version
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add State-Estimation subdirectory
add_subdirectory(State-Estimation)
