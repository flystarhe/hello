cmake_minimum_required(VERSION 3.10)
project(hello-c)

# The first thing do is to tell cmake to find the TorchVision library.
# The package pulls in all the necessary torch libraries,
# so there is no need to also add `find_package(Torch)` here.
find_package(TorchVision REQUIRED)

# This due to LibTorch's version is the one included in the Python
# package that links to Python.
find_package(Python3 COMPONENTS Development)

add_executable(hello-c main.cpp)

# We now need to link the TorchVision library to our executable.
# We can do that by using the TorchVision::TorchVision target,
# which also adds all the necessary torch dependencies.
target_compile_features(hello-c PUBLIC cxx_range_for)
target_link_libraries(hello-c TorchVision::TorchVision)
set_property(TARGET hello-c PROPERTY CXX_STANDARD 14)