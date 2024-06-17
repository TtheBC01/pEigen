cmake_minimum_required(VERSION 3.25)
project(PEIGEN)

# set C++ to version 23 and only use Eigen lib components that are MPL2 licenced
set(CMAKE_CXX_STANDARD 23)
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_MPL2_ONLY")

include_directories(./src ./eigen)

# require Python 3
find_package(PythonLibs 3 REQUIRED)
include_directories("${PYTHON_INCLUDE_DIRS}")

# main library for python bindings
add_library(peigen SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/python_bindings.d/peigen.cxx)

# find the required packages from boost
find_package(Boost 1.74 REQUIRED system python serialization)
if(Boost_FOUND)
	MESSAGE(" -- A library with Boost found.")
        include_directories("${Boost_INCLUDE_DIRS}")
        target_link_libraries(peigen ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
elseif(not Boost_FOUND)
	message(FATAL_ERROR "unable to find correct Boost version.")
endif(Boost_FOUND)

INSTALL(TARGETS peigen LIBRARY DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

configure_file(${PROJECT_SOURCE_DIR}/src/peigen/__init__.py.in ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/src/peigen/__init__.py)

message(" -- Build Type:         " ${CMAKE_BUILD_TYPE})
