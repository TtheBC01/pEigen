cmake_minimum_required(VERSION 3.0)

project(PEIGEN)

set(INC_DIR .)

set(CMAKE_CXX_STANDARD 11)

include_directories("${INC_DIR}")
include_directories(/pEigen/eigen)

# only use Eigen lib components that are MPL2 licenced
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_MPL2_ONLY")

# create directies for binaries and python lib
set(BINDIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(LIBDIR ${CMAKE_CURRENT_BINARY_DIR}/lib)

# require python 3 for building
find_package(PythonLibs 3 REQUIRED)
include_directories("${PYTHON_INCLUDE_DIRS}")

# main library for python bindings
add_library(hypernet SHARED pythonwrapper.d/hypernet.cxx)
set_target_properties(hypernet
                      PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY ${LIBDIR})

# find the required packages from boost
find_package(Boost 1.74 REQUIRED system python)
if(Boost_FOUND)
	MESSAGE(" -- A library with Boost found.")
        include_directories("${Boost_INCLUDE_DIRS}")
	target_link_libraries(a.out    ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
	target_link_libraries(b.out    ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
        target_link_libraries(hypernet ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
elseif(not Boost_FOUND)
	message(FATAL_ERROR "unable to find correct Boost version.")
endif(Boost_FOUND)

set(AllTargets hypernet)

INSTALL(TARGETS ${AllTargets} 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

##add_subdirectory(pythonwrapper.d)
##target_link_libraries(pythonwrapper)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING
		"Choose the type of build, options are: None Debug Release."
		FORCE)
endif(NOT CMAKE_BUILD_TYPE)

message(" -- Build Type:         " ${CMAKE_BUILD_TYPE})
