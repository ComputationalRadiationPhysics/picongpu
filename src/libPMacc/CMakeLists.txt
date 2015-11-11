################################################################################
# PMacc tests
################################################################################
cmake_minimum_required(VERSION 3.3)
project ("pmacc test")


################################################################################
# PMacc
################################################################################
find_package(pmacc REQUIRED CONFIG PATHS ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(SYSTEM ${pmacc_INCLUDE_DIRS})
set(LIBS ${LIBS} ${pmacc_LIBRARIES})
add_definitions(${pmacc_DEFINITIONS})

###############################################################################
# Boost.Test
###############################################################################
find_package(Boost 1.56.0 COMPONENTS unit_test_framework  REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})


###############################################################################
# Targets
###############################################################################

# Test cases
file(GLOB TESTS test/*UT.cu)
cuda_add_executable(check EXCLUDE_FROM_ALL ${TESTS})
target_link_libraries(check ${LIBS})

# CTest
enable_testing()
add_test(pmacc_test_build "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target check)
add_test(pmacc_test_run  mpiexec --allow-run-as-root -n 2 check )
set_tests_properties(pmacc_test_run PROPERTIES DEPENDS graybat_check_build)
