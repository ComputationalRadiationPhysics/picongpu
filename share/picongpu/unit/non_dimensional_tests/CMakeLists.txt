# Handle tests without dimensionality
file(GLOB_RECURSE NON_DIMENSIONAL_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach(testCaseFilepath ${NON_DIMENSIONAL_TESTS})
    get_filename_component(testCaseFilename ${testCaseFilepath} NAME)
    string(REPLACE "UT.cpp" "" testCase ${testCaseFilename})
    set(testExe "${PROJECT_NAME}-${testCase}")
    alpaka_add_executable(${testExe} ${testCaseFilepath})
    target_include_directories(${testExe} PRIVATE BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include")
    target_link_libraries(${testExe} PUBLIC Catch2 Catch2WithMain)
    target_link_libraries(${testExe} PRIVATE pmacc::pmacc)
    add_test(
        NAME "${testCase}"
        COMMAND mpiexec ${MPI_RUNTIME_FLAGS} -n 1 ./${testExe}
    )
endforeach()
