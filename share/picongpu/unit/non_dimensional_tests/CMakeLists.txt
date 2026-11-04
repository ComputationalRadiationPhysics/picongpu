# Handle tests without dimensionality
file(GLOB_RECURSE NON_DIMENSIONAL_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach(testCaseFilepath ${NON_DIMENSIONAL_TESTS})
    get_filename_component(testCaseFilename ${testCaseFilepath} NAME)
    string(REPLACE "UT.cpp" "" testCase ${testCaseFilename})
    set(testExe "${PROJECT_NAME}-${testCase}")
    alpaka_add_executable(${testExe} ${testCaseFilepath})
    target_include_directories(${testExe} PRIVATE BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include")
    if(testCase STREQUAL "poissonStencil")
        # poissonStencilUT.cpp includes fields/poissonSolver/Stencil.hpp, whose
        # functors are defined next to the kernel that pulls in FieldTmpOperations
        # -> MaxwellSolver -> incidentField -> Free.hpp -> nlohmann/json.hpp.
        # The functors are not split into a light header yet (that header lives
        # under include/** and is out of scope here), so this single test needs
        # the nlohmann include directory that the main PIConGPU target already
        # gets. Scoped to this test so other tests still expose missing includes.
        target_include_directories(
            ${testExe}
            PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../../thirdParty/nlohmann_json/single_include"
        )
    endif()
    target_link_libraries(${testExe} PUBLIC Catch2 Catch2WithMain)
    target_link_libraries(${testExe} PRIVATE pmacc::pmacc)
    add_test(
        NAME "${testCase}"
        COMMAND mpiexec ${MPI_RUNTIME_FLAGS} -n 1 ./${testExe}
    )
endforeach()
