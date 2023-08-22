/* Copyright 2022 Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/OmpSchedule.hpp>

#include <catch2/catch_test_macros.hpp>

#include <tuple>

TEST_CASE("ompScheduleDefaultConstructor", "[core]")
{
    std::ignore = alpaka::omp::Schedule{};
}

TEST_CASE("ompScheduleConstructor", "[core]")
{
    std::ignore = alpaka::omp::Schedule{alpaka::omp::Schedule::Static, 5};
    std::ignore = alpaka::omp::Schedule{alpaka::omp::Schedule::Guided};
}

TEST_CASE("ompScheduleConstexprConstructor", "[core]")
{
    std::ignore = alpaka::omp::Schedule{alpaka::omp::Schedule::Dynamic};
}

TEST_CASE("ompGetSchedule", "[core]")
{
    std::ignore = alpaka::omp::getSchedule();
}

TEST_CASE("ompSetSchedule", "[core]")
{
    auto const expectedSchedule = alpaka::omp::Schedule{alpaka::omp::Schedule::Dynamic, 3};
    alpaka::omp::setSchedule(expectedSchedule);
    // The check makes sense only when this feature is supported
#if defined _OPENMP && _OPENMP >= 200805 && defined ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    auto const actualSchedule = alpaka::omp::getSchedule();
    REQUIRE(expectedSchedule.kind == actualSchedule.kind);
    REQUIRE(expectedSchedule.chunkSize == actualSchedule.chunkSize);
#endif
}

TEST_CASE("ompSetNoSchedule", "[core]")
{
    auto const expectedSchedule = alpaka::omp::Schedule{alpaka::omp::Schedule::Guided, 2};
    alpaka::omp::setSchedule(expectedSchedule);
    auto const noSchedule = alpaka::omp::Schedule{alpaka::omp::Schedule::NoSchedule};
    alpaka::omp::setSchedule(noSchedule);
    // The check makes sense only when this feature is supported
#if defined _OPENMP && _OPENMP >= 200805 && defined ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    auto const actualSchedule = alpaka::omp::getSchedule();
    REQUIRE(expectedSchedule.kind == actualSchedule.kind);
    REQUIRE(expectedSchedule.chunkSize == actualSchedule.chunkSize);
#endif
}
