#pragma once

#include "alpaka/acc/Traits.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/vec/Vec.hpp"
#include "alpaka/workdiv/WorkDivMembers.hpp"

#include <type_traits>

// Create an accelerator-dependent work division for 1-dimensional kernels.
template<typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
inline auto makeWorkDiv(alpaka::Idx<TAcc> blocks, alpaka::Idx<TAcc> elements)
    -> alpaka::WorkDivMembers<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>>
{
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    if constexpr(alpaka::isMultiThreadAcc<TAcc>)
    {
        // On thread-parallel backends, each thread is looking at a single element:
        //   - the number of threads per block is "elements";   - the number of elements per thread is always 1.
        return WorkDiv{blocks, elements, Idx{1}};
    }
    else
    {
        // On thread-serial backends, run serially with a single thread per block:
        //   - the number of threads per block is always 1;   - the number of elements per thread is "elements".
        return WorkDiv{blocks, Idx{1}, elements};
    }

    ALPAKA_UNREACHABLE(WorkDiv{blocks, elements, Idx{1}});
}

// Create the accelerator-dependent workdiv for N-dimensional kernels.
template<typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
inline auto makeWorkDiv(
    alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& blocks,
    alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& elements)
    -> alpaka::WorkDivMembers<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>>
{
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;
    using Vec = alpaka::Vec<Dim, Idx>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    if constexpr(alpaka::isMultiThreadAcc<TAcc>)
    {
        // On thread-parallel backends, each thread is looking at a single element:
        //   - the number of threads per block is "elements";   - the number of elements per thread is always 1.
        return WorkDiv{blocks, elements, Vec::ones()};
    }
    else
    {
        // On thread-serial backends, run serially with a single thread per block:
        //   - the number of threads per block is always 1;   - the number of elements per thread is "elements".
        return WorkDiv{blocks, Vec::ones(), elements};
    }

    ALPAKA_UNREACHABLE(WorkDiv{blocks, elements, Vec::ones()});
}
