#pragma once

// include all Acc's because of the struct AccIsEnabled
// if an acc is not include, it will be not enabled independent of the compiler flags
#include "alpaka/acc/AccCpuOmp2Blocks.hpp"
#include "alpaka/acc/AccCpuOmp2Threads.hpp"
#include "alpaka/acc/AccCpuSerial.hpp"
#include "alpaka/acc/AccCpuSycl.hpp"
#include "alpaka/acc/AccCpuTbbBlocks.hpp"
#include "alpaka/acc/AccCpuThreads.hpp"
#include "alpaka/acc/AccFpgaSyclIntel.hpp"
#include "alpaka/acc/AccGpuCudaRt.hpp"
#include "alpaka/acc/AccGpuHipRt.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/meta/Filter.hpp"

#include <type_traits>

namespace alpaka
{
    //! \brief check if the accelerator is enabled for a given tag
    //! \tparam TTag alpaka tag type
    template<typename TTag, typename = void>
    struct AccIsEnabled : std::false_type
    {
    };

    template<typename TTag>
    struct AccIsEnabled<TTag, std::void_t<TagToAcc<TTag, alpaka::DimInt<1>, int>>> : std::true_type
    {
    };

    //! list of all tags where the related accelerator is enabled
    using EnabledAccTags = alpaka::meta::Filter<AccTags, alpaka::AccIsEnabled>;

} // namespace alpaka
