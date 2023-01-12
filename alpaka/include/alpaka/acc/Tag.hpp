/* Copyright 2022 Simeon Ehrig, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>

#include <iostream>
#include <type_traits>

#define CREATE_ACC_TAG(tag_name)                                                                                      \
    struct tag_name                                                                                                   \
    {                                                                                                                 \
        static std::string get_name()                                                                                 \
        {                                                                                                             \
            return #tag_name;                                                                                         \
        }                                                                                                             \
    }

namespace alpaka
{
    CREATE_ACC_TAG(TagCpuOmp2Blocks);
    CREATE_ACC_TAG(TagCpuOmp2Threads);
    CREATE_ACC_TAG(TagCpuSerial);
    CREATE_ACC_TAG(TagCpuSyclIntel);
    CREATE_ACC_TAG(TagCpuTbbBlocks);
    CREATE_ACC_TAG(TagCpuThreads);
    CREATE_ACC_TAG(TagFpgaSyclIntel);
    CREATE_ACC_TAG(TagFpgaSyclXilinx);
    CREATE_ACC_TAG(TagGenericSycl);
    CREATE_ACC_TAG(TagGpuCudaRt);
    CREATE_ACC_TAG(TagGpuHipRt);
    CREATE_ACC_TAG(TagGpuSyclIntel);
    CREATE_ACC_TAG(TagOacc);
    CREATE_ACC_TAG(TagOmp5);

    namespace trait
    {
        template<typename TAcc>
        struct AccToTag;

        template<typename TAcc>
        using AccToTagType = typename AccToTag<TAcc>::type;

        template<typename TTag, typename TDim, typename TIdx>
        struct TagToAcc;

        template<typename TTag, typename TDim, typename TIdx>
        using TagToAccType = typename TagToAcc<TTag, TDim, TIdx>::type;
    } // namespace trait


/** \todo: Remove the following pragmas once support for clang 6 is removed. They are necessary because
        these /  clang versions incorrectly warn about a missing 'extern'. */
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-variable-declarations"
#endif
    template<typename TAcc, typename... TTag>
    inline constexpr bool accMatchesTags = (std::is_same_v<alpaka::trait::AccToTagType<TAcc>, TTag> || ...);
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
} // namespace alpaka
