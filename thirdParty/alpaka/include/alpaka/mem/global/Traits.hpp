/* Copyright 2024 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Tag.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/meta/DependentFalseType.hpp"

namespace alpaka
{

    namespace detail
    {
        template<typename TTag, typename T>
        struct DevGlobalImplGeneric
        {
            // does not make use of TTag
            using Type = std::remove_const_t<T>;
            Type value; // backend specific value

            ALPAKA_FN_HOST_ACC T* operator&()
            {
                return &value;
            }

            ALPAKA_FN_HOST_ACC T& get()
            {
                return value;
            }
        };

        template<typename TTag, typename T>
        struct DevGlobalTrait
        {
            static constexpr bool const IsImplementedFor = alpaka::meta::DependentFalseType<TTag>::value;

            static_assert(IsImplementedFor, "Error: device global variables are not implemented for the given Tag");
        };
    } // namespace detail

    template<typename TAcc, typename T>
    using DevGlobal = typename detail::DevGlobalTrait<typename alpaka::trait::AccToTag<TAcc>::type, T>::Type;
} // namespace alpaka
