/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/queue/Traits.hpp"

#include <type_traits>
#include <vector>

namespace alpaka
{
    struct ConceptPlatform
    {
    };

    //! True if TPlatform is a platform, i.e. if it implements the ConceptPlatform concept.
    template<typename TPlatform>
    inline constexpr bool isPlatform = concepts::ImplementsConcept<ConceptPlatform, TPlatform>::value;

    //! The platform traits.
    namespace trait
    {
        //! The platform type trait.
        template<typename T, typename TSfinae = void>
        struct PlatformType;

        template<typename TPlatform>
        struct PlatformType<
            TPlatform,
            std::enable_if_t<concepts::ImplementsConcept<ConceptPlatform, TPlatform>::value>>
        {
            using type = typename concepts::ImplementationBase<ConceptDev, TPlatform>;
        };

        //! The device count get trait.
        template<typename T, typename TSfinae = void>
        struct GetDevCount;

        //! The device get trait.
        template<typename T, typename TSfinae = void>
        struct GetDevByIdx;
    } // namespace trait

    //! The platform type trait alias template to remove the ::type.
    template<typename T>
    using Platform = typename trait::PlatformType<T>::type;

    //! \return The device identified by its index.
    template<typename TPlatform>
    ALPAKA_FN_HOST auto getDevCount(TPlatform const& platform)
    {
        return trait::GetDevCount<TPlatform>::getDevCount(platform);
    }

    //! \return The device identified by its index.
    template<typename TPlatform>
    ALPAKA_FN_HOST auto getDevByIdx(TPlatform const& platform, std::size_t const& devIdx) -> Dev<TPlatform>
    {
        return trait::GetDevByIdx<TPlatform>::getDevByIdx(platform, devIdx);
    }

    //! \return All the devices available on this accelerator.
    template<typename TPlatform>
    ALPAKA_FN_HOST auto getDevs(TPlatform const& platform) -> std::vector<Dev<TPlatform>>
    {
        std::vector<Dev<TPlatform>> devs;

        std::size_t const devCount = getDevCount(platform);
        devs.reserve(devCount);
        for(std::size_t devIdx(0); devIdx < devCount; ++devIdx)
        {
            devs.push_back(getDevByIdx(platform, devIdx));
        }

        return devs;
    }

    namespace trait
    {
        template<typename TPlatform, typename TProperty>
        struct QueueType<
            TPlatform,
            TProperty,
            std::enable_if_t<concepts::ImplementsConcept<ConceptPlatform, TPlatform>::value>>
        {
            using type = typename QueueType<typename alpaka::trait::DevType<TPlatform>::type, TProperty>::type;
        };
    } // namespace trait
} // namespace alpaka
