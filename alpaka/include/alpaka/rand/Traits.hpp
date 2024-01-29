/* Copyright 2023 Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"

#include <cstdint>
#include <type_traits>

namespace alpaka::rand
{
    struct ConceptRand
    {
    };

    //! The random number generator distribution specifics.
    namespace distribution
    {
        //! The random number generator distribution trait.
        namespace trait
        {
            //! The random number float normal distribution get trait.
            template<typename TRand, typename T, typename TSfinae = void>
            struct CreateNormalReal;

            //! The random number float uniform distribution get trait.
            template<typename TRand, typename T, typename TSfinae = void>
            struct CreateUniformReal;

            //! The random number integer uniform distribution get trait.
            template<typename TRand, typename T, typename TSfinae = void>
            struct CreateUniformUint;
        } // namespace trait

        //! \return A normal float distribution with mean 0.0f and standard deviation 1.0f.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TRand>
        ALPAKA_FN_HOST_ACC auto createNormalReal(TRand const& rand)
        {
            static_assert(std::is_floating_point_v<T>, "The value type T has to be a floating point type!");

            using ImplementationBase = concepts::ImplementationBase<ConceptRand, TRand>;
            return trait::CreateNormalReal<ImplementationBase, T>::createNormalReal(rand);
        }

        //! \return A uniform floating point distribution [0.0, 1.0).
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TRand>
        ALPAKA_FN_HOST_ACC auto createUniformReal(TRand const& rand)
        {
            static_assert(std::is_floating_point_v<T>, "The value type T has to be a floating point type!");

            using ImplementationBase = concepts::ImplementationBase<ConceptRand, TRand>;
            return trait::CreateUniformReal<ImplementationBase, T>::createUniformReal(rand);
        }

        //! \return A uniform integer distribution [0, UINT_MAX].
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TRand>
        ALPAKA_FN_HOST_ACC auto createUniformUint(TRand const& rand)
        {
            static_assert(
                std::is_integral_v<T> && std::is_unsigned_v<T>,
                "The value type T has to be a unsigned integral type!");

            using ImplementationBase = concepts::ImplementationBase<ConceptRand, TRand>;
            return trait::CreateUniformUint<ImplementationBase, T>::createUniformUint(rand);
        }
    } // namespace distribution

    //! The random number generator engine specifics.
    namespace engine
    {
        //! The random number generator engine trait.
        namespace trait
        {
            //! The random number default generator engine get trait.
            template<typename TRand, typename TSfinae = void>
            struct CreateDefault;
        } // namespace trait

        //! \return A default random number generator engine. Its type is guaranteed to be trivially copyable.
        //!         Except HIP accelerator for HIP versions below 5.2 as its internal state was not trivially copyable.
        //!         The limitation was discussed in PR #1778.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TRand>
        ALPAKA_FN_HOST_ACC auto createDefault(
            TRand const& rand,
            std::uint32_t const& seed = 0,
            std::uint32_t const& subsequence = 0,
            std::uint32_t const& offset = 0)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptRand, TRand>;
            return trait::CreateDefault<ImplementationBase>::createDefault(rand, seed, subsequence, offset);
        }
    } // namespace engine
} // namespace alpaka::rand
