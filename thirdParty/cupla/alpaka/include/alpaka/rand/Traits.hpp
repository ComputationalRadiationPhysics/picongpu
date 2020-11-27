/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>

#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The random number generation specifics.
    namespace rand
    {
        struct ConceptRand
        {
        };

        //-----------------------------------------------------------------------------
        //! The random number generator distribution specifics.
        namespace distribution
        {
            //-----------------------------------------------------------------------------
            //! The random number generator distribution traits.
            namespace traits
            {
                //#############################################################################
                //! The random number float normal distribution get trait.
                template<typename TRand, typename T, typename TSfinae = void>
                struct CreateNormalReal;

                //#############################################################################
                //! The random number float uniform distribution get trait.
                template<typename TRand, typename T, typename TSfinae = void>
                struct CreateUniformReal;

                //#############################################################################
                //! The random number integer uniform distribution get trait.
                template<typename TRand, typename T, typename TSfinae = void>
                struct CreateUniformUint;
            } // namespace traits

            //-----------------------------------------------------------------------------
            //! \return A normal float distribution with mean 0.0f and standard deviation 1.0f.
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename T, typename TRand>
            ALPAKA_FN_HOST_ACC auto createNormalReal(TRand const& rand)
            {
                static_assert(std::is_floating_point<T>::value, "The value type T has to be a floating point type!");

                using ImplementationBase = concepts::ImplementationBase<ConceptRand, TRand>;
                return traits::CreateNormalReal<ImplementationBase, T>::createNormalReal(rand);
            }
            //-----------------------------------------------------------------------------
            //! \return A uniform floating point distribution [0.0, 1.0).
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename T, typename TRand>
            ALPAKA_FN_HOST_ACC auto createUniformReal(TRand const& rand)
            {
                static_assert(std::is_floating_point<T>::value, "The value type T has to be a floating point type!");

                using ImplementationBase = concepts::ImplementationBase<ConceptRand, TRand>;
                return traits::CreateUniformReal<ImplementationBase, T>::createUniformReal(rand);
            }
            //-----------------------------------------------------------------------------
            //! \return A uniform integer distribution [0, UINT_MAX].
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename T, typename TRand>
            ALPAKA_FN_HOST_ACC auto createUniformUint(TRand const& rand)
            {
                static_assert(
                    std::is_integral<T>::value && std::is_unsigned<T>::value,
                    "The value type T has to be a unsigned integral type!");

                using ImplementationBase = concepts::ImplementationBase<ConceptRand, TRand>;
                return traits::CreateUniformUint<ImplementationBase, T>::createUniformUint(rand);
            }
        } // namespace distribution

        //-----------------------------------------------------------------------------
        //! The random number generator specifics.
        namespace generator
        {
            //-----------------------------------------------------------------------------
            //! The random number generator traits.
            namespace traits
            {
                //#############################################################################
                //! The random number default generator get trait.
                template<typename TRand, typename TSfinae = void>
                struct CreateDefault;
            } // namespace traits
            //-----------------------------------------------------------------------------
            //! \return A default random number generator.
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TRand>
            ALPAKA_FN_HOST_ACC auto createDefault(
                TRand const& rand,
                std::uint32_t const& seed,
                std::uint32_t const& subsequence)
            {
                using ImplementationBase = concepts::ImplementationBase<ConceptRand, TRand>;
                return traits::CreateDefault<ImplementationBase>::createDefault(rand, seed, subsequence);
            }
        } // namespace generator
    } // namespace rand
} // namespace alpaka
