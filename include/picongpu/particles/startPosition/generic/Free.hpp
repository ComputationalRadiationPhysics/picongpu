/*
 * SPDX-FileCopyrightText: Rene Widera, Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/startPosition/generic/Free.def"

#include <type_traits>
#include <utility>

namespace picongpu
{
    namespace particles
    {
        namespace startPosition
        {
            namespace generic
            {
                namespace acc
                {
                    /** wrapper for the user functor on the accelerator
                     *
                     * @tparam T_Functor user defined functor
                     */
                    template<typename T_Functor>
                    struct Free : private T_Functor
                    {
                        //! type of the user functor
                        using Functor = T_Functor;

                        //! store user functor instance
                        HDINLINE Free(Functor const& functor) : Functor(functor)
                        {
                        }

                        /** execute the user functor
                         *
                         * @tparam T_Args type of the arguments passed to the user functor
                         * @tparam T_Worker lockstep worker type
                         *
                         * @param worker lockstep worker
                         * @param args arguments passed to the user functor
                         */
                        template<typename T_Worker, typename T_Particle>
                        HDINLINE void operator()(T_Worker const&, T_Particle& particle)
                        {
                            Functor::operator()(particle);
                        }

                        template<typename T_Particle>
                        HDINLINE uint32_t numberOfMacroParticles(float_X const realParticlesPerCell)
                        {
                            return Functor::template numberOfMacroParticles<T_Particle>(realParticlesPerCell);
                        }
                    };
                } // namespace acc

                template<typename T_Functor>
                struct Free : protected T_Functor
                {
                    using Functor = T_Functor;

                    template<typename T_SpeciesType>
                    struct apply
                    {
                        using type = Free;
                    };

                    /** constructor
                     *
                     * This constructor is only compiled if the user functor has
                     * a host side constructor with one (uint32_t) argument.
                     *
                     * @tparam DeferFunctor is used to defer the functor type evaluation to enable/disable
                     *                      the constructor
                     * @param currentStep current simulation time step
                     * @param is used to enable/disable the constructor (do not pass any value to this parameter)
                     */
                    template<typename DeferFunctor = Functor>
                    HINLINE Free(
                        uint32_t currentStep,
                        std::enable_if_t<
                            !std::is_default_constructible_v<DeferFunctor>
                            && std::is_constructible_v<DeferFunctor, uint32_t>>* = 0)
                        : Functor(currentStep)
                    {
                    }

                    /** constructor
                     *
                     * This constructor is only compiled if the user functor has a default constructor.
                     *
                     * @tparam DeferFunctor is used to defer the functor type evaluation to enable/disable
                     *                      the constructor
                     * @param current simulation time step
                     * @param is used to enable/disable the constructor (do not pass any value to this parameter)
                     */
                    template<typename DeferFunctor = Functor>
                    HINLINE Free(uint32_t, std::enable_if_t<std::is_default_constructible_v<DeferFunctor>>* = nullptr)
                        : Functor()
                    {
                    }

                    /** create device functor
                     *
                     * @tparam T_Worker lockstep worker type
                     *
                     * @param worker lockstep worker
                     * @param offset (in supercells, without any guards) to the
                     *         origin of the local domain
                     * @param configuration of the worker
                     */
                    template<typename T, typename T_Worker>
                    HDINLINE acc::Free<Functor> operator()(T_Worker const& worker, T const&) const
                    {
                        return acc::Free<Functor>(*static_cast<Functor const*>(this));
                    }
                };

            } // namespace generic
        } // namespace startPosition
    } // namespace particles
} // namespace picongpu
