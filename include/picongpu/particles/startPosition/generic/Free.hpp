/* Copyright 2013-2023 Rene Widera, Axel Huebl
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
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
                            !std::is_default_constructible_v<
                                DeferFunctor> && std::is_constructible_v<DeferFunctor, uint32_t>>* = 0)
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
