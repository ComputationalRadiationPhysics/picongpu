/* Copyright 2015-2022 Rene Widera, Alexander Grund
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/functor/misc/Rng.hpp"
#include "picongpu/particles/startPosition/generic/FreeRng.def"

#include <string>
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
                    template<typename T_Functor, typename T_RngType>
                    struct FreeRng : private T_Functor
                    {
                        using Functor = T_Functor;
                        using RngType = T_RngType;

                        HDINLINE FreeRng(Functor const& functor, RngType const& rng) : T_Functor(functor), m_rng(rng)
                        {
                        }

                        /** call user functor
                         *
                         * The random number generator is initialized with the first call.
                         *
                         * @tparam T_Particle type of the particle to manipulate
                         * @tparam T_Args type of the arguments passed to the user functor
                         * @tparam T_Worker lockstep worker type
                         *
                         * @param worker lockstep worker
                         * @param particle particle which is given to the user functor
                         * @return void is used to enable the operator if the user functor except two arguments
                         */
                        template<typename T_Particle, typename... T_Args, typename T_Worker>
                        HDINLINE void operator()(T_Worker const&, T_Particle& particle, T_Args&&... args)
                        {
                            Functor::operator()(m_rng, particle, args...);
                        }

                        template<typename T_Particle>
                        HDINLINE uint32_t numberOfMacroParticles(float_X const realParticlesPerCell)
                        {
                            return Functor::template numberOfMacroParticles<T_Particle>(realParticlesPerCell);
                        }

                    private:
                        RngType m_rng;
                    };
                } // namespace acc

                template<typename T_Functor, typename T_Distribution>
                struct FreeRng
                    : protected T_Functor
                    , private picongpu::particles::functor::misc::Rng<T_Distribution>
                {
                    template<typename T_SpeciesType>
                    using fn = FreeRng;

                    using RngGenerator = picongpu::particles::functor::misc::Rng<T_Distribution>;

                    using Functor = T_Functor;
                    using Distribution = T_Distribution;

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
                    HINLINE FreeRng(
                        uint32_t currentStep,
                        std::enable_if_t<std::is_constructible_v<DeferFunctor, uint32_t>>* = 0)
                        : Functor(currentStep)
                        , RngGenerator(currentStep)
                    {
                    }

                    /** constructor
                     *
                     * This constructor is only compiled if the user functor has a default constructor.
                     *
                     * @tparam DeferFunctor is used to defer the functor type evaluation to enable/disable
                     *                      the constructor
                     * @param currentStep simulation time step
                     * @param is used to enable/disable the constructor (do not pass any value to this parameter)
                     */
                    template<typename DeferFunctor = Functor>
                    HINLINE FreeRng(uint32_t currentStep, std::enable_if_t<std::is_constructible_v<DeferFunctor>>* = 0)
                        : Functor()
                        , RngGenerator(currentStep)
                    {
                    }

                    /** create functor for the accelerator
                     *
                     * @tparam T_Worker lockstep worker type
                     *
                     * @param worker lockstep worker
                     * @param localSupercellOffset offset (in superCells, without any guards) relative
                     *                        to the origin of the local domain
                     */
                    template<typename T_Worker>
                    HDINLINE auto operator()(T_Worker const& worker, DataSpace<simDim> const& localSupercellOffset)
                        const
                    {
                        auto const rng = (*static_cast<RngGenerator const*>(this))(worker, localSupercellOffset);

                        return acc::FreeRng<Functor, ALPAKA_DECAY_T(decltype(rng))>(
                            *static_cast<Functor const*>(this),
                            rng);
                    }

                    static HINLINE std::string getName()
                    {
                        return std::string("FreeRNG");
                    }
                };

            } // namespace generic
        } // namespace startPosition
    } // namespace particles
} // namespace picongpu
