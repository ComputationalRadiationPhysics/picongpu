/* Copyright 2017-2023 Rene Widera
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/filter/generic/FreeRng.def"
#include "picongpu/particles/functor/User.hpp"
#include "picongpu/particles/functor/misc/Rng.hpp"

#include <string>


namespace picongpu
{
    namespace particles
    {
        namespace filter
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
                        HDINLINE bool operator()(T_Worker const&, T_Particle const& particle)
                        {
                            bool const isValid = particle.isHandleValid();

                            return isValid && Functor::operator()(m_rng, particle);
                        }

                    private:
                        RngType m_rng;
                    };
                } // namespace acc

                template<typename T_Functor, typename T_Distribution>
                struct FreeRng
                    : protected functor::User<T_Functor>
                    , private picongpu::particles::functor::misc::Rng<T_Distribution>
                {
                    template<typename T_SpeciesType>
                    struct apply
                    {
                        using type = FreeRng;
                    };

                    using RngGenerator = picongpu::particles::functor::misc::Rng<T_Distribution>;

                    using Functor = functor::User<T_Functor>;
                    using Distribution = T_Distribution;

                    /** constructor
                     *
                     * @param currentStep current simulation time step
                     */
                    HINLINE FreeRng(uint32_t currentStep, IdGenerator idGen)
                        : Functor(currentStep, idGen)
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
                     * @param blockCfg configuration of the worker
                     */
                    template<typename T_Worker>
                    HDINLINE auto operator()(T_Worker const& worker, DataSpace<simDim> const& localSupercellOffset)
                        const
                    {
                        auto const rng = (*static_cast<RngGenerator const*>(this))(worker, localSupercellOffset);

                        return acc::FreeRng<Functor, std::decay_t<decltype(rng)>>(
                            *static_cast<Functor const*>(this),
                            rng);
                    }

                    HINLINE static std::string getName()
                    {
                        // we provide the name from the param class
                        return Functor::name;
                    }

                    /** A filter is deterministic if the filter outcome is equal between evaluations. If so, set this
                     * variable to true, otherwise to false.
                     *
                     * Example: A filter were results depend on a random number generator must return false.
                     */
                    static constexpr bool isDeterministic = Functor::isDeterministic;
                };

            } // namespace generic
        } // namespace filter
    } // namespace particles
} // namespace picongpu
