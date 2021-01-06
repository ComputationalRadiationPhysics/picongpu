/* Copyright 2015-2021 Rene Widera, Alexander Grund
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
#include "picongpu/particles/manipulators/generic/FreeRng.def"
#include "picongpu/particles/functor/misc/Rng.hpp"
#include "picongpu/particles/functor/User.hpp"

#include <utility>
#include <type_traits>
#include <string>


namespace picongpu
{
    namespace particles
    {
        namespace manipulators
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
                         * @tparam T_Acc alpaka accelerator type
                         *
                         * @param alpaka accelerator
                         * @param particle particle which is given to the user functor
                         * @return void is used to enable the operator if the user functor except two arguments
                         */
                        template<typename T_Particle, typename... T_Args, typename T_Acc>
                        HDINLINE void operator()(T_Acc const&, T_Particle& particle, T_Args&&... args)
                        {
                            Functor::operator()(m_rng, particle, args...);
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

                    using RngType = typename RngGenerator::RandomGen;

                    using Functor = functor::User<T_Functor>;
                    using Distribution = T_Distribution;

                    /** constructor
                     *
                     * @param currentStep current simulation time step
                     */
                    HINLINE FreeRng(uint32_t currentStep) : Functor(currentStep), RngGenerator(currentStep)
                    {
                    }

                    /** create functor for the accelerator
                     *
                     * @tparam T_WorkerCfg pmacc::mappings::threads::WorkerCfg, configuration of the worker
                     * @tparam T_Acc alpaka accelerator type
                     *
                     * @param alpaka accelerator
                     * @param localSupercellOffset offset (in superCells, without any guards) relative
                     *                        to the origin of the local domain
                     * @param workerCfg configuration of the worker
                     */
                    template<typename T_WorkerCfg, typename T_Acc>
                    HDINLINE auto operator()(
                        T_Acc const& acc,
                        DataSpace<simDim> const& localSupercellOffset,
                        T_WorkerCfg const& workerCfg) const -> acc::FreeRng<Functor, RngType>
                    {
                        RngType const rng
                            = (*static_cast<RngGenerator const*>(this))(acc, localSupercellOffset, workerCfg);

                        return acc::FreeRng<Functor, RngType>(*static_cast<Functor const*>(this), rng);
                    }

                    static HINLINE std::string getName()
                    {
                        // we provide the name from the param class
                        return Functor::name;
                    }
                };

            } // namespace generic
        } // namespace manipulators
    } // namespace particles
} // namespace picongpu
