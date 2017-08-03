/* Copyright 2015-2017 Rene Widera, Alexander Grund
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
#include "picongpu/particles/startPosition/generic/FreeRng.def"
#include "picongpu/particles/manipulators/generic/detail/Rng.hpp"

#include <utility>
#include <type_traits>
#include <string>


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
    template<
        typename T_Functor,
        typename T_RngType
    >
    struct FreeRng : private T_Functor
    {

        using Functor = T_Functor;
        using RngType = T_RngType;

        DINLINE FreeRng(
            Functor const & functor,
            RngType const & rng
        ) :
            T_Functor( functor ), m_rng( rng )
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
        template<
            typename T_Particle,
            typename ... T_Args,
            typename T_Acc
        >
        DINLINE
        void operator()(
            T_Acc const &,
            T_Particle& particle,
            T_Args && ... args
        )
        {
            namespace nvrng = nvidia::rng;

            Functor::operator()(
                m_rng,
                particle,
                args ...
            );
        }

        DINLINE uint32_t
        numberOfMacroParticles( float_X const realParticlesPerCell )
        {
            return Functor::numberOfMacroParticles( realParticlesPerCell );
        }

    private:

        RngType m_rng;
    };
} // namespace acc

    template<
        typename T_Functor,
        typename T_Distribution,
        typename T_Seed,
        typename T_SpeciesType
    >
    struct FreeRng :
        protected T_Functor,
        private picongpu::particles::manipulators::generic::detail::Rng<
            T_Distribution,
            T_Seed,
            T_SpeciesType
        >
    {
        using RngGenerator = picongpu::particles::manipulators::generic::detail::Rng<
            T_Distribution,
            T_Seed,
            T_SpeciesType
        >;

        template< typename T_Acc >
        using RngType = typename RngGenerator::template RngType< T_Acc >;

        using Functor = T_Functor;
        using Distribution = T_Distribution;
        using SpeciesType = T_SpeciesType;

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
        template< typename DeferFunctor = Functor >
        HINLINE FreeRng(
            uint32_t currentStep,
            typename std::enable_if<
                std::is_constructible<
                    DeferFunctor,
                    uint32_t
                >::value
            >::type* = 0
        ) :
            Functor( currentStep ),
            RngGenerator( currentStep )
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
        template< typename DeferFunctor = Functor >
        HINLINE FreeRng(
            uint32_t currentStep,
            typename std::enable_if<
                std::is_constructible< DeferFunctor >::value
            >::type* = 0
        ) :
            Functor( ),
            RngGenerator( currentStep )
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
        template<
            typename T_WorkerCfg,
            typename T_Acc
        >
        DINLINE auto
        operator()(
            T_Acc const & acc,
            DataSpace< simDim > const & localSupercellOffset,
            T_WorkerCfg const & workerCfg
        )
        -> acc::FreeRng<
            Functor,
            RngType< T_Acc >
        >
        {
            RngType< T_Acc > const rng = ( *reinterpret_cast< RngGenerator * >( this ) )(
                localSupercellOffset,
                workerCfg
            );

            return acc::FreeRng<
                Functor,
                RngType< T_Acc >
            >(
                *reinterpret_cast< Functor * >( this ),
                rng
            );
        }

        HINLINE std::string
        getName( ) const
        {
            return std::string("FreeRNG");
        }
    };

} // namepsace generic
} // namespace startPosition
} // namespace particles
} // namespace picongpu
