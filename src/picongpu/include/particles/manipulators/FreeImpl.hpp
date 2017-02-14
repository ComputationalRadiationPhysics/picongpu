/**
 * Copyright 2013-2017 Rene Widera, Axel Huebl
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

#include "simulation_defines.hpp"

#include <utility>
#include <type_traits>

namespace picongpu
{
namespace particles
{
namespace manipulators
{

    /** generic manipulator to create user defined manipulators
     *
     * @tparam T_Functor user defined functor
     *              - must implement `void operator()(ParticleType)` **or** `void operator()(ParticleType1, ParticleType2)`
     *              - **optional**: can implement **one** host side constructor `T_Functor()` or `T_Functor(uint32_t currentTimeStep)`
     */
    template< typename T_Functor >
    struct FreeImpl : private T_Functor
    {

        typedef T_Functor Functor;

        template< typename T_SpeciesType >
        struct apply
        {
            typedef FreeImpl< T_Functor > type;
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
        template< typename DeferFunctor = Functor >
        HINLINE FreeImpl(
            uint32_t currentStep,
            typename std::enable_if<
                std::is_constructible<
                    DeferFunctor,
                    uint32_t
                >::value
            >::type* = 0
        ) : Functor( currentStep )
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
        template< typename DeferFunctor = Functor >
        HINLINE FreeImpl(
            uint32_t,
            typename std::enable_if<
                std::is_constructible< DeferFunctor >::value
            >::type* = 0
        ) : Functor( )
        {
        }

        /** call user functor
         *
         * This method is only compiled if the user functor is a binary particle functor.
         * The user functor is called if \p isParticle1 and \p isParticle2 are valid.
         *
         * @param cell index within the local volume
         * @param particleSpecies1 first particle
         * @param particleSpecies2 second particle, can be equal to the first particle
         * @param isParticle1 define if the reference @p particleSpecies1 is valid
         * @param isParticle2 define if the reference @p particleSpecies2 is valid
         * @return void is used to enable the operator if the user functor except two arguments
         */
        template<
            typename T_Particle1,
            typename T_Particle2
        >
        DINLINE auto operator()(
            DataSpace<simDim> const &,
            T_Particle1 & particleSpecies1,
            T_Particle2 & particleSpecies2,
            bool const isParticle1,
            bool const isParticle2
        )
        -> decltype(
            std::declval< Functor >()(
                particleSpecies1,
                particleSpecies2
            )
        )
        {
            if( isParticle1 && isParticle2 )
            {
                Functor::operator()(
                    particleSpecies1,
                    particleSpecies2
                );
            }
        }

        /** call user functor
         *
         * This method is only compiled if the user functor is a binary particle functor.
         * The user functor is called if \p isParticle1 is valid.
         *
         * @param cell index within the local volume
         * @param particleSpecies1 first particle
         * @param unused
         * @param isParticle1 define if the reference @p particleSpecies1 is valid
         * @param unused
         * @return void is used to enable the operator if user the functor except one argument
         */
        template<typename T_Particle1, typename T_Particle2>
        DINLINE auto operator()(
            DataSpace<simDim> const &,
            T_Particle1 & particleSpecies1,
            T_Particle2 &,
            bool const isParticle1,
            bool const
        )
        -> decltype(
            std::declval<Functor>()
            ( particleSpecies1 )
        )
        {
            if( isParticle1 )
            {
                Functor::operator()( particleSpecies1 );
            }
        }

    };

} //namespace manipulators
} //namespace particles
} //namespace picongpu
