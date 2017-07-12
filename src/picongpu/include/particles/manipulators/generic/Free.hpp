/* Copyright 2013-2017 Rene Widera, Axel Huebl
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
namespace generic
{
namespace acc
{
    /** wrapper for the user functor on the accelerator
     *
     * @tparam T_Functor user defined functor
     */
    template< typename T_Functor >
    struct Free : public T_Functor
    {
        //! type of the user functor
        using Functor = T_Functor;

        //! store user functor instance
        DINLINE Free( Functor const & functor ) :
            Functor( functor )
        {
        }

        /** execute the user functor
         *
         * @tparam T_Args type of the arguments passed to the user functor
         *
         * @param args arguments passed to the user functor
         */
        template< typename ... T_Args >
        DINLINE
        void operator( )( T_Args && ... args )
        {
            Functor::operator( )( args ... );
        }
    };
} // namespace acc

    template< typename T_Functor >
    struct Free : private T_Functor
    {

        using Functor = T_Functor;

        template< typename T_SpeciesType >
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
        template< typename DeferFunctor = Functor >
        HINLINE Free(
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
        HINLINE Free(
            uint32_t,
            typename std::enable_if<
                std::is_constructible< DeferFunctor >::value
            >::type* = 0
        ) : Functor( )
        {
        }

        /** create device functor
         *
         * @tparam T_WorkerCfg PMacc::mappings::threads::WorkerCfg, configuration of the worker
         *
         * @param offset (in supercells, without any guards) to the
         *         origin of the local domain
         * @param configuration of the worker
         */
        template< typename T_WorkerCfg >
        DINLINE acc::Free< Functor >
        operator()(
            DataSpace< simDim > const &,
            T_WorkerCfg const &
        )
        {
            return acc::Free< Functor >( *reinterpret_cast< Functor * >( this ) );
        }
    };

} // namespace generic
} // namespace manipulators
} // namespace particles
} // namespace picongpu
