/* Copyright 2017-2023 Axel Huebl, Rene Widera, Sergei Bastrakov
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/lockstep/Variable.hpp"
#include "pmacc/particles/memory/dataTypes/FramePointer.hpp"
#include "pmacc/particles/memory/dataTypes/Particle.hpp"

#include <utility>


namespace pmacc::particles::algorithm::acc
{
    namespace detail
    {
        //! Tag marks that the particle functor interface must be used
        struct CallParticleFunctor;

        //! Tag marks that the frame functor interface must be used
        struct CallFrameFunctor;

        /** Frame functor interface
         *
         * Ensure that a functor can be called with the given lockstep frame variable.
         *
         * @tparam T_FrameFunctor Type of the user functor to operate on frames.
         */
        template<typename T_FrameFunctor>
        struct FrameFunctorInterface
        {
            T_FrameFunctor m_functor;
            DINLINE FrameFunctorInterface(T_FrameFunctor&& functor) : m_functor(std::forward<T_FrameFunctor>(functor))
            {
            }

            /** Invokes the user functor with the given parameters.
             *
             * @tparam T_Worker lockstep worker type
             * @tparam T_FrameType @see FramePointer
             * @tparam T_Config @see lockstep::Variable
             * @param acc alpaka accelerator
             * @param frameIterCtx Lockstep variable containing a frame for each virtual worker. To operate on this
             *                     parameter @see ForEachParticle::lockstepForEach should be used.
             *
             * @{
             */
            template<typename T_Worker, typename T_FrameType, typename T_Config>
            DINLINE void operator()(
                T_Worker const& worker,
                lockstep::Variable<FramePointer<T_FrameType>, T_Config>& frameIterCtx)
            {
                m_functor(worker, frameIterCtx);
            }

            template<typename T_Worker, typename T_FrameType, typename T_Config>
            DINLINE void operator()(
                T_Worker const& worker,
                lockstep::Variable<FramePointer<T_FrameType>, T_Config>& frameIterCtx) const
            {
                m_functor(worker, frameIterCtx);
            }
            /**@}*/
        };

        /** Factory to create a particle interface functor.
         *
         * @tparam T_FrameFunctor Type of the user frame functor.
         * @param frameFunctor Frame functor which should be wrapped to check the interface.
         * @return Callable functor which guarantees the frame interface used by forEach.
         */
        template<typename T_FrameFunctor>
        DINLINE auto makeFrameFunctorInterface(T_FrameFunctor&& frameFunctor)
        {
            return FrameFunctorInterface<T_FrameFunctor>{std::forward<T_FrameFunctor>(frameFunctor)};
        }

        /** Particle functor interface
         *
         * Ensure that a functor can be called for a particle.
         *
         * @tparam T_ParticleFunctor Type of the user particle functor to operate with a single particle.
         */
        template<typename T_ParticleFunctor>
        struct ParticleFunctorInterface
        {
            T_ParticleFunctor m_functor;
            DINLINE ParticleFunctorInterface(T_ParticleFunctor&& functor)
                : m_functor(std::forward<T_ParticleFunctor>(functor))
            {
            }

            /** Invoke the user functor with the given arguments.
             *
             * @tparam T_Worker lockstep worker type
             * @tparam T_FrameType @see Particle
             * @tparam T_ValueTypeSeq @see Particle
             * @param worker lockstep worker
             * @param particle particle to process
             *
             * @{
             */
            template<typename T_Worker, typename T_FrameType, typename T_ValueTypeSeq>
            DINLINE void operator()(T_Worker const& worker, Particle<T_FrameType, T_ValueTypeSeq>& particle)
            {
                m_functor(worker, particle);
            }

            template<typename T_Worker, typename T_FrameType, typename T_ValueTypeSeq>
            DINLINE void operator()(T_Worker const& worker, Particle<T_FrameType, T_ValueTypeSeq>& particle) const
            {
                m_functor(worker, particle);
            }

            /**@}*/
        };

        /** Factory to create a particle interface functor.
         *
         * @tparam T_ParticleFunctor Type of the user particle functor to operate with a single particle.
         * @param particleFunctor Particle functor which should be wrapped to check the interface.
         * @return Callable functor which guarantees the particle interface used by forEach.
         */
        template<typename T_ParticleFunctor>
        DINLINE auto makeParticleFunctorInterface(T_ParticleFunctor&& particleFunctor)
        {
            return ParticleFunctorInterface<T_ParticleFunctor>{std::forward<T_ParticleFunctor>(particleFunctor)};
        }

    } // namespace detail
} // namespace pmacc::particles::algorithm::acc
