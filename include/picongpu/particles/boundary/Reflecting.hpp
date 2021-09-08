/* Copyright 2021 Sergei Bastrakov
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

#include "picongpu/particles/Manipulate.hpp"
#include "picongpu/particles/boundary/ApplyImpl.hpp"
#include "picongpu/particles/boundary/Kind.hpp"

#include <pmacc/mappings/kernel/ExchangeMapping.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>

#include <cstdint>
#include <stdexcept>


namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            namespace detail
            {
                //! Our functor that actually processes particles on device
                struct ReflectParticle
                {
                    //! Some name is required
                    static constexpr char const* name = "reflectParticle";

                    /** Process the current particle located in the given supercell
                     *
                     * @param localSupercellOffset supercell offset (in superCells, without any guards) relative
                     *                             to the origin of the local domain
                     *                             when running in GUARD on "left" side will be negative
                     * @param particle handle of particle to process (can be used to change attribute values)
                     */
                    template<typename T_Particle>
                    HDINLINE void operator()(DataSpace<simDim> const localSupercellOffset, T_Particle& particle)
                    {
                        /// @Lennert implement the reflection logic here
                        /// If you need more additional data, do similar to how localSupercellOffset is passed
                        /// (see FreeSupercellOffset and acc::FreeSupercellOffset below)

                        /* Change attributes of particle here:
                         * probably position_, momentum_, localCellIdx_, etc.
                         * Note that you can't directly delete or move a particle between supercells here,
                         * instead you set such status via multiMask_
                         */
                        // Translate from localCellIdx_ to a 2d/3d cell index inside supercell
                        DataSpace<simDim> const cellInSuperCell(
                            DataSpaceOperations<simDim>::template map<SuperCellSize>(particle[localCellIdx_]));
                    }
                };

                namespace acc
                {
                    /** This functor will be created on the device.
                     *
                     * It wraps our functor given as template parameter.
                     * It saves index of the current supercell and pass it to our functor in operator().
                     */
                    template<typename T_Functor>
                    struct FreeSupercellOffset : private T_Functor
                    {
                        using Functor = T_Functor;

                        /** @param localSupercellOffset offset (in superCells, without any guards) relative
                         *                             to the origin of the local domain
                         */
                        HDINLINE FreeSupercellOffset(
                            Functor const& functor,
                            DataSpace<simDim> const& localSupercellOffset)
                            : T_Functor(functor)
                            , m_localSupercellOffset(localSupercellOffset)
                        {
                        }

                        /** call user functor
                         *
                         * @tparam T_Particle type of the particle to manipulate
                         * @tparam T_Args type of the arguments passed to the user functor
                         * @tparam T_Acc alpaka accelerator type
                         *
                         * @param alpaka accelerator
                         * @param particle particle which is given to the user functor
                         * @return void is used to enable the operator if the user functor expects two arguments
                         */
                        template<typename T_Particle, typename T_Acc>
                        HDINLINE void operator()(T_Acc const&, T_Particle& particle)
                        {
                            Functor::operator()(m_localSupercellOffset, particle);
                        }

                    private:
                        //! offset (in superCells, without any guards) relative to the origin of the local domain
                        DataSpace<simDim> const m_localSupercellOffset;
                    };
                } // namespace acc

                /** Wrapper to pass to Manipulate
                 *
                 * This is basically like particles::manipulators::unary::FreeTotalCellOffset.
                 * Maybe we will move this away to something similar.
                 * However, as we may need more data than just supercell offset, for now it's easier to keep it here
                 */
                template<typename T_Functor>
                struct FreeSupercellOffset : protected functor::User<T_Functor>
                {
                    using Functor = functor::User<T_Functor>;

                    template<typename T_SpeciesType>
                    struct apply
                    {
                        using type = FreeSupercellOffset;
                    };

                    /** constructor
                     *
                     * @param currentStep current simulation time step
                     */
                    HINLINE FreeSupercellOffset(uint32_t currentStep) : Functor(currentStep)
                    {
                    }

                    /** create functor for the accelerator
                     *
                     * @tparam T_WorkerCfg lockstep::Worker, configuration of the worker
                     * @tparam T_Acc alpaka accelerator type
                     *
                     * @param alpaka accelerator
                     * @param localSupercellOffset offset (in superCells, without any guards) relative
                     *                             to the origin of the local domain
                     * @param workerCfg configuration of the worker
                     */
                    template<typename T_WorkerCfg, typename T_Acc>
                    HDINLINE auto operator()(
                        T_Acc const& acc,
                        DataSpace<simDim> const& localSupercellOffset,
                        T_WorkerCfg const& workerCfg) const
                    {
                        return acc::FreeSupercellOffset<Functor>(
                            *static_cast<Functor const*>(this),
                            localSupercellOffset);
                    }

                    static HINLINE std::string getName()
                    {
                        // we provide the name from the param class
                        return Functor::name;
                    }
                };

                /** Wrap our functor so that it is accepted by manipulate()
                 *
                 * Note that if the signature of our operator() changes, we need to change the wrapper as well.
                 */
                using ReflectParticleManipulator = FreeSupercellOffset<ReflectParticle>;
            } // namespace detail

            //! Functor to reflecting boundary to particle species
            template<>
            struct ApplyImpl<Kind::Reflecting>
            {
                /** Apply reflecting boundary conditions along the given outer boundary
                 *
                 * @tparam T_Species particle species type
                 *
                 * @param species particle species
                 * @param exchangeType exchange describing the active boundary
                 * @param currentStep current time iteration
                 */
                template<typename T_Species>
                void operator()(T_Species& species, uint32_t exchangeType, uint32_t currentStep)
                {
                    /* Factory to create a mapper in GUARD for the given exchange.
                     * We pass it to the for-each algorithm to only process particles in this area
                     */
                    auto mapperFactory = pmacc::ExchangeMapperFactory<GUARD>{exchangeType};

                    // Apply  our functor to all particles (of the species) in the area
                    particles::manipulate<detail::ReflectParticleManipulator, T_Species>(currentStep, mapperFactory);

                    /* This has to be called to move particles between supercells (and fill gaps)
                     * according to what you set in multiMask_ of particles.
                     * It is not optimal to call it for full GUARD, but for now is tolerable.
                     */
                    /// For now this won't compile with GUARD due to some missing specializations in
                    /// StrideMappingMethods. This will be fixed soon, this is independent of particle BCs For now call
                    /// it everywhere, which is stupid but works
                    species.template shiftBetweenSupercells<CORE + BORDER + GUARD /* GUARD */>();
                }
            };

        } // namespace boundary
    } // namespace particles
} // namespace picongpu
