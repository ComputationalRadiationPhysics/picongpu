/* Copyright 2015-2022 Rene Widera, Alexander Grund, Axel Huebl, Sergei Bastrakov
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

#include "picongpu/particles/functor/misc/TotalCellOffset.hpp"
#include "picongpu/particles/manipulators/unary/FreeTotalCellOffset.def"

#include <string>


namespace picongpu
{
    namespace particles
    {
        namespace manipulators
        {
            namespace unary
            {
                namespace acc
                {
                    /** Device-side functor
                     *
                     * @tparam T_Functor user-defined unary functor
                     * @tparam T_RngType rng functor type
                     */
                    template<typename T_Functor, typename T_RngType>
                    struct FreeTotalCellOffsetRng : private T_Functor
                    {
                        using Functor = T_Functor;
                        using RngType = T_RngType;

                        HDINLINE FreeTotalCellOffsetRng(
                            Functor const& functor,
                            DataSpace<simDim> const& superCellToLocalOriginCellOffset,
                            RngType const& rng)
                            : T_Functor(functor)
                            , m_superCellToLocalOriginCellOffset(superCellToLocalOriginCellOffset)
                            , m_rng(rng)
                        {
                        }

                        /** call user functor
                         *
                         * @tparam T_Particle type of the particle to manipulate
                         * @tparam T_Worker lockstep worker type
                         *
                         * @param worker lockstep worker
                         * @param particle particle which is given to the user functor
                         */
                        template<typename T_Particle, typename T_Worker>
                        HDINLINE void operator()(T_Worker const&, T_Particle& particle)
                        {
                            DataSpace<simDim> const cellInSuperCell(
                                DataSpaceOperations<simDim>::template map<SuperCellSize>(particle[localCellIdx_]));
                            Functor::operator()(m_superCellToLocalOriginCellOffset + cellInSuperCell, m_rng, particle);
                        }

                    private:
                        DataSpace<simDim> const m_superCellToLocalOriginCellOffset;
                        RngType m_rng;
                    };
                } // namespace acc

                template<typename T_Functor, typename T_Distribution>
                struct FreeTotalCellOffsetRng
                    : protected functor::User<T_Functor>
                    , private functor::misc::TotalCellOffset
                    , private functor::misc::Rng<T_Distribution>
                {
                    using CellOffsetFunctor = functor::misc::TotalCellOffset;
                    using Functor = functor::User<T_Functor>;

                    using RngGenerator = functor::misc::Rng<T_Distribution>;
                    using Distribution = T_Distribution;

                    template<typename T_SpeciesType>
                    using fn = FreeTotalCellOffsetRng;

                    /** constructor
                     *
                     * @param currentStep current simulation time step
                     */
                    HINLINE FreeTotalCellOffsetRng(uint32_t currentStep)
                        : Functor(currentStep)
                        , CellOffsetFunctor(currentStep)
                        , RngGenerator(currentStep)
                    {
                    }

                    /** Create functor for the accelerator
                     *
                     * @tparam T_Worker lockstep worker type
                     *
                     * @param worker lockstep worker
                     * @param localSupercellOffset offset (in superCells, without any guards) relative
                     *                             to the origin of the local domain
                     * @param workerCfg configuration of the worker
                     */
                    template<typename T_Worker>
                    HDINLINE auto operator()(T_Worker const& worker, DataSpace<simDim> const& localSupercellOffset)
                        const
                    {
                        auto& cellOffsetFunctor = *static_cast<CellOffsetFunctor const*>(this);
                        auto const rng = (*static_cast<RngGenerator const*>(this))(worker, localSupercellOffset);
                        return acc::FreeTotalCellOffsetRng<Functor, ALPAKA_DECAY_T(decltype(rng))>(
                            *static_cast<Functor const*>(this),
                            cellOffsetFunctor(worker, localSupercellOffset),
                            rng);
                    }

                    static HINLINE std::string getName()
                    {
                        return Functor::name;
                    }
                };

            } // namespace unary
        } // namespace manipulators
    } // namespace particles
} // namespace picongpu
