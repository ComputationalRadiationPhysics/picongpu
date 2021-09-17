/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch, Ilja Goethel,
 *                     Anton Helm, Alexander Debus, Sergei Bastrakov
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

#include <cstdint>

namespace picongpu
{
    namespace fields
    {
        namespace laserProfiles
        {
            namespace acc
            {
                /** Base device-side functor for laser profile implementations
                 *
                 * Stores common data for all such implementations.
                 * Implements the logic of contributing the stored value to the grid value of E.
                 * Has two modes of operation depending on T_initPlaneY:
                 * For T_initPlaneY == 0 acts as a classic hard source.
                 * For T_initPlaneY > 0 acts as a middle ground between hard and soft source.
                 *
                 * @tparam T_initPlaneY laser initialization plane in y,
                 *                      value in cells in the global coordinates
                 */
                template<uint32_t T_initPlaneY>
                class BaseFunctor
                {
                public:
                    //! Type of data box for field E
                    using DataBoxE = typename FieldE::DataBoxType;

                    /** Device-side constructor
                     *
                     * @param dataBoxE global data box for field E
                     * @param superCellToLocalOriginCellOffset offset of the active supercell to the local domain
                     * origin; in cells, does not account for guards
                     * @param offsetToTotalDomain offset of local domain to total domain, in cells
                     * @param elong initial laser-induced value of E
                     */
                    HDINLINE BaseFunctor(
                        DataBoxE const& dataBoxE,
                        DataSpace<simDim> const& superCellToLocalOriginCellOffset,
                        DataSpace<simDim> const& offsetToTotalDomain,
                        float3_X const& elong)
                        : m_dataBoxE(dataBoxE)
                        , m_offsetToTotalDomain(offsetToTotalDomain)
                        , m_superCellToLocalOriginCellOffset(superCellToLocalOriginCellOffset)
                        , m_elong(elong)
                    {
                    }

                    /** Contribute the value stored in m_elong (calculated by a child class) to the grid value of E
                     *
                     * The way of contributing depends on whether T_initPlaneY == 0.
                     * However, the difference is contained here, it does not affect calculating m_elong.
                     *
                     * @param localCell index of a local cell in the local domain (without guards)
                     */
                    HDINLINE void operator()(DataSpace<simDim> const& localCell)
                    {
                        // Take into account guard, as m_dataBoxE is stored like that
                        auto const gridIdx = localCell + SuperCellSize::toRT() * GuardSize::toRT();
                        if(T_initPlaneY != 0)
                        {
                            /* If the laser is not initialized in the first cell we emit a
                             * negatively and positively propagating wave. Therefore we need to multiply the
                             * amplitude with a correction factor depending of the cell size in
                             * propagation direction.
                             * The negatively propagating wave is damped by the absorber.
                             *
                             * The `correctionFactor` assume that the wave is moving in y direction.
                             */
                            auto const correctionFactor = (SPEED_OF_LIGHT * DELTA_T) / CELL_HEIGHT * 2._X;
                            m_dataBoxE(gridIdx) += correctionFactor * m_elong;
                        }
                        else
                            m_dataBoxE(gridIdx) = m_elong;
                    }

                public:
                    //! Global data box for field E
                    DataBoxE m_dataBoxE;

                    /** Offset of the active supercell to the local domain origin
                     *
                     * In cells, does not account for guards.
                     */
                    DataSpace<simDim> m_superCellToLocalOriginCellOffset;

                    //! Offset of local domain to total domain, in cells
                    DataSpace<simDim> m_offsetToTotalDomain;

                    /** Laser-induced value of E
                     *
                     * To be modified by child classes before calling operator().
                     */
                    float3_X m_elong;
                };

            } // namespace acc
        } // namespace laserProfiles
    } // namespace fields
} // namespace picongpu
