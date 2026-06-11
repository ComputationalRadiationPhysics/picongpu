/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/functor/misc/TotalCellOffset.def"
#include "picongpu/simulation/control/MovingWindow.hpp"

namespace picongpu
{
    namespace particles
    {
        namespace functor
        {
            namespace misc
            {
                struct TotalCellOffset
                {
                    /** constructor
                     *
                     * @param currentStep current simulation time step
                     */
                    HINLINE TotalCellOffset(uint32_t currentStep)
                    {
                        uint32_t const numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
                        SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                        DataSpace<simDim> const localCells = subGrid.getLocalDomain().size;
                        gpuCellOffsetToTotalOrigin = subGrid.getLocalDomain().offset;
                        gpuCellOffsetToTotalOrigin.y() += numSlides * localCells.y();
                    }

                    /** get cell offset of the supercell
                     *
                     * @tparam T_Worker lockstep worker type
                     *
                     * @param worker lockstep worker
                     * @param localSupercellOffset (in supercells, without any guards) to the
                     *         origin of the local domain
                     */
                    template<typename T_Worker>
                    HDINLINE DataSpace<simDim> operator()(
                        T_Worker const& worker,
                        DataSpace<simDim> const& localSupercellOffset) const
                    {
                        DataSpace<simDim> const superCellToLocalOriginCellOffset(
                            localSupercellOffset * SuperCellSize::toRT());

                        return gpuCellOffsetToTotalOrigin + superCellToLocalOriginCellOffset;
                    }

                private:
                    //! offset in cells to the total domain origin
                    DataSpace<simDim> gpuCellOffsetToTotalOrigin;
                };

            } // namespace misc
        } // namespace functor
    } // namespace particles
} // namespace picongpu
