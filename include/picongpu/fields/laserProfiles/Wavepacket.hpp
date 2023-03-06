/* Copyright 2013-2022 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Stefan Tietze
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

#include "picongpu/fields/laserProfiles/BaseFunctor.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>


namespace picongpu
{
    namespace fields
    {
        namespace laserProfiles
        {
            namespace wavepacket
            {
                template<typename T_Params>
                struct Unitless : public T_Params
                {
                    using Params = T_Params;

                    static constexpr float_X WAVE_LENGTH
                        = float_X(Params::WAVE_LENGTH_SI / UNIT_LENGTH); // unit: meter
                    static constexpr float_X PULSE_LENGTH
                        = float_X(Params::PULSE_LENGTH_SI / UNIT_TIME); // unit: seconds (1 sigma)
                    static constexpr float_X LASER_NOFOCUS_CONSTANT
                        = float_X(Params::LASER_NOFOCUS_CONSTANT_SI / UNIT_TIME); // unit: seconds
                    static constexpr float_X AMPLITUDE
                        = float_X(Params::AMPLITUDE_SI / UNIT_EFIELD); // unit: Volt /meter
                    static constexpr float_X W0_X = float_X(Params::W0_X_SI / UNIT_LENGTH); // unit: meter
                    static constexpr float_X W0_Z = float_X(Params::W0_Z_SI / UNIT_LENGTH); // unit: meter
                    static constexpr float_X INIT_TIME = float_X(
                        Params::PULSE_INIT * PULSE_LENGTH
                        + LASER_NOFOCUS_CONSTANT); // unit: seconds (full initialization length)
                    static constexpr float_X endUpramp = -0.5_X * LASER_NOFOCUS_CONSTANT; // unit: seconds
                    static constexpr float_X startDownramp = 0.5_X * LASER_NOFOCUS_CONSTANT; // unit: seconds

                    /* initialize the laser not in the first cell is equal to a negative shift
                     * in time
                     */
                    static constexpr float_X laserTimeShift = Params::initPlaneY * CELL_HEIGHT / SPEED_OF_LIGHT;

                    static constexpr float_64 f = SPEED_OF_LIGHT / WAVE_LENGTH;
                    static constexpr float_64 w = 2.0 * PI * f;
                };
            } // namespace wavepacket

            namespace acc
            {
                template<typename T_Unitless>
                struct Wavepacket
                    : public T_Unitless
                    , public acc::BaseFunctor<T_Unitless::initPlaneY>
                {
                    using Unitless = T_Unitless;
                    using BaseFunctor = acc::BaseFunctor<T_Unitless::initPlaneY>;

                    /** Device-Side Constructor
                     *
                     * @param superCellToLocalOriginCellOffset local offset in cells to current supercell
                     * @param offsetToTotalDomain offset to origin of global (@todo: total) coordinate system (possibly
                     * after transform to centered origin)
                     */
                    HDINLINE Wavepacket(
                        typename FieldE::DataBoxType const& dataBoxE,
                        DataSpace<simDim> const& superCellToLocalOriginCellOffset,
                        DataSpace<simDim> const& offsetToTotalDomain,
                        float3_X const& elong)
                        : BaseFunctor(dataBoxE, superCellToLocalOriginCellOffset, offsetToTotalDomain, elong)
                    {
                    }

                    /** device side manipulation for init plane (transversal)
                     *
                     * @tparam T_Worker lockstep worker type
                     *
                     * @param cellIndexInSuperCell ND cell index in current supercell
                     */
                    template<typename T_Worker>
                    HDINLINE void operator()(T_Worker const&, DataSpace<simDim> const& cellIndexInSuperCell)
                    {
                        // coordinate system to global simulation as origin
                        DataSpace<simDim> const localCell(
                            cellIndexInSuperCell + this->m_superCellToLocalOriginCellOffset);

                        // transform coordinate system to center of x-z plane of initialization
                        constexpr uint8_t planeNormalDir = 1u;
                        DataSpace<simDim> offsetToCenterOfPlane(this->m_offsetToTotalDomain);
                        offsetToCenterOfPlane[planeNormalDir] = 0; // do not shift origin of plane normal
                        floatD_X const pos
                            = precisionCast<float_X>(localCell + offsetToCenterOfPlane) * cellSize.shrink<simDim>();
                        // @todo add half-cells via traits::FieldPosition< Solver::NumicalCellType, FieldE >()

                        // transversal position only
                        float3_X const w0_3D(Unitless::W0_X, 0._X, Unitless::W0_Z);
                        auto const w0(w0_3D.shrink<simDim>().remove<planeNormalDir>());
                        auto const pos_trans(pos.remove<planeNormalDir>());
                        auto const exp_compos(pos_trans * pos_trans / (w0 * w0));
                        float_X const exp_arg(exp_compos.sumOfComponents());

                        this->m_elong *= math::exp(-1.0_X * exp_arg);
                        BaseFunctor::operator()(localCell);
                    }
                };
            } // namespace acc

            template<typename T_Params>
            struct Wavepacket : public wavepacket::Unitless<T_Params>
            {
                using Unitless = wavepacket::Unitless<T_Params>;

                float3_X elong;
                float_X phase;
                typename FieldE::DataBoxType dataBoxE;
                DataSpace<simDim> offsetToTotalDomain;

                /** constructor
                 *
                 * @param currentStep current simulation time step
                 */
                HINLINE Wavepacket(float_X currentStep)
                {
                    // get data
                    DataConnector& dc = Environment<>::get().DataConnector();
                    dataBoxE = dc.get<FieldE>(FieldE::getName())->getDeviceDataBox();

                    // get meta data for offsets
                    SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                    // const DataSpace< simDim > totalCellOffset( subGrid.getGlobalDomain().offset );
                    DataSpace<simDim> const globalCellOffset(subGrid.getLocalDomain().offset);
                    DataSpace<simDim> const halfSimSize(subGrid.getGlobalDomain().size / 2);

                    // transform coordinate system to center of global simulation as origin [cells]
                    offsetToTotalDomain = /* totalCellOffset + */ globalCellOffset - halfSimSize;

                    // @todo reset origin of direction of moving window
                    // offsetToTotalDomain.y() = 0

                    // a symmetric pulse will be initialized at position z=0 for
                    // a time of PULSE_INIT * PULSE_LENGTH + LASER_NOFOCUS_CONSTANT = INIT_TIME.
                    // we shift the complete pulse for the half of this time to start with
                    // the front of the laser pulse.
                    const float_64 mue = 0.5 * Unitless::INIT_TIME;

                    float_64 const runTime = DELTA_T * currentStep - Unitless::laserTimeShift - mue;

                    elong = float3_X::create(0.0_X);
                    float_X envelope = float_X(Unitless::AMPLITUDE);

                    const float_64 tau = Unitless::PULSE_LENGTH * math::sqrt(2.0_X);

                    float_64 correctionFactor = 0.0;

                    if(runTime > Unitless::startDownramp)
                    {
                        // downramp = end
                        const float_64 exponent
                            = ((runTime - Unitless::startDownramp) / Unitless::PULSE_LENGTH / math::sqrt(2.0));
                        envelope *= math::exp(-0.5 * exponent * exponent);
                        correctionFactor = (runTime - Unitless::startDownramp) / (tau * tau * Unitless::w);
                    }
                    else if(runTime < Unitless::endUpramp)
                    {
                        // upramp = start
                        const float_X exponent
                            = ((runTime - Unitless::endUpramp) / Unitless::PULSE_LENGTH / math::sqrt(2.0_X));
                        envelope *= math::exp(-0.5_X * exponent * exponent);
                        correctionFactor = (runTime - Unitless::endUpramp) / (tau * tau * Unitless::w);
                    }

                    phase += float_X(Unitless::w * runTime) + Unitless::LASER_PHASE;

                    if(Unitless::Polarisation == Unitless::LINEAR_X)
                    {
                        elong.x() = envelope * (math::sin(phase) + correctionFactor * math::cos(phase));
                    }
                    else if(Unitless::Polarisation == Unitless::LINEAR_Z)
                    {
                        elong.z() = envelope * (math::sin(phase) + correctionFactor * math::cos(phase));
                    }
                    else if(Unitless::Polarisation == Unitless::CIRCULAR)
                    {
                        elong.x()
                            = envelope / math::sqrt(2.0_X) * (math::sin(phase) + correctionFactor * math::cos(phase));
                        elong.z()
                            = envelope / math::sqrt(2.0_X) * (math::cos(phase) + correctionFactor * math::sin(phase));
                    }
                }

                /** create device manipulator functor
                 *
                 * @tparam T_Worker lockstep worker type
                 *
                 * @param worker lockstep worker
                 * @param localSupercellOffset (in supercells, without guards) to the
                 *        origin of the local domain
                 * @param configuration of the worker
                 */
                template<typename T_Worker>
                HDINLINE acc::Wavepacket<Unitless> operator()(
                    T_Worker const&,
                    DataSpace<simDim> const& localSupercellOffset) const
                {
                    auto const superCellToLocalOriginCellOffset = localSupercellOffset * SuperCellSize::toRT();
                    return acc::Wavepacket<Unitless>(
                        dataBoxE,
                        superCellToLocalOriginCellOffset,
                        offsetToTotalDomain,
                        elong);
                }

                //! get the name of the laser profile
                static HINLINE std::string getName()
                {
                    return "Wavepacket";
                }
            };

        } // namespace laserProfiles
    } // namespace fields
} // namespace picongpu
