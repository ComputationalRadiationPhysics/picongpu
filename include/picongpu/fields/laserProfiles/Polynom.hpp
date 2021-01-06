/* Copyright 2013-2021 Heiko Burau, Rene Widera, Richard Pausch, Axel Huebl
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

#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>


namespace picongpu
{
    namespace fields
    {
        namespace laserProfiles
        {
            namespace polynom
            {
                template<typename T_Params>
                struct Unitless : public T_Params
                {
                    using Params = T_Params;

                    static constexpr float_X WAVE_LENGTH
                        = float_X(Params::WAVE_LENGTH_SI / UNIT_LENGTH); // unit: meter
                    static constexpr float_X PULSE_LENGTH
                        = float_X(Params::PULSE_LENGTH_SI / UNIT_TIME); // unit: seconds (1 sigma)
                    static constexpr float_X AMPLITUDE
                        = float_X(Params::AMPLITUDE_SI / UNIT_EFIELD); // unit: Volt /meter
                    static constexpr float_X W0_X = float_X(Params::W0_X_SI / UNIT_LENGTH); // unit: meter
                    static constexpr float_X W0_Z = float_X(Params::W0_Z_SI / UNIT_LENGTH); // unit: meter
                    static constexpr float_X INIT_TIME
                        = float_X(Params::PULSE_LENGTH_SI / UNIT_TIME); // unit: seconds (full initialization length)

                    /* initialize the laser not in the first cell is equal to a negative shift
                     * in time
                     */
                    static constexpr float_X laserTimeShift = Params::initPlaneY * CELL_HEIGHT / SPEED_OF_LIGHT;

                    static constexpr float_64 f = SPEED_OF_LIGHT / WAVE_LENGTH;
                };
            } // namespace polynom

            namespace acc
            {
                template<typename T_Unitless>
                struct Polynom : public T_Unitless
                {
                    using Unitless = T_Unitless;

                    float3_X m_elong;
                    float_X m_phase;
                    typename FieldE::DataBoxType m_dataBoxE;
                    DataSpace<simDim> m_offsetToTotalDomain;
                    DataSpace<simDim> m_superCellToLocalOriginCellOffset;

                    /** Device-Side Constructor
                     *
                     * @param superCellToLocalOriginCellOffset local offset in cells to current supercell
                     * @param offsetToTotalDomain offset to origin of global (@todo: total) coordinate system (possibly
                     * after transform to centered origin)
                     */
                    HDINLINE Polynom(
                        typename FieldE::DataBoxType const& dataBoxE,
                        DataSpace<simDim> const& superCellToLocalOriginCellOffset,
                        DataSpace<simDim> const& offsetToTotalDomain,
                        float3_X const& elong,
                        float_X const phase)
                        : m_elong(elong)
                        , m_phase(phase)
                        , m_dataBoxE(dataBoxE)
                        , m_offsetToTotalDomain(offsetToTotalDomain)
                        , m_superCellToLocalOriginCellOffset(superCellToLocalOriginCellOffset)
                    {
                    }

                    /** device side manipulation for init plane (transversal)
                     *
                     * @tparam T_Args type of the arguments passed to the user manipulator functor
                     *
                     * @param cellIndexInSuperCell ND cell index in current supercell
                     */
                    template<typename T_Acc>
                    HDINLINE void operator()(T_Acc const&, DataSpace<simDim> const& cellIndexInSuperCell)
                    {
                        // coordinate system to global simulation as origin
                        DataSpace<simDim> const localCell(cellIndexInSuperCell + m_superCellToLocalOriginCellOffset);

                        // transform coordinate system to center of x-z plane of initialization
                        constexpr uint8_t planeNormalDir = 1u;
                        DataSpace<simDim> offsetToCenterOfPlane(m_offsetToTotalDomain);
                        offsetToCenterOfPlane[planeNormalDir] = 0; // do not shift origin of plane normal
                        floatD_X const pos
                            = precisionCast<float_X>(localCell + offsetToCenterOfPlane) * cellSize.shrink<simDim>();
                        // @todo add half-cells via traits::FieldPosition< Solver::NumicalCellType, FieldE >()

                        // transversal position only
                        float3_X const w0_3D(Unitless::W0_X, 0., Unitless::W0_Z);
                        auto const w0(w0_3D.shrink<simDim>().remove<planeNormalDir>());
                        auto const pos_trans(pos.remove<planeNormalDir>());
                        auto const exp_compos(pos_trans * pos_trans / (w0 * w0));
                        float_X const exp_arg(exp_compos.sumOfComponents());

                        m_elong *= math::exp(-1.0_X * exp_arg);

                        if(Unitless::initPlaneY != 0) // compile time if
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

                            // jump over the guard of the electric field
                            m_dataBoxE(localCell + SuperCellSize::toRT() * GuardSize::toRT())
                                += correctionFactor * m_elong;
                        }
                        else
                        {
                            // jump over the guard of the electric field
                            m_dataBoxE(localCell + SuperCellSize::toRT() * GuardSize::toRT()) = m_elong;
                        }
                    }
                };
            } // namespace acc

            template<typename T_Params>
            struct Polynom : public polynom::Unitless<T_Params>
            {
                using Unitless = polynom::Unitless<T_Params>;

                float3_X elong;
                float_X phase;
                typename FieldE::DataBoxType dataBoxE;
                DataSpace<simDim> offsetToTotalDomain;

                HDINLINE float_X Tpolynomial(float_X const tau)
                {
                    float_X result(0.0_X);
                    if(tau >= 0.0_X && tau <= 1.0_X)
                        result = tau * tau * tau * (10.0_X - 15.0_X * tau + 6.0_X * tau * tau);
                    else if(tau > 1.0_X && tau <= 2.0_X)
                        result = (2.0_X - tau) * (2.0_X - tau) * (2.0_X - tau)
                            * (4.0_X - 9.0_X * tau + 6.0_X * tau * tau);

                    return result;
                }

                /** constructor
                 *
                 * @param currentStep current simulation time step
                 */
                HINLINE Polynom(uint32_t currentStep) : phase(0.0_X)
                {
                    // get data
                    DataConnector& dc = Environment<>::get().DataConnector();
                    dataBoxE = dc.get<FieldE>(FieldE::getName(), true)->getDeviceDataBox();

                    // get meta data for offsets
                    SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                    // const DataSpace< simDim > totalCellOffset( subGrid.getGlobalDomain().offset );
                    DataSpace<simDim> const globalCellOffset(subGrid.getLocalDomain().offset);
                    DataSpace<simDim> const halfSimSize(subGrid.getGlobalDomain().size / 2);

                    // transform coordinate system to center of global simulation as origin [cells]
                    offsetToTotalDomain = /* totalCellOffset + */ globalCellOffset - halfSimSize;

                    // @todo reset origin of direction of moving window
                    // offsetToTotalDomain.y() = 0

                    float_64 const runTime = DELTA_T * currentStep - Unitless::laserTimeShift;

                    elong = float3_X::create(0.0_X);

                    /* a symmetric pulse will be initialized at position z=0
                     * the laser amplitude rises  for t_rise
                     * and falls for t_rise
                     * making the laser pulse 2*t_rise long
                     */

                    const float_X t_rise = 0.5_X * Unitless::PULSE_LENGTH;
                    const float_X tau = runTime / t_rise;

                    const float_X omegaLaser = 2.0_X * PI * Unitless::f;

                    if(Unitless::Polarisation == Unitless::LINEAR_X)
                    {
                        elong.x() = Unitless::AMPLITUDE * Tpolynomial(tau)
                            * math::sin(omegaLaser * (runTime - t_rise) + Unitless::LASER_PHASE);
                    }
                    else if(Unitless::Polarisation == Unitless::LINEAR_Z)
                    {
                        elong.z() = Unitless::AMPLITUDE * Tpolynomial(tau)
                            * math::sin(omegaLaser * (runTime - t_rise) + Unitless::LASER_PHASE);
                    }
                    else if(Unitless::Polarisation == Unitless::CIRCULAR)
                    {
                        elong.x() = Unitless::AMPLITUDE * Tpolynomial(tau) / math::sqrt(2.0_X)
                            * math::sin(omegaLaser * (runTime - t_rise) + Unitless::LASER_PHASE);
                        elong.z() = Unitless::AMPLITUDE * Tpolynomial(tau) / math::sqrt(2.0_X)
                            * math::cos(omegaLaser * (runTime - t_rise) + Unitless::LASER_PHASE);
                    }
                }

                /** create device manipulator functor
                 *
                 * @tparam T_WorkerCfg pmacc::mappings::threads::WorkerCfg, configuration of the worker
                 * @tparam T_Acc alpaka accelerator type
                 *
                 * @param alpaka accelerator
                 * @param localSupercellOffset (in supercells, without guards) to the
                 *        origin of the local domain
                 * @param configuration of the worker
                 */
                template<typename T_WorkerCfg, typename T_Acc>
                HDINLINE acc::Polynom<Unitless> operator()(
                    T_Acc const&,
                    DataSpace<simDim> const& localSupercellOffset,
                    T_WorkerCfg const&) const
                {
                    auto const superCellToLocalOriginCellOffset = localSupercellOffset * SuperCellSize::toRT();
                    return acc::Polynom<Unitless>(
                        dataBoxE,
                        superCellToLocalOriginCellOffset,
                        offsetToTotalDomain,
                        elong,
                        phase);
                }

                //! get the name of the laser profile
                static HINLINE std::string getName()
                {
                    return "Polynom";
                }
            };

        } // namespace laserProfiles
    } // namespace fields
} // namespace picongpu
