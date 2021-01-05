/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch
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
            namespace planeWave
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
                    static constexpr float_X INIT_TIME = float_X(
                        (Params::RAMP_INIT * Params::PULSE_LENGTH_SI + Params::LASER_NOFOCUS_CONSTANT_SI)
                        / UNIT_TIME); // unit: seconds (full inizialisation length)

                    /* initialize the laser not in the first cell is equal to a negative shift
                     * in time
                     */
                    static constexpr float_X laserTimeShift = Params::initPlaneY * CELL_HEIGHT / SPEED_OF_LIGHT;

                    static constexpr float_64 f = SPEED_OF_LIGHT / WAVE_LENGTH;
                };
            } // namespace planeWave

            namespace acc
            {
                template<typename T_Unitless>
                struct PlaneWave : public T_Unitless
                {
                    using Unitless = T_Unitless;

                    float3_X m_elong;
                    typename FieldE::DataBoxType m_dataBoxE;
                    DataSpace<simDim> m_offsetToTotalDomain;
                    DataSpace<simDim> m_superCellToLocalOriginCellOffset;

                    /** Device-Side Constructor
                     *
                     * @param superCellToLocalOriginCellOffset local offset in cells to current supercell
                     * @param offsetToTotalDomain offset to origin of global (@todo: total) coordinate system (possibly
                     * after transform to centered origin)
                     */
                    HDINLINE PlaneWave(
                        typename FieldE::DataBoxType const& dataBoxE,
                        DataSpace<simDim> const& superCellToLocalOriginCellOffset,
                        DataSpace<simDim> const& offsetToTotalDomain,
                        float3_X const& elong)
                        : m_elong(elong)
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
            struct PlaneWave : public planeWave::Unitless<T_Params>
            {
                using Unitless = planeWave::Unitless<T_Params>;

                float3_X elong;
                float_X phase;
                typename FieldE::DataBoxType dataBoxE;
                DataSpace<simDim> offsetToTotalDomain;

                /** constructor
                 *
                 * @param currentStep current simulation time step
                 */
                HINLINE PlaneWave(uint32_t currentStep) : phase(0.0_X)
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

                    elong = float3_X::create(0.0);

                    float_64 envelope = float_64(Unitless::AMPLITUDE);

                    float_64 const mue = 0.5 * Unitless::RAMP_INIT * Unitless::PULSE_LENGTH;

                    float_64 const w = 2.0 * PI * Unitless::f;
                    float_64 const tau = Unitless::PULSE_LENGTH * math::sqrt(2.0);

                    float_64 const endUpramp = mue;
                    float_64 const startDownramp = mue + Unitless::LASER_NOFOCUS_CONSTANT;

                    float_64 integrationCorrectionFactor = 0.0;

                    if(runTime > startDownramp)
                    {
                        // downramp = end
                        float_64 const exponent = (runTime - startDownramp) / tau;
                        envelope *= exp(-0.5 * exponent * exponent);
                        integrationCorrectionFactor = (runTime - startDownramp) / (w * tau * tau);
                    }
                    else if(runTime < endUpramp)
                    {
                        // upramp = start
                        float_64 const exponent = (runTime - endUpramp) / tau;
                        envelope *= exp(-0.5 * exponent * exponent);
                        integrationCorrectionFactor = (runTime - endUpramp) / (w * tau * tau);
                    }

                    float_64 const timeOszi = runTime - endUpramp;
                    float_64 const t_and_phase = w * timeOszi + Unitless::LASER_PHASE;
                    // to understand both components [sin(...) + t/tau^2 * cos(...)] see description above
                    if(Unitless::Polarisation == Unitless::LINEAR_X)
                    {
                        elong.x() = float_X(
                            envelope
                            * (math::sin(t_and_phase) + math::cos(t_and_phase) * integrationCorrectionFactor));
                    }
                    else if(Unitless::Polarisation == Unitless::LINEAR_Z)
                    {
                        elong.z() = float_X(
                            envelope
                            * (math::sin(t_and_phase) + math::cos(t_and_phase) * integrationCorrectionFactor));
                    }
                    else if(Unitless::Polarisation == Unitless::CIRCULAR)
                    {
                        elong.x() = float_X(
                            envelope / math::sqrt(2.0)
                            * (math::sin(t_and_phase) + math::cos(t_and_phase) * integrationCorrectionFactor));
                        elong.z() = float_X(
                            envelope / math::sqrt(2.0)
                            * (math::cos(t_and_phase) - math::sin(t_and_phase) * integrationCorrectionFactor));
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
                HDINLINE acc::PlaneWave<Unitless> operator()(
                    T_Acc const&,
                    DataSpace<simDim> const& localSupercellOffset,
                    T_WorkerCfg const&) const
                {
                    auto const superCellToLocalOriginCellOffset = localSupercellOffset * SuperCellSize::toRT();
                    return acc::PlaneWave<Unitless>(
                        dataBoxE,
                        superCellToLocalOriginCellOffset,
                        offsetToTotalDomain,
                        elong);
                }

                //! get the name of the laser profile
                static HINLINE std::string getName()
                {
                    return "PlaneWave";
                }
            };

        } // namespace laserProfiles
    } // namespace fields
} // namespace picongpu
