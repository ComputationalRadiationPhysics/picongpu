/* Copyright 2018-2021 Ilja Goethel, Axel Huebl
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
            namespace expRampWithPrepulse
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

                    static constexpr float_64 TIME_PREPULSE = float_64(Params::TIME_PREPULSE_SI / UNIT_TIME);
                    static constexpr float_64 TIME_PEAKPULSE = float_64(Params::TIME_PEAKPULSE_SI / UNIT_TIME);
                    static constexpr float_64 TIME_1 = float_64(Params::TIME_POINT_1_SI / UNIT_TIME);
                    static constexpr float_64 TIME_2 = float_64(Params::TIME_POINT_2_SI / UNIT_TIME);
                    static constexpr float_64 TIME_3 = float_64(Params::TIME_POINT_3_SI / UNIT_TIME);
                    static constexpr float_X endUpramp = TIME_PEAKPULSE - 0.5_X * LASER_NOFOCUS_CONSTANT;
                    static constexpr float_X startDownramp = TIME_PEAKPULSE + 0.5_X * LASER_NOFOCUS_CONSTANT;

                    static constexpr float_X INIT_TIME
                        = float_X((TIME_PEAKPULSE + Params::RAMP_INIT * PULSE_LENGTH) / UNIT_TIME);

                    // compile-time checks for physical sanity:
                    static_assert(
                        (TIME_1 < TIME_2) && (TIME_2 < TIME_3) && (TIME_3 < endUpramp),
                        "The times in the parameters TIME_POINT_1/2/3 and the beginning of the plateau (which is at "
                        "TIME_PEAKPULSE - 0.5*RAMP_INIT*PULSE_LENGTH) should be in ascending order");

                    // some prerequisites for check of intensities (approximate check, because I can't use exp and log)
                    static constexpr float_X ratio_dt
                        = (endUpramp - TIME_3) / (TIME_3 - TIME_2); // ratio of time intervals
                    static constexpr float_X ri1
                        = Params::INT_RATIO_POINT_3 / Params::INT_RATIO_POINT_2; // first intensity ratio
                    static constexpr float_X ri2
                        = 0.2_X / Params::INT_RATIO_POINT_3; // second intensity ratio (0.2 is an arbitrary upper
                                                             // border for the intensity of the exp ramp)

                    /* Approximate check, if ri1 ^ ratio_dt > ri2. That would mean, that the exponential curve through
                     * (time2, int2) and (time3, int3) lies above (endUpramp, 0.2) the power function is emulated by
                     * "rounding" the exponent to a rational number and expanding both sides by the common denominator,
                     * to get integer powers, see below for this, the range for ratio_dt is split into parts; the
                     * checked condition is "rounded down", i.e. it's weaker in every point of those ranges except one.
                     */
                    static constexpr bool intensity_too_big = (ratio_dt >= 3._X && ri1 * ri1 * ri1 > ri2)
                        || (ratio_dt >= 2._X && ri1 * ri1 > ri2) || (ratio_dt >= 1.5_X && ri1 * ri1 * ri1 > ri2 * ri2)
                        || (ratio_dt >= 1._X && ri1 > ri2)
                        || (ratio_dt >= 0.8_X && ri1 * ri1 * ri1 * ri1 > ri2 * ri2 * ri2 * ri2 * ri2)
                        || (ratio_dt >= 0.75_X && ri1 * ri1 * ri1 > ri2 * ri2 * ri2 * ri2)
                        || (ratio_dt >= 0.67_X && ri1 * ri1 > ri2 * ri2 * ri2)
                        || (ratio_dt >= 0.6_X && ri1 * ri1 * ri1 > ri2 * ri2 * ri2 * ri2 * ri2)
                        || (ratio_dt >= 0.5_X && ri1 > ri2 * ri2)
                        || (ratio_dt >= 0.4_X && ri1 * ri1 > ri2 * ri2 * ri2 * ri2 * ri2)
                        || (ratio_dt >= 0.33_X && ri1 > ri2 * ri2 * ri2)
                        || (ratio_dt >= 0.25_X && ri1 > ri2 * ri2 * ri2 * ri2)
                        || (ratio_dt >= 0.2_X && ri1 > ri2 * ri2 * ri2 * ri2 * ri2);
                    static_assert(
                        !intensity_too_big,
                        "The intensities of the ramp are very large - the extrapolation to the time of the main pulse "
                        "would give more than half of the pulse amplitude. This is not a Gaussian pulse at all "
                        "anymore - probably some of the parameters are different from what you think!?");

                    /* initialize the laser not in the first cell is equal to a negative shift
                     * in time
                     */
                    static constexpr float_X laserTimeShift = Params::initPlaneY * CELL_HEIGHT / SPEED_OF_LIGHT;

                    /* a symmetric pulse will be initialized at position z=0 for
                     * a time of RAMP_INIT * PULSE_LENGTH + LASER_NOFOCUS_CONSTANT = INIT_TIME.
                     * we shift the complete pulse for the half of this time to start with
                     * the front of the laser pulse.
                     */
                    static constexpr float_X time_start_init = TIME_1 - (0.5 * Params::RAMP_INIT * PULSE_LENGTH);
                    static constexpr float_64 f = SPEED_OF_LIGHT / WAVE_LENGTH;
                    static constexpr float_64 w = 2.0 * PI * f;
                };
            } // namespace expRampWithPrepulse

            namespace acc
            {
                template<typename T_Unitless>
                struct ExpRampWithPrepulse : public T_Unitless
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
                    HDINLINE ExpRampWithPrepulse(
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
            struct ExpRampWithPrepulse : public expRampWithPrepulse::Unitless<T_Params>
            {
                using Unitless = expRampWithPrepulse::Unitless<T_Params>;

                float3_X elong;
                float_X phase;
                typename FieldE::DataBoxType dataBoxE;
                DataSpace<simDim> offsetToTotalDomain;

                /** takes time t relative to the center of the Gaussian and returns value
                 * between 0 and 1, i.e. as multiple of the max value.
                 * use as: amp_t = amp_0 * gauss( t - t_0 )
                 */
                HDINLINE float_X gauss(float_X const t)
                {
                    float_X const exponent = t / float_X(Unitless::PULSE_LENGTH);
                    return math::exp(-0.25_X * exponent * exponent);
                }

                /** get value of exponential curve through two points at given t
                 * t/t1/t2 given as float_X, since the envelope doesn't need the accuracy
                 */
                HDINLINE float_X extrapolate_expo(
                    float_X const t1,
                    float_X const a1,
                    float_X const t2,
                    float_X const a2,
                    float_X const t)
                {
                    const float_X log1 = (t2 - t) * math::log(a1);
                    const float_X log2 = (t - t1) * math::log(a2);
                    return math::exp((log1 + log2) / (t2 - t1));
                }

                HINLINE float_X get_envelope(float_X runTime)
                {
                    /* workaround for clang 5 linker issues
                     * `undefined reference to
                     * `picongpu::fields::laserProfiles::ExpRampWithPrepulseParam::INT_RATIO_POINT_1'`
                     */
                    constexpr auto int_ratio_prepule = Unitless::INT_RATIO_PREPULSE;
                    constexpr auto int_ratio_point_1 = Unitless::INT_RATIO_POINT_1;
                    constexpr auto int_ratio_point_2 = Unitless::INT_RATIO_POINT_2;
                    constexpr auto int_ratio_point_3 = Unitless::INT_RATIO_POINT_3;
                    float_X const AMP_PREPULSE = float_X(math::sqrt(int_ratio_prepule) * Unitless::AMPLITUDE);
                    float_X const AMP_1 = float_X(math::sqrt(int_ratio_point_1) * Unitless::AMPLITUDE);
                    float_X const AMP_2 = float_X(math::sqrt(int_ratio_point_2) * Unitless::AMPLITUDE);
                    float_X const AMP_3 = float_X(math::sqrt(int_ratio_point_3) * Unitless::AMPLITUDE);

                    float_X env = 0.0;
                    bool const before_preupramp = runTime < Unitless::time_start_init;
                    bool const before_start = runTime < Unitless::TIME_1;
                    bool const before_peakpulse = runTime < Unitless::endUpramp;
                    bool const during_first_exp = (Unitless::TIME_1 < runTime) && (runTime < Unitless::TIME_2);
                    bool const after_peakpulse = Unitless::startDownramp <= runTime;

                    if(before_preupramp)
                        env = 0.;
                    else if(before_start)
                    {
                        env = AMP_1 * gauss(runTime - Unitless::TIME_1);
                    }
                    else if(before_peakpulse)
                    {
                        float_X const ramp_when_peakpulse
                            = extrapolate_expo(Unitless::TIME_2, AMP_2, Unitless::TIME_3, AMP_3, Unitless::endUpramp)
                            / Unitless::AMPLITUDE;

                        if(ramp_when_peakpulse > 0.5)
                        {
                            log<picLog::PHYSICS>(
                                "Attention, the intensities of the laser upramp are very large! "
                                "The extrapolation of the last exponential to the time of "
                                "the peakpulse gives more than half of the amplitude of "
                                "the peak Gaussian. This is not a Gaussian at all anymore, "
                                "and physically very unplausible, check the params for misunderstandings!");
                        }

                        env += Unitless::AMPLITUDE * (1._X - ramp_when_peakpulse)
                            * gauss(runTime - Unitless::endUpramp);
                        env += AMP_PREPULSE * gauss(runTime - Unitless::TIME_PREPULSE);
                        if(during_first_exp)
                            env += extrapolate_expo(Unitless::TIME_1, AMP_1, Unitless::TIME_2, AMP_2, runTime);
                        else
                            env += extrapolate_expo(Unitless::TIME_2, AMP_2, Unitless::TIME_3, AMP_3, runTime);
                    }
                    else if(!after_peakpulse)
                        env = Unitless::AMPLITUDE;
                    else // after startDownramp
                        env = Unitless::AMPLITUDE * gauss(runTime - Unitless::startDownramp);
                    return env;
                }

                /** constructor
                 *
                 * @param currentStep current simulation time step
                 */
                HINLINE ExpRampWithPrepulse(uint32_t currentStep)
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

                    elong = float3_X::create(0.0);

                    /* initialize the laser not in the first cell is equal to a negative shift
                     * in time
                     */
                    const float_64 runTime
                        = Unitless::time_start_init - Unitless::laserTimeShift + DELTA_T * currentStep;

                    phase = float_X(Unitless::w * runTime) + Unitless::LASER_PHASE;

                    float_X const envelope = get_envelope(runTime);

                    if(Unitless::Polarisation == Unitless::LINEAR_X)
                    {
                        elong.x() = envelope * math::sin(phase);
                    }
                    else if(Unitless::Polarisation == Unitless::LINEAR_Z)
                    {
                        elong.z() = envelope * math::sin(phase);
                    }
                    else if(Unitless::Polarisation == Unitless::CIRCULAR)
                    {
                        elong.x() = envelope / math::sqrt(2.0_X) * math::sin(phase);
                        elong.z() = envelope / math::sqrt(2.0_X) * math::cos(phase);
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
                HDINLINE acc::ExpRampWithPrepulse<Unitless> operator()(
                    T_Acc const&,
                    DataSpace<simDim> const& localSupercellOffset,
                    T_WorkerCfg const&) const
                {
                    auto const superCellToLocalOriginCellOffset = localSupercellOffset * SuperCellSize::toRT();
                    return acc::ExpRampWithPrepulse<Unitless>(
                        dataBoxE,
                        superCellToLocalOriginCellOffset,
                        offsetToTotalDomain,
                        elong);
                }

                //! get the name of the laser profile
                static HINLINE std::string getName()
                {
                    return "ExpRampWithPrepulse";
                }
            };

        } // namespace laserProfiles
    } // namespace fields
} // namespace picongpu
