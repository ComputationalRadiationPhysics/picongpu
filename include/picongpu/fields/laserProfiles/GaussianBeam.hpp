/* Copyright 2013-2022 Axel Huebl, Heiko Burau, Anton Helm, Rene Widera,
 *                     Richard Pausch, Alexander Debus
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
            namespace gaussianBeam
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
                    static constexpr float_X W0 = float_X(Params::W0_SI / UNIT_LENGTH); // unit: meter
                    static constexpr float_X FOCUS_POS = float_X(Params::FOCUS_POS_SI / UNIT_LENGTH); // unit: meter
                    static constexpr float_X INIT_TIME = float_X(
                        (Params::PULSE_INIT * Params::PULSE_LENGTH_SI)
                        / UNIT_TIME); // unit: seconds (full initialization length)

                    /* initialize the laser not in the first cell is equal to a negative shift
                     * in time
                     */
                    static constexpr float_X laserTimeShift = Params::initPlaneY * CELL_HEIGHT / SPEED_OF_LIGHT;

                    static constexpr float_64 f = SPEED_OF_LIGHT / WAVE_LENGTH;
                };
            } // namespace gaussianBeam

            namespace acc
            {
                template<typename T_Unitless>
                struct GaussianBeam
                    : public T_Unitless
                    , public acc::BaseFunctor<T_Unitless::initPlaneY>
                {
                    using Unitless = T_Unitless;
                    using BaseFunctor = acc::BaseFunctor<T_Unitless::initPlaneY>;

                    float_X m_phase;

                    /** Simple iteration algorithm to implement Laguerre polynomials for GPUs.
                     *
                     *  @param n order of the Laguerre polynomial
                     *  @param x coordinate at which the polynomial is evaluated
                     *  @return ...
                     */
                    HDINLINE float_X simpleLaguerre(const uint32_t n, const float_X x)
                    {
                        // Result for special case n == 0
                        if(n == 0)
                            return 1.0_X;
                        uint32_t currentN = 1;
                        float_X laguerreNMinus1 = 1.0_X;
                        float_X laguerreN = 1.0_X - x;
                        float_X laguerreNPlus1(0.0_X);
                        while(currentN < n)
                        {
                            // Core statement of the algorithm
                            laguerreNPlus1 = ((2.0_X * float_X(currentN) + 1.0_X - x) * laguerreN
                                              - float_X(currentN) * laguerreNMinus1)
                                / float_X(currentN + 1u);
                            // Advance by one order
                            laguerreNMinus1 = laguerreN;
                            laguerreN = laguerreNPlus1;
                            currentN++;
                        }
                        return laguerreN;
                    }

                    /** Device-Side Constructor
                     *
                     * @param superCellToLocalOriginCellOffset local offset in cells to current supercell
                     * @param offsetToTotalDomain offset to origin of global (@todo: total) coordinate system (possibly
                     * after transform to centered origin)
                     */
                    HDINLINE GaussianBeam(
                        typename FieldE::DataBoxType const& dataBoxE,
                        DataSpace<simDim> const& superCellToLocalOriginCellOffset,
                        DataSpace<simDim> const& offsetToTotalDomain,
                        float3_X const& elong,
                        float_X const phase)
                        : BaseFunctor(dataBoxE, superCellToLocalOriginCellOffset, offsetToTotalDomain, elong)
                        , m_phase(phase)
                    {
                    }

                    /** device side manipulation for init plane (transversal)
                     *
                     * @tparam T_Worker lockstep worker type
                     *
                     * @param cellIndexInSuperCell ND cell index in current supercell
                     *
                     * The transverse spatial laser modes are given as a decomposition of Gauss-Laguerre modes
                     * GLM(m,r,z) : Sum_{m=0}^{m_max} := Snorm * a_m * GLM(m,r,z)
                     * with a_m being complex-valued coefficients: a_m := |a_m| * exp(I Arg(a_m) )
                     * |a_m| are equivalent to the LAGUERREMODES vector entries.
                     * Arg(a_m) are equivalent to the LAGUERREPHASES vector entries.
                     * The implicit beam properties w0, lambda0, etc... equally apply to all GLM-modes.
                     * The on-axis, in-focus field value of the mode decomposition is normalized to unity:
                     * Snorm := 1 / ( Sum_{m=0}^{m_max}GLM(m,0,0) )
                     *
                     * Spatial mode: Arg(a_m) * GLM(m,r,z) := w0/w(zeta) * L_m( 2*r^2/(w(zeta))^2 ) \
                     *     * exp( I*k*z - I*(2*m+1)*ArcTan(zeta) - r^2 / ( w0^2*(1+I*zeta) ) + I*Arg(a_m) ) )
                     * with w(zeta) = w0*sqrt(1+zeta^2)
                     * with zeta = z / zR
                     * with zR = PI * w0^2 / lambda0
                     *
                     * Uses only radial modes (m) of Laguerre-Polynomials: L_m(x)=L_m^n=0(x)
                     * z is the direction of laser propagation. In PIConGPU, the direction of propagation is y.
                     *
                     * References:
                     * F. Pampaloni et al. (2004), Gaussian, Hermite-Gaussian, and Laguerre-Gaussian beams: A primer
                     * https://arxiv.org/pdf/physics/0410021
                     *
                     * Allen, L. (June 1, 1992). "Orbital angular momentum of light
                     *      and the transformation of Laguerre-Gaussian laser modes"
                     * https://doi.org/10.1103/physreva.45.8185
                     *
                     * Wikipedia on Gaussian laser beams
                     * https://en.wikipedia.org/wiki/Gaussian_beam
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
                        floatD_X planeNoNormal = floatD_X::create(1.0_X);
                        planeNoNormal[planeNormalDir] = 0.0_X;
                        float_X const r2 = pmacc::math::abs2(pos * planeNoNormal);

                        // calculate focus position relative to the laser initialization plane
                        float_X const focusPos = Unitless::FOCUS_POS - pos.y();

                        // rayleigh length (in y-direction)
                        float_X const y_R = float_X(PI) * Unitless::W0 * Unitless::W0 / Unitless::WAVE_LENGTH;

                        // inverse radius of curvature of the beam's  wavefronts
                        float_X const R_y_inv = -focusPos / (y_R * y_R + focusPos * focusPos);

                        // initialize temporary variables
                        float_X etrans(0.0_X);
                        float_X etrans_norm(0.0_X);
                        PMACC_CASSERT_MSG(
                            MODENUMBER_must_be_smaller_than_number_of_entries_in_LAGUERREMODES_vector,
                            Unitless::MODENUMBER < Unitless::LAGUERREMODES_t::dim);
                        PMACC_CASSERT_MSG(
                            MODENUMBER_must_be_smaller_than_number_of_entries_in_LAGUERREPHASES_vector,
                            Unitless::MODENUMBER < Unitless::LAGUERREPHASES_t::dim);
                        for(uint32_t m = 0; m <= Unitless::MODENUMBER; ++m)
                            etrans_norm += typename Unitless::LAGUERREMODES_t{}[m];

                        // beam waist in the near field: w_y(y=0) == W0
                        float_X const w_y = Unitless::W0 * math::sqrt(1.0_X + (focusPos / y_R) * (focusPos / y_R));
                        //! the Gouy phase shift
                        float_X const xi_y = math::atan(-focusPos / y_R);

                        if(Unitless::Polarisation == Unitless::LINEAR_X
                           || Unitless::Polarisation == Unitless::LINEAR_Z)
                        {
                            for(uint32_t m = 0; m <= Unitless::MODENUMBER; ++m)
                            {
                                etrans += typename Unitless::LAGUERREMODES_t{}[m]
                                    * simpleLaguerre(m, 2.0_X * r2 / w_y / w_y) * math::exp(-r2 / w_y / w_y)
                                    * math::cos(
                                              2.0_X * float_X(PI) / Unitless::WAVE_LENGTH * focusPos
                                              - 2.0_X * float_X(PI) / Unitless::WAVE_LENGTH * r2 / 2.0_X * R_y_inv
                                              + (2._X * float_X(m) + 1._X) * xi_y + m_phase +
                                              typename Unitless::LAGUERREPHASES_t{}[m])
                                    * math::exp(
                                              -(r2 / 2.0_X * R_y_inv - focusPos
                                                - m_phase / 2.0_X / float_X(PI) * Unitless::WAVE_LENGTH)
                                              * (r2 / 2.0_X * R_y_inv - focusPos
                                                 - m_phase / 2.0_X / float_X(PI) * Unitless::WAVE_LENGTH)
                                              / SPEED_OF_LIGHT / SPEED_OF_LIGHT / (2.0_X * Unitless::PULSE_LENGTH)
                                              / (2.0_X * Unitless::PULSE_LENGTH));
                            }
                            this->m_elong *= etrans / etrans_norm;
                        }
                        else if(Unitless::Polarisation == Unitless::CIRCULAR)
                        {
                            for(uint32_t m = 0; m <= Unitless::MODENUMBER; ++m)
                            {
                                etrans += typename Unitless::LAGUERREMODES_t{}[m]
                                    * simpleLaguerre(m, 2.0_X * r2 / w_y / w_y) * math::exp(-r2 / w_y / w_y)
                                    * math::cos(
                                              2.0_X * float_X(PI) / Unitless::WAVE_LENGTH * focusPos
                                              - 2.0_X * float_X(PI) / Unitless::WAVE_LENGTH * r2 / 2.0_X * R_y_inv
                                              + (2._X * float_X(m) + 1._X) * xi_y + m_phase +
                                              typename Unitless::LAGUERREPHASES_t{}[m])
                                    * math::exp(
                                              -(r2 / 2.0_X * R_y_inv - focusPos
                                                - m_phase / 2.0_X / float_X(PI) * Unitless::WAVE_LENGTH)
                                              * (r2 / 2.0_X * R_y_inv - focusPos
                                                 - m_phase / 2.0_X / float_X(PI) * Unitless::WAVE_LENGTH)
                                              / SPEED_OF_LIGHT / SPEED_OF_LIGHT / (2.0_X * Unitless::PULSE_LENGTH)
                                              / (2.0_X * Unitless::PULSE_LENGTH));
                            }
                            this->m_elong.x() *= etrans / etrans_norm;
                            m_phase += float_X(PI / 2.0);
                            etrans = 0.0_X;
                            for(uint32_t m = 0; m <= Unitless::MODENUMBER; ++m)
                            {
                                etrans += typename Unitless::LAGUERREMODES_t{}[m]
                                    * simpleLaguerre(m, 2.0_X * r2 / w_y / w_y) * math::exp(-r2 / w_y / w_y)
                                    * math::cos(
                                              2.0_X * float_X(PI) / Unitless::WAVE_LENGTH * focusPos
                                              - 2.0_X * float_X(PI) / Unitless::WAVE_LENGTH * r2 / 2.0_X * R_y_inv
                                              + (2._X * float_X(m) + 1._X) * xi_y + m_phase +
                                              typename Unitless::LAGUERREPHASES_t{}[m])
                                    * math::exp(
                                              -(r2 / 2.0_X * R_y_inv - focusPos
                                                - m_phase / 2.0_X / float_X(PI) * Unitless::WAVE_LENGTH)
                                              * (r2 / 2.0_X * R_y_inv - focusPos
                                                 - m_phase / 2.0_X / float_X(PI) * Unitless::WAVE_LENGTH)
                                              / SPEED_OF_LIGHT / SPEED_OF_LIGHT / (2.0_X * Unitless::PULSE_LENGTH)
                                              / (2.0_X * Unitless::PULSE_LENGTH));
                            }
                            this->m_elong.z() *= etrans / etrans_norm;
                            // reminder: if you want to use phase below, substract pi/2
                            // m_phase -= float_X( PI / 2.0 );
                        }

                        BaseFunctor::operator()(localCell);
                    }
                };
            } // namespace acc

            template<typename T_Params>
            struct GaussianBeam : public gaussianBeam::Unitless<T_Params>
            {
                using Unitless = gaussianBeam::Unitless<T_Params>;

                float3_X elong;
                float_X phase;
                typename FieldE::DataBoxType dataBoxE;
                DataSpace<simDim> offsetToTotalDomain;

                /** constructor
                 *
                 * @param currentStep current simulation time step
                 */
                HINLINE GaussianBeam(float_X currentStep)
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

                    float_64 const runTime = DELTA_T * currentStep - Unitless::laserTimeShift;

                    // calculate focus position relative to the laser initialization plane
                    float_X const focusPos = Unitless::FOCUS_POS - Unitless::initPlaneY * CELL_HEIGHT;

                    elong = float3_X::create(0.0_X);

                    // This check is done here on HOST, since std::numeric_limits<float_X>::epsilon() does not compile
                    // on laserTransversal(), which is on DEVICE.
                    float_X etrans_norm(0.0_X);

                    PMACC_CASSERT_MSG(
                        MODENUMBER_must_be_smaller_than_number_of_entries_in_LAGUERREMODES_vector,
                        Unitless::MODENUMBER < Unitless::LAGUERREMODES_t::dim);
                    for(uint32_t m = 0; m <= Unitless::MODENUMBER; ++m)
                        etrans_norm += typename Unitless::LAGUERREMODES_t{}[m];
                    PMACC_VERIFY_MSG(
                        math::abs(etrans_norm) > std::numeric_limits<float_X>::epsilon(),
                        "Sum of LAGUERREMODES can not be 0.");


                    // a symmetric pulse will be initialized at position z=0 for
                    // a time of PULSE_INIT * PULSE_LENGTH = INIT_TIME.
                    // we shift the complete pulse for the half of this time to start with
                    // the front of the laser pulse.
                    constexpr float_64 mue = 0.5 * Unitless::INIT_TIME;

                    // rayleigh length (in y-direction)
                    constexpr float_64 y_R = PI * Unitless::W0 * Unitless::W0 / Unitless::WAVE_LENGTH;
                    // gaussian beam waist in the nearfield: w_y(y=0) == W0
                    float_64 const w_y = Unitless::W0 * math::sqrt(1.0 + (focusPos / y_R) * (focusPos / y_R));

                    float_64 envelope = float_64(Unitless::AMPLITUDE);
                    if(simDim == DIM2)
                        envelope *= math::sqrt(float_64(Unitless::W0) / w_y);
                    else if(simDim == DIM3)
                        envelope *= float_64(Unitless::W0) / w_y;
                    /* no 1D representation/implementation */

                    if(Unitless::Polarisation == Unitless::LINEAR_X)
                    {
                        elong.x() = float_X(envelope);
                    }
                    else if(Unitless::Polarisation == Unitless::LINEAR_Z)
                    {
                        elong.z() = float_X(envelope);
                    }
                    else if(Unitless::Polarisation == Unitless::CIRCULAR)
                    {
                        elong.x() = float_X(envelope) / math::sqrt(2.0_X);
                        elong.z() = float_X(envelope) / math::sqrt(2.0_X);
                    }

                    phase = 2.0_X * float_X(PI) * float_X(Unitless::f)
                            * (runTime - float_X(mue) - focusPos / SPEED_OF_LIGHT)
                        + Unitless::LASER_PHASE;
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
                HDINLINE acc::GaussianBeam<Unitless> operator()(
                    T_Worker const&,
                    DataSpace<simDim> const& localSupercellOffset) const
                {
                    auto const superCellToLocalOriginCellOffset = localSupercellOffset * SuperCellSize::toRT();
                    return acc::GaussianBeam<Unitless>(
                        dataBoxE,
                        superCellToLocalOriginCellOffset,
                        offsetToTotalDomain,
                        elong,
                        phase);
                }

                //! get the name of the laser profile
                static HINLINE std::string getName()
                {
                    return "GaussianBeam";
                }
            };

        } // namespace laserProfiles
    } // namespace fields
} // namespace picongpu
