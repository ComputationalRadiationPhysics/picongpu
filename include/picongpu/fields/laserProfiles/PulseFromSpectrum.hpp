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
            namespace pulseFromSpectrum
            {
                template<typename T_Params>
                struct Unitless : public T_Params
                {
                    using Params = T_Params;

                    static constexpr float_X WAVE_LENGTH
                        = float_X(Params::WAVE_LENGTH_SI / UNIT_LENGTH); // unit: meter
                    static constexpr float_X v0
                        = (::picongpu::SI::SPEED_OF_LIGHT_SI / Params::WAVE_LENGTH_SI) * UNIT_TIME; // unit: seconds^-1
                    static constexpr float_X PULSE_DURATION
                        = float_X(Params::PULSE_DURATION_SI / UNIT_TIME); // unit: seconds (1 sigma)
                    static constexpr float_X AMPLITUDE
                        = float_X(Params::AMPLITUDE_SI / UNIT_EFIELD); // unit: volt / meter
                    static constexpr float_X PULSE_INIT = float_X(Params::PULSE_INIT); // unit: none
                    static constexpr float_X GDD
                        = float_X(Params::GDD_SI / (UNIT_TIME * UNIT_TIME)); // unit: seconds^2
                    static constexpr float_X TOD
                        = float_X(Params::TOD_SI / (UNIT_TIME * UNIT_TIME * UNIT_TIME)); // unit: seconds^3
                    static constexpr float_X W0 = float_X(Params::W0_SI / UNIT_LENGTH); // unit: meter
                    static constexpr float_X FOCUS_POS = float_X(Params::FOCUS_POS_SI / UNIT_LENGTH); // unit: meter
                    static constexpr float_X INIT_TIME = float_X(
                        (Params::PULSE_INIT * Params::PULSE_DURATION_SI)
                        / UNIT_TIME); // unit: seconds (full inizialisation duration)

                    /* Rayleigh-Length in y-direction
                     */
                    static constexpr float_X Y_R = float_X(PI * W0 * W0 / WAVE_LENGTH); // unit: meter
                };
            } // namespace pulseFromSpectrum

            namespace acc
            {
                template<typename T_Unitless>
                struct PulseFromSpectrum
                    : public T_Unitless
                    , public acc::BaseFunctor<T_Unitless::initPlaneY>
                {
                    using Unitless = T_Unitless;
                    using BaseFunctor = acc::BaseFunctor<T_Unitless::initPlaneY>;

                    float_X m_currentStep;

                    /** gauss spectrum
                     * v0 is the central frequency
                     * PULSE_DURATION is sigma of std. gauss for intensity (E^2) in time domain
                     * @param v frequency
                     */
                    HDINLINE float_X gauss_spectrum(float_X const v) const
                    {
                        // Norm is choosen so that the maximum after fourier transformation is unity.
                        float_X const norm = math::sqrt(float_X(PI)) * Unitless::PULSE_DURATION;
                        float_X const exponent = 2._X * Unitless::PULSE_DURATION * float_X(PI) * (v - Unitless::v0);
                        return norm * math::exp(-exponent * exponent);
                    }

                    /** transversal spectrum at the init plane
                     * to implement a laser-pulse with a transverse profile, the spectrum needs to depend on the
                     * distance to the beam axis: spectrum(v) --> spectrum(v, r) (the transversal profile is radial
                     * symmetric)
                     * @param v frequency
                     * @param r2 distance to beam axis to the power of 2
                     * @param y_to_focus y-coordinate relative to focus-position
                     */
                    HDINLINE float_X
                    transversal_spectrum(float_X const v, float_X const r2, float_X const y_to_focus) const
                    {
                        float_X const y_rel = y_to_focus / Unitless::Y_R;
                        float_X const waist = Unitless::W0 * math::sqrt(1._X + y_rel * y_rel);
                        float_X const transversal_spectrum = gauss_spectrum(v) * math::exp(-r2 / (waist * waist));
                        return transversal_spectrum;
                    }

                    /** Spectral phase phi of the pulse with complex electric field
                     * E = E_amp * e^[-I phi].
                     * Depends on frequency and radius to beam axis at the init plane.
                     * The phase is altered for two purposes:
                     * 1. to be able to choose freely the GDD/TOD of the laser-pulse
                     * 2. to implement the transversal profile (thus the phase depends on the distance to the beam
                     * axis)
                     * @param v frequency
                     * @param r2 distance to beam axis to the power of 2
                     * @param y_to_focus y-coordinate relative to focus-position
                     */
                    HDINLINE float_X phase_v(float_X const v, float_X const r2, float_X const y_to_focus) const
                    {
                        float_X const dOmega = 2._X * float_X(PI) * (v - Unitless::v0);
                        float_X const TD = -y_to_focus / SPEED_OF_LIGHT;
                        float_X const phase_shift_TD = TD * dOmega;
                        float_X const phase_shift_GDD = 0.5_X * Unitless::GDD * dOmega * dOmega;
                        float_X const phase_shift_TOD = (1._X / 6._X) * Unitless::TOD * dOmega * dOmega * dOmega;
                        float_X const phase_shift_transversal_1 = -math::atan(y_to_focus / Unitless::Y_R);
                        float_X const phase_shift_transversal_2 = 2._X * float_X(PI) * v * y_to_focus / SPEED_OF_LIGHT;
                        float_X const R_inv = y_to_focus / (y_to_focus * y_to_focus + Unitless::Y_R * Unitless::Y_R);
                        float_X const phase_shift_transversal_3 = r2 * float_X(PI) * v * R_inv / SPEED_OF_LIGHT;
                        float_X const phi = phase_shift_TD + phase_shift_GDD + phase_shift_TOD
                            + phase_shift_transversal_1 + phase_shift_transversal_2 + phase_shift_transversal_3;
                        return phi;
                    }

                    /** fourier-transformation from frequency domain to time domain
                     * @param currentStep current simulation time step
                     * @param r2 radius to beam axis to the power of 2
                     * @param y_to_focus y-coordinate relative to focus-position
                     * @return E_t electric field derived from spectrum for the current timestep
                     */
                    HDINLINE float_X
                    E_t_dft(uint32_t const currentStep, float_X const r2, float_X const y_to_focus) const
                    {
                        // number of steps for the fourier-transformation
                        // needs to be signed to ease difference calculation in next line
                        int32_t const N
                            = int32_t(.5_X * (Unitless::PULSE_INIT * Unitless::PULSE_DURATION / DELTA_T - 1._X));

                        // timesteps for DFT range from -N*dt to N*dt -> 2N+1 timesteps, equally spaced
                        float_X const runTime = float_X(int32_t(currentStep) - N) * DELTA_T;

                        /* Calculation of the E(t) using trigonometric Interpolation.
                         * Coefficients can be easily determined cause the spectrum is given.
                         */
                        float_X E_a = (1._X / DELTA_T) * transversal_spectrum(0._X, r2, y_to_focus)
                            * math::cos(phase_v(0._X, r2, y_to_focus)); // from first symm. coeff. (E_a(k=0) = a/2)
                        float_X E_b = 0._X; // from first antisymm. coeff. (E_b(k=0) = 0)

                        // rest of trig. Interpolation
                        for(int32_t k = 1; k < N + 1; ++k)
                        {
                            float_X const v_k
                                = float_X(k) / (float_X(2 * N + 1) * DELTA_T); // discrete values of frequency
                            float_X const a = (2._X / DELTA_T) * transversal_spectrum(v_k, r2, y_to_focus)
                                * math::cos(phase_v(v_k, r2, y_to_focus)); // symm. coeff. trig. Interpolation
                            float_X const b = (2._X / DELTA_T) * transversal_spectrum(v_k, r2, y_to_focus)
                                * math::sin(phase_v(v_k, r2, y_to_focus)); // antisymm. coeff. trig. Interpolation

                            E_a += a * math::cos(2._X * float_X(PI) * v_k * runTime);
                            E_b += b * math::sin(2._X * float_X(PI) * v_k * runTime);
                        }

                        // E(t)-Field derived from Spectrum
                        float_X const E_t = (E_a + E_b) / float_X(2 * N + 1);

                        return E_t;
                    }

                    /** Device-Side Constructor
                     *
                     * @param superCellToLocalOriginCellOffset local offset in cells to current supercell
                     * @param offsetToTotalDomain offset to origin of global (@todo: total) coordinate system (possibly
                     * after transform to centered origin)
                     */
                    HDINLINE PulseFromSpectrum(
                        typename FieldE::DataBoxType const& dataBoxE,
                        DataSpace<simDim> const& superCellToLocalOriginCellOffset,
                        DataSpace<simDim> const& offsetToTotalDomain,
                        float3_X const& elong,
                        uint32_t const currentStep)
                        : BaseFunctor(dataBoxE, superCellToLocalOriginCellOffset, offsetToTotalDomain, elong)
                        , m_currentStep(currentStep)
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
                        floatD_X planeNoNormal = floatD_X::create(1._X);
                        planeNoNormal[planeNormalDir] = 0._X;
                        float_X const r2 = pmacc::math::abs2(pos * planeNoNormal);

                        // position of the init plane relative to the focus position
                        float_X const y_to_focus = pos.y() - Unitless::FOCUS_POS;

                        // polarisation (only linear polarisation!)
                        if(Unitless::Polarisation == Unitless::LINEAR_X
                           || Unitless::Polarisation == Unitless::LINEAR_Z)
                        {
                            this->m_elong *= E_t_dft(m_currentStep, r2, y_to_focus);
                        }

                        BaseFunctor::operator()(localCell);
                    }
                };
            } // namespace acc

            template<typename T_Params>
            struct PulseFromSpectrum : public pulseFromSpectrum::Unitless<T_Params>
            {
                using Unitless = pulseFromSpectrum::Unitless<T_Params>;

                float3_X elong;
                uint32_t m_currentStep;
                typename FieldE::DataBoxType dataBoxE;
                DataSpace<simDim> offsetToTotalDomain;

                /** constructor
                 *
                 * @param currentStep current simulation time step
                 */
                HINLINE PulseFromSpectrum(uint32_t const currentStep) : m_currentStep(currentStep)
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

                    // set e-field to 0
                    elong = float3_X::create(0._X);

                    // calculate focus position relative to the laser initialization plane
                    float_X const focusPos = Unitless::FOCUS_POS - Unitless::initPlaneY * CELL_HEIGHT;

                    // gaussian beam waist in the nearfield: w_y(y=0) == W0
                    float_64 const y_rel = focusPos / Unitless::Y_R;
                    float_64 const waist = Unitless::W0 * math::sqrt(1.0 + y_rel * y_rel);

                    float_64 envelope = float_64(Unitless::AMPLITUDE);
                    if(simDim == DIM2)
                        envelope *= math::sqrt(float_64(Unitless::W0) / waist);
                    else if(simDim == DIM3)
                        envelope *= float_64(Unitless::W0) / waist;
                    /* no 1D representation/implementation */

                    if(Unitless::Polarisation == Unitless::LINEAR_X)
                    {
                        elong.x() = float_X(envelope);
                    }
                    else if(Unitless::Polarisation == Unitless::LINEAR_Z)
                    {
                        elong.z() = float_X(envelope);
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
                HDINLINE acc::PulseFromSpectrum<Unitless> operator()(
                    T_Worker const&,
                    DataSpace<simDim> const& localSupercellOffset) const
                {
                    auto const superCellToLocalOriginCellOffset = localSupercellOffset * SuperCellSize::toRT();
                    return acc::PulseFromSpectrum<Unitless>(
                        dataBoxE,
                        superCellToLocalOriginCellOffset,
                        offsetToTotalDomain,
                        elong,
                        m_currentStep);
                }

                //! get the name of the laser profile
                static HINLINE std::string getName()
                {
                    return "PulseFromSpectrum";
                }
            };

        } // namespace laserProfiles
    } // namespace fields
} // namespace picongpu
