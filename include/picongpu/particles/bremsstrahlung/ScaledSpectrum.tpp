/* Copyright 2016-2021 Heiko Burau
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

#include "picongpu/simulation_defines.hpp"
#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/cuSTL/container/HostBuffer.hpp>

namespace picongpu
{
    namespace particles
    {
        namespace bremsstrahlung
        {
            namespace detail
            {
                /** constructor
                 *
                 * @param linInterpCursor
                 */
                HDINLINE LookupTableFunctor::LookupTableFunctor(LinInterpCursor linInterpCursor)
                    : linInterpCursor(linInterpCursor)
                {
                    float_X const lnEMinTmp(electron::MIN_ENERGY);
                    float_X const lnEMaxTmp(electron::MAX_ENERGY);
                    this->lnEMin = math::log(lnEMinTmp);
                    this->lnEMax = math::log(lnEMaxTmp);
                }

                /** scaled differential cross section
                 *
                 * @param Ekin kinetic energy of the incident electron
                 * @param kappa energy loss normalized to Ekin
                 */
                HDINLINE float_X LookupTableFunctor::operator()(const float_X Ekin, const float_X kappa) const
                {
                    const float_X lnE = math::log(Ekin);

                    const float_X binE = (lnE - this->lnEMin) / (this->lnEMax - this->lnEMin)
                        * static_cast<float_X>(electron::NUM_SAMPLES_EKIN - 1);
                    // in the low-energy limit Bremsstrahlung is not taken into account
                    if(binE < float_X(0.0))
                        return float_X(0.0);
                    const float_X binKappa = kappa * static_cast<float_X>(electron::NUM_SAMPLES_KAPPA - 1);

                    if(picLog::log_level & picLog::CRITICAL::lvl)
                    {
                        if(Ekin < electron::MIN_ENERGY || Ekin > electron::MAX_ENERGY)
                        {
                            const float_64 Ekin_SI = Ekin * UNIT_ENERGY;
                            printf(
                                "[Bremsstrahlung] error lookup table: Ekin=%g MeV is out of range.\n",
                                float_X(Ekin_SI * UNITCONV_Joule_to_keV * float_X(1.0e-3)));
                        }
                        if(kappa < float_X(0.0) || kappa > float_X(1.0))
                            printf("[Bremsstrahlung] error lookup table: kappa=%f is out of range.\n", kappa);
                    }

                    return this->linInterpCursor[float2_X(binE, binKappa)];
                }


            } // namespace detail


            /** differential cross section: cross section per unit energy
             *
             * This is the screened Bethe-Heitler cross section. See e.g.:
             * Salvat, F., et al. "Monte Carlo simulation of bremsstrahlung emission by electrons."
             * Radiation Physics and Chemistry 75.10 (2006): 1201-1219.
             *
             * @param Ekin kinetic electron energy
             * @param kappa energy loss normalized to Ekin
             */
            float_64 ScaledSpectrum::dcs(const float_64 Ekin, const float_64 kappa, const float_64 targetZ) const
            {
                constexpr float_64 pi = pmacc::math::Pi<float_64>::value;
                constexpr float_64 bohrRadius
                    = pi * 4.0 * EPS0 * HBAR * HBAR / (float_64(ELECTRON_MASS) * ELECTRON_CHARGE * ELECTRON_CHARGE);
                constexpr float_64 classicalElRadius = float_64(ELECTRON_CHARGE * ELECTRON_CHARGE)
                    / (pi * 4.0 * EPS0 * ELECTRON_MASS * SPEED_OF_LIGHT * SPEED_OF_LIGHT);
                constexpr float_64 fineStructureConstant
                    = float_64(ELECTRON_CHARGE * ELECTRON_CHARGE) / (pi * 4.0 * EPS0 * HBAR * SPEED_OF_LIGHT);

                constexpr float_64 c = SPEED_OF_LIGHT;
                constexpr float_64 c2 = c * c;
                constexpr float_64 m_e = ELECTRON_MASS;
                constexpr float_64 r_e = classicalElRadius;
                constexpr float_64 alpha = fineStructureConstant;

                const float_64 W = kappa * Ekin;
                const float_64 eps = W / (Ekin + m_e * c2);
                const float_64 R = math::pow(targetZ, float_64(-1.0 / 3.0)) * bohrRadius;
                const float_64 gamma = Ekin / (m_e * c2) + float_64(1.0);
                const float_64 b = R * m_e * c / HBAR / (float_64(2.0) * gamma) * eps / (float_64(1.0) - eps);

                const float_64 phi_1 = float_64(4.0) * math::log(R * m_e * c / HBAR) + float_64(2.0)
                    - float_64(2.0) * math::log(float_64(1.0) + b * b)
                    - float_64(4.0) * b * math::atan(float_64(1.0) / b);
                const float_64 phi_2 = float_64(4.0) * math::log(R * m_e * c / HBAR) + float_64(7.0) / float_64(3.0)
                    - float_64(2.0) * math::log(float_64(1.0) + b * b)
                    - float_64(6.0) * b * math::atan(float_64(1.0) / b)
                    - b * b
                        * (float_64(4.0) - float_64(4.0) * b * math::atan(float_64(1.0) / b)
                           - float_64(3.0) * math::log(float_64(1.0) + float_64(1.0) / (b * b)));

                return r_e * r_e * alpha * targetZ * targetZ / W
                    * (eps * eps * phi_1 + float_64(4.0) / float_64(3.0) * (float_64(1.0) - eps) * phi_2);
            }


            void ScaledSpectrum::init(const float_64 targetZ)
            {
                namespace odeint = boost::numeric::odeint;

                // there is a margin of one cell to make the linear interpolation valid for border cells.
                this->dBufScaledSpectrum = MyBuf(new pmacc::container::DeviceBuffer<float_X, DIM2>(
                    electron::NUM_SAMPLES_EKIN + 1,
                    electron::NUM_SAMPLES_KAPPA + 1));
                this->dBufStoppingPower = MyBuf(new pmacc::container::DeviceBuffer<float_X, DIM2>(
                    electron::NUM_SAMPLES_EKIN + 1,
                    electron::NUM_SAMPLES_KAPPA + 1));

                pmacc::container::HostBuffer<float_X, DIM2> hBufScaledSpectrum(this->dBufScaledSpectrum->size());
                pmacc::container::HostBuffer<float_X, DIM2> hBufStoppingPower(this->dBufStoppingPower->size());
                hBufScaledSpectrum.assign(float_X(0.0));
                hBufStoppingPower.assign(float_X(0.0));

                auto curScaledSpectrum = hBufScaledSpectrum.origin();
                auto curStoppingPower = hBufStoppingPower.origin();

                const float_64 lnEMin = math::log(electron::MIN_ENERGY);
                const float_64 lnEMax = math::log(electron::MAX_ENERGY);

                using state_type = boost::array<float_64, 1>;

                for(uint32_t EkinIdx = 0; EkinIdx < electron::NUM_SAMPLES_EKIN; EkinIdx++)
                {
                    for(uint32_t kappaIdx = 0; kappaIdx < electron::NUM_SAMPLES_KAPPA; kappaIdx++)
                    {
                        float_64 kappa
                            = static_cast<float_64>(kappaIdx) / static_cast<float_64>(electron::NUM_SAMPLES_KAPPA - 1);
                        if(kappa == 0.0)
                            kappa = electron::MIN_KAPPA;

                        const float_64 lnE_norm
                            = static_cast<float_64>(EkinIdx) / static_cast<float_64>(electron::NUM_SAMPLES_EKIN - 1);
                        const float_64 Ekin = math::exp(lnEMin + (lnEMax - lnEMin) * lnE_norm);

                        *curScaledSpectrum(EkinIdx, kappaIdx)
                            = Ekin * kappa * static_cast<float_X>(this->dcs(Ekin, kappa, targetZ));

                        state_type integral_result = {0.0};
                        const float_64 lowerLimit = electron::MIN_KAPPA * Ekin;
                        const float_64 upperLimit = kappa * Ekin;
                        const float_64 stepwidth = upperLimit / electron::NUM_STEPS_STOPPING_POWER_INTERGRAL;
                        StoppingPowerIntegrand integrand(Ekin, *this, targetZ);
                        odeint::integrate(integrand, integral_result, lowerLimit, upperLimit, stepwidth);
                        *curStoppingPower(EkinIdx, kappaIdx) = static_cast<float_X>(integral_result[0]);

                        // check for nans
                        if(*curScaledSpectrum(EkinIdx, kappaIdx) != *curScaledSpectrum(EkinIdx, kappaIdx))
                        {
                            const float_64 Ekin_SI = Ekin * UNIT_ENERGY;
                            const float_64 Ekin_MeV = Ekin_SI * UNITCONV_Joule_to_keV / 1.0e3;
                            std::stringstream errMsg;
                            errMsg << "[Bremsstrahlung] lookup table (scaled spectrum) has NaN-entry at Ekin = "
                                   << Ekin_MeV << " MeV, kappa = " << kappa << std::endl;
                            throw std::runtime_error(errMsg.str().c_str());
                        }
                        if(*curStoppingPower(EkinIdx, kappaIdx) != *curStoppingPower(EkinIdx, kappaIdx))
                        {
                            const float_64 Ekin_SI = Ekin * UNIT_ENERGY;
                            const float_64 Ekin_MeV = Ekin_SI * UNITCONV_Joule_to_keV / 1.0e3;
                            std::stringstream errMsg;
                            errMsg << "[Bremsstrahlung] lookup table (stopping power) has NaN-entry at Ekin = "
                                   << Ekin_MeV << " MeV, kappa = " << kappa << std::endl;
                            throw std::runtime_error(errMsg.str().c_str());
                        }
                    }
                }

                *this->dBufScaledSpectrum = hBufScaledSpectrum;
                *this->dBufStoppingPower = hBufStoppingPower;
            }

            /** Return a functor representing the scaled differential cross section
             *
             * scaled differential cross section = electron energy loss times cross section per unit energy
             */
            detail::LookupTableFunctor ScaledSpectrum::getScaledSpectrumFunctor() const
            {
                LookupTableFunctor::LinInterpCursor linInterpCursor
                    = pmacc::cursor::tools::LinearInterp<float_X>()(this->dBufScaledSpectrum->origin());

                return LookupTableFunctor(linInterpCursor);
            }

            /** Return a functor representing the stopping power
             *
             * stopping power = energy loss per unit length
             */
            detail::LookupTableFunctor ScaledSpectrum::getStoppingPowerFunctor() const
            {
                LookupTableFunctor::LinInterpCursor linInterpCursor
                    = pmacc::cursor::tools::LinearInterp<float_X>()(this->dBufStoppingPower->origin());

                return LookupTableFunctor(linInterpCursor);
            }


        } // namespace bremsstrahlung
    } // namespace particles
} // namespace picongpu
