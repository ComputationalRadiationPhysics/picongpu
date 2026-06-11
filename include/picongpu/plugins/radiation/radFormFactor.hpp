/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Richard Pausch, Sergei Bastrakov, Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once


#include "picongpu/plugins/radiation/VectorTypes.hpp"
#include "picongpu/plugins/radiation/utilities.hpp"

#include <cstdint>

namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            /** Base radiation form factor functor
             *
             * Each functor instance is used for fixed frequency and observation direction
             *
             * This class serves to define the interface requirements for radiation form factor implementations.
             * So if roughly defines a "concept", does not have to be inherited.
             *
             * Most implementations would precompute normalized coherent amplification for the given frequency and
             * observation direction in the constructor. It will be then used by operator() called in the hot loop of
             * the radiation kernel.
             */
            struct RadFormFactorConcept
            {
                /** Create a functor for the given frequency and observation direction, is required
                 *
                 * @param omega frequency
                 * @param observerUnitVec unit vector of observation direction
                 */
                HDINLINE RadFormFactorConcept(float_X const omega, vector_64 const& observerUnitVec);

                /** Calculate form factor value \f$ \sqrt( | \mathcal{F} |^2 ) \f$
                 *
                 * @param N macro particle weighting
                 */
                HDINLINE float_X operator()(float_X const N) const;
            };

            namespace radFormFactor_baseShape_3D
            {
                /** general form factor class of discrete charge distribution of PIC particle shape of order
                 * T_shapeOrder
                 *
                 * Adheres to the RadFormFactorConcept concept
                 *
                 * @tparam T_shapeOrder order of charge distribution shape in PIC code used for radiation form factor
                 */
                template<uint32_t T_shapeOrder>
                struct RadFormFactor
                {
                    /** Construct the form factor functor for the given frequency and observation direction
                     *
                     * @param omega frequency
                     * @param observerUnitVec unit vector of observation direction
                     */
                    HDINLINE RadFormFactor(float_X const omega, vector_64 const& observerUnitVec)
                        : normalizedCoherentAmplification(getNormalizedCoherentAmplification(omega, observerUnitVec))
                    {
                    }

                    /** Form Factor for T_shapeOrder-order particle shape charge distribution of N discrete electrons:
                     * \f[ | \mathcal{F} |^2 = N + (N*N - N) * (sinc^2(n_x * L_x * \omega) * sinc^2(n_y * L_y * \omega)
                     * * sinc^2(n_z * L_z * \omega))^T_shapeOrder \f]
                     *
                     * with observation direction (unit vector) \f$ \vec{n} = (n_x, n_y, n_z) \f$
                     * and with:
                     *
                     * @param N = macro particle weighting
                     * @param omega = frequency at which to calculate the  form factor
                     *
                     * @return the Form Factor: \f$ \sqrt( | \mathcal{F} |^2 ) \f$
                     */
                    HDINLINE float_X operator()(float_X const N) const
                    {
                        return math::sqrt(N + (N * N - N) * normalizedCoherentAmplification);
                    }

                private:
                    //! Precomputed value of normalized coherent amplification for the given frequency
                    float_X const normalizedCoherentAmplification;

                    /** Calculate normalized coherent amplification for the given frequency and observation direction
                     *
                     * @param omega frequency
                     * @param observerUnitVec unit vector of observation direction
                     */
                    HDINLINE float_X
                    getNormalizedCoherentAmplification(float_X const omega, vector_64 const& observerUnitVec) const
                    {
                        // here we combine sinc^2(..) with (...)^T_shapeOrder to ...^(2 * T_shapeOrder)
                        float_X sincValue = 1.0_X;
                        for(uint32_t d = 0; d < DIM3; ++d)
                            sincValue *= pmacc::math::sinc(
                                observerUnitVec[d] * sim.pic.getCellSize()[d] / (sim.pic.getSpeedOfLight() * 2.0_X)
                                * omega);
                        return pmacc::math::cPow(sincValue, static_cast<uint32_t>(2u) * T_shapeOrder);
                    }
                };
            } // namespace radFormFactor_baseShape_3D

            namespace radFormFactor_CIC_3D
            {
                //! Adheres to the RadFormFactorConcept concept
                struct RadFormFactor : public radFormFactor_baseShape_3D::RadFormFactor<1>
                {
                    /** Construct the form factor functor for the given frequency and observation direction
                     *
                     * @param omega frequency
                     * @param observerUnitVec unit vector of observation direction
                     */
                    HDINLINE RadFormFactor(float_X const omega, vector_64 const& observerUnitVec)
                        : radFormFactor_baseShape_3D::RadFormFactor<1>(omega, observerUnitVec)
                    {
                    }
                };
            } // namespace radFormFactor_CIC_3D

            namespace radFormFactor_TSC_3D
            {
                //! Adheres to the RadFormFactorConcept concept
                struct RadFormFactor : public radFormFactor_baseShape_3D::RadFormFactor<2>
                {
                    /** Construct the form factor functor for the given frequency and observation direction
                     *
                     * @param omega frequency
                     * @param observerUnitVec unit vector of observation direction
                     */
                    HDINLINE RadFormFactor(float_X const omega, vector_64 const& observerUnitVec)
                        : radFormFactor_baseShape_3D::RadFormFactor<2>(omega, observerUnitVec)
                    {
                    }
                };
            } // namespace radFormFactor_TSC_3D

            namespace radFormFactor_PCS_3D
            {
                //! Adheres to the RadFormFactorConcept concept
                struct RadFormFactor : public radFormFactor_baseShape_3D::RadFormFactor<3>
                {
                    /** Construct the form factor functor for the given frequency and observation direction
                     *
                     * @param omega frequency
                     * @param observerUnitVec unit vector of observation direction
                     */
                    HDINLINE RadFormFactor(float_X const omega, vector_64 const& observerUnitVec)
                        : radFormFactor_baseShape_3D::RadFormFactor<3>(omega, observerUnitVec)
                    {
                    }
                };
            } // namespace radFormFactor_PCS_3D

            namespace radFormFactor_CIC_1Dy
            {
                //! Adheres to the RadFormFactorConcept concept
                struct RadFormFactor
                {
                    /** Construct the form factor functor for the given frequency
                     *
                     * @param omega frequency
                     * @param observerUnitVec unit vector of observation direction,
                     *                        not used for this form factor but requried by RadFormFactorConcept
                     */
                    HDINLINE RadFormFactor(float_X const omega, vector_64 const&)
                        : normalizedCoherentAmplification(
                              util::square(
                                  pmacc::math::sinc(
                                      sim.pic.getCellSize().y() / (sim.pic.getSpeedOfLight() * 2.0_X) * omega)))
                    {
                    }

                    /** Form Factor for 1-d CIC charge distribution iy y of N discrete electrons:
                     * \f[ | \mathcal{F} |^2 = N + (N*N - N) * sinc^2(n_y * L_y * \omega) \f]
                     *
                     * with observation direction (unit vector) \f$ \vec{n} = (n_x, n_y, n_z) \f$
                     * and with:
                     *
                     * @param N macro particle weighting
                     *
                     * @return the Form Factor: \f$ \sqrt( | \mathcal{F} |^2 ) \f$
                     */
                    HDINLINE float_X operator()(float_X const N) const
                    {
                        return math::sqrt(N + (N * N - N) * normalizedCoherentAmplification);
                    }

                private:
                    //! Precomputed value of normalized coherent amplification for the given frequency
                    float_X const normalizedCoherentAmplification;
                };
            } // namespace radFormFactor_CIC_1Dy

            namespace radFormFactor_Gauss_spherical
            {
                //! Adheres to the RadFormFactorConcept concept
                struct RadFormFactor
                {
                    /** Construct the form factor functor for the given frequency
                     *
                     * Currently a fixed sigma of sim.pic.getDt() * c is used to describe the distribution - might
                     * become a parameter.
                     *
                     * @param omega frequency
                     * @param observerUnitVec unit vector of observation direction,
                     *                        not used for this form factor but requried by RadFormFactorConcept
                     */
                    HDINLINE RadFormFactor(float_X const omega, vector_64 const&)
                        : normalizedCoherentAmplification(
                              util::square(math::exp(-0.5_X * util::square(omega * 0.5_X * sim.pic.getDt()))))
                    {
                    }

                    /** Form Factor for point-symmetric Gauss-shaped charge distribution of N discrete electrons:
                     * \f[ <rho(r)> = N*q_e* 1/sqrt(2*pi*sigma^2) * exp(-0.5 * r^2/sigma^2) \f]
                     * with sigma = 0.5*c/delta_t (0.5 because sigma is defined around center)
                     *
                     * @param N macro particle weighting
                     *
                     * @return the Form Factor: \f$ \sqrt( | \mathcal{F} |^2 ) \f$
                     */
                    HDINLINE float_X operator()(float_X const N) const
                    {
                        return math::sqrt(N + (N * N - N) * normalizedCoherentAmplification);
                    }

                private:
                    //! Precomputed value of normalized coherent amplification for the given frequency
                    float_X const normalizedCoherentAmplification;
                };
            } // namespace radFormFactor_Gauss_spherical

            namespace radFormFactor_Gauss_cell
            {
                //! Adheres to the RadFormFactorConcept concept
                struct RadFormFactor
                {
                    /** Construct the form factor functor for the given frequency and observation direction
                     *
                     * @param omega frequency
                     * @param observerUnitVec unit vector of observation direction
                     */
                    HDINLINE RadFormFactor(float_X const omega, vector_64 const& observerUnitVec)
                        : normalizedCoherentAmplification(getNormalizedCoherentAmplification(omega, observerUnitVec))
                    {
                    }

                    /** Form Factor for per-dimension Gauss-shaped charge distribution of N discrete electrons:
                     * \f[ <rho(r)> = N*q_e* product[d={x,y,z}](1/sqrt(2*pi*sigma_d^2) * exp(-0.5 * d^2/sigma_d^2)) \f]
                     * with sigma_d = 0.5*cell_width_d*n_d
                     *
                     * @param N macro particle weighting
                     *
                     * @return the Form Factor: \f$ \sqrt( | \mathcal{F} |^2 ) \f$
                     */
                    HDINLINE float_X operator()(float_X const N) const
                    {
                        return math::sqrt(N + (N * N - N) * normalizedCoherentAmplification);
                    }

                private:
                    //! Precomputed value of normalized coherent amplification for the given frequency
                    float_X const normalizedCoherentAmplification;

                    /** Calculate normalized coherent amplification for the given frequency
                     *
                     * @param omega frequency
                     * @param observerUnitVec unit vector of observation direction
                     */
                    HDINLINE float_X
                    getNormalizedCoherentAmplification(float_X const omega, vector_64 const& observerUnitVec)
                    {
                        return util::square(
                            math::exp(
                                -0.5_X
                                * (util::square(
                                       observerUnitVec.x() * sim.pic.getCellSize().x()
                                       / (sim.pic.getSpeedOfLight() * 2.0_X) * omega)
                                   + util::square(
                                       observerUnitVec.y() * sim.pic.getCellSize().y()
                                       / (sim.pic.getSpeedOfLight() * 2.0_X) * omega)
                                   + util::square(
                                       observerUnitVec.z() * sim.pic.getCellSize().z()
                                       / (sim.pic.getSpeedOfLight() * 2.0_X) * omega))));
                    }
                };
            } // namespace radFormFactor_Gauss_cell

            namespace radFormFactor_incoherent
            {
                //! Adheres to the RadFormFactorConcept concept
                struct RadFormFactor
                {
                    /** Construct the form factor functor for the given frequency
                     *
                     * @param omega frequency
                     *              not used for this form factor but requried by RadFormFactorConcept
                     * @param observerUnitVec unit vector of observation direction,
                     *                        not used for this form factor but requried by RadFormFactorConcept
                     */
                    HDINLINE RadFormFactor(float_X const, vector_64 const&)
                    {
                    }

                    /** Form Factor for an incoherent charge distribution:
                     *
                     * @param N macro particle weighting
                     *
                     * @return the Form Factor: \f$ \sqrt( | \mathcal{F} |^2 == \sqrt(weighting) \f$
                     */
                    HDINLINE float_X operator()(float_X const N) const
                    {
                        return math::sqrt(N);
                    }
                };
            } // namespace radFormFactor_incoherent

            namespace radFormFactor_coherent
            {
                //! Adheres to the RadFormFactorConcept concept
                struct RadFormFactor
                {
                    /** Construct the form factor functor for the given frequency
                     *
                     * @param omega frequency
                     *              not used for this form factor but requried by RadFormFactorConcept
                     * @param observerUnitVec unit vector of observation direction,
                     *                        not used for this form factor but requried by RadFormFactorConcept
                     */
                    HDINLINE RadFormFactor(float_X const, vector_64 const&)
                    {
                    }

                    /** Form Factor for a coherent charge distribution:
                     *
                     * @param N = macro particle weighting
                     *
                     * @return the Form Factor: \f$ \sqrt( | \mathcal{F} |^2 == \sqrt(weighting) \f$
                     */
                    HDINLINE float_X operator()(float_X const N) const
                    {
                        return N;
                    }
                };
            } // namespace radFormFactor_coherent

        } // namespace radiation
    } // namespace plugins
} // namespace picongpu
