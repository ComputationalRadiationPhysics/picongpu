/* Copyright 2013-2021 Heiko Burau, Rene Widera, Richard Pausch
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


#include "picongpu/plugins/radiation/utilities.hpp"


namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            namespace radFormFactor_baseShape_3D
            {
                /** general form factor class of discrete charge distribution of PIC particle shape of order
                 * T_shapeOrder
                 *
                 * @tparam T_shapeOrder order of charge distribution shape in PIC code used for radiation form factor
                 */

                template<uint32_t T_shapeOrder>
                struct radFormFactor
                {
                    /** Form Factor for T_shapeOrder-order particle shape charge distribution of N discrete electrons:
                     * \f[ | \mathcal{F} |^2 = N + (N*N - N) * (sinc^2(n_x * L_x * \omega) * sinc^2(n_y * L_y * \omega)
                     * * sinc^2(n_z * L_z * \omega))^T_shapeOrder \f]
                     *
                     * with observation direction (unit vector) \f$ \vec{n} = (n_x, n_y, n_z) \f$
                     * and with:
                     * @param N     = weighting
                     * @param omega = frequency
                     * @param L_d   = the size of the CIC-particle / cell in dimension d
                     *
                     * @param N = macro particle weighting
                     * @param omega = frequency at which to calculate the  form factor
                     * @param observer_unit_vec = observation direction
                     * @return the Form Factor: \f$ \sqrt( | \mathcal{F} |^2 ) \f$
                     */
                    HDINLINE float_X
                    operator()(const float_X N, const float_X omega, vector_X const& observer_unit_vec) const
                    {
                        float_X sincValue = float_X(1.0);
                        for(uint32_t d = 0; d < DIM3; ++d)
                            sincValue *= pmacc::math::sinc(
                                observer_unit_vec[d] * cellSize[d] / (SPEED_OF_LIGHT * float_X(2.0)) * omega);

                        // here we combine sinc^2(..) with (...)^T_shapeOrder to ...^(2 * T_shapeOrder)
                        return math::sqrt(N + (N * N - N) * util::pow(sincValue, 2 * T_shapeOrder));
                    }
                };
            } // namespace radFormFactor_baseShape_3D


            namespace radFormFactor_CIC_3D
            {
                struct radFormFactor : public radFormFactor_baseShape_3D::radFormFactor<1>
                {
                };
            } // namespace radFormFactor_CIC_3D

            namespace radFormFactor_TSC_3D
            {
                struct radFormFactor : public radFormFactor_baseShape_3D::radFormFactor<2>
                {
                };
            } // namespace radFormFactor_TSC_3D

            namespace radFormFactor_PCS_3D
            {
                struct radFormFactor : public radFormFactor_baseShape_3D::radFormFactor<3>
                {
                };
            } // namespace radFormFactor_PCS_3D


            namespace radFormFactor_CIC_1Dy
            {
                struct radFormFactor
                {
                    /** Form Factor for 1-d CIC charge distribution iy y of N discrete electrons:
                     * \f[ | \mathcal{F} |^2 = N + (N*N - N) * sinc^2(n_y * L_y * \omega) \f]
                     *
                     * with observation direction (unit vector) \f$ \vec{n} = (n_x, n_y, n_z) \f$
                     * and with:
                     * @param N     = weighting
                     * @param omega = frequency
                     * @param L_d   = the size of the CIC-particle / cell in dimension d
                     *
                     * @param N = macro particle weighting
                     * @param omega = frequency at which to calculate the  form factor
                     * @param observer_unit_vec = observation direction
                     * @return the Form Factor: \f$ \sqrt( | \mathcal{F} |^2 ) \f$
                     */
                    HDINLINE float_X
                    operator()(const float_X N, const float_X omega, const vector_X observer_unit_vec) const
                    {
                        return math::sqrt(
                            N
                            + (N * N - N)
                                * util::square(
                                    pmacc::math::sinc(CELL_HEIGHT / (SPEED_OF_LIGHT * float_X(2.0)) * omega)));
                    }
                };
            } // namespace radFormFactor_CIC_1Dy


            namespace radFormFactor_Gauss_spherical
            {
                struct radFormFactor
                {
                    /** Form Factor for point-symmetric Gauss-shaped charge distribution of N discrete electrons:
                     * \f[ <rho(r)> = N*q_e* 1/sqrt(2*pi*sigma^2) * exp(-0.5 * r^2/sigma^2) \f]
                     * with sigma = 0.5*c/delta_t (0.5 because sigma is defined around center)
                     *
                     * @param N = macro particle weighting
                     * @param omega = frequency at which to calculate the  form factor
                     * @param observer_unit_vec = observation direction
                     * @return the Form Factor: \f$ \sqrt( | \mathcal{F} |^2 ) \f$
                     */
                    HDINLINE float_X
                    operator()(const float_X N, const float_X omega, const vector_X observer_unit_vec) const
                    {
                        /* currently a fixed sigma of DELTA_T * c is used to describe the distribution - might become a
                         * parameter */
                        return math::sqrt(
                            N
                            + (N * N - N)
                                * util::square(
                                    math::exp(float_X(-0.5) * util::square(omega * float_X(0.5) * DELTA_T))));
                    }
                };
            } // namespace radFormFactor_Gauss_spherical


            namespace radFormFactor_Gauss_cell
            {
                struct radFormFactor
                {
                    /** Form Factor for per-dimension Gauss-shaped charge distribution of N discrete electrons:
                     * \f[ <rho(r)> = N*q_e* product[d={x,y,z}](1/sqrt(2*pi*sigma_d^2) * exp(-0.5 * d^2/sigma_d^2)) \f]
                     * with sigma_d = 0.5*cell_width_d*n_d
                     *
                     * @param N = macro particle weighting
                     * @param omega = frequency at which to calculate the  form factor
                     * @param observer_unit_vec = observation direction
                     * @return the Form Factor: \f$ \sqrt( | \mathcal{F} |^2 ) \f$
                     */
                    HDINLINE float_X
                    operator()(const float_X N, const float_X omega, const vector_X observer_unit_vec) const
                    {
                        return math::sqrt(
                            N
                            + (N * N - N)
                                * util::square(math::exp(
                                    float_X(-0.5)
                                    * (util::square(
                                           observer_unit_vec.x() * CELL_WIDTH / (SPEED_OF_LIGHT * float_X(2.0))
                                           * omega)
                                       + util::square(
                                           observer_unit_vec.y() * CELL_HEIGHT / (SPEED_OF_LIGHT * float_X(2.0))
                                           * omega)
                                       + util::square(
                                           observer_unit_vec.z() * CELL_DEPTH / (SPEED_OF_LIGHT * float_X(2.0))
                                           * omega)))));
                    }
                };
            } // namespace radFormFactor_Gauss_cell


            namespace radFormFactor_incoherent
            {
                struct radFormFactor
                {
                    /** Form Factor for an incoherent charge distribution:
                     *
                     * @param N = macro particle weighting
                     * @param omega = frequency at which to calculate the  form factor
                     * @param observer_unit_vec = observation direction
                     * @return the Form Factor: \f$ \sqrt( | \mathcal{F} |^2 == \sqrt(weighting) \f$
                     */
                    HDINLINE float_X
                    operator()(const float_X N, const float_X omega, const vector_X observer_unit_vec) const
                    {
                        return math::sqrt(N);
                    }
                };
            } // namespace radFormFactor_incoherent


            namespace radFormFactor_coherent
            {
                struct radFormFactor
                {
                    /** Form Factor for a coherent charge distribution:
                     *
                     * @param N = macro particle weighting
                     * @param omega = frequency at which to calculate the  form factor
                     * @param observer_unit_vec = observation direction
                     * @return the Form Factor: \f$ \sqrt( | \mathcal{F} |^2 == \sqrt(weighting) \f$
                     */
                    HDINLINE float_X
                    operator()(const float_X N, const float_X omega, const vector_X observer_unit_vec) const
                    {
                        return N;
                    }
                };
            } // namespace radFormFactor_coherent

        } // namespace radiation
    } // namespace plugins
} // namespace picongpu
