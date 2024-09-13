/* Copyright 2020-2023 Sergei Bastrakov
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

namespace picongpu::fields::incidentField::traits::detail
{
    /** Get type of incident field functor for the given profile type
     *
     * The resulting functor is set as ::type.
     * These traits have to be specialized by all profiles.
     *
     * @tparam T_Profile profile type
     *
     * @{
     */

    //! Get functor for incident E values
    template<typename T_Profile>
    struct GetFunctorIncidentE;

    //! Get functor for incident B values
    template<typename T_Profile>
    struct GetFunctorIncidentB;

    /** @} */

    /** Type of incident E/B functor for the given profile type
     *
     * These are helper aliases to wrap GetFunctorIncidentE/B.
     * The latter present customization points.
     *
     * @tparam T_Profile profile type
     *
     * @{
     */

    //! Functor for incident E values
    template<typename T_Profile>
    using FunctorIncidentE = typename GetFunctorIncidentE<T_Profile>::type;

    //! Functor for incident B values
    template<typename T_Profile>
    using FunctorIncidentB = typename GetFunctorIncidentB<T_Profile>::type;

    /** @} */
} // namespace picongpu::fields::incidentField::traits::detail
