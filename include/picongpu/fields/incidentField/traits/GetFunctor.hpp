/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
