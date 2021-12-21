/* Copyright 2019-2022 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once


namespace pmacc
{
    /** prevent a type for boost apply
     *
     * Avoid that boost apply is called for a given type.
     *
     * @tparam T_LockedType type where a boost::apply should not be performed
     *
     * @code{.cpp}
     *
     * struct Foo{};
     * struct Bar{};
     *
     * using Result = boost::apply1<
     *     ApplyGuard< Foo >,
     *     Bar
     * >;
     * PMACC_CASSERT(
     *  boost::is_same<
     *      Foo,
     *      Result
     *  >::value
     * );
     * @endcode
     */
    template<typename T_LockedType>
    struct ApplyGuard
    {
        template<typename... T_Args>
        using fn = T_LockedType;
    };

} // namespace pmacc
