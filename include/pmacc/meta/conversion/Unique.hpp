/* Copyright 2022 Sergei Bastrakov
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

#include "pmacc/meta/conversion/ToSeq.hpp"

#include <boost/mpl/copy_if.hpp>
#include <boost/mpl/count.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/size.hpp>


namespace pmacc
{
    /** boost::mpl predicate to check if the given type is present exactly once in the given sequence
     *
     * Defines result as ::type.
     *
     * @tparam T_Seq type sequence
     * @tparam T target type
     */
    template<typename T_Seq, typename T>
    struct IsPresentOnce
    {
        using Count = typename bmpl::count<T_Seq, T>::type;
        using type = typename bmpl::equal_to<Count, bmpl::int_<1>>::type;
    };

    /** Make a sequence out of the input sequence with the duplicate elements removed
     *
     * This operation turned out surprisingly tricky to implement with boost::mpl, see #4078 for details.
     *
     * @tparam T_Seq source sequence
     */
    template<typename T_Seq>
    struct Unique
    {
        using type = typename bmpl::copy_if<T_Seq, IsPresentOnce<T_Seq, bmpl::_>>::type;
    };

    //! Helper alias for @see Unique<>
    template<typename T_Seq>
    using Unique_t = typename Unique<T_Seq>::type;

    /** Flag whether the given sequence is unique (has no duplicate elements)
     *
     * @tparam T_Seq sequence
     */
    template<typename T_Seq>
    static constexpr bool isUnique = (bmpl::size<T_Seq>::value == bmpl::size<Unique_t<T_Seq>>::value);

} // namespace pmacc
