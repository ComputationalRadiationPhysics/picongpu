/* Copyright 2013-2021 Heiko Burau, Rene Widera
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

#include <boost/mpl/void.hpp>
#include "pmacc/math/vector/Int.hpp"
#include "pmacc/types.hpp"
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include "pmacc/cuSTL/cursor/traits.hpp"

namespace mpl = boost::mpl;

namespace pmacc
{
    namespace cursor
    {
        /** A cursor is used to access a single datum and to jump to another one.
         * It is always located at a certain datum. Think of a generalized iterator.
         * \tparam _Accessor Policy functor class that is called inside operator*().
         * It typically returns a reference to the current selected datum.
         * \tparam _Navigator Policy functor class that is called inside operator()().
         * It jumps to another datum.
         * \tparam _Marker Runtime data that is used by the accessor and the navigator.
         * This is typically a data pointer.
         */
        template<typename _Accessor, typename _Navigator, typename _Marker>
        class Cursor
            : private _Accessor
            , _Navigator
        {
        public:
            typedef typename _Accessor::type type;
            typedef typename boost::remove_reference<type>::type ValueType;
            typedef _Accessor Accessor;
            typedef _Navigator Navigator;
            typedef _Marker Marker;
            typedef Cursor<Accessor, Navigator, Marker> This;
            typedef This result_type;

        protected:
            Marker marker;

        public:
            HDINLINE
            Cursor(const Accessor& accessor, const Navigator& navigator, const Marker& marker)
                : Accessor(accessor)
                , Navigator(navigator)
                , marker(marker)
            {
            }

            /** access
             * \return Accessor's return type.
             * Typically a reference to the current selected single datum.
             */
            HDINLINE
            type operator*()
            {
                return Accessor::operator()(this->marker);
            }

            /* This is a const method which is called for a const cursor object.
             * A const cursor object does *not* mean that the data it points to
             * is neccessarily constant too. This is why here the return type is
             * the same as for the non-const method above.
             */
            HDINLINE
            type operator*() const
            {
                return Accessor::operator()(this->marker);
            }

            /** jumping
             * \param jump Specifies a jump relative to the current selected datum.
             * This is usually a int vector but may be any type that navigator accepts.
             * \return A new cursor, which has jumped according to the jump param.
             */
            template<typename Jump>
            HDINLINE This operator()(const Jump& jump) const
            {
                Navigator newNavigator(getNavigator());
                Marker newMarker = newNavigator(this->marker, jump);
                return This(getAccessor(), newNavigator, newMarker);
            }

            /* convenient method which is available if the navigator accepts a Int<1> */
            HDINLINE This operator()(int x) const
            {
                return (*this)(math::Int<1>(x));
            }

            /* convenient method which is available if the navigator accepts a Int<2> */
            HDINLINE This operator()(int x, int y) const
            {
                return (*this)(math::Int<2u>(x, y));
            }

            /* convenient method which is available if the navigator accepts a Int<3> */
            HDINLINE This operator()(int x, int y, int z) const
            {
                return (*this)(math::Int<3>(x, y, z));
            }

            /* convenient method which is available if the navigator implements operator++ */
            HDINLINE void operator++()
            {
                Navigator::operator++;
            }
            /* convenient method which is available if the navigator implements operator-- */
            HDINLINE void operator--()
            {
                Navigator::operator--;
            }

            /* jump and access in one call */
            template<typename Jump>
            HDINLINE type operator[](const Jump& jump)
            {
                return *((*this)(jump));
            }

            template<typename Jump>
            HDINLINE type operator[](const Jump& jump) const
            {
                return *((*this)(jump));
            }

            /* This is a dirty workaround to enable and disable safe-cursor checks.*/
            /** \todo: Can be substituted by ordinary functions instead of methods.*/
            HDINLINE void enableChecking()
            {
                this->marker.enableChecking();
            }
            HDINLINE void disableChecking()
            {
                this->marker.disableChecking();
            }

            /* getters */
            HDINLINE
            const _Accessor& getAccessor() const
            {
                return *this;
            }
            HDINLINE
            const _Navigator& getNavigator() const
            {
                return *this;
            }
            HDINLINE
            const Marker& getMarker() const
            {
                return this->marker;
            }
        };

        /* convenient function to construct a cursor by passing its constructor arguments */
        template<typename Accessor, typename Navigator, typename Marker>
        HDINLINE Cursor<Accessor, Navigator, Marker> make_Cursor(
            const Accessor& accessor,
            const Navigator& navigator,
            const Marker& marker)
        {
            return Cursor<Accessor, Navigator, Marker>(accessor, navigator, marker);
        }

        namespace traits
        {
            /* type trait to get the cursor's dimension if it has one */
            template<typename _Accessor, typename _Navigator, typename _Marker>
            struct dim<pmacc::cursor::Cursor<_Accessor, _Navigator, _Marker>>
            {
                static constexpr int value
                    = pmacc::cursor::traits::dim<typename Cursor<_Accessor, _Navigator, _Marker>::Navigator>::value;
            };

        } // namespace traits

    } // namespace cursor
} // namespace pmacc
