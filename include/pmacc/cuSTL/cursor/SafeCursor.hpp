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

#include "pmacc/cuSTL/cursor/traits.hpp"
#include "pmacc/math/vector/Int.hpp"
#include "pmacc/cuSTL/cursor/Cursor.hpp"

namespace pmacc
{
    namespace cursor
    {
        /** A SafeCursor is like a cursor, except that it checks its validity before each access.
         */
        template<typename Cursor>
        class SafeCursor : public Cursor
        {
        public:
            static constexpr int dim = pmacc::cursor::traits::dim<Cursor>::value;

        private:
            /* \todo: Use a zone instead of lowerExtent and UpperExtent */
            const math::Int<dim> lowerExtent;
            const math::Int<dim> upperExtent;
            math::Int<dim> offset;
            bool enabled;

        public:
            /**
             * \param cursor Base cursor
             * \param lowerExtent Top left corner of valid range, inside the range.
             * \param upperExtent Bottom right corner of valid range, inside the range.
             */
            HDINLINE SafeCursor(
                const Cursor& cursor,
                const math::Int<dim>& lowerExtent,
                const math::Int<dim>& upperExtent)
                : Cursor(cursor)
                , lowerExtent(lowerExtent)
                , upperExtent(upperExtent)
                , offset(math::Int<dim>(0))
                , enabled(true)
            {
            }

            HDINLINE void enableChecking()
            {
                this->enabled = true;
            }
            HDINLINE void disableChecking()
            {
                this->enabled = false;
            }

            HDINLINE
            typename Cursor::type operator*()
            {
                checkValidity();
                return Cursor::operator*();
            }

            HDINLINE
            typename boost::add_const<typename Cursor::type>::type operator*() const
            {
                checkValidity();
                return Cursor::operator*();
            }

            template<typename Jump>
            HDINLINE SafeCursor<Cursor> operator()(const Jump& jump) const
            {
                SafeCursor<Cursor> result(Cursor::operator()(jump), this->lowerExtent, this->upperExtent);
                result.offset = this->offset + jump;
                result.enabled = this->enabled;
                return result;
            }

            HDINLINE
            SafeCursor<Cursor> operator()(int x) const
            {
                return (*this)(math::Int<1>(x));
            }

            HDINLINE
            SafeCursor<Cursor> operator()(int x, int y) const
            {
                return (*this)(math::Int<2>(x, y));
            }

            HDINLINE
            SafeCursor<Cursor> operator()(int x, int y, int z) const
            {
                return (*this)(math::Int<3>(x, y, z));
            }

            HDINLINE void operator++()
            {
                this->jump[0]++;
                Cursor::operator++;
            }
            HDINLINE void operator--()
            {
                this->jump[0]--;
                Cursor::operator--;
            }

            template<typename Jump>
            HDINLINE typename Cursor::type operator[](const Jump& jump)
            {
                return *((*this)(jump));
            }

            template<typename Jump>
            HDINLINE typename Cursor::type operator[](const Jump& jump) const
            {
                return *((*this)(jump));
            }

        private:
            HDINLINE void checkValidity() const
            {
                if(!this->enabled)
                    return;
#pragma unroll
                for(int i = 0; i < dim; i++)
                {
                    if(this->offset[i] < this->lowerExtent[i] || this->offset[i] > this->upperExtent[i])
                        printf(
                            "error[cursor]: index %d out of range: %d is not within [%d, %d]\n",
                            i,
                            this->offset[i],
                            this->lowerExtent[i],
                            this->upperExtent[i]);
                }
            }
        };

        namespace traits
        {
            /* type trait to get the safe-cursor's dimension if it has one */
            template<typename Cursor>
            struct dim<SafeCursor<Cursor>>
            {
                static constexpr int value = SafeCursor<Cursor>::dim;
            };

        } // namespace traits

        /* convenient function to construct a safe-cursor by passing its constructor arguments */
        template<typename Cursor>
        HDINLINE SafeCursor<Cursor> make_SafeCursor(
            const Cursor& cursor,
            const math::Int<traits::dim<SafeCursor<Cursor>>::value>& lowerExtent,
            const math::Int<traits::dim<SafeCursor<Cursor>>::value>& upperExtent)
        {
            return SafeCursor<Cursor>(cursor, lowerExtent, upperExtent);
        }

    } // namespace cursor
} // namespace pmacc
