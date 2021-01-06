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

namespace pmacc
{
    namespace cursor
    {
        namespace CT
        {
            /** Compile-time version of cursor::SafeCursor where LowerExtent and UpperExtent are
             * compile-time vectors.
             */
            template<typename Cursor, typename LowerExtent, typename UpperExtent>
            class SafeCursor : public Cursor
            {
            private:
                typedef SafeCursor<Cursor, LowerExtent, UpperExtent> This;
                static constexpr int dim = pmacc::cursor::traits::dim<Cursor>::value;
                math::Int<dim> offset;

            public:
                HDINLINE SafeCursor(const Cursor& cursor) : Cursor(cursor), offset(math::Int<dim>(0))
                {
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
                HDINLINE This operator()(const Jump& jump) const
                {
                    This result(Cursor::operator()(jump));
                    result.offset = this->offset + jump;
                    return result;
                }

                HDINLINE
                This operator()(int x) const
                {
                    return (*this)(math::Int<1>(x));
                }

                HDINLINE
                This operator()(int x, int y) const
                {
                    return (*this)(math::Int<2>(x, y));
                }

                HDINLINE
                This operator()(int x, int y, int z) const
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
#pragma unroll
                    for(int i = 0; i < dim; i++)
                    {
                        if(this->offset[i] < LowerExtent().toRT()[i] || this->offset[i] > UpperExtent().toRT()[i])
                            printf(
                                "error[cursor]: index %d out of range: %d is not within [%d, %d]\n",
                                i,
                                this->offset[i],
                                LowerExtent().toRT()[i],
                                UpperExtent().toRT()[i]);
                    }
                }
            };

            template<typename Cursor, typename LowerExtent, typename UpperExtent>
            HDINLINE SafeCursor<Cursor, LowerExtent, UpperExtent> make_SafeCursor(
                const Cursor& cursor,
                LowerExtent,
                UpperExtent)
            {
                return SafeCursor<Cursor, LowerExtent, UpperExtent>(cursor);
            }

        } // namespace CT
    } // namespace cursor
} // namespace pmacc
