/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch
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

#include <iostream>
#include <pmacc/types.hpp>


namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            template<typename V, typename T>
            struct cuda_vec : public V
            {
                // constructor

                HDINLINE cuda_vec(T x, T y, T z)
                {
                    this->x() = x;
                    this->y() = y;
                    this->z() = z;
                }

                // default constructor

                HDINLINE cuda_vec()
                {
                }

                // constructor

                HDINLINE cuda_vec(const V& other)
                {
                    this->x() = other.x();
                    this->y() = other.y();
                    this->z() = other.z();
                }

                HDINLINE static cuda_vec<V, T> zero()
                {
                    return cuda_vec(0, 0, 0);
                }


                // conversion between two cuda vectors with different types

                template<typename O, typename Q>
                HDINLINE cuda_vec(const cuda_vec<O, Q>& other)
                {
                    this->x() = (T) other.x();
                    this->y() = (T) other.y();
                    this->z() = (T) other.z();
                }

                HDINLINE cuda_vec<V, T>& operator=(const cuda_vec<V, T>& other)
                {
                    this->x() = other.x();
                    this->y() = other.y();
                    this->z() = other.z();
                    return (*this);
                }

                HDINLINE T& operator[](uint32_t dim)
                {
                    return (&(this->x()))[dim];
                }

                HDINLINE const T& operator[](uint32_t dim) const
                {
                    return (&(this->x()))[dim];
                }


                // addition

                HDINLINE cuda_vec<V, T> operator+(const cuda_vec<V, T>& other) const
                {
                    return cuda_vec<V, T>(this->x() + other.x(), this->y() + other.y(), this->z() + other.z());
                }

                // difference

                HDINLINE cuda_vec<V, T> operator-(const cuda_vec<V, T>& other) const
                {
                    return cuda_vec<V, T>(this->x() - other.x(), this->y() - other.y(), this->z() - other.z());
                }

                // vector multiplication

                HDINLINE T operator*(const cuda_vec<V, T>& other) const
                {
                    return this->x() * other.x() + this->y() * other.y() + this->z() * other.z();
                }

                // scalar multiplication

                HDINLINE cuda_vec<V, T> operator*(const T scalar) const
                {
                    return cuda_vec(scalar * this->x(), scalar * this->y(), scalar * this->z());
                }

                // division (scalar)

                HDINLINE cuda_vec<V, T> operator/(const T scalar) const
                {
                    return cuda_vec(this->x() / scalar, this->y() / scalar, this->z() / scalar);
                }

                // cross product (vector)

                HDINLINE cuda_vec<V, T> operator%(const cuda_vec<V, T>& other) const
                {
                    return cuda_vec(
                        this->y() * other.z() - this->z() * other.y(),
                        this->z() * other.x() - this->x() * other.z(),
                        this->x() * other.y() - this->y() * other.x());
                }

                // magnitude of vector (length of vector)

                HDINLINE T magnitude(void) const
                {
                    return picongpu::math::sqrt(this->x() * this->x() + this->y() * this->y() + this->z() * this->z());
                }

                // unit vector in the direction of the vector

                HDINLINE cuda_vec<V, T> unit_vec(void) const
                {
                    return *this / magnitude();
                }

                // assign add

                HDINLINE void operator+=(const cuda_vec<V, T>& other)
                {
                    this->x() += other.x();
                    this->y() += other.y();
                    this->z() += other.z();
                }

                // assign multiply

                HDINLINE void operator*=(const T scalar)
                {
                    this->x() *= scalar;
                    this->y() *= scalar;
                    this->z() *= scalar;
                }
            };

        } // namespace radiation
    } // namespace plugins
} // namespace picongpu

// print

template<typename V, typename T>
HINLINE std::ostream& operator<<(std::ostream& os, const picongpu::plugins::radiation::cuda_vec<V, T>& v)
{
    os << " ( " << v.x() << " , " << v.y() << " , " << v.z() << " ) ";
    return os;
}
