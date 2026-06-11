/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/math/math.hpp>
#include <pmacc/types.hpp>

#include <iostream>

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

                HDINLINE cuda_vec() = default;

                // constructor

                HDINLINE cuda_vec(V const& other)
                {
                    this->x() = other.x();
                    this->y() = other.y();
                    this->z() = other.z();
                }

                HDINLINE cuda_vec(cuda_vec const& other) = default;

                HDINLINE static cuda_vec<V, T> zero()
                {
                    return cuda_vec(0, 0, 0);
                }

                // conversion between two cuda vectors with different types

                template<typename O, typename Q>
                HDINLINE cuda_vec(cuda_vec<O, Q> const& other)
                {
                    this->x() = (T) other.x();
                    this->y() = (T) other.y();
                    this->z() = (T) other.z();
                }

                HDINLINE cuda_vec& operator=(cuda_vec const& other) = default;

                HDINLINE T& operator[](uint32_t dim)
                {
                    return (&(this->x()))[dim];
                }

                HDINLINE const T& operator[](uint32_t dim) const
                {
                    return (&(this->x()))[dim];
                }

                // addition

                HDINLINE cuda_vec<V, T> operator+(cuda_vec<V, T> const& other) const
                {
                    return cuda_vec<V, T>(this->x() + other.x(), this->y() + other.y(), this->z() + other.z());
                }

                // difference

                HDINLINE cuda_vec<V, T> operator-(cuda_vec<V, T> const& other) const
                {
                    return cuda_vec<V, T>(this->x() - other.x(), this->y() - other.y(), this->z() - other.z());
                }

                // vector multiplication

                HDINLINE T operator*(cuda_vec<V, T> const& other) const
                {
                    return this->x() * other.x() + this->y() * other.y() + this->z() * other.z();
                }

                // scalar multiplication

                HDINLINE cuda_vec<V, T> operator*(T const scalar) const
                {
                    return cuda_vec(scalar * this->x(), scalar * this->y(), scalar * this->z());
                }

                // division (scalar)

                HDINLINE cuda_vec<V, T> operator/(T const scalar) const
                {
                    return cuda_vec(this->x() / scalar, this->y() / scalar, this->z() / scalar);
                }

                // cross product (vector)

                HDINLINE cuda_vec<V, T> operator%(cuda_vec<V, T> const& other) const
                {
                    return cuda_vec(
                        this->y() * other.z() - this->z() * other.y(),
                        this->z() * other.x() - this->x() * other.z(),
                        this->x() * other.y() - this->y() * other.x());
                }

                // magnitude of vector (length of vector)

                HDINLINE T magnitude(void) const
                {
                    return pmacc::math::sqrt(this->x() * this->x() + this->y() * this->y() + this->z() * this->z());
                }

                // unit vector in the direction of the vector

                HDINLINE cuda_vec<V, T> unitVec(void) const
                {
                    return *this / magnitude();
                }

                // assign add

                HDINLINE void operator+=(cuda_vec<V, T> const& other)
                {
                    this->x() += other.x();
                    this->y() += other.y();
                    this->z() += other.z();
                }

                // assign multiply

                HDINLINE void operator*=(T const scalar)
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
HINLINE std::ostream& operator<<(std::ostream& os, picongpu::plugins::radiation::cuda_vec<V, T> const& v)
{
    os << " ( " << v.x() << " , " << v.y() << " , " << v.z() << " ) ";
    return os;
}
