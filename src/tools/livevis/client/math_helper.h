/**
 * Copyright 2013-2016 Benjamin Schneider
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

#ifndef INSITUVOLUME_MATHHELPER_HPP
#define INSITUVOLUME_MATHHELPER_HPP

#include <cmath>
#include "transferfunctions.h"

struct float3
{
    float3() : x(0.0f), y(0.0f), z(0.0f) { }
    float3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) { }
    float3(float4 f4) : x(f4.x), y(f4.y), z(f4.z) { }

    float x, y, z;
};

/**
 * A collection of math classes and functions needed for the Volume Renderer.
 */
class float4x4;

/// Negation
inline float3 operator-(const float3& a)
{
    return float3(-a.x, -a.y, -a.z);
}

/// Addition
inline float3 operator+(float3 a, float3 b)
{
    return float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline float4 operator+(float4 a, float4 b)
{
    return float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline float4 operator+(float4 a, float b)
{
    return float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

inline void operator+=(float4& a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline void operator+=(volatile float4& a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

/// Subtract
inline float3 operator-(float3 a, float b)
{
    return float3(a.x - b, a.y - b, a.z - b);
}

inline float3 operator-(float3 a, float3 b)
{
    return float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

/// Multiply
inline float3 operator*(float3 a, float b)
{
    return float3(a.x * b, a.y * b, a.z * b);
}

inline float3 operator*(float3 a, float3 b)
{
    return float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline float4 operator*(float4 a, float b)
{
    return float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

/// Divide
inline float3 operator/(float3 a, float3 b)
{
    return float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

/// Min and Max
inline float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline float3 fminf(float3 a, float3 b)
{
    return float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

inline float3 fmaxf(float3 a, float3 b)
{
    return float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

/// Dot Product
inline float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/// Cross Product
inline float3 cross(float3 a, float3 b)
{
    return float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

/// Normalization
inline float3 normalize(float3 v)
{
    float invLen = 1.0f / sqrt(dot(v, v));
    return v * invLen;
}

/// clamping functions
inline float clamp(float v, float min, float max)
{
    if (v <= min) return min;
    if (v >= max) return max;
    return v;
}

inline float saturate(float v)
{
    return clamp(v, 0.0f, 1.0f);
}

inline float4 saturate(float4 val)
{
    float4 out;

    out.x = clamp(val.x, 0.0f, 1.0f);
    out.y = clamp(val.y, 0.0f, 1.0f);
    out.z = clamp(val.z, 0.0f, 1.0f);
    out.w = clamp(val.w, 0.0f, 1.0f);

    return out;
}

/**
 * Helper class for matrices. Uses column-major memory layout (like OpenGL).
 */
class float4x4
{
private:

    float m[16];

public:

#ifndef __CUDACC__

    /// constructors
    float4x4() { }

    float4x4(float4x4 const& rhs)
    {
        for (int i = 0; i < 16; i++) m[i] = rhs[i];
    }

    float4x4(const float* rhs)
    {
        for (int i = 0; i < 16; i++) m[i] = rhs[i];
    }

    explicit float4x4(const float v)
    {
        for (int i = 0; i < 16; i++) m[i] = v;
    }

#endif

    float& operator[](int i)
    {
        return m[i];
    }

    const float& operator[](int i) const
    {
        return m[i];
    }

    float& get(int row, int col)
    {
        return m[row + (col * 4)];
    }

    const float& get(int row, int col) const
    {
        return m[row + (col * 4)];
    }

    float4 getRow(int row) const
    {
        float4 v;
        v.x = get(row, 0);
        v.y = get(row, 1);
        v.z = get(row, 2);
        v.w = get(row, 3);
        return v;
    }

    float4 getCol(int col) const
    {
        float4 v;
        v.x = get(0, col);
        v.y = get(1, col);
        v.z = get(2, col);
        v.w = get(3, col);
        return v;
    }

    operator float*()
    {
        return m;
    }

    operator const float*() const
    {
        return m;
    }

    float4x4& operator=(float4x4 const& rhs)
    {
        for (int i = 0; i < 16; i++) m[i] = rhs[i];
            return (*this);
    }

    float4x4& operator+=(float4x4 const& rhs)
    {
        for(int i = 0; i < 16; i++) m[i] += rhs[i];
        return (*this);
    }

    float4x4& operator-=(float4x4 const& rhs)
    {
        for(int i = 0; i < 16; i++) m[i] -= rhs[i];
        return (*this);
    }

    float4x4 operator*(float4x4 const & rhs)
    {
        float4x4 v;

        v[0] = m[0] * rhs[0] + m[4] * rhs[1] + m[8] * rhs[2] + m[12] * rhs[3];
        v[1] = m[1] * rhs[0] + m[5] * rhs[1] + m[9] * rhs[2] + m[13] * rhs[3];
        v[2] = m[2] * rhs[0] + m[6] * rhs[1] + m[10] * rhs[2] + m[14] * rhs[3];
        v[3] = m[3] * rhs[0] + m[7] * rhs[1] + m[11] * rhs[2] + m[15] * rhs[3];

        v[4] = m[0] * rhs[4] + m[4] * rhs[5] + m[8] * rhs[6] + m[12] * rhs[7];
        v[5] = m[1] * rhs[4] + m[5] * rhs[5] + m[9] * rhs[6] + m[13] * rhs[7];
        v[6] = m[2] * rhs[4] + m[6] * rhs[5] + m[10] * rhs[6] + m[14] * rhs[7];
        v[7] = m[3] * rhs[4] + m[7] * rhs[5] + m[11] * rhs[6] + m[15] * rhs[7];

        v[8] = m[0] * rhs[8] + m[4] * rhs[9] + m[8] * rhs[10] + m[12] * rhs[11];
        v[9] = m[1] * rhs[8] + m[5] * rhs[9] + m[9] * rhs[10] + m[13] * rhs[11];
        v[10] = m[2] * rhs[8] + m[6] * rhs[9] + m[10] * rhs[10] + m[14] * rhs[11];
        v[11] = m[3] * rhs[8] + m[7] * rhs[9] + m[11] * rhs[10] + m[15] * rhs[11];

        v[12] = m[0] * rhs[12] + m[4] * rhs[13] + m[8] * rhs[14] + m[12] * rhs[15];
        v[13] = m[1] * rhs[12] + m[5] * rhs[13] + m[9] * rhs[14] + m[13] * rhs[15];
        v[14] = m[2] * rhs[12] + m[6] * rhs[13] + m[10] * rhs[14] + m[14] * rhs[15];
        v[15] = m[3] * rhs[12] + m[7] * rhs[13] + m[11] * rhs[14] + m[15] * rhs[15];

        return v;
    }

    const float4 operator*(float4 const & rhs) const
    {
        return float4(
            m[0] * rhs.x + m[4] * rhs.y + m[8] * rhs.z + m[12] * rhs.w,
            m[1] * rhs.x + m[5] * rhs.y + m[9] * rhs.z + m[13] * rhs.w,
            m[2] * rhs.x + m[6] * rhs.y + m[10] * rhs.z + m[14] * rhs.w,
            m[3] * rhs.x + m[7] * rhs.y + m[11] * rhs.z + m[15] * rhs.w);
    }

    /// helper methods
    static void make_identity(float4x4& v)
    {
        v[0] = v[5] = v[10] = v[15] = 1.0f;
        v[1] = v[2] = v[3] = v[4] = v[6] = v[7] = v[8] = v[9] = v[11] = v[12] = v[13] = v[14] = 0.0f;
    }

    static float4x4 make_view(float3 eye, float3 target, float3 up)
    {
        float3 zaxis = normalize(eye - target);

        /// if up and zaxis are nearly parallel choose an alternative up axis
        if (fabs(dot(up, zaxis)) > 0.9999f)
        {
            /// turn up by 90 degree
            up = normalize(float3(-up.y, up.x, up.z));
        }

        float3 xaxis = normalize(cross(up, zaxis));
        float3 yaxis = normalize(cross(zaxis, xaxis));

        float4x4 rotation;

        rotation[0] = xaxis.x;
        rotation[1] = yaxis.x;
        rotation[2] = zaxis.x;
        rotation[3] = 0.0f;

        rotation[4] = xaxis.y;
        rotation[5] = yaxis.y;
        rotation[6] = zaxis.y;
        rotation[7] = 0.0f;

        rotation[8] = xaxis.z;
        rotation[9] = yaxis.z;
        rotation[10] = zaxis.z;
        rotation[11] = 0.0f;

        rotation[12] = 0.0f;
        rotation[13] = 0.0f;
        rotation[14] = 0.0f;
        rotation[15] = 1.0f;

        float4x4 translation;

        translation[0] = 1.0f;
        translation[1] = 0.0f;
        translation[2] = 0.0f;
        translation[3] = 0.0f;

        translation[4] = 0.0f;
        translation[5] = 1.0f;
        translation[6] = 0.0f;
        translation[7] = 0.0f;

        translation[8] = 0.0f;
        translation[9] = 0.0f;
        translation[10] = 1.0f;
        translation[11] = 0.0f;

        translation[12] = -eye.x;
        translation[13] = -eye.y;
        translation[14] = -eye.z;
        translation[15] = 1.0f;

        return (rotation * translation);
    }

    static float4x4 make_inv_view(float3 eye, float3 target, float3 up)
    {
        float3 zaxis = normalize(eye - target);

        /// if up and zaxis are nearly parallel choose an alternative up axis
        if (fabs(dot(up, zaxis)) > 0.9999f)
        {
            /// turn up by 90 degree
            up = normalize(float3(-up.y, up.x, up.z));
        }

        float3 xaxis = normalize(cross(up, zaxis));
        float3 yaxis = normalize(cross(zaxis, xaxis));

        float4x4 rotation;

        rotation[0] = xaxis.x;
        rotation[1] = xaxis.y;
        rotation[2] = xaxis.z;
        rotation[3] = 0.0f;

        rotation[4] = yaxis.x;
        rotation[5] = yaxis.y;
        rotation[6] = yaxis.z;
        rotation[7] = 0.0f;

        rotation[8] = zaxis.x;
        rotation[9] = zaxis.y;
        rotation[10] = zaxis.z;
        rotation[11] = 0.0f;

        rotation[12] = 0.0f;
        rotation[13] = 0.0f;
        rotation[14] = 0.0f;
        rotation[15] = 1.0f;

        float4x4 translation;

        translation[0] = 1.0f;
        translation[1] = 0.0f;
        translation[2] = 0.0f;
        translation[3] = 0.0f;

        translation[4] = 0.0f;
        translation[5] = 1.0f;
        translation[6] = 0.0f;
        translation[7] = 0.0f;

        translation[8] = 0.0f;
        translation[9] = 0.0f;
        translation[10] = 1.0f;
        translation[11] = 0.0f;

        translation[12] = eye.x;
        translation[13] = eye.y;
        translation[14] = eye.z;
        translation[15] = 1.0f;

        return (translation * rotation);
    }

    static float4x4 make_rotation_x(float degree)
    {
        static const float PI = 3.14159265f;

        float4x4 rotation;

        float s = sin(PI * degree / 180.0f);
        float c = cos(PI * degree / 180.0f);

        rotation[0] = 1.0f;
        rotation[1] = 0.0f;
        rotation[2] = 0.0f;
        rotation[3] = 0.0f;

        rotation[4] = 0.0f;
        rotation[5] = c;
        rotation[6] = s;
        rotation[7] = 0.0f;

        rotation[8] = 0.0f;
        rotation[9] = -s;
        rotation[10] = c;
        rotation[11] = 0.0f;

        rotation[0] = 0.0f;
        rotation[0] = 0.0f;
        rotation[0] = 0.0f;
        rotation[0] = 1.0f;

        return rotation;
    }

    static float4x4 make_rotation_around_axis(float3 axis, float degree)
    {
        static const float PI = 3.14159265f;

        float4x4 rotation;

        float cos = cosf(PI * degree / 180.0f);
        float sin = sinf(PI * degree / 180.0f);

        rotation[0] = cos + axis.x * axis.x * (1.0f - cos);
        rotation[1] = axis.y * axis.x * (1.0f - cos) + axis.z * sin;
        rotation[2] = axis.z * axis.x * (1.0f - cos) - axis.y * sin;
        rotation[3] = 0.0f;

        rotation[4] = axis.x * axis.y * (1.0f - cos) - axis.z * sin;
        rotation[5] = cos + axis.y * axis.y * (1.0f - cos);
        rotation[6] = axis.z * axis.y * (1.0f - cos) + axis.x * sin;
        rotation[7] = 0.0f;

        rotation[8] = axis.x * axis.z * (1.0f - cos) + axis.y * sin;
        rotation[9] = axis.y * axis.z * (1.0f - cos) - axis.x * sin;
        rotation[10] = cos + axis.z * axis.z * (1.0f - cos);
        rotation[11] = 0.0f;

        rotation[12] = 0.0f;
        rotation[13] = 0.0f;
        rotation[14] = 0.0f;
        rotation[15] = 1.0f;

        return rotation;
    }

    static float4x4 make_translation(float3 t)
    {
        float4x4 translation;

        translation[0] = 0.0f;
        translation[1] = 0.0f;
        translation[2] = 0.0f;
        translation[3] = 0.0f;

        translation[4] = 0.0f;
        translation[5] = 0.0f;
        translation[6] = 0.0f;
        translation[7] = 0.0f;

        translation[8] = 0.0f;
        translation[9] = 0.0f;
        translation[10] = 0.0f;
        translation[11] = 0.0f;

        translation[12] = t.x;
        translation[13] = t.y;
        translation[14] = t.z;
        translation[15] = 1.0f;

        return translation;
    }
};

#endif /* INSITUVOLUME_MATHHELPER_HPP */
