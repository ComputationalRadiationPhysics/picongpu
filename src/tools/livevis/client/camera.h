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

#ifndef CAMERA_H
#define CAMERA_H

#include "math_helper.h"

/**
 * Simple 3D Camera class. Stores the camera position and look at point
 * as well as the field of view and aspect ratio.
 */
class Camera
{
public:

    Camera(float3 pos, float3 target)
     : m_position(pos),
       m_focalPoint(target),
       m_up(float3(0,1,0))
    { }

    float3 getPosition() { return m_position; }
    float3 getFocalPoint() { return m_focalPoint; }
    float3 getUp() { return m_up; }

    void setPosition(float3 pos) { m_position = pos; }
    void setFocalPoint(float3 focus) { m_focalPoint = focus; }
    void setUp(float3 up) { m_up = up; }

    /**
     * Moves the camera and the focal point by the same vector along the local camera axes.
     * Useful for following the sliding simulation window.
     */
    void slide(float3 offset)
    {
        float4x4 invView = this->getInvViewMatrix();

        float3 move = float3(invView * float4(offset.x, offset.y, offset.z, 0.0f));

        m_position += move;
        m_focalPoint += move;
    }

    /**
     * Moves the camera and the focal point by the same vector along the world axes.
     * Useful for following the sliding simulation window.
     */
    void follow(float3 offset)
    {
        m_position += offset;
        m_focalPoint += offset;
    }

    /**
     * Moves the camera towards or away from the focal point.
     *
     * @param amount A value in the range [0;1] will move the camera towards the focal point.
     * A value greater one will move the camera away from the focal point.
     */
    void dolly(float amount)
    {
        if (amount < 0.000001f) return;

        float3 dir = m_focalPoint - m_position;
        float dolly = sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z) / amount;

        m_position = m_focalPoint - (normalize(dir) * dolly);
    }

    /**
     * Rotates the camera around its focal point on all three axes.
     *
     * @param yaw The rotation around the Y axis in degree.
     * @param pitch The rotation around the X axis in degree.
     * @param roll The rotation around the Z axis in degree.
     */
    void orbitYawPitchRoll(float yaw, float pitch, float roll)
    {
        float3 fp_to_cam = m_position - m_focalPoint;
        float3 dir = normalize(fp_to_cam);
        float3 right = normalize(cross(m_up, dir));
        float3 up = normalize(cross(dir, right));

        float4x4 rot_yaw = float4x4::make_rotation_around_axis(up, yaw);
        float4x4 rot_pitch = float4x4::make_rotation_around_axis(right, pitch);
        float4x4 rot_roll = float4x4::make_rotation_around_axis(dir, roll);

        float3 newpos = float3( (rot_yaw * rot_pitch * rot_roll) * float4(fp_to_cam.x, fp_to_cam.y, fp_to_cam.z, 1.0f) );

        m_position = m_focalPoint + newpos;
    }

    /**
     * Orbit the focal point around the camera position to pan the camera.
     *
     * @param yaw The rotation around the Y axis.
     * @param pitch The rotation around the X axis.
     */
    void panYawPitch(float yaw, float pitch)
    {
        float3 zaxis = normalize(m_position - m_focalPoint);
        float3 xaxis = normalize(cross(m_up, zaxis));
        float3 yaxis = normalize(cross(zaxis, xaxis));

        float4x4 rot_yaw = float4x4::make_rotation_around_axis(yaxis, yaw);
        float4x4 rot_pitch = float4x4::make_rotation_around_axis(xaxis, pitch);

        m_focalPoint = float3( (rot_yaw * rot_pitch) * float4(m_focalPoint.x, m_focalPoint.y, m_focalPoint.z, 1.0f) );
    }

    /**
     * Returns the inverse view matrix. This is useful to construct rays.
     */
    float4x4 getInvViewMatrix()
    {
        return float4x4::make_inv_view(m_position, m_focalPoint, m_up);
    }

private:

    /// TODO: add aspect ratio and field of view to construct rays correctly for non square images.
    float3 m_position;
    float3 m_focalPoint;
    float3 m_up;
};

#endif // CAMERA_H
