#include "transferfunctions.h"
#include <cmath>

float4 RedGreenTransferFunction::sample(float samplingPoint)
{
    if (samplingPoint < 0.0f) samplingPoint = 0.0f;
    if (samplingPoint > 1.0f) samplingPoint = 1.0f;

    float4 result;

    result.r = 1.0f - samplingPoint;
    result.g = samplingPoint;
    result.b = 0.0f;

    result.a = m_offsetY * std::erff( m_slope * (samplingPoint - m_offsetX) ) + m_offsetY;

    return result;
}

float4 TemperatureTransferFunction::sample(float samplingPoint)
{
    if (samplingPoint < 0.0f) samplingPoint = 0.0f;
    if (samplingPoint > 1.0f) samplingPoint = 1.0f;

    float4 result;

    if (samplingPoint < 0.4f) result.r = 2.5f * samplingPoint;
    else result.r = 1.0f;

    if (samplingPoint < 0.4f) result.g = -1.95f * samplingPoint + 0.78f;
    else if (samplingPoint >= 0.4f && samplingPoint < 0.75f) result.g = 2.857f * samplingPoint - 1.143f;
    else result.g = 1.0f;

    if (samplingPoint < 0.4f) result.b = -2.35f * samplingPoint + 0.94f;
    else if (samplingPoint >= 0.4f && samplingPoint < 0.75f) result.b = 0.0f;
    else result.b = 4.0f * samplingPoint - 3.0f;

    result.a = m_offsetY * std::erff( m_slope * (samplingPoint - m_offsetX) ) + m_offsetY;

    return result;
}

float4 TwoHueTransferFunction::sample(float samplingPoint)
{
    if (samplingPoint < 0.0f) samplingPoint = 0.0f;
    if (samplingPoint > 1.0f) samplingPoint = 1.0f;

    float4 result;

    if (samplingPoint < 0.5f) result.r = 2.0f * samplingPoint;
    else result.r = -0.76f * samplingPoint + 1.38f;

    if (samplingPoint < 0.5f) result.g = 2.0f * samplingPoint;
    else result.g = -2.0f * samplingPoint + 2.0f;

    result.b = 1.0f - samplingPoint;

    result.a = m_offsetY * std::erff( m_slope * (samplingPoint - m_offsetX) ) + m_offsetY;

    return result;
}
