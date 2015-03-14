#ifndef TRANSFERFUNCTIONS_H
#define TRANSFERFUNCTIONS_H

const int TF_RESOLUTION = 64; // number of samples taken from the gradients (widgets) used to adjust the TF

struct float4
{
    union
    {
        struct { float r, g, b, a; };
        struct { float x, y, z, w; };
        float c[4];
    };

    float4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) { }
    float4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) { }
};

/**
 * @brief The ITransferFunction class
 *
 * Interface for Transferfunction classes.
 */
class ITransferFunction
{
public:

    ITransferFunction()
        : m_offsetX(0.20f),
          m_offsetY(0.005f),
          m_slope(9.5f)
    { }

    virtual ~ITransferFunction()
    { }

    /**
     * @brief setOffsetX Offsets the opacity curve along the X-Axis, e.g. f(x) -> f(x - offsetX)
     * @param offsetX
     */
    void setOffsetX(float offsetX) { m_offsetX = offsetX; }

    /**
     * @brief setOffsetY Offsets and scales the opacity curve along the Y-Axis, e.g. f(x) -> offsetY * f(x) + offsetY
     * @param offsetY
     */
    void setOffsetY(float offsetY) { m_offsetY = offsetY; }

    /**
     * @brief setSlope Determines the slope of the opacity curve, e.g. f(x) -> f(slope * x)
     * @param slope
     */
    void setSlope(float slope) { m_slope = slope; }

    float getOffsetX() { return m_offsetX; }
    float getOffsetY() { return m_offsetY; }
    float getSlope() { return m_slope; }

    /**
     * @brief sample Get the RGBA value of the Transferfunction at a given sampling point in range (0; 1).
     * @param samplingPoint A sampling point between 0 and 1.
     * @return An RGBA tuple.
     */
    virtual float4 sample(float samplingPoint) = 0;

protected:

    float m_offsetX;
    float m_offsetY;
    float m_slope;
};

/**
 * @brief The RedGreenTransferFunction class
 *
 * A simple color scale ranging from red for low values to green for high values.
 */
class RedGreenTransferFunction : public ITransferFunction
{
public:

    RedGreenTransferFunction()
        : ITransferFunction()
    { }

    virtual float4 sample(float samplingPoint);
};

/**
 * @brief The TemperatureTransferFunction class
 *
 *  A temperature color scale ranging from blue (cold) over red and yellow to white (darn hot!).
 */
class TemperatureTransferFunction : public ITransferFunction
{
public:

    TemperatureTransferFunction()
        : ITransferFunction()
    { }

    virtual float4 sample(float samplingPoint);
};

/**
 * @brief The TwoHueTransferFunction class
 *
 *  A color scale ranging from dark blue to dark red.
 */
class TwoHueTransferFunction : public ITransferFunction
{
public:

    TwoHueTransferFunction()
        : ITransferFunction()
    { }

    virtual float4 sample(float samplingPoint);
};

#endif // TRANSFERFUNCTIONS_H
