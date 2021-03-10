template<typename T_Function>
static T_Value getDerivative(T_Function const& function, T_Argument argument, uint8_t orderDerivative, )
{
    /**returns derivative of a callable function at the argument
     *
     * @tparam T_Function ... type of function
     *
     * @param function ... callable function,
     *      must define: T_Value operator()( T_Argument x )
     *      returning the value of function for x, must at least cover all
     *      sample points.
     * @param argument ... point where to calculate the derivative
     * @param sampleIntervalWidth ... width of the intervall centered
     * @param orderDerivative ... order of derivative to be calculated
     */

    T_Value result = static_cast<T_Value>(0);

    for(uint8_t samplePoint = 0; samplePoint <= T_numSamples; samplePoint++)
    {
        result += this->coefficients[samplePoint] * function(samplePoints[i]);
    }

    return result;
}

// relative error function used
DINLINE static float_X relativeErrorFunction(float_X binWidth, float_X centralValue){T_RelativeErrorFunction()}


float_X samplePoints[2]
    = {0, 1};
// weight used in numerical Differentiation
using T_WeightsDifferentiation = FornbergNumericalDifferentation<2u, 1u, float_X, float_X>;
T_WeightsDifferentiation weights = T_WeightsDifferentiation(samplePoints);