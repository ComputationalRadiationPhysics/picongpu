/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#include <cuda_to_cupla.hpp>
#include <stdio.h>
#if !(BOOST_LANG_CUDA || BOOST_LANG_HIP)
struct float2{
    float x;
    float y;
    float2(float x, float y) : y(y), x(x) { }
};
float2 make_float2(float x, float y){
    return float2(x,y);
}
#endif

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
template<typename T_Acc>
ALPAKA_FN_ACC
float cndGPU(T_Acc const & acc, float d)
{
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
    K = __fdividef(1.0f, (1.0f + 0.2316419f * cupla::abs(d)));
    float cnd = RSQRT2PI * cupla::exp(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
template<typename T_Acc>
ALPAKA_FN_ACC void BlackScholesBodyGPU(
    T_Acc const & acc,
    float &CallResult,
    float &PutResult,
    float S, //Stock price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V  //Volatility rate
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;
    sqrtT = cupla::sqrt(T); /// __fdividef(1.0F, rsqrtf(T));
    d1 = __fdividef(cupla::log(S / X) + (R + 0.5f * V * V) * T, V * sqrtT);

    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(acc, d1);
    CNDD2 = cndGPU(acc, d2);

    //Calculate Call and Put simultaneously
    expRT = cupla::exp(- R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
//__launch_bounds__(128)
struct BlackScholesGPU {
    template< typename T_Acc>
    ALPAKA_FN_HOST_ACC
    void operator()(
            T_Acc const & acc,
            float2 *__restrict d_CallResult,
            float2 *__restrict d_PutResult,
            float2 *__restrict d_StockPrice,
            float2 *__restrict d_OptionStrike,
            float2 *__restrict d_OptionYears,
            float Riskfree,
            float Volatility,
            int optN
    ) const
    {
        ////Thread index
        //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
        ////Total number of threads in execution grid
        //const int THREAD_N = blockDim.x * gridDim.x;

        const int opt_begin = blockDim.x * blockIdx.x * elemDim.x + threadIdx.x * elemDim.x;

        // Calculating 2 options per thread to increase ILP (instruction level parallelism)
        if (opt_begin < (optN / 2)) {
            const int opt_end = (opt_begin + elemDim.x < optN / 2) ? opt_begin + elemDim.x : optN / 2;
            for (int opt = opt_begin; opt < opt_end; opt++) {
                float callResult1, callResult2;
                float putResult1, putResult2;
                BlackScholesBodyGPU(
                        acc,
                        callResult1,
                        putResult1,
                        d_StockPrice[opt].x,
                        d_OptionStrike[opt].x,
                        d_OptionYears[opt].x,
                        Riskfree,
                        Volatility
                );
                BlackScholesBodyGPU(
                        acc,
                        callResult2,
                        putResult2,
                        d_StockPrice[opt].y,
                        d_OptionStrike[opt].y,
                        d_OptionYears[opt].y,
                        Riskfree,
                        Volatility
                );
                d_CallResult[opt] = make_float2(callResult1, callResult2);
                d_PutResult[opt] = make_float2(putResult1, putResult2);
            }
        }
    }
};
