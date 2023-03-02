#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/simulation/control/Window.hpp"

#include <pmacc/assert.hpp>
#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/cuSTL/container/HostBuffer.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/math/vector/Float.hpp>
#include <pmacc/math/vector/Int.hpp>
#include <pmacc/math/vector/Size_t.hpp>

#include <cmath> // what

#include <fftw3.h>
#include <stdio.h>

namespace picongpu
{
    namespace plugins
    {
        namespace shadowgraphy
        {
            using namespace std::complex_literals;

            class Helper
            {
            private:
                using complex_64 = alpaka::Complex<float_64>;

                typedef std::vector<std::vector<std::vector<complex_64>>> vec3c;
                typedef std::vector<std::vector<complex_64>> vec2c;
                typedef std::vector<complex_64> vec1c;
                typedef std::vector<std::vector<std::vector<float_64>>> vec3r;
                typedef std::vector<std::vector<float_64>> vec2r;
                typedef std::vector<float_64> vec1r;

                // Arrays to store Ex, Ey, Bx and Bz per time step temporarily
                vec2r tmpEx, tmpEy;
                vec2r tmpBx, tmpBy;

                // Arrays for FFTW
                fftw_complex* fftwInF; // @TODO: Can this be real? Issue is forward / backward FFT
                fftw_complex* fftwOutF;
                fftw_complex* fftwInB;
                fftw_complex* fftwOutB;

                fftw_plan planForward;
                fftw_plan planBackward;

                // Arrays for DFT sum
                vec3c ExOmega;
                vec3c EyOmega;
                vec3c BxOmega;
                vec3c ByOmega;

                // Arrays for propagated fields
                vec3c ExOmegaPropagated;
                vec3c EyOmegaPropagated;
                vec3c BxOmegaPropagated;
                vec3c ByOmegaPropagated;

                vec2r shadowgram;

                // Size of arrays
                int pluginNumX, pluginNumY;
                int omegaMinIndex, omegaMaxIndex, numOmegas;

                int yTotalMinIndex, yTotalMaxIndex;
                int cellsPerGpu;
                bool isSlidingWindowActive;

                // Variables for omega calculations
                float dt;
                int pluginNumT;
                int duration;

                float propagationDistance;

                bool fourierOutputEnabled;
                bool intermediateOutputEnabled;

                std::shared_ptr<pmacc::container::HostBuffer<float_64, DIM2>> retBuffer;

                

            public:
                /** Constructor of shadowgraphy helper class
                 * To be called at the first timestep when the shadowgraphy time integration starts
                 *
                 * @param currentStep current simulation timestep
                 * @param slicePoint value between 0.0 and 1.0 where the fields are extracted
                 * @param focusPos focus position of lens system, e.g. the propagation distance of the vacuum
                 * propagator
                 * @param duration duration of time extraction in simulation time steps
                 */
                Helper(
                    int currentStep,
                    float slicePoint,
                    float focusPos,
                    int duration,
                    bool fourierOutputEnabled,
                    bool intermediateOutputEnabled)
                    : duration(duration)
                    , fourierOutputEnabled(fourierOutputEnabled)
                    , intermediateOutputEnabled(intermediateOutputEnabled)
                {
                    dt = params::tRes * SI::DELTA_T_SI;
                    pluginNumT = duration / params::tRes;

                    omegaMinIndex = getOmegaMinIndex();
                    omegaMaxIndex = getOmegaMaxIndex();
                    numOmegas = getNumOmegas();


                    propagationDistance = focusPos;

                    const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();

                    pmacc::math::Size_t<simDim> globalGridSize = subGrid.getGlobalDomain().size;

                    pmacc::GridController<simDim>& con = pmacc::Environment<simDim>::get().GridController();
                    int const nGpus = con.getGpuNodes()[1];
                    cellsPerGpu = int(globalGridSize[1] / nGpus);

                    pluginNumX = globalGridSize[0] / params::xRes - 2;

                    int const startSlideCount = MovingWindow::getInstance().getSlideCounter(currentStep);

                    // These are currently not allowed to change during plugin run!
                    isSlidingWindowActive = MovingWindow::getInstance().isSlidingWindowActive(currentStep);
                    bool const isSlidingWindowEnabled = MovingWindow::getInstance().isEnabled();

                    // If the sliding window is enabled, the resulting shadowgram will be smaller to not show the
                    // "invisible" GPU fields in the shadowgram
                    int yWindowSize;
                    if(isSlidingWindowEnabled)
                    {
                        yWindowSize = int((float(nGpus - 1) / float(nGpus)) * globalGridSize[1]);
                    }
                    else
                    {
                        yWindowSize = globalGridSize[1];
                    }

                    // If the sliding window is active, the resulting shadowgram will also be smaller to adjust for the
                    // laser propagation distance
                    float slidingWindowCorrection;
                    if(isSlidingWindowActive)
                    {
                        int const cellsUntilIntegrationPlane = slicePoint * globalGridSize[2];
                        slidingWindowCorrection = cellsUntilIntegrationPlane * SI::CELL_DEPTH_SI
                            + pluginNumT * dt * float_64(SI::SPEED_OF_LIGHT_SI);
                    }
                    else
                    {
                        slidingWindowCorrection = 0.0;
                    }

                    pluginNumY = math::floor(
                        (yWindowSize - slidingWindowCorrection / SI::CELL_HEIGHT_SI) / (params::yRes) -2);

                    // Don't use fields inside the field absorber
                    pluginNumX
                        -= (fields::absorber::NUM_CELLS[0][0] + fields::absorber::NUM_CELLS[0][1]) / (params::xRes);
                    pluginNumY
                        -= (fields::absorber::NUM_CELLS[1][0] + fields::absorber::NUM_CELLS[1][1]) / (params::yRes);

                    // Make sure spatial grid is even
                    pluginNumX = pluginNumX % 2 == 0 ? pluginNumX : pluginNumX - 1;
                    pluginNumY = pluginNumY % 2 == 0 ? pluginNumY : pluginNumY - 1;

                    int const yGlobalOffset
                        = (MovingWindow::getInstance().getWindow(currentStep).globalDimensions.offset)[1];
                    int const yTotalOffset = int(startSlideCount * globalGridSize[1] / nGpus);

                    // The total domain indices of the integration slice are constant, because the screen is not
                    // co-propagating with the moving window
                    yTotalMinIndex
                        = yTotalOffset + yGlobalOffset + math::floor(slidingWindowCorrection / SI::CELL_HEIGHT_SI) - 1;
                    yTotalMaxIndex = yTotalMinIndex + pluginNumY;

                    // Initialization of storage arrays
                    ExOmega = vec3c(pluginNumX, vec2c(pluginNumY, vec1c(numOmegas)));
                    EyOmega = vec3c(pluginNumX, vec2c(pluginNumY, vec1c(numOmegas)));
                    BxOmega = vec3c(pluginNumX, vec2c(pluginNumY, vec1c(numOmegas)));
                    ByOmega = vec3c(pluginNumX, vec2c(pluginNumY, vec1c(numOmegas)));

                    ExOmegaPropagated = vec3c(pluginNumX, vec2c(pluginNumY, vec1c(numOmegas)));
                    EyOmegaPropagated = vec3c(pluginNumX, vec2c(pluginNumY, vec1c(numOmegas)));
                    BxOmegaPropagated = vec3c(pluginNumX, vec2c(pluginNumY, vec1c(numOmegas)));
                    ByOmegaPropagated = vec3c(pluginNumX, vec2c(pluginNumY, vec1c(numOmegas)));

                    tmpEx = vec2r(pluginNumX, vec1r(pluginNumY));
                    tmpEy = vec2r(pluginNumX, vec1r(pluginNumY));
                    tmpBx = vec2r(pluginNumX, vec1r(pluginNumY));
                    tmpBy = vec2r(pluginNumX, vec1r(pluginNumY));

                    shadowgram = vec2r(pluginNumX, vec1r(pluginNumY));

                    init_fftw();
                }

                /** Destructor of the shadowgraphy helper class
                 * To be called at the last time step when the shadowgraphy time integration ends
                 */
                ~Helper()
                {
                    fftw_free(fftwInF);
                    fftw_free(fftwOutF);
                    fftw_free(fftwInB);
                    fftw_free(fftwOutB);
                }

                /** Store fields in helper class with proper resolution and fixed Yee offset
                 *
                 * @tparam F Field
                 * @param t current plugin timestep (simulation timestep - plugin start)
                 * @param currentStep current simulation timestep
                 * @param fieldBuffer1 2D array of field at slicePos
                 * @param fieldBuffer2 2D array of field at slicePos with 1 offset (to fix Yee offset)
                 */
                template<typename F>
                void storeField(
                    int t,
                    int currentStep,
                    pmacc::container::HostBuffer<float3_64, 2>* fieldBuffer1,
                    pmacc::container::HostBuffer<float3_64, 2>* fieldBuffer2)
                {
                    int const currentSlideCount = MovingWindow::getInstance().getSlideCounter(currentStep);

                    for(int i = 0; i < pluginNumX; i++)
                    {
                        int const simI = fields::absorber::NUM_CELLS[0][0] + i * params::xRes;
                        for(int j = 0; j < pluginNumY; ++j)
                        {
                            // Transform the total coordinates of the fixed shadowgraphy screen to the global
                            // coordinates of the field-buffers
                            int const simJ = fields::absorber::NUM_CELLS[0][1] + yTotalMinIndex
                                - currentSlideCount * cellsPerGpu + j * params::yRes;

                            float_64 const wf
                                = masks::positionWf(i, j, pluginNumX, pluginNumY) * masks::timeWf(t, duration);

                            // fix yee offset
                            if(F::getName() == "E")
                            {
                                tmpEx[i][j] = wf
                                    * ((*(fieldBuffer2->origin()(simI, simJ + 1))).x()
                                       + (*(fieldBuffer2->origin()(simI + 1, simJ + 1))).x())
                                    / 2.0;
                                tmpEy[i][j] = wf
                                    * ((*(fieldBuffer2->origin()(simI + 1, simJ))).y()
                                       + (*(fieldBuffer2->origin()(simI + 1, simJ + 1))).y())
                                    / 2.0;
                            }
                            else
                            {
                                tmpBx[i][j] = wf
                                    * ((*(fieldBuffer1->origin()(simI + 1, simJ))).x()
                                       + (*(fieldBuffer1->origin()(simI + 1, simJ + 1))).x()
                                       + (*(fieldBuffer2->origin()(simI + 1, simJ))).x()
                                       + (*(fieldBuffer2->origin()(simI + 1, simJ + 1))).x())
                                    / 4.0;
                                tmpBy[i][j] = wf
                                    * ((*(fieldBuffer1->origin()(simI, simJ + 1))).y()
                                       + (*(fieldBuffer1->origin()(simI + 1, simJ + 1))).y()
                                       + (*(fieldBuffer2->origin()(simI, simJ + 1))).y()
                                       + (*(fieldBuffer2->origin()(simI + 1, simJ + 1))).y())
                                    / 4.0;
                            }
                        }
                    }
                }

                /** Time domain window function that will be multiplied with the electric and magnetic fields
                 * in time-position domain to reduce ringing artifacts in the omega domain after the DFT.
                 * The implemented window function is a Tukey-Window with sinusoidal slopes.
                 *
                 * @param t timestep from 0 to t_n
                 */
                void calculate_dft(int t)
                {
                    float_64 const tSI = t * int(params::tRes) * float_64(picongpu::SI::DELTA_T_SI);

                    for(int o = 0; o < numOmegas; ++o)
                    {
                        int const omegaIndex = getOmegaIndex(o);
                        float_64 const omegaSI = omega(omegaIndex);

                        complex_64 const phase = complex_64(0, omegaSI * tSI);
                        complex_64 const exponential = math::exp(phase);

                        for(int i = 0; i < pluginNumX; ++i)
                        {
                            for(int j = 0; j < pluginNumY; ++j)
                            {
                                ExOmega[i][j][o] += tmpEx[i][j] * exponential;
                                EyOmega[i][j][o] += tmpEy[i][j] * exponential;
                                BxOmega[i][j][o] += tmpBx[i][j] * exponential;
                                ByOmega[i][j][o] += tmpBy[i][j] * exponential;
                            }
                        }
                    }
                }

                /** Propagate the electric and magnetic field from the extraction position
                 * to the focus position of the virtual lens system. This is done by Fourier transforming the field
                 * with a FFT into $(k_\perp, \omega)$-domain, applying the defined masks in the .param files,
                 * multiplying with the propagator $\exp^{i \Delta z k \sqrt{\omega^2/c^2 - k_x^2 - k_y^2}}$ and
                 * transforming it back into
                 * $(x, y, \omega)$-domain.
                 */
                void propagateFields()
                {
                    for(int fieldIndex = 0; fieldIndex < 4; fieldIndex++)
                    {
                        for(int o = 0; o < numOmegas; ++o)
                        {
                            int const omegaIndex = getOmegaIndex(o);
                            float_64 const omegaSI = omega(omegaIndex);
                            float_64 const kSI = omegaSI / float_64(SI::SPEED_OF_LIGHT_SI);

                            // put field into fftw array
                            for(int i = 0; i < pluginNumX; ++i)
                            {
                                for(int j = 0; j < pluginNumY; ++j)
                                {
                                    int const index = i + j * pluginNumX;

                                    if(fieldIndex == 0)
                                    {
                                        fftwInF[index][0] = ExOmega[i][j][o].real();
                                        fftwInF[index][1] = ExOmega[i][j][o].imag();
                                    }
                                    else if(fieldIndex == 1)
                                    {
                                        fftwInF[index][0] = EyOmega[i][j][o].real();
                                        fftwInF[index][1] = EyOmega[i][j][o].imag();
                                    }
                                    else if(fieldIndex == 2)
                                    {
                                        fftwInF[index][0] = BxOmega[i][j][o].real();
                                        fftwInF[index][1] = BxOmega[i][j][o].imag();
                                    }
                                    else if(fieldIndex == 3)
                                    {
                                        fftwInF[index][0] = ByOmega[i][j][o].real();
                                        fftwInF[index][1] = ByOmega[i][j][o].imag();
                                    }
                                }
                            }
                            if(intermediateOutputEnabled)
                                writeIntermediateFile(o, fieldIndex);

                            fftw_execute(planForward);

                            if(fourierOutputEnabled)
                                writeFourierFile(o, fieldIndex, false);

                            // put field into fftw array
                            for(int i = 0; i < pluginNumX; ++i)
                            {
                                // Put origin into center of array with this, necessary due to FFT
                                int const iffs = (i + pluginNumX / 2) % pluginNumX;

                                for(int j = 0; j < pluginNumY; ++j)
                                {
                                    int const index = i + j * pluginNumX;
                                    int const jffs = (j + pluginNumY / 2) % pluginNumY;

                                    float_64 const sqrt1 = kSI * kSI;
                                    float_64 const sqrt2 = kx(i) * kx(i);
                                    float_64 const sqrt3 = ky(j) * ky(j);
                                    float_64 const sqrtContent
                                        = (kSI == 0.0) ? 0.0 : 1 - sqrt2 / sqrt1 - sqrt3 / sqrt1;

                                    if(sqrtContent >= 0.0)
                                    {
                                        // Put origin into center of array with this, necessary due to FFT
                                        int const indexffs = iffs + jffs * pluginNumX;

                                        complex_64 const field
                                            = complex_64(fftwOutF[indexffs][0], fftwOutF[indexffs][1]);

                                        float_64 const phase
                                            = (kSI == 0.0) ? 0.0 : propagationDistance * kSI * math::sqrt(sqrtContent);
                                        complex_64 const propagator = math::exp(complex_64(0, phase));

                                        complex_64 const propagatedField
                                            = masks::maskFourier(kx(i), ky(j), omega(omegaIndex)) * field * propagator;

                                        fftwInB[index][0] = propagatedField.real();
                                        fftwInB[index][1] = propagatedField.imag();
                                    }
                                    else
                                    {
                                        fftwInB[index][0] = 0.0;
                                        fftwInB[index][1] = 0.0;
                                    }
                                }
                            }

                            if(fourierOutputEnabled)
                                writeFourierFile(o, fieldIndex, true);

                            fftw_execute(planBackward);

                            // yoink fields from fftw array
                            for(int i = 0; i < pluginNumX; ++i)
                            {
                                for(int j = 0; j < pluginNumY; ++j)
                                {
                                    int const index = i + j * pluginNumX;

                                    if(fieldIndex == 0)
                                    {
                                        ExOmegaPropagated[i][j][o]
                                            = complex_64(fftwOutB[index][0], fftwOutB[index][1]);
                                    }
                                    else if(fieldIndex == 1)
                                    {
                                        EyOmegaPropagated[i][j][o]
                                            = complex_64(fftwOutB[index][0], fftwOutB[index][1]);
                                    }
                                    else if(fieldIndex == 2)
                                    {
                                        BxOmegaPropagated[i][j][o]
                                            = complex_64(fftwOutB[index][0], fftwOutB[index][1]);
                                    }
                                    else if(fieldIndex == 3)
                                    {
                                        ByOmegaPropagated[i][j][o]
                                            = complex_64(fftwOutB[index][0], fftwOutB[index][1]);
                                    }
                                }
                            }
                        }
                    }
                }

                /** Perform an inverse Fourier transform into time domain of both the electric and magnetic field
                 * and then perform a time integration to generate a 2D image out of the 3D array.
                 */
                void calculate_shadowgram()
                {
                    // Loop over all timesteps
                    for(int t = 0; t < pluginNumT; ++t)
                    {
                        float_64 const tSI = t * int(params::tRes) * float_64(picongpu::SI::DELTA_T_SI);

                        // Initialization of storage arrays
                        vec2c ExTmpSum = vec2c(pluginNumX, vec1c(pluginNumY));
                        vec2c EyTmpSum = vec2c(pluginNumX, vec1c(pluginNumY));
                        vec2c BxTmpSum = vec2c(pluginNumX, vec1c(pluginNumY));
                        vec2c ByTmpSum = vec2c(pluginNumX, vec1c(pluginNumY));

                        // DFT loop to time domain
                        for(int o = 0; o < numOmegas; ++o)
                        {
                            int const omegaIndex = getOmegaIndex(o);
                            float_64 const omegaSI = omega(omegaIndex);

                            complex_64 const phase = complex_64(0, -tSI * omegaSI);
                            complex_64 const exponential = math::exp(phase);

                            for(int i = 0; i < pluginNumX; ++i)
                            {
                                for(int j = 0; j < pluginNumY; ++j)
                                {
                                    complex_64 const Ex = ExOmegaPropagated[i][j][o] * exponential;
                                    complex_64 const Ey = EyOmegaPropagated[i][j][o] * exponential;
                                    complex_64 const Bx = BxOmegaPropagated[i][j][o] * exponential;
                                    complex_64 const By = ByOmegaPropagated[i][j][o] * exponential;
                                    shadowgram[i][j] += (dt
                                                         / (SI::MUE0_SI * pluginNumT * pluginNumT * pluginNumX
                                                            * pluginNumX * pluginNumY * pluginNumY))
                                        * (Ex * By - Ey * Bx + ExTmpSum[i][j] * By + Ex * ByTmpSum[i][j]
                                           - EyTmpSum[i][j] * Bx - Ey * BxTmpSum[i][j])
                                              .real();

                                    if(o < (numOmegas - 1))
                                    {
                                        ExTmpSum[i][j] += Ex;
                                        EyTmpSum[i][j] += Ey;
                                        BxTmpSum[i][j] += Bx;
                                        ByTmpSum[i][j] += By;
                                    }
                                }
                            }
                        }
                    }
                }

                //! Get shadowgram as a 2D image
                vec2r getShadowgram() const
                {
                    return shadowgram;
                }

                std::vector<float_64> getShadowgram1D()
                {
                    std::vector<float_64> retVec(getSizeX() * getSizeY());

                    for (int i = 0; i < getSizeX(); ++i){
                        for (int j = 0; j < getSizeY(); ++j){
                            retVec[i + j * getSizeX()] = shadowgram[i][j];
                        }
                    }

                    return retVec;
                }

                std::shared_ptr<pmacc::container::HostBuffer<float_64, DIM2>> getShadowgramBuf()
                {
                    //pmacc::container::HostBuffer<float_64, DIM2> retBuffer(getSizeX(), getSizeY());
                    retBuffer = std::make_shared<pmacc::container::HostBuffer<float_64, DIM2>>(getSizeX(), getSizeY());

                    for (int j = 0; j < getSizeY(); ++j){
                        for (int i = 0; i < getSizeX(); ++i){
                            *(retBuffer->origin()(i,j)) = static_cast<float_64>(shadowgram[i][j]);
                        }
                    }

                    return retBuffer;
                }

                //! Get amount of shadowgram pixels in x direction
                int getSizeX() const
                {
                    return pluginNumX;
                }

                //! Get amount of shadowgram pixels in y direction
                int getSizeY() const
                {
                    return pluginNumY;
                }

            private:
                //! Initialize fftw memory things, supposed to be called once at the start of the plugin loop
                void init_fftw()
                {
                    // Input and output arrays for the FFT transforms
                    fftwInF = fftw_alloc_complex(pluginNumX * pluginNumY);
                    fftwOutF = fftw_alloc_complex(pluginNumX * pluginNumY);

                    fftwInB = fftw_alloc_complex(pluginNumX * pluginNumY);
                    fftwOutB = fftw_alloc_complex(pluginNumX * pluginNumY);

                    // Create fftw plan for transverse fft for real to complex
                    // Many ffts will be performed -> use FFTW_MEASURE as flag
                    planForward
                        = fftw_plan_dft_2d(pluginNumY, pluginNumX, fftwInF, fftwOutF, FFTW_FORWARD, FFTW_MEASURE);

                    // Create fftw plan for transverse ifft for complex to complex
                    // Even more iffts will be performed -> use FFTW_MEASURE as flag
                    planBackward
                        = fftw_plan_dft_2d(pluginNumY, pluginNumX, fftwInB, fftwOutB, FFTW_BACKWARD, FFTW_MEASURE);
                }

                /** Store fields in helper class with proper resolution and fixed Yee offset in (k_x, k_y,
                 * \omega)-domain
                 *
                 * @param o omega index from trimmed array in plugin
                 * @param fieldIndex value from 0 to 3 (Ex, Ey, Bx, By)
                 * @param masksApplied have masks been applied in Fourier space yet
                 */
                void writeFourierFile(int o, int fieldIndex, bool masksApplied)
                {
                    int const omegaIndex = getOmegaIndex(o);
                    std::ofstream outFile;
                    std::ostringstream fileName;

                    if(fieldIndex == 0)
                    {
                        fileName << "Ex";
                    }
                    else if(fieldIndex == 1)
                    {
                        fileName << "Ey";
                    }
                    else if(fieldIndex == 2)
                    {
                        fileName << "Bx";
                    }
                    else if(fieldIndex == 3)
                    {
                        fileName << "By";
                    }

                    fileName << "_fourierspace";
                    if(masksApplied)
                    {
                        fileName << "_with_masks";
                    }
                    fileName << "_" << omegaIndex << ".dat";

                    outFile.open(fileName.str(), std::ofstream::out | std::ostream::trunc);

                    if(!outFile)
                    {
                        std::cerr << "Can't open file [" << fileName.str()
                                  << "] for output, disable plugin output. Chuchu" << std::endl;
                    }
                    else
                    {
                        for(unsigned int i = 0; i < getSizeX(); ++i) // over all x
                        {
                            int const iffs = (i + pluginNumX / 2) % pluginNumX;
                            for(unsigned int j = 0; j < getSizeY(); ++j) // over all y
                            {
                                int const index = i + j * pluginNumX;
                                int const jffs = (j + pluginNumY / 2) % pluginNumY;
                                int const indexffs = iffs + jffs * pluginNumX;
                                if(!masksApplied)
                                {
                                    outFile << fftwOutF[indexffs][0] << "+" << fftwOutF[indexffs][1] << "j"
                                            << "\t";
                                }
                                else
                                {
                                    outFile << fftwInB[index][0] << "+" << fftwInB[index][1] << "j"
                                            << "\t";
                                }
                            } // for loop over all y

                            outFile << std::endl;
                        } // for loop over all x

                        outFile.flush();
                        outFile << std::endl; // now all data is written to file

                        if(outFile.fail())
                            std::cerr << "Error on flushing file [" << fileName.str() << "]. " << std::endl;

                        outFile.close();
                    }
                }

                /** Store fields in helper class with proper resolution and fixed Yee offset in (x, y, \omega)-domain
                 * directly after loading the field
                 *
                 * @param o omega index from trimmed array in plugin
                 * @param fieldIndex value from 0 to 3 (Ex, Ey, Bx, By)
                 */
                void writeIntermediateFile(int o, int fieldIndex)
                {
                    int const omegaIndex = getOmegaIndex(o);
                    std::ofstream outFile;
                    std::ostringstream fileName;

                    if(fieldIndex == 0)
                    {
                        fileName << "Ex";
                    }
                    else if(fieldIndex == 1)
                    {
                        fileName << "Ey";
                    }
                    else if(fieldIndex == 2)
                    {
                        fileName << "Bx";
                    }
                    else if(fieldIndex == 3)
                    {
                        fileName << "By";
                    }

                    fileName << "_omegaspace";
                    fileName << "_" << omegaIndex << ".dat";

                    outFile.open(fileName.str(), std::ofstream::out | std::ostream::trunc);

                    if(!outFile)
                    {
                        std::cerr << "Can't open file [" << fileName.str()
                                  << "] for output, disable plugin output. Chuchu" << std::endl;
                    }
                    else
                    {
                        for(unsigned int i = 0; i < getSizeX(); ++i) // over all x
                        {
                            for(unsigned int j = 0; j < getSizeY(); ++j) // over all y
                            {
                                int const index = i + j * pluginNumX;
                                outFile << fftwInF[index][0] << "+" << fftwInF[index][1] << "j"
                                        << "\t";
                            } // for loop over all y

                            outFile << std::endl;
                        } // for loop over all x

                        outFile.flush();
                        outFile << std::endl; // now all data are written to file

                        if(outFile.fail())
                            std::cerr << "Error on flushing file [" << fileName.str() << "]. " << std::endl;

                        outFile.close();
                    }
                }

                //! Return minimum omega index for trimmed arrays in omega dimension
                int getOmegaMinIndex() const
                {
                    float_64 const actualStep = params::tRes * SI::DELTA_T_SI;
                    int const tmpIndex
                        = math::floor(pluginNumT * ((actualStep * params::omegaWfMin) / (2.0 * PI) + 0.5));
                    int retIndex = tmpIndex > pluginNumT / 2 + 1 ? tmpIndex : pluginNumT / 2 + 1;
                    return retIndex;
                }

                //! Return maximum omega index for trimmed arrays in omega dimension
                int getOmegaMaxIndex() const
                {
                    float_64 const actualStep = params::tRes * SI::DELTA_T_SI;
                    int const tmpIndex
                        = math::ceil(pluginNumT * ((actualStep * params::omegaWfMax) / (2.0 * PI) + 0.5));
                    int retIndex = tmpIndex <= pluginNumT ? tmpIndex : pluginNumT;
                    return retIndex + 1;
                }

                //! Return size of trimmed arrays in omega dimension
                int getNumOmegas() const
                {
                    return 2 * (getOmegaMaxIndex() - getOmegaMinIndex());
                }

                /** Return omega index for a matrix that doesn't remove the zero-valued frequencies.
                 * Used for properly treating raw Fourier space data of this plugin.
                 *
                 * @param i frequency index for trimmed plugin array
                 *
                 * @return index for non-trimmed array in frequency domain
                 */
                int getOmegaIndex(int i) const
                {
                    if(i < numOmegas / 2)
                    {
                        return duration / params::tRes - getOmegaMinIndex() - numOmegas / 2 + i + 1;
                    }
                    else
                    {
                        return (i % numOmegas / 2) + getOmegaMinIndex();
                    }
                }

                /** angular frequency in SI units
                 *
                 * @param i frequency index for trimmed plugin array
                 *
                 * @return angular frequency in SI units
                 */
                float omega(int i) const
                {
                    float const actualStep = dt;
                    return 2.0 * PI * (float(i) - float(pluginNumT) / 2.0) / float(pluginNumT) / actualStep;
                }

                /** x component of k vector in SI units for FFTs
                 *
                 * @param i k vector index
                 *
                 * @return x component of k vector in SI units
                 */
                float kx(int i) const
                {
                    float const actualStep = params::xRes * SI::CELL_WIDTH_SI;
                    return 2.0 * PI * (float(i) - float(pluginNumX) / 2.0) / float(pluginNumX) / actualStep;
                }

                /** y component of k vector in SI units for FFTs
                 *
                 * @param i k vector index
                 *
                 * @return y component of k vector in SI units
                 */
                float ky(int i) const
                {
                    float const actualStep = params::yRes * SI::CELL_HEIGHT_SI;
                    return 2.0 * PI * (float(i) - float(pluginNumY) / 2.0) / float(pluginNumY) / actualStep;
                }
            }; // class Helper
        } // namespace shadowgraphy
    } // namespace plugins
} // namespace picongpu