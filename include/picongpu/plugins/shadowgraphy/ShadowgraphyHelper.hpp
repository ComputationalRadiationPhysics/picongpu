/* Copyright 2023 Finn-Ole Carstens, Rene Widera
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

// required for SIMDIM definition
#include "picongpu/defines.hpp"

#if(SIMDIM == DIM3 && PIC_ENABLE_FFTW3 == 1 && ENABLE_OPENPMD == 1)

#    include "picongpu/simulation/control/Window.hpp"

#    include <pmacc/algorithms/math/defines/pi.hpp>
#    include <pmacc/assert.hpp>
#    include <pmacc/mappings/simulation/GridController.hpp>
#    include <pmacc/math/Vector.hpp>
#    include <pmacc/memory/buffers/Buffer.hpp>

#    include <chrono>
#    include <cmath> // what
#    include <complex>
#    include <iostream>

#    include <fftw3.h>
#    include <stdio.h>

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

                using vec3c = std::vector<std::vector<std::vector<complex_64>>>;
                using vec2c = std::vector<std::vector<complex_64>>;
                using vec1c = std::vector<complex_64>;
                using vec3r = std::vector<std::vector<std::vector<float_64>>>;
                using vec2r = std::vector<std::vector<float_64>>;
                using vec1r = std::vector<float_64>;

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

                vec2r shadowgram;

                // Size of arrays
                int pluginNumX, pluginNumY;
                int numOmegas;

                float_X xMin;
                float_X xStep;
                float_X yMin;
                float_X yStep;

                // Variables to adjust window size for moving windows
                int yTotalMinIndex;
                int cellsPerGpuY;
                bool isSlidingWindowActive;

                // Variables for omega calculations
                float_X dt;
                int pluginNumT;
                int duration;

                float_X propagationDistance;

                bool fourierOutputEnabled;

                bool initializedDataBox = false;

            public:
                enum class FieldType : uint32_t
                {
                    E,
                    B
                };

                /** Constructor of shadowgraphy helper class
                 * To be called at the first timestep when the shadowgraphy time integration starts
                 *
                 * @param currentStep current simulation timestep
                 * @param slicePoint value between 0.0 and 1.0 where the fields are extracted
                 * @param focusPos focus position of lens system, e.g. the propagation distance of the vacuum
                 * propagator
                 * @param duration duration of time extraction in simulation time steps
                 * @param fourierOutputEnabled whether to output the fourier transform of the fields
                 */
                Helper(int currentStep, float_X slicePoint, float_X focusPos, int duration, bool fourierOutputEnabled)
                    : duration(duration)
                    , fourierOutputEnabled(fourierOutputEnabled)
                {
                    dt = params::tRes * sim.si.getDt();
                    pluginNumT = duration / params::tRes;
                    numOmegas = getNumOmegas();

                    propagationDistance = focusPos;

                    const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                    pmacc::math::Size_t<simDim> globalGridSize = subGrid.getGlobalDomain().size;

                    pmacc::GridController<simDim>& con = pmacc::Environment<simDim>::get().GridController();
                    int const nGpus = con.getGpuNodes()[1];
                    cellsPerGpuY = int(globalGridSize[1] / nGpus);

                    int const startSlideCount = MovingWindow::getInstance().getSlideCounter(currentStep);

                    // These are currently not allowed to change during plugin run!
                    isSlidingWindowActive = MovingWindow::getInstance().isSlidingWindowActive(currentStep);
                    bool const isSlidingWindowEnabled = MovingWindow::getInstance().isEnabled();

                    // If the sliding window is enabled, the resulting shadowgram will be smaller to not show the
                    // "invisible" GPU fields in the shadowgram
                    int yWindowSize;
                    if(isSlidingWindowEnabled)
                    {
                        yWindowSize = int((float_X(nGpus - 1) / float_X(nGpus)) * globalGridSize[1]);
                    }
                    else
                    {
                        yWindowSize = globalGridSize[1];
                    }

                    // If the sliding window is active, the resulting shadowgram will also be smaller to adjust for the
                    // laser propagation distance
                    float_X slidingWindowCorrection;
                    if(isSlidingWindowActive)
                    {
                        int const cellsUntilIntegrationPlane = slicePoint * globalGridSize[2];
                        slidingWindowCorrection = cellsUntilIntegrationPlane * sim.si.getCellSize().z()
                            + pluginNumT * dt * float_64(sim.si.getSpeedOfLight());
                    }
                    else
                    {
                        slidingWindowCorrection = 0.0;
                    }

                    pluginNumX = globalGridSize[0] / params::xRes;

                    pluginNumY = math::floor(
                        (yWindowSize - slidingWindowCorrection / sim.si.getCellSize().y()) / (params::yRes));

                    // Don't use fields inside the field absorber
                    pluginNumX
                        -= (fields::absorber::NUM_CELLS[0][0] + fields::absorber::NUM_CELLS[0][1]) / (params::xRes);
                    pluginNumY
                        -= (fields::absorber::NUM_CELLS[1][0] + fields::absorber::NUM_CELLS[1][1]) / (params::yRes);

                    // Make sure spatial grid is even
                    pluginNumX = pluginNumX % 2 == 0 ? pluginNumX : pluginNumX - 1;
                    pluginNumY = pluginNumY % 2 == 0 ? pluginNumY : pluginNumY - 1;

                    // Offsets for the moving window
                    int const yGlobalOffset
                        = (MovingWindow::getInstance().getWindow(currentStep).globalDimensions.offset)[1];
                    int const yTotalOffset = int(startSlideCount * globalGridSize[1] / nGpus);

                    // The total domain indices of the integration slice are constant, because the screen is not
                    // co-propagating with the moving window
                    yTotalMinIndex = yTotalOffset + yGlobalOffset
                        + math::floor(slidingWindowCorrection / sim.si.getCellSize().y());

                    // Initialization of storage arrays
                    ExOmega = vec3c(pluginNumX, vec2c(pluginNumY, vec1c(numOmegas)));
                    EyOmega = vec3c(pluginNumX, vec2c(pluginNumY, vec1c(numOmegas)));
                    BxOmega = vec3c(pluginNumX, vec2c(pluginNumY, vec1c(numOmegas)));
                    ByOmega = vec3c(pluginNumX, vec2c(pluginNumY, vec1c(numOmegas)));

                    tmpEx = vec2r(pluginNumX, vec1r(pluginNumY));
                    tmpEy = vec2r(pluginNumX, vec1r(pluginNumY));
                    tmpBx = vec2r(pluginNumX, vec1r(pluginNumY));
                    tmpBy = vec2r(pluginNumX, vec1r(pluginNumY));

                    shadowgram = vec2r(pluginNumX, vec1r(pluginNumY));
                }


                /** Compute cross product of a field
                 *
                 * @tparam T_fieldType Field type (E or B)
                 * @tparam T_FieldDataBox Field data box type
                 *
                 * @param field Field data box
                 *
                 * @return Cross product of the field
                 */
                template<FieldType T_fieldType, typename T_FieldDataBox>
                float2_X cross(T_FieldDataBox field)
                {
                    // Adjust for Yee offset
                    if constexpr(T_fieldType == FieldType::E)
                    {
                        // Calculate cross product of electric field at (0, 0, 0)
                        return float2_X(
                            (field(DataSpace<DIM3>(0, 0, 0)).x() + field(DataSpace<DIM3>(-1, 0, 0)).x()) / 2.0_X,
                            (field(DataSpace<DIM3>(0, 0, 0)).y() + field(DataSpace<DIM3>(0, -1, 0)).y()) / 2.0_X);
                    }
                    else if constexpr(T_fieldType == FieldType::B)
                    {
                        // Calculate cross product of magnetic field at (0, 0, 0)
                        return float2_X(
                            (field(DataSpace<DIM3>(0, 0, 0)).x() + field(DataSpace<DIM3>(0, -1, 0)).x()
                             + field(DataSpace<DIM3>(0, 0, -1)).x() + field(DataSpace<DIM3>(0, -1, -1)).x())
                                / 4.0_X,
                            (field(DataSpace<DIM3>(0, 0, 0)).y() + field(DataSpace<DIM3>(-1, 0, 0)).y()
                             + field(DataSpace<DIM3>(-1, 0, 0)).y() + field(DataSpace<DIM3>(-1, 0, -1)).y())
                                / 4.0_X);
                    }
                    else
                    {
                        static_assert(!sizeof(T_FieldDataBox), "Unknown field description used");
                    }

                    ALPAKA_UNREACHABLE(float2_X{});
                }

                /** Store fields in helper class with proper resolution
                 *
                 * @tparam T_fieldType Field type (E or B)
                 * @tparam T_SliceBuffer 3D data box type
                 *
                 * @param t current plugin timestep (simulation timestep - plugin start)
                 * @param currentStep current simulation timestep
                 * @param sliceBuffer 3D data box shifted to the local simulation origin (no guard)
                 */
                template<FieldType T_fieldType, typename T_SliceBuffer>
                void storeField(int t, int currentStep, T_SliceBuffer sliceBuffer)
                {
                    auto globalFieldBox = sliceBuffer->getDataBox();
                    int const currentSlideCount = MovingWindow::getInstance().getSlideCounter(currentStep);
                    if(!initializedDataBox)
                    {
                        xMin = fields::absorber::NUM_CELLS[0][0] * sim.si.getCellSize().x();
                        yMin = (fields::absorber::NUM_CELLS[1][0] + yTotalMinIndex) * sim.si.getCellSize().y();

                        xStep = params::xRes * sim.si.getCellSize().x();
                        yStep = params::yRes * sim.si.getCellSize().y();
                        initializedDataBox = true;
                    }


                    for(int i = 0; i < pluginNumX; i++)
                    {
                        int const simI = fields::absorber::NUM_CELLS[0][0] + i * params::xRes;
                        for(int j = 0; j < pluginNumY; ++j)
                        {
                            // Transform the total coordinates of the fixed shadowgraphy screen to the global
                            // coordinates of the field-buffers
                            int const simJ = fields::absorber::NUM_CELLS[1][0] + yTotalMinIndex
                                - currentSlideCount * cellsPerGpuY + j * params::yRes;

                            float_64 const wf
                                = masks::positionWf(i, j, pluginNumX, pluginNumY) * masks::timeWf(t, duration);

                            auto value = globalFieldBox(DataSpace<DIM2>{simI, simJ});

                            if constexpr(T_fieldType == FieldType::E)
                            {
                                tmpEx[i][j] = sim.unit.eField() * wf * value.x();
                                tmpEy[i][j] = sim.unit.eField() * wf * value.y();
                            }
                            else if constexpr(T_fieldType == FieldType::B)
                            {
                                tmpBx[i][j] = sim.unit.bField() * wf * value.x();
                                tmpBy[i][j] = sim.unit.bField() * wf * value.y();
                            }
                        }
                    }
                }

                /** Calculate Discrete Fourier Transform (DFT) from time domain to omega domain.
                 *
                 * @param t timestep from 0 to t_n
                 */
                void computeDFT(int t)
                {
                    float_64 const tSI = t * int(params::tRes) * float_64(picongpu::sim.si.getDt());

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
                void propagateFieldsAndCalculateShadowgram()
                {
                    initFFTW();

                    // Arrays for propagated fields
                    auto ExOmegaPropagated = vec3c(pluginNumX, vec2c(pluginNumY, vec1c(numOmegas)));
                    auto EyOmegaPropagated = vec3c(pluginNumX, vec2c(pluginNumY, vec1c(numOmegas)));
                    auto BxOmegaPropagated = vec3c(pluginNumX, vec2c(pluginNumY, vec1c(numOmegas)));
                    auto ByOmegaPropagated = vec3c(pluginNumX, vec2c(pluginNumY, vec1c(numOmegas)));

                    // fieldIndex 0: Ex, 1: Ey, 2: Bx, 3: By
                    for(int fieldIndex = 0; fieldIndex < 4; fieldIndex++)
                    {
                        for(int o = 0; o < numOmegas; ++o)
                        {
                            int const omegaIndex = getOmegaIndex(o);
                            float_64 const omegaSI = omega(omegaIndex);
                            float_64 const kSI = omegaSI / float_64(sim.si.getSpeedOfLight());

                            // Put field into fftw array
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

                            fftw_execute(planForward);

                            // Put field into fftw array
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

                            fftw_execute(planBackward);

                            // Get fields from fftw array
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

                    computeShadowgram(ExOmegaPropagated, EyOmegaPropagated, BxOmegaPropagated, ByOmegaPropagated);

                    freeFFTW();
                }

                //! Get shadowgram as a 2D image
                vec2r getShadowgram() const
                {
                    return shadowgram;
                }

                //! Get x coordinate in SI units for openPMD output
                float_X getX(int i) const
                {
                    return xMin + i * xStep;
                }

                //! Get y coordinate in SI units for openPMD output
                float_X getY(int j) const
                {
                    return yMin + j * yStep;
                }

                //! Return Shadowgram buffer for openPMD output
                auto getShadowgramBuf()
                {
                    auto retBuffer
                        = std::make_shared<HostBuffer<float_64, DIM2>>(DataSpace<DIM2>(getSizeX(), getSizeY()));
                    auto dataBox = retBuffer->getDataBox();

                    for(int j = 0; j < getSizeY(); ++j)
                    {
                        for(int i = 0; i < getSizeX(); ++i)
                        {
                            dataBox({i, j}) = static_cast<float_64>(shadowgram[i][j]);
                        }
                    }

                    return retBuffer;
                }

                /** Return Fourier domain buffer for openPMD output
                 *
                 * @param index description of parameter from 0 to 7. The indices stand for:
                 * 0: Ex positive
                 * 1: Ex negative
                 * 2: Ey positive
                 * 3: Ey negative
                 * 4: Bx positive
                 * 5: Bx negative
                 * 6: By positive
                 * 7: By negative
                 *
                 * @return description of return value
                 */
                auto getFourierBuf(uint32_t index)
                {
                    const int nOmegasHalf = numOmegas / 2;
                    auto retBufferF = std::make_shared<HostBuffer<std::complex<float_64>, DIM3>>(
                        DataSpace<DIM3>(getSizeX(), getSizeY(), nOmegasHalf));
                    auto dataBox = retBufferF->getDataBox();

                    // The fields are split into 2 parts in the output, because the omega-domain
                    // is not necessarily continuous due to band-pass filters
                    vec3c* retField = nullptr;
                    if(index <= 1)
                        retField = &ExOmega;
                    else if(index <= 3)
                        retField = &EyOmega;
                    else if(index <= 5)
                        retField = &BxOmega;
                    else if(index <= 7)
                        retField = &ByOmega;
                    else
                        throw std::runtime_error(
                            std::string("getFourierBuf() called with unknown index '") + std::to_string(index) + "'");

                    for(int i = 0; i < getSizeX(); ++i)
                    {
                        for(int j = 0; j < getSizeY(); ++j)
                        {
                            for(int o = 0; o < nOmegasHalf; ++o)
                            {
                                int const oSigned = ((index % 2) == 0) ? o : o + nOmegasHalf;
                                dataBox({i, j, o}) = static_cast<std::complex<float_64>>((*retField)[i][j][oSigned]);
                            }
                        }
                    }

                    return retBufferF;
                }


                /** Calculate angular frequency in SI units
                 *
                 * @param i frequency index for trimmed plugin array
                 *
                 * @return angular frequency in SI units
                 */
                float_X omega(int i) const
                {
                    float_X const actualStep = dt;
                    return 2.0_X * float_X(PI) * (float_X(i) - float_X(pluginNumT) / 2.0_X) / float_X(pluginNumT)
                        / actualStep;
                }


                //! Return size of trimmed arrays in omega dimension
                int getNumOmegas() const
                {
                    PMACC_VERIFY_MSG(
                        getOmegaMaxIndex() > getOmegaMinIndex(),
                        "Shadowgraphy: omega max <= omega min is not allowed!");
                    return 2 * (getOmegaMaxIndex() - getOmegaMinIndex());
                }

                /** Return omega index for a non trimmed frequency array.
                 * Used for properly treating raw Fourier space data of this plugin.
                 *
                 * @param i frequency index for trimmed plugin array
                 *
                 * @return index for non-trimmed array in frequency domain
                 */
                int getOmegaIndex(int i) const
                {
                    const int nOmegasHalf = numOmegas / 2;
                    if(i < nOmegasHalf)
                    {
                        return duration / params::tRes - getOmegaMinIndex() - nOmegasHalf + i + 1;
                    }
                    else
                    {
                        return (i % (nOmegasHalf)) + getOmegaMinIndex();
                    }
                }

                //! Return size of trimmed arrays in omega dimension
                int getNumT() const
                {
                    return duration / params::tRes;
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

                //! Returns openPMD data structure names for detector directions
                std::string dataLabelsFieldComponent(int index) const
                {
                    const int localIndex = index / 2;
                    const std::string dataLabelList[] = {
                        "Ex",
                        "Ey",
                        "Bx",
                        "By",
                    };
                    return dataLabelList[localIndex];
                }

            private:
                /**
                 * Initialize FFTW, supposed to be called once at the start of the plugin loop.
                 *
                 * This function allocates memory for the input and output arrays for the FFT transforms,
                 * and creates the FFTW plans for transverse fft for real to complex and transverse ifft
                 * for complex to complex.
                 */
                void initFFTW()
                {
                    // Allocate memory for the input and output arrays for the FFT transforms
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


                //! Free the memory allocated for the input and output arrays of FFTW.
                void freeFFTW()
                {
                    fftw_free(fftwInF);
                    fftw_free(fftwOutF);
                    fftw_free(fftwInB);
                    fftw_free(fftwOutB);
                }

                /** Perform an inverse Fourier transform into time domain of both the electric and magnetic field
                 * and then perform a time integration to generate a 2D image out of the 3D array.
                 *
+                * @param ExOmegaPropagated vector of Ex-fields in (x, y, omega)-domain
+                * @param EyOmegaPropagated vector of Ey-fields in (x, y, omega)-domain
+                * @param BxOmegaPropagated vector of Bx-fields in (x, y, omega)-domain
+                * @param ByOmegaPropagated vector of By-fields in (x, y, omega)-domain
                 */
                void computeShadowgram(
                    vec3c const& ExOmegaPropagated,
                    vec3c const& EyOmegaPropagated,
                    vec3c const& BxOmegaPropagated,
                    vec3c const& ByOmegaPropagated)
                {
                    // Loop over all timesteps
                    for(int t = 0; t < pluginNumT; ++t)
                    {
                        float_64 const tSI = t * int(params::tRes) * float_64(picongpu::sim.si.getDt());

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
                                                         / (sim.si.getMue0() * pluginNumT * pluginNumT * pluginNumX
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

                //! Returns the minimum index for trimmed arrays in the omega dimension
                int getOmegaMinIndex() const
                {
                    double const stepSize = params::tRes * sim.si.getDt();
                    int const tmpIndex = static_cast<int>(
                        std::floor(pluginNumT * ((stepSize * params::omegaWfMin) / (2.0 * PI) + 0.5)));
                    return std::max(tmpIndex, pluginNumT / 2 + 1);
                }

                //! Return maximum omega index for trimmed arrays in the omega dimension.
                int getOmegaMaxIndex() const
                {
                    float_64 const actualStep = params::tRes * sim.si.getDt();
                    int const tmpIndex
                        = math::ceil(pluginNumT * ((actualStep * params::omegaWfMax) / (2.0 * PI) + 0.5));
                    return std::min(tmpIndex, pluginNumT) + 1;
                }


                /** x component of k vector in SI units for FFTs
                 *
                 * @param i k vector index
                 *
                 * @return x component of k vector in SI units
                 */
                float_X kx(int i) const
                {
                    float_X const actualStep = params::xRes * sim.si.getCellSize().x();
                    return 2.0_X * float_X(PI) * (float_X(i) - float_X(pluginNumX) / 2.0_X) / float_X(pluginNumX)
                        / actualStep;
                }

                /** y component of k vector in SI units for FFTs
                 *
                 * @param i k vector index
                 *
                 * @return y component of k vector in SI units
                 */
                float_X ky(int i) const
                {
                    float_X const actualStep = params::yRes * sim.si.getCellSize().y();
                    return 2.0_X * float_X(PI) * (float_X(i) - float_X(pluginNumY) / 2.0_X) / float_X(pluginNumY)
                        / actualStep;
                }
            }; // class Helper
        } // namespace shadowgraphy
    } // namespace plugins
} // namespace picongpu

#endif
