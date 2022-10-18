 #pragma once

#include <fftw3.h>
#include <pmacc/algorithms/math/defines/pi.hpp>
#include "picongpu/simulation_defines.hpp"
#include <cmath> // what
#include "pmacc/assert.hpp"
#include <stdio.h>
#include <pmacc/mappings/simulation/GridController.hpp>

#include <pmacc/math/Vector.hpp>
#include <pmacc/math/vector/Float.hpp>
#include <pmacc/math/vector/Int.hpp>
#include <pmacc/math/vector/Size_t.hpp>
#include "picongpu/simulation/control/Window.hpp"

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

                typedef std::vector< std::vector< std::vector< complex_64 > > > vec3c;
                typedef std::vector< std::vector< complex_64 > > vec2c;
                typedef std::vector< complex_64 > vec1c;
                typedef std::vector< std::vector< std::vector< float_64 > > > vec3r;
                typedef std::vector< std::vector< float_64 > > vec2r;
                typedef std::vector< float_64 > vec1r;

                // Arrays to store Ex, Ey, Bx and Bz per time step temporarily
                vec2r tmpEx, tmpEy;
                vec2r tmpBx, tmpBy;

                // Arrays for FFTW
                fftw_complex *fftw_in_f; // @TODO: Can this be real? Issue is forward / backward FFT
                fftw_complex *fftw_out_f;
                fftw_complex *fftw_in_b;
                fftw_complex *fftw_out_b;

                fftw_plan plan_forward;
                fftw_plan plan_backward;

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
                int n_x, n_y;
                int omega_min_index, omega_max_index, n_omegas;

                // Variables for omega calculations @TODO some initializations and bla
                float dt;
                int nt;
                int duration;

                int cellsPerGpu;

                float propagationDistance;

                bool isSlidingWindowActive;

                int yTotalMinIndex, yTotalMaxIndex;

            public:
                /** Constructor of shadowgraphy helper class
                 * To be called at the first timestep when the shadowgraphy time integration starts
                 *
                 * @param currentStep current simulation timestep
                 * @param slicePoint value between 0.0 and 1.0 where the fields are extracted
                 * @param focusPos focus position of lens system, e.g. the propagation distance of the vacuum propagator
                 * @param duration duration of time extraction in simulation time steps
                 */
                Helper(int currentStep, float slicePoint, float focusPos, int duration):
                    duration(duration)
                {
                    dt = params::t_res * SI::DELTA_T_SI;
                    nt = duration / params::t_res;

                    // Same amount of omegas as ts 
                    // @TODO int division
                    // n_omegas = params::omega_n;
                    omega_min_index = get_omega_min_index();
                    omega_max_index = get_omega_max_index();
                    n_omegas = get_n_omegas(); //omega_max_index - omega_min_index;
                    
                    printf("minindex: %d, maxindex: %d, n: %d \n", omega_min_index, omega_max_index, n_omegas);

                    propagationDistance = focusPos;
                    printf("propagationDistance = %e \n", propagationDistance);
                    
                    const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();

                    pmacc::math::Size_t<simDim> globalGridSize = subGrid.getGlobalDomain().size;
                    std::cout << "globalgridsize: " << subGrid.getGlobalDomain().toString() << std::endl;

                    pmacc::GridController<simDim>& con = pmacc::Environment<simDim>::get().GridController();
                    int const nGpus = con.getGpuNodes()[1]; //((pmacc::math::Size_t<simDim>) con.getGpuNodes())[1];
                    cellsPerGpu = int(globalGridSize[1] / nGpus);
                    printf("nGpus = %d\n",nGpus);

                    n_x = globalGridSize[0] / params::x_res - 2;

                    int const startSlideCount = MovingWindow::getInstance().getSlideCounter(currentStep);

                    // These are currently not allowed to change during plugin run!
                    isSlidingWindowActive = MovingWindow::getInstance().isSlidingWindowActive(currentStep);
                    bool const isSlidingWindowEnabled = MovingWindow::getInstance().isEnabled();

                    // If the sliding window is enabled, the resulting shadowgram will be smaller to not show the "invisible" GPU fields in the shadowgram
                    int yWindowSize;
                    if(isSlidingWindowEnabled){
                        yWindowSize = int((float(nGpus - 1) / float(nGpus)) * globalGridSize[1]);
                    } else {
                        yWindowSize = globalGridSize[1];
                    }

                    // If the sliding window is active, the resulting shadowgram will also be smaller to adjust for the laser propagation distance
                    float slidingWindowCorrection;
                    if(isSlidingWindowActive){
                        int const cellsUntilIntegrationPlane = slicePoint * globalGridSize[2];
                        slidingWindowCorrection =  cellsUntilIntegrationPlane * SI::CELL_DEPTH_SI + nt * dt * float_64(SI::SPEED_OF_LIGHT_SI);
                    } else {
                        slidingWindowCorrection = 0.0;
                    }

                    n_y = math::floor(( yWindowSize - slidingWindowCorrection / SI::CELL_HEIGHT_SI) / (params::y_res) - 2);

                    // Make sure spatial grid is even
                    n_x = n_x % 2 == 0 ? n_x : n_x - 1;
                    n_y = n_y % 2 == 0 ? n_y : n_y - 1;

                    printf("n_y: %d\n", n_y);
                    printf("yWindowSize: %d\n", yWindowSize);
                    printf("slidingWindowCorrection: %e\n", slidingWindowCorrection);
                    printf("slidingWindowCorrection: %d\n", int(slidingWindowCorrection / SI::CELL_HEIGHT_SI));
                    int const yGlobalOffset = (MovingWindow::getInstance().getWindow(currentStep).globalDimensions.offset)[1];
                    int const yTotalOffset = int(startSlideCount * globalGridSize[1] / nGpus);

                    // The total domain indices of the integration slice are constant, because the screen is not co-propagating with the moving window
                    yTotalMinIndex = yTotalOffset + yGlobalOffset + math::floor(slidingWindowCorrection / SI::CELL_HEIGHT_SI) - 1; // yTotalMaxIndex - n_y;
                    yTotalMaxIndex = yTotalMinIndex + n_y; // yTotalOffset + yGlobalOffset + n_y - 1;
                    printf("slidingWindowCorrection: %e\n", slidingWindowCorrection);
                    printf("n_y: %d\n", n_y);
                    printf("ytotalmaxindex: %d\n", yTotalMaxIndex);
                    printf("ytotalminindex: %d\n", yTotalMinIndex);

                    std::cout << "initialized with "<< n_x << ", " << n_y << std::endl;

                    printf("nGpus: %d, (%e) \n", nGpus, (float(nGpus - 1) / float(nGpus)) * globalGridSize[1]);
                    printf("things: %e\n",  slidingWindowCorrection / SI::CELL_HEIGHT_SI);

                    // Initialization of storage arrays
                    ExOmega = vec3c(n_x, vec2c(n_y, vec1c(n_omegas)));
                    EyOmega = vec3c(n_x, vec2c(n_y, vec1c(n_omegas)));
                    BxOmega = vec3c(n_x, vec2c(n_y, vec1c(n_omegas)));
                    ByOmega = vec3c(n_x, vec2c(n_y, vec1c(n_omegas)));

                    ExOmegaPropagated = vec3c(n_x, vec2c(n_y, vec1c(n_omegas)));
                    EyOmegaPropagated = vec3c(n_x, vec2c(n_y, vec1c(n_omegas)));
                    BxOmegaPropagated = vec3c(n_x, vec2c(n_y, vec1c(n_omegas)));
                    ByOmegaPropagated = vec3c(n_x, vec2c(n_y, vec1c(n_omegas)));

                    tmpEx = vec2r(n_x, vec1r(n_y));
                    tmpEy = vec2r(n_x, vec1r(n_y));
                    tmpBx = vec2r(n_x, vec1r(n_y));
                    tmpBy = vec2r(n_x, vec1r(n_y));

                    shadowgram = vec2r(n_x, vec1r(n_y));

                    //printf("min lambda= %e, max lambda = %e \n", params::min_lambda, params::max_lambda);
                    printf("min omega wf = %e, max omega wf = %e \n", params::omega_min_wf, params::omega_max_wf);
                    printf("min omega = %e, max omega = %e \n", params::omega_min, params::omega_max);

                    init_fftw();
                }

                /** Destructor of the shadowgraphy helper class
                 * To be called at the last time step when the shadowgraphy time integration ends
                 */
                ~Helper()
                {
                    fftw_free(fftw_in_f);
                    fftw_free(fftw_out_f);
                    fftw_free(fftw_in_b);
                    fftw_free(fftw_out_b);
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
                void store_field(int t, int currentStep, 
                                 pmacc::container::HostBuffer<float3_64, 2>* fieldBuffer1, 
                                 pmacc::container::HostBuffer<float3_64, 2>* fieldBuffer2)
                {   
                    int const currentSlideCount = MovingWindow::getInstance().getSlideCounter(currentStep);
                    printf("currentslidecount = %d\n", currentSlideCount);

                    for(int i = 0; i < n_x; i++){
                        int const grid_i = i * params::x_res;
                        for(int j = 0; j < n_y; ++j){
                            // Transform the total coordinates of the fixed shadowgraphy screen to the global coordinates of the field-buffers
                            int const grid_j = yTotalMinIndex - currentSlideCount * cellsPerGpu + j * params::y_res;

                            //if(j == (n_y - 1)){
                            //    printf("grid_i = %d, grid_j = %d\n", grid_i, grid_j);
                            //}
                            
                            float_64 const wf = masks::position_wf(i, j, n_x, n_y) * masks::t_wf(t, duration);
                            //float_64 const wf = 1.0;

                            // fix yee offset
                            if(F::getName() == "E"){
                                tmpEx[i][j] = wf * ((*(fieldBuffer2->origin()(grid_i, grid_j+1))).x() 
                                                    + (*(fieldBuffer2->origin()(grid_i+1, grid_j+1))).x()) / 2.0; 
                                tmpEy[i][j] = wf * ((*(fieldBuffer2->origin()(grid_i+1, grid_j))).y() 
                                                    + (*(fieldBuffer2->origin()(grid_i+1, grid_j+1))).y()) / 2.0;
                            } else {
                                tmpBx[i][j] = wf * ((*(fieldBuffer1->origin()(grid_i+1, grid_j))).x() 
                                                    + (*(fieldBuffer1->origin()(grid_i+1, grid_j+1))).x() 
                                                    + (*(fieldBuffer2->origin()(grid_i+1, grid_j))).x() 
                                                    + (*(fieldBuffer2->origin()(grid_i+1, grid_j+1))).x()) / 4.0; 
                                tmpBy[i][j] = wf * ((*(fieldBuffer1->origin()(grid_i, grid_j+1))).y() 
                                                    + (*(fieldBuffer1->origin()(grid_i+1, grid_j+1))).y() 
                                                    + (*(fieldBuffer2->origin()(grid_i, grid_j+1))).y() 
                                                    + (*(fieldBuffer2->origin()(grid_i+1, grid_j+1))).y()) / 4.0;
                            }
                            //printf("  done \n"); 
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
                    float_64 const tSI = t * int(params::t_res) * float_64(picongpu::SI::DELTA_T_SI);

                    for(int o = 0; o < n_omegas; ++o){
                        int const omegaIndex = get_omega_index(o);
                        float_64 const omegaSI = omega(omegaIndex);

                        complex_64 const phase = complex_64(0, omegaSI * tSI);
                        complex_64 const exponential = math::exp(phase);

                        for(int i = 0; i < n_x; ++i){
                            for(int j = 0; j < n_y; ++j){
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
                 * with a FFT into $(k_\perp, \omega)$-domain, applying the defined masks in the .param files, multiplying with 
                 * the propagator $\exp^{i \Delta z k \sqrt{\omega^2/c^2 - k_x^2 - k_y^2}}$ and transforming it back into
                 * $(x, y, \omega)$-domain.
                 */
                void propagate_fields()
                {
                    for(int fieldIndex = 0; fieldIndex < 4; fieldIndex++){
                        for(int o = 0; o < n_omegas; ++o){
                            
                            int const omegaIndex = get_omega_index(o);
                            float_64 const omegaSI = omega(omegaIndex);
                            float_64 const kSI = omegaSI / float_64(SI::SPEED_OF_LIGHT_SI) ;
                            printf("omegaIndex: %d \n", omegaIndex);

                            // put field into fftw array
                            for(int i = 0; i < n_x; ++i){
                                for(int j = 0; j < n_y; ++j){
                                int const index = i + j * n_x;

                                if(fieldIndex == 0){
                                    fftw_in_f[index][0] = ExOmega[i][j][o].real();
                                    fftw_in_f[index][1] = ExOmega[i][j][o].imag();
                                } else if(fieldIndex == 1){
                                    fftw_in_f[index][0] = EyOmega[i][j][o].real();
                                    fftw_in_f[index][1] = EyOmega[i][j][o].imag();
                                } else if(fieldIndex == 2){
                                    fftw_in_f[index][0] = BxOmega[i][j][o].real();
                                    fftw_in_f[index][1] = BxOmega[i][j][o].imag();
                                } else if(fieldIndex == 3){
                                    fftw_in_f[index][0] = ByOmega[i][j][o].real();
                                    fftw_in_f[index][1] = ByOmega[i][j][o].imag();
                                }

                                }
                            }
                            //writeIntermediateFile(o, fieldIndex);

                            fftw_execute(plan_forward);
                            writeFourierFile(o, fieldIndex, false);

                            // put field into fftw array
                            for(int i = 0; i < n_x; ++i){
                                // Put origin into center of array with this, necessary due to FFT
                                int const iffs = (i + n_x/2) % n_x;

                                for(int j = 0; j < n_y; ++j){
                                    int const index = i + j * n_x;
                                    int const jffs = (j  + n_y / 2) % n_y;
                                    
                                    float_64 const sqrt1 = kSI * kSI;
                                    float_64 const sqrt2 = kx(i) * kx(i);
                                    float_64 const sqrt3 = ky(j) * ky(j);
                                    float_64 const sqrtContent = (kSI == 0.0) ? 0.0 : 1 - sqrt2 / sqrt1 - sqrt3 / sqrt1;
                                    
                                    if(sqrtContent >= 0.0)
                                    {
                                        // Put origin into center of array with this, necessary due to FFT
                                        int const indexffs = iffs + jffs * n_x;

                                        complex_64 const field = complex_64(fftw_out_f[indexffs][0], fftw_out_f[indexffs][1]);

                                        float_64 const phase = (kSI == 0.0) ? 0.0 : propagationDistance * kSI * math::sqrt(sqrtContent);
                                        complex_64 const propagator = math::exp(complex_64(0, phase));

                                        complex_64 const propagatedField = masks::mask_f(kx(i), ky(j), omega(omegaIndex)) * field * propagator;

                                        fftw_in_b[index][0] = propagatedField.real();
                                        fftw_in_b[index][1] = propagatedField.imag();
                                    } else {
                                        fftw_in_b[index][0] = 0.0;
                                        fftw_in_b[index][1] = 0.0;
                                    }
                                }
                            }

                            writeFourierFile(o, fieldIndex, true);
                            fftw_execute(plan_backward);

                            // yoink fields from fftw array
                            for(int i = 0; i < n_x; ++i){
                                for(int j = 0; j < n_y; ++j){
                                    int const index = i + j * n_x;

                                    if(fieldIndex == 0){
                                        ExOmegaPropagated[i][j][o] = complex_64(fftw_out_b[index][0], fftw_out_b[index][1]);
                                    } else if(fieldIndex == 1){
                                        EyOmegaPropagated[i][j][o] = complex_64(fftw_out_b[index][0], fftw_out_b[index][1]);
                                    } else if(fieldIndex == 2){
                                        BxOmegaPropagated[i][j][o] = complex_64(fftw_out_b[index][0], fftw_out_b[index][1]);
                                    } else if(fieldIndex == 3){
                                        ByOmegaPropagated[i][j][o] = complex_64(fftw_out_b[index][0], fftw_out_b[index][1]);
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
                    for(int t = 0; t < nt; ++t){
                        float_64 const tSI = t * int(params::t_res) * float_64(picongpu::SI::DELTA_T_SI);

                        printf("step %d (from %d) of backwards integration \n", t, nt);

                        // Initialization of storage arrays
                        vec2c Ex_tmpsum = vec2c(n_x, vec1c(n_y));
                        vec2c Ey_tmpsum = vec2c(n_x, vec1c(n_y));
                        vec2c Bx_tmpsum = vec2c(n_x, vec1c(n_y));
                        vec2c By_tmpsum = vec2c(n_x, vec1c(n_y));

                        // DFT loop to time domain
                        for(int o = 0; o < n_omegas; ++o){
                            int const omegaIndex = get_omega_index(o);
                            float_64 const omegaSI = omega(omegaIndex);
                                
                            complex_64 const phase = complex_64(0, -tSI * omegaSI);
                            complex_64 const exponential = math::exp(phase);

                            for(int i = 0; i < n_x; ++i){
                                for(int j = 0; j < n_y; ++j){
                                    complex_64 const Ex = ExOmegaPropagated[i][j][o] * exponential;
                                    complex_64 const Ey = EyOmegaPropagated[i][j][o] * exponential;
                                    complex_64 const Bx = BxOmegaPropagated[i][j][o] * exponential;
                                    complex_64 const By = ByOmegaPropagated[i][j][o] * exponential;
                                    shadowgram[i][j] += (dt / (SI::MUE0_SI * nt * nt * n_x * n_x * n_y * n_y)) * (Ex * By - Ey * Bx 
                                                        + Ex_tmpsum[i][j] * By + Ex * By_tmpsum[i][j]
                                                        - Ey_tmpsum[i][j] * Bx - Ey * Bx_tmpsum[i][j]).real();

                                    if(o < (n_omegas - 1)){
                                        Ex_tmpsum[i][j] += Ex;
                                        Ey_tmpsum[i][j] += Ey; 
                                        Bx_tmpsum[i][j] += Bx;
                                        By_tmpsum[i][j] += By;
                                    }
                                } 
                            }
                            
                        }
                    }
                }

                //! Get shadowgram as a 2D image
                vec2r get_shadowgram() const{
                    return shadowgram;
                }

                //! Get amount of shadowgram pixels in x direction
                int get_n_x() const{
                    return n_x;
                }

                //! Get amount of shadowgram pixels in y direction
                int get_n_y() const{
                    return n_y;
                }

            private:
                //! Initialize fftw memory things, supposed to be called once at the start of the plugin loop
                void init_fftw()
                {
                    std::cout << "init fftw" << std::endl;
                    // Input and output arrays for the FFT transforms
                    fftw_in_f = fftw_alloc_complex(n_x * n_y);
                    fftw_out_f = fftw_alloc_complex(n_x * n_y);

                    fftw_in_b = fftw_alloc_complex(n_x * n_y);
                    fftw_out_b = fftw_alloc_complex(n_x * n_y);

                    // Create fftw plan for transverse fft for real to complex
                    // Many ffts will be performed -> use FFTW_MEASURE as flag
                    plan_forward = fftw_plan_dft_2d(n_y, n_x, fftw_in_f, fftw_out_f, FFTW_FORWARD, FFTW_MEASURE);

                    // Create fftw plan for transverse ifft for complex to complex
                    // Even more iffts will be performed -> use FFTW_MEASURE as flag
                    plan_backward = fftw_plan_dft_2d(n_y, n_x, fftw_in_b, fftw_out_b, FFTW_BACKWARD, FFTW_MEASURE);

                    std::cout << "fftw inited" << std::endl;
                }

                /** Store fields in helper class with proper resolution and fixed Yee offset in (k_x, k_y, \omega)-domain
                 *
                 * @param o omega index from trimmed array in plugin
                 * @param fieldIndex value from 0 to 3 (Ex, Ey, Bx, By)
                 * @param masksApplied have masks been applied in Fourier space yet
                 */
                void writeFourierFile(int o, int fieldIndex, bool masksApplied)
                {
                    int const omegaIndex = get_omega_index(o);
                    std::ofstream outFile;
                    std::ostringstream fileName;

                    if(fieldIndex == 0){
                        fileName <<"Ex";
                    } else if(fieldIndex == 1){
                        fileName <<"Ey";
                    } else if(fieldIndex == 2){
                        fileName <<"Bx";
                    } else if(fieldIndex == 3){
                        fileName <<"By";
                    }

                    fileName << "_fourierspace";
                    if(masksApplied){
                        fileName << "_with_masks";
                    }
                    fileName << "_" << omegaIndex << ".dat";

                    outFile.open(fileName.str(), std::ofstream::out | std::ostream::trunc);

                    if(!outFile)
                    {
                        std::cerr << "Can't open file [" << fileName.str() << "] for output, disable plugin output. Chuchu"
                                << std::endl;
                    }
                    else
                    {
                        for( unsigned int i = 0; i < get_n_x(); ++i ) // over all x
                        {
                            int const iffs = (i + n_x/2) % n_x;
                            for(unsigned int j = 0;  j < get_n_y(); ++j) // over all y
                            {
                                int const index = i + j * n_x;
                                int const jffs = (j + n_y / 2) % n_y;
                                int const indexffs = iffs + jffs * n_x;
                                if(!masksApplied){
                                    outFile << fftw_out_f[indexffs][0] << "+" << fftw_out_f[indexffs][1] << "j" << "\t";
                                }
                                else {
                                    outFile << fftw_in_b[index][0] << "+" << fftw_in_b[index][1] << "j" << "\t";
                                }
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

                /** Store fields in helper class with proper resolution and fixed Yee offset in (x, y, \omega)-domain directly 
                 * after loading the field
                 *
                 * @param o omega index from trimmed array in plugin
                 * @param fieldIndex value from 0 to 3 (Ex, Ey, Bx, By)
                 */
                void writeIntermediateFile(int o, int fieldIndex)
                {
                    int const omegaIndex = get_omega_index(o);
                    std::ofstream outFile;
                    std::ostringstream fileName;

                    if(fieldIndex == 0){
                        fileName <<"Ex";
                    } else if(fieldIndex == 1){
                        fileName <<"Ey";
                    } else if(fieldIndex == 2){
                        fileName <<"Bx";
                    } else if(fieldIndex == 3){
                        fileName <<"By";
                    }

                    fileName << "_omegaspace";
                    fileName << "_" << omegaIndex << ".dat";

                    outFile.open(fileName.str(), std::ofstream::out | std::ostream::trunc);

                    if(!outFile)
                    {
                        std::cerr << "Can't open file [" << fileName.str() << "] for output, disable plugin output. Chuchu"
                                << std::endl;
                    }
                    else
                    {
                        for( unsigned int i = 0; i < get_n_x(); ++i ) // over all x
                        {
                            for(unsigned int j = 0;  j < get_n_y(); ++j) // over all y
                            {
                                int const index = i + j * n_x;
                                outFile << fftw_in_f[index][0] << "+" << fftw_in_f[index][1] << "j" << "\t";
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
                int get_omega_min_index() const {
                    float_64 const actual_step = params::t_res * SI::DELTA_T_SI;
                    int const tmpindex = math::floor(nt * ((actual_step * params::omega_min_wf) / (2.0 * PI) + 0.5));
                    int retindex = tmpindex > nt / 2 + 1 ? tmpindex : nt / 2 + 1;
                    return retindex;
                }

                //! Return maximum omega index for trimmed arrays in omega dimension
                int get_omega_max_index() const {
                    float_64 const actual_step = params::t_res * SI::DELTA_T_SI;
                    int const tmpindex = math::ceil(nt * ((actual_step * params::omega_max_wf) / (2.0 * PI) + 0.5));
                    int retindex =  tmpindex <= nt ? tmpindex : nt;
                    return retindex + 1;
                }

                //! Return size of trimmed arrays in omega dimension
                int get_n_omegas() const {
                    return 2 * (get_omega_max_index() - get_omega_min_index());
                }

                /** Return omega index for a matrix that doesn't remove the zero-valued frequencies.
                 * Used for properly treating raw Fourier space data of this plugin.
                 *
                 * @param i frequency index for trimmed plugin array
                 *
                 * @return index for non-trimmed array in frequency domain
                 */
                int get_omega_index(int i) const {
                    int const n_omegas = get_n_omegas() / 2;
                    if (i < n_omegas){
                        return duration / params::t_res - get_omega_min_index() - n_omegas + i + 1;//(get_omega_min_index() % n_omegas) + i + 1;
                    } else {
                        return (i % n_omegas) + get_omega_min_index();
                    }
                }
                
                /** angular frequency in SI units
                 *
                 * @param i frequency index for trimmed plugin array
                 *
                 * @return angular frequency in SI units
                 */
                float omega(int i) const {
                    float const actual_step = dt;
                    return 2.0 * PI  * (float(i) - float(nt) / 2.0) / float(nt) / actual_step;
                }
                
                /** x component of k vector in SI units for FFTs
                 *
                 * @param i k vector index
                 *
                 * @return x component of k vector in SI units
                 */
                float kx(int i) const { 
                    float const actual_step = params::x_res * SI::CELL_WIDTH_SI;
                    return 2.0 * PI  * (float(i) - float(n_x) / 2.0) / float(n_x) / actual_step;
                }

                /** y component of k vector in SI units for FFTs
                 *
                 * @param i k vector index
                 *
                 * @return y component of k vector in SI units
                 */
                float ky(int i) const { 
                    float const actual_step = params::y_res * SI::CELL_HEIGHT_SI;
                    return 2.0 * PI  * (float(i) - float(n_y) / 2.0) / float(n_y) / actual_step;
                }
            }; // class Helper
        } // namespace shadowgraphy
    } // namespace plugins
} //namespace picongpu