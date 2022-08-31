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
                vec3c Ex_omega;
                vec3c Ey_omega;
                vec3c Bx_omega;
                vec3c By_omega;

                // Arrays for propagated fields
                vec3c Ex_omega_propagated;
                vec3c Ey_omega_propagated;
                vec3c Bx_omega_propagated;
                vec3c By_omega_propagated;

                vec2r shadowgram;

                // Size of arrays
                int n_x, n_y;
                int omega_min_index, omega_max_index, n_omegas;

                float_64 movingWindowCorrection;

                // Variables for omega calculations @TODO some initializations and bla
                float dt;
                int nt;
                int duration;

                int nGpus;
                int cellsPerGpu;

                int y_mw_stop_offset;
                float left_border;

                float propagationDistance;

                bool isSlidingWindowActive;

                int startSlideCount;

                int amount_of_moving_windows = -1;

                int yTotalMinIndex, yTotalMaxIndex;

            public:
                // Constructor of the shadowgraphy helper class
                // To be called at the first time step when the shadowgraphy time integration starts
                Helper(int currentStep, float slicePoint, float focusPos, int duration):
                    duration(duration)
                {
                    const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();

                    pmacc::math::Size_t<simDim> globalGridSize = subGrid.getGlobalDomain().size;
                    std::cout << "globalgridsize: " << subGrid.getGlobalDomain().toString() << std::endl;


                    // Same amount of omegas as ts 
                    // @TODO int division
                    // n_omegas = params::omega_n;
                    omega_min_index = fourierhelper::get_omega_min_index(duration);
                    omega_max_index = fourierhelper::get_omega_max_index(duration);
                    n_omegas = fourierhelper::get_n_omegas(duration); //omega_max_index - omega_min_index;
                    
                    printf("minindex: %d, maxindex: %d, n: %d \n", omega_min_index, omega_max_index, n_omegas);

                    dt = params::t_res * SI::DELTA_T_SI;
                    nt = duration / params::t_res;

                    propagationDistance = focusPos;
                    printf("propagationDistance = %e \n", propagationDistance);

                    pmacc::GridController<simDim>& con = pmacc::Environment<simDim>::get().GridController();
                    nGpus = con.getGpuNodes()[1]; //((pmacc::math::Size_t<simDim>) con.getGpuNodes())[1];
                    cellsPerGpu = int(globalGridSize[1] / nGpus);
                    printf("nGpus = %d\n",nGpus);

                    n_x = globalGridSize[0] / params::x_res - 2;
                    int const cellsUntilIntegrationPlane = slicePoint * globalGridSize[2];

                    startSlideCount = MovingWindow::getInstance().getSlideCounter(currentStep);

                    // This is currently not allowed to change during plugin run!
                    isSlidingWindowActive = MovingWindow::getInstance().isSlidingWindowActive(currentStep);
                    bool const isSlidingWindowEnabled = MovingWindow::getInstance().isEnabled();

                    int yWindowSize;

                    if(isSlidingWindowEnabled){
                        movingWindowCorrection =  cellsUntilIntegrationPlane * SI::CELL_DEPTH_SI + nt * dt * float_64(SI::SPEED_OF_LIGHT_SI);
                        yWindowSize = int((float(nGpus - 1) / float(nGpus)) * globalGridSize[1]);
                    } else {
                        movingWindowCorrection = 0.0;
                        yWindowSize = globalGridSize[1];
                    }

                    n_y = math::floor(( yWindowSize - movingWindowCorrection / SI::CELL_HEIGHT_SI) / (params::y_res) - 2);

                    printf("n_y: %d\n", n_y);
                    printf("yWindowSize: %d\n", yWindowSize);
                    printf("movingWindowCorrection: %e\n", movingWindowCorrection);
                    printf("movingWindowCorrection: %d\n", int(movingWindowCorrection / SI::CELL_HEIGHT_SI));
                    int yGlobalOffset = (MovingWindow::getInstance().getWindow(currentStep).globalDimensions.offset)[1];
                    int yTotalOffset = int(startSlideCount * globalGridSize[1] / nGpus);


                    // Make sure spatial grid is even
                    n_x = n_x % 2 == 0 ? n_x : n_x - 1;
                    n_y = n_y % 2 == 0 ? n_y : n_y - 1;

                    // The total domain indices of the integration slice are constant, because the screen is not co-propagating with the moving window
                    yTotalMinIndex = yTotalOffset + yGlobalOffset + math::floor(movingWindowCorrection / SI::CELL_HEIGHT_SI) - 1; // yTotalMaxIndex - n_y;
                    yTotalMaxIndex = yTotalMinIndex + n_y; // yTotalOffset + yGlobalOffset + n_y - 1;
                    printf("movingWindowCorrection: %e\n", movingWindowCorrection);
                    printf("n_y: %d\n", n_y);
                    printf("ytotalmaxindex: %d\n", yTotalMaxIndex);
                    printf("ytotalminindex: %d\n", yTotalMinIndex);

                    std::cout << "initialized with "<< n_x << ", " << n_y << std::endl;

                    printf("nGpus: %d, (%e) \n", nGpus, (float(nGpus - 1) / float(nGpus)) * globalGridSize[1]);
                    printf("things: %e\n",  movingWindowCorrection / SI::CELL_HEIGHT_SI);

                    // Initialization of storage arrays
                    Ex_omega = vec3c(n_x, vec2c(n_y, vec1c(n_omegas)));
                    Ey_omega = vec3c(n_x, vec2c(n_y, vec1c(n_omegas)));
                    Bx_omega = vec3c(n_x, vec2c(n_y, vec1c(n_omegas)));
                    By_omega = vec3c(n_x, vec2c(n_y, vec1c(n_omegas)));

                    Ex_omega_propagated = vec3c(n_x, vec2c(n_y, vec1c(n_omegas)));
                    Ey_omega_propagated = vec3c(n_x, vec2c(n_y, vec1c(n_omegas)));
                    Bx_omega_propagated = vec3c(n_x, vec2c(n_y, vec1c(n_omegas)));
                    By_omega_propagated = vec3c(n_x, vec2c(n_y, vec1c(n_omegas)));

                    tmpEx = vec2r(n_x, vec1r(n_y));
                    tmpEy = vec2r(n_x, vec1r(n_y));
                    tmpBx = vec2r(n_x, vec1r(n_y));
                    tmpBy = vec2r(n_x, vec1r(n_y));

                    shadowgram = vec2r(n_x, vec1r(n_y));

                    //printf("min lambda= %e, max lambda = %e \n", params::min_lambda, params::max_lambda);
                    printf("probe omega = %e \n", params::probe_omega);
                    printf("tbp omega = %e \n", params::delta_omega);
                    printf("min omega wf = %e, max omega wf = %e \n", params::omega_min_wf, params::omega_max_wf);
                    printf("min omega = %e, max omega = %e \n", params::omega_min, params::omega_max);

                    init_fftw();
                }

                // Destructor of the shadowgraphy helper class
                // To be called at the last time step when the shadowgraphy time integration ends
                ~Helper()
                {
                    fftw_free(fftw_in_f);
                    fftw_free(fftw_out_f);
                    fftw_free(fftw_in_b);
                    fftw_free(fftw_out_b);
                }

                // Store fields in helper class with proper resolution
                template<typename F>
                void store_field(int t, int currentStep, pmacc::container::HostBuffer<float3_64, 2>* fieldBuffer1, pmacc::container::HostBuffer<float3_64, 2>* fieldBuffer2)
                {   
                    int currentSlideCount = MovingWindow::getInstance().getSlideCounter(currentStep);
                    printf("currentslidecount = %d\n", currentSlideCount);
                    //printf("fieldbuffer1 size: %zu \n", fieldBuffer1->origin().getCurrentSize());
                    //printf("fieldbuffer2 size: %zu \n", fieldBuffer2->origin().getCurrentSize());
                    //std::cout << "fieldbuffer1 size: " << fieldBuffer1.getCurrentSize() <<std::endl;
                    //std::cout << "fieldbuffer2 size: " << fieldBuffer2.getCurrentSize() <<std::endl;

                    for(int i = 0; i < n_x; i++){
                        int const grid_i = i * params::x_res;
                        for(int j = 0; j < n_y; ++j){
                            // Transform the total coordinates of the fixed shadowgraphy screen to the global coordinates of the field-buffers
                            int const grid_j = yTotalMinIndex - currentSlideCount * cellsPerGpu + j * params::y_res;

                            printf("grid_i = %d, grid_j = %d", grid_i, grid_j);
                            
                            //float_64 const wf = masks::position_wf(i, j, n_x, n_y) * masks::t_wf(t, duration);
                            float_64 const wf = 1.0;

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
                            printf("  done \n"); 
                        }
                    }
                }

                void calculate_dft(int t)
                {
                    float_64 const tSI = t * int(params::t_res) * float_64(picongpu::SI::DELTA_T_SI);

                    for(int o = 0; o < n_omegas; ++o){
                        int const omegaIndex = fourierhelper::get_omega_index(o, duration);
                        float_64 const omegaSI = omega(omegaIndex);

                        complex_64 const phase = complex_64(0, omegaSI * tSI);
                        complex_64 const exponential = math::exp(phase);

                        for(int i = 0; i < n_x; ++i){
                            for(int j = 0; j < n_y; ++j){
                                Ex_omega[i][j][o] += tmpEx[i][j] * exponential;
                                Ey_omega[i][j][o] += tmpEy[i][j] * exponential;
                                Bx_omega[i][j][o] += tmpBx[i][j] * exponential;
                                By_omega[i][j][o] += tmpBy[i][j] * exponential;
                            }
                        }
                    }
                }

                void propagate_fields()
                {
                    for(int fieldindex = 0; fieldindex < 4; fieldindex++){
                        for(int o = 0; o < n_omegas; ++o){
                            
                            int const omegaIndex = fourierhelper::get_omega_index(o, duration);
                            float_64 const omegaSI = omega(omegaIndex);
                            float_64 const kSI = omegaSI / float_64(SI::SPEED_OF_LIGHT_SI) ;
                            printf("omegaIndex: %d \n", omegaIndex);

                            // put field into fftw array
                            for(int i = 0; i < n_x; ++i){
                                for(int j = 0; j < n_y; ++j){
                                int const index = i + j * n_x;

                                if(fieldindex == 0){
                                    fftw_in_f[index][0] = Ex_omega[i][j][o].real();
                                    fftw_in_f[index][1] = Ex_omega[i][j][o].imag();
                                } else if(fieldindex == 1){
                                    fftw_in_f[index][0] = Ey_omega[i][j][o].real();
                                    fftw_in_f[index][1] = Ey_omega[i][j][o].imag();
                                } else if(fieldindex == 2){
                                    fftw_in_f[index][0] = Bx_omega[i][j][o].real();
                                    fftw_in_f[index][1] = Bx_omega[i][j][o].imag();
                                } else if(fieldindex == 3){
                                    fftw_in_f[index][0] = By_omega[i][j][o].real();
                                    fftw_in_f[index][1] = By_omega[i][j][o].imag();
                                }

                                }
                            }
                            //writeIntermediateFile(o, fieldindex);

                            fftw_execute(plan_forward);
                            writeFourierFile(o, fieldindex, false);

                            // put field into fftw array
                            for(int i = 0; i < n_x; ++i){
                                // Put origin into center of array with this, necessary due to FFT
                                int const i_ffs = (i + n_x/2) % n_x;

                                for(int j = 0; j < n_y; ++j){
                                    int const index = i + j * n_x;
                                    int const j_ffs = (j  + n_y / 2) % n_y;
                                    
                                    float_64 const sqrt1 = kSI * kSI;
                                    float_64 const sqrt2 = kx(i) * kx(i);
                                    float_64 const sqrt3 = ky(j) * ky(j);
                                    float_64 const sqrtContent = (kSI == 0.0) ? 0.0 : 1 - sqrt2 / sqrt1 - sqrt3 / sqrt1;
                                    
                                    if(sqrtContent >= 0.0)
                                    {
                                        // Put origin into center of array with this, necessary due to FFT
                                        int const index_ffs = i_ffs + j_ffs * n_x;

                                        complex_64 const field = complex_64(fftw_out_f[index_ffs][0], fftw_out_f[index_ffs][1]);

                                        float_64 const phase = (kSI == 0.0) ? 0.0 : propagationDistance * kSI * math::sqrt(sqrtContent);
                                        complex_64 const propagator = math::exp(complex_64(0, phase));

                                        complex_64 const propagated_field = masks::mask_f(kx(i), ky(j), omega(omegaIndex)) * field * propagator;

                                        fftw_in_b[index][0] = propagated_field.real();
                                        fftw_in_b[index][1] = propagated_field.imag();
                                    } else {
                                        fftw_in_b[index][0] = 0.0;
                                        fftw_in_b[index][1] = 0.0;
                                    }
                                }
                            }

                            writeFourierFile(o, fieldindex, true);
                            fftw_execute(plan_backward);

                            // yoink fields from fftw array
                            for(int i = 0; i < n_x; ++i){
                                for(int j = 0; j < n_y; ++j){
                                int const index = i + j * n_x;

                                if(fieldindex == 0){
                                    Ex_omega_propagated[i][j][o] = complex_64(fftw_out_b[index][0], fftw_out_b[index][1]);
                                } else if(fieldindex == 1){
                                    Ey_omega_propagated[i][j][o] = complex_64(fftw_out_b[index][0], fftw_out_b[index][1]);
                                } else if(fieldindex == 2){
                                    Bx_omega_propagated[i][j][o] = complex_64(fftw_out_b[index][0], fftw_out_b[index][1]);
                                } else if(fieldindex == 3){
                                    By_omega_propagated[i][j][o] = complex_64(fftw_out_b[index][0], fftw_out_b[index][1]);
                                }
                                }
                            }
                        }
                    }
                }

                void calculate_shadowgram()
                {
                    // Loop through all omega
                    for(int t = 0; t < nt; ++t){
                        float_64 const tSI = t * int(params::t_res) * float_64(picongpu::SI::DELTA_T_SI);

                        printf("step %d (from %d) of backwards integration \n", t, nt);

                        // Initialization of storage arrays
                        vec2c Ex_tmpsum = vec2c(n_x, vec1c(n_y));
                        vec2c Ey_tmpsum = vec2c(n_x, vec1c(n_y));
                        vec2c Bx_tmpsum = vec2c(n_x, vec1c(n_y));
                        vec2c By_tmpsum = vec2c(n_x, vec1c(n_y));

                        for(int o = 0; o < n_omegas; ++o){
                            int const omegaIndex = fourierhelper::get_omega_index(o, duration);
                            float_64 const omegaSI = omega(omegaIndex);
                                
                            complex_64 const phase = complex_64(0, -tSI * omegaSI);
                            complex_64 const exponential = math::exp(phase);

                            for(int i = 0; i < n_x; ++i){
                                for(int j = 0; j < n_y; ++j){
                                    complex_64 const Ex = Ex_omega_propagated[i][j][o] * exponential;
                                    complex_64 const Ey = Ey_omega_propagated[i][j][o] * exponential;
                                    complex_64 const Bx = Bx_omega_propagated[i][j][o] * exponential;
                                    complex_64 const By = By_omega_propagated[i][j][o] * exponential;
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

                vec2r get_shadowgram()
                {
                    return shadowgram;
                }


                int get_n_x(){
                    return n_x;
                }

                int get_n_y(){
                    return n_y;
                }

            private:
                // Initialize fftw memory things, supposed to be called once per plugin loop
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


                void writeFourierFile(int o, int fieldindex, bool masksapplied)
                {
                    int const omegaIndex = fourierhelper::get_omega_index(o, duration);
                    std::ofstream outFile;
                    std::ostringstream filename;

                    if(fieldindex == 0){
                        filename <<"Ex";
                    } else if(fieldindex == 1){
                        filename <<"Ey";
                    } else if(fieldindex == 2){
                        filename <<"Bx";
                    } else if(fieldindex == 3){
                        filename <<"By";
                    }

                    filename << "_fourierspace";
                    if(masksapplied){
                        filename << "_with_masks";
                    }
                    filename << "_" << omegaIndex << ".dat";

                    outFile.open(filename.str(), std::ofstream::out | std::ostream::trunc);

                    if(!outFile)
                    {
                        std::cerr << "Can't open file [" << filename.str() << "] for output, disable plugin output. Chuchu"
                                << std::endl;
                    }
                    else
                    {
                        for( unsigned int i = 0; i < get_n_x(); ++i ) // over all x
                        {
                            int const i_ffs = (i + n_x/2) % n_x;
                            for(unsigned int j = 0;  j < get_n_y(); ++j) // over all y
                            {
                                int const index = i + j * n_x;
                                int const j_ffs = (j + n_y / 2) % n_y;
                                int const index_ffs = i_ffs + j_ffs * n_x;
                                if(!masksapplied){
                                    outFile << fftw_out_f[index_ffs][0] << "+" << fftw_out_f[index_ffs][1] << "j" << "\t";
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
                            std::cerr << "Error on flushing file [" << filename.str() << "]. " << std::endl;

                        outFile.close();
                    }
                }

                void writeIntermediateFile(int o, int fieldindex)
                {
                    int const omegaIndex = fourierhelper::get_omega_index(o, duration);
                    std::ofstream outFile;
                    std::ostringstream filename;

                    if(fieldindex == 0){
                        filename <<"Ex";
                    } else if(fieldindex == 1){
                        filename <<"Ey";
                    } else if(fieldindex == 2){
                        filename <<"Bx";
                    } else if(fieldindex == 3){
                        filename <<"By";
                    }

                    filename << "_omegaspace";
                    filename << "_" << omegaIndex << ".dat";

                    outFile.open(filename.str(), std::ofstream::out | std::ostream::trunc);

                    if(!outFile)
                    {
                        std::cerr << "Can't open file [" << filename.str() << "] for output, disable plugin output. Chuchu"
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
                            std::cerr << "Error on flushing file [" << filename.str() << "]. " << std::endl;

                        outFile.close();
                    }
                    //}
                }

                float_64 omega(int i){
                    int const actual_n = nt;
                    float const actual_step = dt;
                    return 2.0 * PI  * (float(i) - float(actual_n) / 2.0) / float(actual_n) / actual_step;
                }
                
                // kx so that it is the proper kx for FFTs
                float_64 kx(int i){ // @TODO x_n
                    int const actual_n = n_x;
                    float const actual_step = params::x_res * SI::CELL_WIDTH_SI;
                    return 2.0 * PI  * (float(i) - float(actual_n) / 2.0) / float(actual_n) / actual_step;
                }

                // ky so that it is the proper ky for FFTs
                float_64 ky(int i){ // @TODO y_n
                    int const actual_n = n_y;
                    float const actual_step = params::y_res * SI::CELL_HEIGHT_SI;
                    return 2.0 * PI  * (float(i) - float(actual_n) / 2.0) / float(actual_n) / actual_step;
                }

                

            }; // class Helper
        } // namespace shadowgraphy
    } // namespace plugins
} //namespace picongpu