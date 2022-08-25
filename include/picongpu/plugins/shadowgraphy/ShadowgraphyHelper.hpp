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
                vec2r tmp_Ex, tmp_Ey;
                vec2r tmp_Bx, tmp_By;

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

                float slicepoint;
                float_64 movingWindowCorrection;

                // Variables for omega calculations @TODO some initializations and bla
                float dt;
                int nt;
                int duration;

                int n_z;
                int ngpus;
                float cellspergpu;
                float mwstartStep;
                float mwstart;
                float globalgridsizey;

                int y_mw_stop_offset;
                float left_border;

                float delta_z;

                float domega;

                int startTime;
                bool isSlidingWindowActive;

                int amount_of_moving_windows = -1;

            public:
                // Constructor of the shadowgraphy helper class
                // To be called at the first time step when the shadowgraphy time integration starts
                Helper(float slicepoint, float mwstart, float focuspos, int duration, int pluginstart, int movingwindowstop):
                    duration(duration),
                    startTime(pluginstart),
                    slicepoint(slicepoint),
                    mwstart(mwstart)
                {
                    const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();

                    pmacc::math::Size_t<simDim> globalGridSize = subGrid.getGlobalDomain().size;
                    std::cout << "globalgridsize: " << subGrid.getGlobalDomain().to_string() << std::endl;

                    n_x = globalGridSize[0] / params::x_res - 2;
                    globalgridsizey = float(globalGridSize[1]);
                    // Same amount of omegas as ts 
                    // @TODO int division
                    // n_omegas = params::omega_n;
                    omega_min_index = fourierhelper::get_omega_min_index(duration);
                    omega_max_index = fourierhelper::get_omega_max_index(duration);
                    n_omegas = fourierhelper::get_n_omegas(duration); //omega_max_index - omega_min_index;
                    
                    printf("minindex: %d, maxindex: %d, n: %d \n", omega_min_index, omega_max_index, n_omegas);

                    dt = params::t_res * SI::DELTA_T_SI;
                    nt = duration / params::t_res;

                    delta_z = focuspos;
                    printf("deltaz = %e \n", delta_z);

                    pmacc::GridController<simDim>& con = pmacc::Environment<simDim>::get().GridController();
                    //pmacc::math::Size_t<simDim> gpuDim = (pmacc::math::Size_t<simDim>) con.getGpuNodes();
                    //vec::Size_t<simDim> globalGridSize = gpuDim * field.size();

                    ngpus = con.getGpuNodes()[1];//((pmacc::math::Size_t<simDim>) con.getGpuNodes())[1];
                    printf("ngpus = %d\n",ngpus);

                    //time step when moving window started:
                    mwstartStep = (mwstart * float(globalGridSize[1]) * (ngpus - 1) / float(ngpus) / SI::SPEED_OF_LIGHT_SI) / SI::DELTA_T_SI;
                    

                    domega = math::abs(omega(1) - omega(0));

                    cellspergpu = float(globalGridSize[1]) / float(ngpus);

                    n_z = slicepoint * globalGridSize[2];

                    // This is currently not allowed to change during plugin run!
                    isSlidingWindowActive = MovingWindow::getInstance().isSlidingWindowActive(pluginstart);
                    bool const isSlidingWindowEnabled = MovingWindow::getInstance().isEnabled();

                    if(isSlidingWindowEnabled){
                        printf("pluginstart %d\n", pluginstart);

                        DataSpace<3> localDomainOffset(subGrid.getLocalDomain().offset);
                        printf("gdo 0: %d\n", localDomainOffset[0]);
                        printf("gdo 1: %d\n", localDomainOffset[1]);
                        printf("gdo 2: %d\n", localDomainOffset[2]);
                        printf("gdo 3: %d\n", localDomainOffset[3]);
                        std::cout << subGrid.getLocalDomain().globalDimensions.toString() << std::endl;

                        DataSpace<3> movingWindowGlobalOffset(MovingWindow::getInstance().getWindow(pluginstart).globalDimensions.offset);
                        printf("gdo 0: %d\n", movingWindowGlobalOffset[0]);
                        printf("gdo 1: %d\n", movingWindowGlobalOffset[1]);
                        printf("gdo 2: %d\n", movingWindowGlobalOffset[2]);
                        printf("gdo 3: %d\n", movingWindowGlobalOffset[3]);

                        printf("current slides: %d\n", MovingWindow::getInstance().getSlideCounter(pluginstart));

                        // moving window is enabled
                        if(isSlidingWindowActive){
                            //moving window is active
                        } else {

                        }

                        printf("what are we doing here\n");
                        movingWindowCorrection =  n_z * SI::CELL_DEPTH_SI + nt * dt * float_64(SI::SPEED_OF_LIGHT_SI);

                        int yWindowSize = int((float(ngpus - 1) / float(ngpus)) * globalGridSize.y());
                        n_y = math::ceil(( yWindowSize - movingWindowCorrection / SI::CELL_HEIGHT_SI) / (params::y_res) - 2);
                        int yGlobalOffset = (MovingWindow::getInstance().getWindow(pluginstart).globalDimensions.offset)[1];
                        int yTotalOffset = int(MovingWindow::getInstance().getSlideCounter(pluginstart) * globalGridSize[1] / ngpus);

                        // The total domain indices of the integration slice are constant, because the screen is not co-propagating with the moving window
                        int yTotalMaxIndex = yTotalOffset + yGlobalOffset + yWindowSize;
                        int yTotalMinIndex = yTotalMaxIndex - n_y;
                        y_mw_stop_offset = 0;
                    } else {
                        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                        //subGrid.getGlobalDomain();
                        DataSpace<3> globalDomainOffset(subGrid.getGlobalDomain().offset);
                        printf("gdo 0: %.4e\n", globalDomainOffset[0]);
                        printf("gdo 1: %.4e\n", globalDomainOffset[1]);
                        printf("gdo 2: %.4e\n", globalDomainOffset[2]);
                        printf("this should appear\n");
                        // this means that moving window was never on
                        n_y = globalGridSize.y() / params::y_res - 2;
                        y_mw_stop_offset = 0;
                        left_border = 0.0;
                    }

                    if(false){
                        // movingWindowCorrection makes the resulting shadowgram smaller if the moving Window is enabled
                        // The resulting loss in the size of the shadowgram comes from the duration of the time integration
                        movingWindowCorrection =  n_z * SI::CELL_DEPTH_SI + nt * dt * float_64(SI::SPEED_OF_LIGHT_SI);
                        printf("1: %e \n", n_z * SI::CELL_DEPTH_SI);
                        printf("2: %e \n", nt * dt * float_64(SI::SPEED_OF_LIGHT_SI));
                        // @TODO
                        n_y = math::ceil(( (float(ngpus - 1) / float(ngpus)) * globalGridSize[1] - movingWindowCorrection / SI::CELL_HEIGHT_SI) / (params::y_res) - 2);
                        PMACC_ASSERT_MSG(n_y > 0, "n_y must be larger than 0, your moving window goes too fast brrrr \n");
                        printf("moving window enabled \n");
                    } else {
                        if(movingwindowstop == 0){
                            // this means that moving window was never on
                            n_y = globalGridSize.y() / params::y_res - 2;
                            y_mw_stop_offset = 0;
                        } else {
                            printf("moving window stopped at %d \n", movingwindowstop);
                            n_y = (float(ngpus - 1) / float(ngpus)) * globalGridSize[1] / params::y_res - 2;
                            y_mw_stop_offset = math::fmod((movingwindowstop * SI::DELTA_T_SI * SI::SPEED_OF_LIGHT_SI 
                                                            - mwstart * globalgridsizey * SI::CELL_HEIGHT_SI * float(ngpus - 1) / float(ngpus)) 
                                                            / SI::CELL_HEIGHT_SI, cellspergpu);
                            printf("ymw_stop_offset %e\n", y_mw_stop_offset);
                            printf("1 %e\n", movingwindowstop * SI::DELTA_T_SI * SI::SPEED_OF_LIGHT_SI);
                            printf("2 %e\n", mwstart * globalgridsizey * SI::CELL_HEIGHT_SI * float(ngpus - 1) / float(ngpus));
                            printf("3 %e\n", (movingwindowstop * SI::DELTA_T_SI * SI::SPEED_OF_LIGHT_SI - mwstart * globalgridsizey * SI::CELL_HEIGHT_SI * float(ngpus - 1) / float(ngpus)) / SI::CELL_HEIGHT_SI);
                            printf("4 %e\n", cellspergpu);
                            printf("ymw_stop_offset %f \n", y_mw_stop_offset);
                            printf("ymw_stop_offset %e \n", y_mw_stop_offset);
                        }
                    }

                    // Make sure spatial grid is even
                    n_x = n_x % 2 == 0 ? n_x : n_x - 1;
                    n_y = n_y % 2 == 0 ? n_y : n_y - 1;

                    std::cout << "initialized with "<< n_x << ", " << n_y << std::endl;

                    printf("ngpus: %d, (%e) \n", ngpus, (float(ngpus - 1) / float(ngpus)) * globalGridSize[1]);
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

                    tmp_Ex = vec2r(n_x, vec1r(n_y));
                    tmp_Ey = vec2r(n_x, vec1r(n_y));
                    tmp_Bx = vec2r(n_x, vec1r(n_y));
                    tmp_By = vec2r(n_x, vec1r(n_y));

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
                    float const tmax_sim_c = SI::SPEED_OF_LIGHT_SI * SI::DELTA_T_SI * (startTime + duration);
                    float const mwsy = mwstart * globalgridsizey * SI::CELL_HEIGHT_SI * float(ngpus - 1) / float(ngpus);

                    float const lost_index_from_sw = math::fmod((tmax_sim_c - mwsy) / SI::CELL_HEIGHT_SI, cellspergpu);

                    float const jumped_gpu_cells = (math::floor((tmax_sim_c - mwsy) / SI::CELL_HEIGHT_SI / cellspergpu) 
                                                - math::floor((currentStep * SI::SPEED_OF_LIGHT_SI * SI::DELTA_T_SI - mwsy) / SI::CELL_HEIGHT_SI / cellspergpu)) * cellspergpu;// / params::y_res;

                    for(int i = 0; i < n_x; ++i){
                        int const grid_i = i * params::x_res;
                        for(int j = 0; j < n_y; ++j){
                            if(isSlidingWindowActive){
                                float const gridPos = float(j * params::y_res) + jumped_gpu_cells + lost_index_from_sw;
                                float const  wr = math::fmod(gridPos, 1.0);
                                float const wl = 1.0 - wr;
                                int const grid_j = math::floor(gridPos);

                                float_64 const wf = masks::position_wf(i, j, n_x, n_y) * masks::t_wf(t, duration);

                                if(F::getName() == "E"){
                                    if(t == 0 && j == 0 && i == 0){
                                        printf("grid_j: %d, j: %d \t", grid_j, j);
                                        //printf("gridPos: %e, nzthing: %e, ntf: %e, nti: %e \n", gridPos, n_z * SI::CELL_DEPTH_SI, nt * SI::SPEED_OF_LIGHT_SI * dt, (float_64(nt - 1.0 - t) / float_64(nt - 1.0)));
                                        printf("jumped gpu cells: %e, gridpos: %e \n", jumped_gpu_cells, gridPos);
                                        printf("end time: %d", startTime + duration);
                                        printf("lost index: %f", lost_index_from_sw);
                                    } else if( t == 0 && j == n_y-1 && i == 0) {
                                        printf("grid_j: %d, j: %d \t", grid_j, j);
                                        printf("jumped gpu cells: %e, gridpos: %e \n", jumped_gpu_cells, gridPos);
                                        //printf("gridPos: %e, nzthing: %e, ntf: %e, nti: %e  -- end \n", gridPos, n_z * SI::CELL_DEPTH_SI, nt * SI::SPEED_OF_LIGHT_SI * dt,(float_64(nt - 1.0 - t) / float_64(nt - 1.0)));
                                    } else if(j == 0 && i == 0){
                                        printf("grid_j: %d, j: %d \t", grid_j, j);
                                        printf("jumped gpu cells: %e, gridpos: %e \n", jumped_gpu_cells, gridPos);
                                        //printf("gridPos: %f \n", gridPos);
                                    } else if(j == n_y-1 && i == 0) {
                                        printf("grid_j: %d, j: %d \t", grid_j, j);
                                        printf("jumped gpu cells: %e, gridpos: %e \n", jumped_gpu_cells, gridPos);
                                        //printf("gridPos: %f \n", gridPos);
                                    }
                                    float_64 const Ex011 = (*(fieldBuffer2->origin()(grid_i, grid_j+1))).x(); //
                                    float_64 const Ex111 = (*(fieldBuffer2->origin()(grid_i+1, grid_j+1))).x(); //
                                    float_64 const Ex021 = (*(fieldBuffer2->origin()(grid_i, grid_j+2))).x();  //
                                    float_64 const Ex121 = (*(fieldBuffer2->origin()(grid_i+1, grid_j+2))).x(); //
                                    tmp_Ex[i][j] = wf * ( wl * (Ex011 + Ex111) + wr * (Ex021 + Ex121) ) / 2.0;
                                    
                                    if(wr < 0.5){
                                        float_64 const wrEy = wr + 0.5;
                                        float_64 const wlEy = wl - 0.5;
                                        float_64 const Ey101 = (*(fieldBuffer2->origin()(grid_i+1, grid_j))).y(); //
                                        float_64 const Ey111 = (*(fieldBuffer2->origin()(grid_i+1, grid_j+1))).y(); //
                                        tmp_Ey[i][j] = wf * (wlEy * Ey101 + wrEy * Ey111);
                                    } else {
                                        float_64 const wrEy = wr - 0.5;
                                        float_64 const wlEy = wl + 0.5;
                                        float_64 const Ey111 = (*(fieldBuffer2->origin()(grid_i+1, grid_j+1))).y();
                                        float_64 const Ey121 = (*(fieldBuffer2->origin()(grid_i+1, grid_j+2))).y();
                                        tmp_Ey[i][j] = wf * (wlEy * Ey111 + wrEy * Ey121);
                                    }
                                } else {
                                    if(wr < 0.5){
                                        float_64 wrBx = wr + 0.5;
                                        float_64 wlBx = wl - 0.5;
                                        float_64 const Bx101 = (*(fieldBuffer2->origin()(grid_i+1, grid_j))).x(); //
                                        float_64 const Bx111 = (*(fieldBuffer2->origin()(grid_i+1, grid_j+1))).x(); //
                                        float_64 const Bx100 = (*(fieldBuffer1->origin()(grid_i+1, grid_j))).x(); //
                                        float_64 const Bx110 = (*(fieldBuffer1->origin()(grid_i+1, grid_j+1))).x(); //

                                        tmp_Bx[i][j] = wf * (wlBx * (Bx100 + Bx101) + wrBx * (Bx111 + Bx110)) / 2.0;
                                    } else {
                                        float_64 wrBx = wr - 0.5;
                                        float_64 wlBx = wl + 0.5;
                                        float_64 const Bx111 = (*(fieldBuffer2->origin()(grid_i+1, grid_j+1))).x();
                                        float_64 const Bx121 = (*(fieldBuffer2->origin()(grid_i+1, grid_j+2))).x();
                                        float_64 const Bx110 = (*(fieldBuffer1->origin()(grid_i+1, grid_j+1))).x();
                                        float_64 const Bx120 = (*(fieldBuffer1->origin()(grid_i+1, grid_j+2))).x();

                                        tmp_Bx[i][j] = wf * (wlBx * (Bx110 + Bx111) + wrBx * (Bx121 + Bx120)) / 2.0;
                                    }

                                    float_64 const By111 = (*(fieldBuffer2->origin()(grid_i+1, grid_j+1))).y(); //
                                    float_64 const By011 = (*(fieldBuffer2->origin()(grid_i, grid_j+1))).y(); //
                                    float_64 const By121 = (*(fieldBuffer2->origin()(grid_i+1, grid_j+2))).y(); //
                                    float_64 const By021 = (*(fieldBuffer2->origin()(grid_i, grid_j+2))).y(); //
                                    float_64 const By110 = (*(fieldBuffer1->origin()(grid_i+1, grid_j+1))).y(); //
                                    float_64 const By010 = (*(fieldBuffer1->origin()(grid_i, grid_j+1))).y(); //
                                    float_64 const By120 = (*(fieldBuffer1->origin()(grid_i+1, grid_j+2))).y(); //
                                    float_64 const By020 = (*(fieldBuffer1->origin()(grid_i, grid_j+2))).y(); //

                                    tmp_By[i][j] = wf * (wl * (By111 + By011 + By110 + By010) + wr * (By121 + By021 + By120 + By020)) / 4.0;
                                }
                            } else {
                                int const grid_j = j * params::y_res + int(math::floor(y_mw_stop_offset));
                                if((t == 0  || t == duration - 1) && (i == 0) && (j == 0)){
                                    printf("slidecounter %d", MovingWindow::getInstance().getSlideCounter(currentStep));
                                    printf("grid_j, mw off, %d \n", grid_j);
                                    printf("ymwoffset 2: %d\n", y_mw_stop_offset);
                                    printf("ymwoffset 3: %e\n", y_mw_stop_offset);
                                    printf("ymwoffset 4: %f\n", y_mw_stop_offset);
                                }

                                float_64 const wf = masks::position_wf(i, j, n_x, n_y) * masks::t_wf(t, duration);

                                // fix yee offset
                                if(F::getName() == "E"){
                                    tmp_Ex[i][j] = wf * ((*(fieldBuffer2->origin()(grid_i, grid_j+1))).x() 
                                                       + (*(fieldBuffer2->origin()(grid_i+1, grid_j+1))).x()) / 2.0; 
                                    tmp_Ey[i][j] = wf * ((*(fieldBuffer2->origin()(grid_i+1, grid_j))).y() 
                                                       + (*(fieldBuffer2->origin()(grid_i+1, grid_j+1))).y()) / 2.0;
                                } else {
                                    tmp_Bx[i][j] = wf * ((*(fieldBuffer1->origin()(grid_i+1, grid_j))).x() 
                                                       + (*(fieldBuffer1->origin()(grid_i+1, grid_j+1))).x() 
                                                       + (*(fieldBuffer2->origin()(grid_i+1, grid_j))).x() 
                                                       + (*(fieldBuffer2->origin()(grid_i+1, grid_j+1))).x()) / 4.0; 
                                    tmp_By[i][j] = wf * ((*(fieldBuffer1->origin()(grid_i, grid_j+1))).y() 
                                                       + (*(fieldBuffer1->origin()(grid_i+1, grid_j+1))).y() 
                                                       + (*(fieldBuffer2->origin()(grid_i, grid_j+1))).y() 
                                                       + (*(fieldBuffer2->origin()(grid_i+1, grid_j+1))).y()) / 4.0;
                                } 
                            }
                        }
                    }
                    
                }

                void calculate_dft(int t)
                {
                    float_64 const t_SI = t * int(params::t_res) * float_64(picongpu::SI::DELTA_T_SI);

                    for(int o = 0; o < n_omegas; ++o){
                        int const omegaIndex = fourierhelper::get_omega_index(o, duration);
                        float_64 const omega_SI = omega(omegaIndex);

                        complex_64 const phase = complex_64(0, omega_SI * t_SI);
                        complex_64 const exponential = math::exp(phase);

                        for(int i = 0; i < n_x; ++i){
                            for(int j = 0; j < n_y; ++j){
                                Ex_omega[i][j][o] += tmp_Ex[i][j] * exponential;
                                Ey_omega[i][j][o] += tmp_Ey[i][j] * exponential;
                                Bx_omega[i][j][o] += tmp_Bx[i][j] * exponential;
                                By_omega[i][j][o] += tmp_By[i][j] * exponential;
                            }
                        }
                    }
                }

                void propagate_fields()
                {
                    for(int fieldindex = 0; fieldindex < 4; fieldindex++){
                        for(int o = 0; o < n_omegas; ++o){
                            
                            int const omegaIndex = fourierhelper::get_omega_index(o, duration);
                            float_64 const omega_SI = omega(omegaIndex);
                            float_64 const k_SI = omega_SI / float_64(SI::SPEED_OF_LIGHT_SI) ;
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
                                    
                                    float_64 const sqrt1 = k_SI * k_SI;
                                    float_64 const sqrt2 = kx(i) * kx(i);
                                    float_64 const sqrt3 = ky(j) * ky(j);
                                    float_64 const sqrtContent = (k_SI == 0.0) ? 0.0 : 1 - sqrt2 / sqrt1 - sqrt3 / sqrt1;
                                    
                                    if(sqrtContent >= 0.0)
                                    {
                                        // Put origin into center of array with this, necessary due to FFT
                                        int const index_ffs = i_ffs + j_ffs * n_x;

                                        complex_64 const field = complex_64(fftw_out_f[index_ffs][0], fftw_out_f[index_ffs][1]);

                                        float_64 const phase = (k_SI == 0.0) ? 0.0 : delta_z * k_SI * math::sqrt(sqrtContent);
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
                        float_64 const t_SI = t * int(params::t_res) * float_64(picongpu::SI::DELTA_T_SI);

                        printf("step %d (from %d) of backwards integration \n", t, nt);

                        // Initialization of storage arrays
                        vec2c Ex_tmpsum = vec2c(n_x, vec1c(n_y));
                        vec2c Ey_tmpsum = vec2c(n_x, vec1c(n_y));
                        vec2c Bx_tmpsum = vec2c(n_x, vec1c(n_y));
                        vec2c By_tmpsum = vec2c(n_x, vec1c(n_y));

                        for(int o = 0; o < n_omegas; ++o){
                            int const omegaIndex = fourierhelper::get_omega_index(o, duration);
                            float_64 const omega_SI = omega(omegaIndex);
                                
                            complex_64 const phase = complex_64(0, -t_SI * omega_SI);
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