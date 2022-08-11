#pragma once

#include <fftw3.h>

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
        
                // Size of arrays
                int n_x, n_y, n_omegas;

                // Variables for omega calculations @TODO some initializations and bla
                float dt;
                int nt;

                float delta_z;

            public:
                // Constructor of the shadowgraphy helper class
                // To be called at the first time step when the shadowgraphy time integration starts
                Helper(int n_x, int n_y, float delta_z):
                    n_x(n_x),
                    n_y(n_y),
                    delta_z(delta_z)
                {
                }

                // Destructor of the shadowgraphy helper class
                // To be called at the last time step when the shadowgraphy time integration ends
                ~Helper()
                {
                }
                
                // Energy flux calculation loop
                template<typename T>
                void calculate_energy_flux(T const& sim_E, T const& sim_B, int t, bool is_first_summand)
                /**
                sim_E: E from simulation, real 2d array
                sim_B: B from simulation, real 2d array
                t: current time step, from 0 to (nt-1)
                is_first_summand: bool, true if first part of poynting vector, false if second part of poynting vector
                **/
                {
                    
                }

                // Calculate the shadowgram after the independent energy fluxes have been calculated for all time steps
                vec2r calculate_shadowgram()
                {
                    vec2r shadowgram(n_x, vec1c(n_y));

                    // Loop through all omega
                    for(int i = 0; i < n_x; i++){
                        for(int j = 0; j < n_y; j++){
                            for(int o = 0; o < n_omegas; ++o)
                            {
                                // shadowgram(x, y) += real(EF1(x, y, zo, omega, tmax)) - real(EF2(x, y, zo, omega, tmax))
                                shadowgram[i][j] += (energydensity_ExBy[i][j][o] - energydensity_EyBx[i][j][o]).real();
                            }
                        }
                    }

                    return shadowgram;
                }

            private:

            }
        }
    }
}