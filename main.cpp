//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//  Quantum Monte Carlo Simulation for Kitaev Models
//  written by: Tim Eschmann, June 2016
//  Modified version: February 2019
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <sstream>
#include <mpi.h>
#include <cmath>

#include "simulation.hpp"

using namespace arma;

int main(int argc, char *argv[])
{
    int me, np;                         // MPI rank, number of ranks

    //////////////////////////////////////////////////////////////
    // Simulation parameters:  
    //////////////////////////////////////////////////////////////

    int sweeps = 100;               // # of simulation sweeps
    int therm = 100;               // # of Thermalization sweeps
    int sweeps_per_swap = 1;            // What is says ...
    int sweeps_per_save = 100;         // ...
    double T_min = 6e-2;                // Lowest simulation temperature
    double T_max = 1e-1;                  // etc.
    // Temperature distribution
    // Choose from "external", "lin", "log", "double_log":
    std::string dist = "log";
    
    /// Further declarations:
    double temp;

    //////////////////////////////////////////////////////////////
    // MPI Initialization:
    //////////////////////////////////////////////////////////////
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    //////////////////////////////////////////////////////////////
    // Calculate temperature (-> hpp/temp.hpp):

    if (me == 0)
        temp = 10000; // dummy value
    else
        temp = calc_temp(T_min, T_max, me, np, dist);

    std::stringstream output_name;
    output_name << "mc.temp_" << temp <<".h5";

    Simulation sim(np, me, T_min, T_max, temp, dist, output_name.str());

    if (me == 0) // Master process (only needed for parallel tempering)
    {
        sim.master(therm, sweeps, sweeps_per_save, sweeps_per_swap);
    }
    else // Replica Monte Carlo processes
    { 
        sim.run(sweeps, therm, sweeps_per_swap, sweeps_per_save);
    }

    //////////////////////////////////////////////////////////////    

    MPI_Finalize();
    
    return 0;
}



