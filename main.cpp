//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//  Quantum Monte Carlo Simulation for Kitaev Models
//  written by: Tim Eschmann, June 2016
//  Modified version: February 2019
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//
//  The code skeleton of this file was derived from "ising_skeleton.cpp"   
//  which is part of the ALPS libraries:
//
/*****************************************************************************
 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations
 *
 * ALPS Libraries
 *
 * Copyright (C) 2003 by Brigitte Surer
 *                       and Jan Gukelberger
 *
 * This software is part of the ALPS libraries, published under the ALPS
 * Library License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 * 
 * You should have received a copy of the ALPS Library License along with
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
 * available from http://alps.comp-phys.org/.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 *
/*****************************************************************************/

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



