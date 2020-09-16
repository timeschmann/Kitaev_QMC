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
/*****************************************************************************
//  This software incorporates the Armadillo C++ Library
//  Armadillo C++ Linear Algebra Library
//  Copyright 2008-2020 Conrad Sanderson (http://conradsanderson.id.au)
//  Copyright 2008-2016 National ICT Australia (NICTA)
//  Copyright 2017-2020 Arroyo Consortium
//  Copyright 2017-2020 Data61, CSIRO

//  This product includes software developed by Conrad Sanderson (http://conradsanderson.id.au)
//  This product includes software developed at National ICT Australia (NICTA)
//  This product includes software developed at Arroyo Consortium
//  This product includes software developed at Data61, CSIRO

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <sstream>
#include <vector>
#include <complex>
#include <cmath>
#include <armadillo>
#include <mpi.h>

#include <alps/scheduler/montecarlo.h>
#include <alps/alea.h>
#include <boost/random.hpp>
#include <boost/multi_array.hpp>
#include <boost/filesystem.hpp>

// Include functions package
#include "functions.hpp"

// Include the desired lattice (read in header file from .hpp folder)
#include "Lattices/honeycomb_8_sites.hpp"

#ifndef _temp_hpp_
#define _temp_hpp_
double calc_temp(double T_min, double T_max, int me, int np, std::string dist);
#endif

using namespace arma;

class Simulation
{
public:
    Simulation(int np, int me, double T_min, double T_max, double T, std::string dist, std::string output_file)
    // Define interaction matrix A, parameters and measurable observables:
    :   eng_(3*me) // Random generator engine (different seed for each replica)
    ,   rng_(eng_, dist_) // Random generator
    ,   np_(np) // # of processes
    ,   me_(me) // process number / parallelization rank
    ,   T_min(T_min) // minimal temperature
    ,   T_max(T_max) // maximal temperature
    ,   dist(dist) // Temperature distribution
    ,   temp_(T)  // Replica Temperature
    ,   beta_(1/T) // Inverse replica Temperature
    ,   ham() // Interaction matrix / Z2 configuration 
    ,   N_() // # system sites (IMPORTANT: THIS HAS CHANGED W.R.T. FORMER VERSIONS !!!)
    ,   v_() // Vector with coordinates of nonzero matrix entries
    ,   length_() // Length of this vector
    ,   eigval() // Eigenvalues of interaction matrix
    ,   F() // Free energy
    ,   plaquettes() // matrix with elementary plaquettes
    ,   energy_("E") // Measurement data: energy
    ,   e2_("E2") // Measurement data: squared energy
    //,   e4_("E4") // Measurement data: energy^4
    ,   dE_db("dE_db") // Measurement data: dE / d(beta) = fermionic part of specific heat
    //,   p_("p") // Disorder
    ,   flux_real("Flreal") // Measurement data: average plaquet flux (real part)
    ,   flux_imag("Flimag") // "" (imaginary part)
    ,   flux_real_squared("Flreal2") // Measurement data: average plaquet flux (real part)
    ,   flux_imag_squared("Flimag2") // "" (imaginary part)
    ,   spin_corr("Spin_corr")
    ,   flip_rate() // Single flip acceptance rate
    ,   filename_(output_file) // Filename for data saving
      
    {  
    
    // Locate nonzero entries of interaction matrix (.hpp file):
    v_ = non_zeros();
    length_ = v_.size();
    
    // Fill interaction matrix A with coefficients due to lattice symmetry (.hpp file)
    ham = randomize(def_matrix());

    N_ = size(ham)[0];

    // Vector of eigenvalues of A
    eigval = eig_sym(ham);

    F = free_en(eigval, beta_);

    plaquettes = create_plaquettes();

    // Initialize single flip acceptance to 0:
    flip_rate = 0;

    } 

    // Replica Monte Carlo iteration
    void run(int n, int ntherm, int sweeps_per_swap, int sweeps_per_save)
    {
        engine_type loaded, saved; // needed to load and save random engine status
        sweeps_ = n;
        thermalization_ = ntherm; // thermalization steps

        double fr;
        int n_tot = n + ntherm;
        double tau_en, tau_fl; // autocorrelation times
        
        std::stringstream matrix_output; // needed for saving configurations
        matrix_output << "matrix_temp_" << 1/beta_ << ".saved";

        std::stringstream rng_save; // needed for saving random generator status
        rng_save << "rng_" << me_ << ".saved";
         
        //Load so-far-obtained measurement data (ALPS)
        if (boost::filesystem::exists(filename_))
        {
            load(filename_);
        }
        
        // Load random generator status:
        std::ifstream rngfile(rng_save.str().c_str(), std::ifstream::in);
        if(rngfile.good())
        {
            rngfile >> loaded;
            rngfile.close();
            eng_ = loaded;
        }

        // Load last Z2 configuration from file (-> skip thermalization)
        std::ifstream matfile(matrix_output.str().c_str(), std::ifstream::in);
        if(matfile.good())
            ham.load(matrix_output.str().c_str());

        // Thermalize for ntherm steps
        if (me_ == 1)
            std::cout << "Thermalizing ..." << std::endl;
        
        while(ntherm--)
        {
            step();

            if (ntherm % sweeps_per_swap == 0)
            {
                swap(); 

                if (((ntherm + n) / sweeps_per_swap) % sweeps_per_save == 0)
                {    
                    // Calculate single flip acceptance rate and send it to Master process:
                    fr = double(flip_rate)/(double(n_tot - n - ntherm)*N_);
                    MPI_Send(&fr, 1, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
                } 
            }
            
            if (ntherm % sweeps_per_save == 0)
            {
                // Save Z2 configuration:
                ham.save(matrix_output.str().c_str());

                // Save random generator status:
                saved = eng_;
                std::ofstream file(rng_save.str().c_str(), std::ofstream::trunc);
                file << saved;
                
                // Tell that everything is saved ...
                if (me_ == 1)
                    std::cout << "SAVE " << ntherm << std::endl; 

            }
        }

        if (me_ == 1)
        {
            std::cout << "###############################" << std::endl;
            std::cout << "Sweeping ..." << std::endl;
        }

        // Run n steps
        while(n--)
        {   
            step();
            
            // Output eigenvalue and flux configurations:
            //outputZ2gauge();
            //output_eigenvalues();
            //output_flux_confs();

            // Measure observables:
            measure();

            // Swap:
            if (n % sweeps_per_swap == 0)
            {
                swap();  

                if (((n) / sweeps_per_swap) % sweeps_per_save == 0)
                {
                    // Calculate single flip acceptance rate and send it to Master process:
                    fr = double(flip_rate)/(double(n_tot - n - ntherm)*N_);
                    MPI_Send(&fr, 1, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
                }
            }
                  
            // Save all simulation data:
            if (n % sweeps_per_save == 0)
            {
                // Save results:
                save(filename_);
                
                // Save Z2 configuration:
                ham.save(matrix_output.str().c_str());

                // Save random generator status:
                saved = eng_;
                std::ofstream file(rng_save.str().c_str(), std::ofstream::trunc);
                file << saved;
                
                // Tell that everything is saved ...
                if (me_ == 1)
                    std::cout << "SAVE " << n << std::endl; 
            }
        }

        //Save the observables to file
        save(filename_);

        tau_en = energy_.tau();
        tau_fl = flux_real.tau();
        MPI_Send(&tau_en, 1, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
        MPI_Send(&tau_fl, 1, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);

     
        // Print observables      
	    /*
        std::cout << temp_ << std::endl;
	    std::cout.precision(17);
       	std::cout << energy_.name() << ":\t" << energy_.mean()
            << " +- " << energy_.error() << ";\ttau = " << energy_.tau() 
            << ";\tconverged: " << alps::convergence_to_text(energy_.converged_errors())     
            << std::endl;
        
	    std::cout << flux_real.name() << ":\t" << flux_real.mean()
            << " +- " << flux_real.error() << ";\ttau = " << flux_real.tau() 
            << ";\tconverged: " << alps::convergence_to_te        //std::cout << double(counter)/double(tries) << std::endl;xt(flux_real.converged_errors())
            << std::endl;
        std::cout << flux_imag.name() << ":\t" << flux_imag.mean()
            << " +- " << flux_imag.error() << ";\ttau = " << flux_imag.tau() 
            << ";\tconverged: " << alps::convergence_to_text(flux_imag.converged_errors())
            << std::endl;*/

    }
    
    // Iteration step (= "Metropolis sweep"): 
    void step()
    {
        int kk, i, j; // Running indices
        cx_mat ham_new(N_,N_);
        vec eigval_new(N_); // Vector for eigenvalues of new matrix (= update proposal)
        
        double F_new; // New free energy
        
        int coord1, coord2; // Random bond coordinate
        double alpha, gamma; // Monte Carlo variables
        
        int count = 0;

        // One sweep = N tries (# lattice sites)
        for (kk = 0; kk < N_; kk++)
        {       
            // Alternative matrix for sampling:
            ham_new = ham;
        
            // Switch sign of random matrix entry:
            int die = roll_die(length_);
            coord1 = v_[die]/(N_);
            coord2 = v_[die]%(N_);
            ham_new(coord1, coord2) *= -1;
            ham_new(coord2, coord1) *= -1;
            
            // Measure free energy of changed Hamiltonian:
            eigval_new = eig_sym(ham_new);    
            F_new = free_en(eigval_new, beta_);

            // Accept change with probability according to Boltzmann distribution:  
            //alpha = exp(- beta_ * (F_new - F));
            // Use Gibbs weights instead of Metropolis weights:
            alpha = 1./ (1. + exp(beta_ * (F_new - F)));
            gamma = rng_();

            // Accepted?
            if (gamma <= alpha) // accept
            {
                ham = ham_new;
                eigval = eigval_new;
                F = F_new;

                // Single flip acceptance rate + 1
                flip_rate += 1;
            }

            count += 1;
        }
    }
    
    // Does what it says ...
    void measure()
    {      
        std::complex <double> fl;
        double E_, dE_, s;
        //double p;
        double fl_real;
        double fl_imag;
        double corr;

        // Measure energy etc.:
        E_ = en(eigval, beta_);
        dE_ = diffE(eigval, beta_);
        
        // Measure average flux per plaquet / disorder:
        fl = flux(ham, plaquettes);
        fl_real = std::real(fl);
        fl_imag = std::imag(fl);
        //p = get_p(ham, plaquettes);

        // Measure spin-spin correlation:
        corr = correlation(ham, v_, beta_);

        // Add sample to observables:
        energy_ << E_/double(N_); // Energy per site
        e2_ << E_/double(N_)*E_/double(N_); // Squared energy per site
        //e4_ << E_/double(N_)*E_/double(N_)*E_/double(N_)*E_/double(N_);
        dE_db << dE_ / double(N_); // dE/d(beta)
        //p_ << p;
        flux_real << fl_real;
        flux_imag << fl_imag;
        flux_real_squared << fl_real*fl_real;
        flux_imag_squared << fl_imag*fl_imag;
        spin_corr << corr;
    }

    ///////////////////////////////////////////////////////////////
    // Output Z2 gauge field configuration:
    ///////////////////////////////////////////////////////////////

    void outputZ2gauge()
    {
        int coord1, coord2;
        std::stringstream gauge_output; // needed for saving configurations
        gauge_output << "gauge_configuration_temp_" << 1/beta_ << ".saved";

        std::ofstream gauge(gauge_output.str().c_str(), std::ofstream::app);
        for (int j = 0; j < v_.size(); j++)
        {
            coord1 = v_[j]/N_;
            coord2 = v_[j]%N_;

            gauge << std::setprecision(17) << std::imag(ham(coord1, coord2) / std::abs(ham(coord1, coord2))) << "   ";
        }
        gauge << std::endl;
    }

    
    // Output flux configurations:
    void output_flux_confs()
    {
        cx_vec fl_confs = flux_confs(ham, plaquettes);
        
        std::stringstream flux_output; // needed for saving configurations
        flux_output << "flux_configuration_temp_" << 1/beta_ << ".saved";

        std::ofstream flux(flux_output.str().c_str(), std::ofstream::app);
        for(int iii = 0; iii < size(plaquettes)[0]; iii++)
        {
            // Switch between real and imaginary fluxes:
            flux << std::setprecision(17) << std::real(fl_confs[iii]) << "   ";
            //flux << std::setprecision(17) << std::imag(fl_confs[iii]) << std::endl;
        }
        flux << std::endl;
    }

    // Output eigenvalue configurations:
    void output_eigenvalues()
    {        
        std::stringstream output; // needed for saving energies
        output << "eigenvalues_temp_" << 1/beta_ << ".saved";

        std::ofstream eig(output.str().c_str(), std::ofstream::app);
        for(int iii = 0; iii < size(eigval)[0]/2; iii++)
        {
            eig << std::setprecision(17) << eigval[iii] << "   ";
        }
        eig << std::endl;
    }
    
    // Swap replica with left neighbour ...
    void swapleft()
    {
        MPI_Status status;
        int control = 0;
        int jj,kk;
        double beta_alt = 1/calc_temp(T_min, T_max, me_ - 1, np_, dist);
        cx_mat H_a(N_,N_); // receive
        cx_mat H_b = ham; // send

        double f2 = -beta_alt * free_en(eigval, beta_alt);
        double f3 = beta_ * F;

        MPI_Send(&f2, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&f3, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

        MPI_Recv(&control, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);

        if (control == 1)
        {
            // Receive replica from left neighbour
            for (jj = 0; jj < N_; jj++)
            {
                for (kk = 0; kk < N_; kk++)
                {
                    MPI_Recv(&H_a(jj,kk), 1, MPI_DOUBLE_COMPLEX, me_- 1, 3, MPI_COMM_WORLD, &status);
                }
            }
            
            // Send own replica to left neighbour
            for (jj = 0; jj < N_; jj++)
            {
                for (kk = 0; kk < N_; kk++)
                {
                    MPI_Send(&H_b(jj,kk), 1, MPI_DOUBLE_COMPLEX, me_- 1, 4, MPI_COMM_WORLD);
                }
            }

            ham = H_a;
            eigval = eig_sym(ham);
            F = free_en(eigval, beta_);
        }
    }

    // Swap replica with right neighbour ...
    void swapright()
    {
        MPI_Status status;
        int control = 0;
        int jj, kk;
        double beta_alt = 1/calc_temp(T_min, T_max, me_ + 1, np_, dist);
        cx_mat H_b(N_,N_); // receive (here it's the other way round!!!)
        cx_mat H_a = ham; // send

        double f1 = -beta_alt * free_en(eigval, beta_alt);
        double f4 = beta_ * F;

        MPI_Send(&f1, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&f4, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

        MPI_Recv(&control, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

        if (control == 1)
        {
            // Send own replica to right neighbour
            for (jj = 0; jj < N_; jj++)
            {
                for (kk = 0; kk < N_; kk++)
                {
                    MPI_Send(&H_a(jj,kk), 1, MPI_DOUBLE_COMPLEX, me_+ 1, 3, MPI_COMM_WORLD);
                }
            }

            // Receive replica from right neighbour
            for (jj = 0; jj < N_; jj++)
            {
                for (kk = 0; kk < N_; kk++)
                {
                    MPI_Recv(&H_b(jj,kk), 1, MPI_DOUBLE_COMPLEX, me_+ 1, 4, MPI_COMM_WORLD, &status);
                }
            }

            ham = H_b;
            eigval = eig_sym(ham);
            F = free_en(eigval, beta_);
        }

    }

    // Parallel Tempering for each temperature point (= "Swap"")
    void swap()
    {
        if (me_ != 1)   
            swapleft();
        if (me_ != np_ - 1)     
            swapright();
    }

    // Master process for managing swaps:
    void master(int therm, int sweeps, int sweeps_per_save, int sweeps_per_swap)
    {
        MPI_Status status;
        engine_type loaded, saved; // needed to load and save random engine status
        int counts = (therm + sweeps)/sweeps_per_swap; // How many swaps in total?
        int control = 0;  // Signal for accepting / rejecting swap
        int sign_[np_];   // needed for ensemble optimization 
        int nplus_[np_];  // n+ histogram
        int nminus_[np_]; // n- histogram
        double counter[np_]; // How many accepted swaps?
        double den = 0;
        int s_i, s_i_plus_1; // sign for each replica (was it at T_min or T_max latest?)
        double f1, f2, f3, f4; // free energy variables
        double alpha_pt, gamma_pt; // Monte Carlo variables
        
        double fr; // Single flip acceptance rate

        // Autocorrelation times:
        double tau_en;
        double tau_fl;

        std::stringstream rng_save; // needed for saving random generator status
        rng_save << "rng_" << me_ << ".saved";

        std::stringstream sfar; // needed for saving single flip acceptance rates
        sfar << "single_flip_rate.saved";

        std::stringstream nplus_ratio; // needed for saving ratio function f = n_plus / n_tot
        nplus_ratio << "n_plus_ratio.saved";

        std::stringstream swap_ratio; // needed for saving replica exchange ratio
        swap_ratio << "swap_ratio.saved";

        std::stringstream tau_energy; // needed for saving energy autocorrelation time
        tau_energy << "tau_energy.saved";

        std::stringstream tau_flux; // needed for saving flux autocorrelation time
        tau_flux << "tau_flux.saved";

        // Load random generator status:
        std::ifstream rngfile(rng_save.str().c_str(), std::ifstream::in);
        if(rngfile.good())
        {
            rngfile >> loaded;
            rngfile.close();
            eng_ = loaded;
        }

        // Initialize sign array and histograms
        for (int k = 0; k < np_; k++)
        {
            sign_[k] = 0;
            nplus_[k] = 0;
            nminus_[k] = 0;
            counter[k] = 0;
        }

        sign_[1] = 1;
        sign_[np_ - 1] = -1;

        // PT iteration
        while(counts--)
        {
            den += 2;

            // Regard temperature points from T_min to T_max
            for (int i = 1; i < np_ - 1; i++)
            {
                // Receive free energies from replicas
                MPI_Recv(&f1, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(&f4, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(&f2, 1, MPI_DOUBLE, i+1, 2, MPI_COMM_WORLD, &status);
                MPI_Recv(&f3, 1, MPI_DOUBLE, i+1, 2, MPI_COMM_WORLD, &status);

                // Decide if replicas are swapped
                alpha_pt = exp(f1 + f2 + f3 + f4);
		        gamma_pt = rng_();

                if (gamma_pt <= alpha_pt) // accept
                {                    
                    control = 1;

                    MPI_Send(&control, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                    MPI_Send(&control, 1, MPI_INT, i+1, 2, MPI_COMM_WORLD);
                    
                    // Record histogram for ensemble optimization:
                    s_i = check_sign_i(i, np_,  sign_[i], sign_[i+1]);
                    s_i_plus_1 = check_sign_i_plus_1(i, np_,  sign_[i], sign_[i+1]);
                    sign_[i] = s_i;
                    sign_[i+1] = s_i_plus_1;
                    if (counts < sweeps + therm - 100) // Start recording after a couple of steps ...
                    {
                        if (s_i == 1)
                            nplus_[i] += 1;
                        else if (s_i = -1)  
                            nminus_[i] += 1;
                        if (s_i_plus_1 == 1)
                            nplus_[i+1] += 1;
                        else if (s_i_plus_1 == -1)
                            nminus_[i+1] += 1;
                    }

                    // Record replica exchange rate:
                    counter[i] += 1;
                    counter[i+1] += 1;
                
                }
                else // refuse
                {
                    //std::cout << "NO SWAP " << i << " " << i + 1 << std::endl;
                    control = 0;
                    MPI_Send(&control, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                    MPI_Send(&control, 1, MPI_INT, i+1, 2, MPI_COMM_WORLD);
                }  
            }   

            if (counts % sweeps_per_save == 0)
            {
                // Give single flip acceptance rates as output ...
                std::ofstream sf__(sfar.str().c_str(), std::ofstream::trunc);
                for (int jj = 1; jj < np_; jj++)
                {
                    MPI_Recv(&fr, 1, MPI_DOUBLE, jj, 5, MPI_COMM_WORLD, &status);
                    sf__ << std::setprecision(17) << calc_temp(T_min, T_max, jj, np_, dist) << " " << fr << std::endl;
                }


                // Give histogram as output ...
                std::ofstream nplus__(nplus_ratio.str().c_str(), std::ofstream::trunc);
                for (int kk = 1; kk < np_ ; kk++)
                {
                    nplus__ << std::setprecision(17) << calc_temp(T_min, T_max, kk, np_, dist) << " " << nplus_[kk] / double(nplus_[kk] + nminus_[kk]) << std::endl;
                }

                // Give swap ratio as output ...
                std::ofstream swap_ratio__(swap_ratio.str().c_str(), std::ofstream::trunc);
                for (int ll = 1; ll < np_ ; ll++)
                {
                    swap_ratio__ << std::setprecision(17) << calc_temp(T_min, T_max, ll, np_, dist) << " " << counter[ll] / den << std::endl;
                }

                // Save random generator status:
                saved = eng_;
                std::ofstream file(rng_save.str().c_str(), std::ofstream::trunc);
                file << saved;
            }         
        }

        // Give autocorrelation times as output ...
        std::ofstream tau1(tau_energy.str().c_str(), std::ofstream::trunc);
        std::ofstream tau2(tau_flux.str().c_str(), std::ofstream::trunc);
        for (int ll = 1; ll < np_; ll++)
        {
            MPI_Recv(&tau_en, 1, MPI_DOUBLE, ll, 6, MPI_COMM_WORLD, &status);
            tau1 << std::setprecision(17) << calc_temp(T_min, T_max, ll, np_, dist) << " " << tau_en << std::endl;

            MPI_Recv(&tau_fl, 1, MPI_DOUBLE, ll, 7, MPI_COMM_WORLD, &status);
            tau2 << std::setprecision(17) << calc_temp(T_min, T_max, ll, np_, dist) << " " << tau_fl << std::endl;
        }
    }


    void load(std::string const & filename)
    {
        alps::hdf5::archive ar(filename, "a");
        ar["/simulation/results/"+energy_.representation()] >> energy_;
        ar["/simulation/results/"+e2_.representation()] >> e2_;
        //ar["/simulation/results/"+e4_.representation()] >> e4_;
        ar["/simulation/results/"+dE_db.representation()] >> dE_db;
        //ar["/simulation/results/"+p_.representation()] >> p_;
        ar["/simulation/results/"+flux_real.representation()] >> flux_real;
        ar["/simulation/results/"+flux_imag.representation()] >> flux_imag;
        ar["/simulation/results/"+flux_real_squared.representation()] >> flux_real_squared;
        ar["/simulation/results/"+flux_imag_squared.representation()] >> flux_imag_squared;
        ar["/simulation/results/"+spin_corr.representation()] >> spin_corr;
        ar["/parameters/T"] >> temp_;
        ar["/parameters/BETA"] >> beta_;
        ar["/parameters/SWEEPS"] >> sweeps_;
        ar["/parameters/THERMALIZATION"] >> thermalization_;
    }
    
    void save(std::string const & filename)
    {               
        alps::hdf5::archive ar(filename, "a");
        ar["/simulation/results/"+energy_.representation()] << energy_;
        ar["/simulation/results/"+e2_.representation()] << e2_;
        //ar["/simulation/results/"+e4_.representation()] << e4_;
        ar["/simulation/results/"+dE_db.representation()] << dE_db;
        //ar["/simulation/results/"+p_.representation()] << p_;
        ar["/simulation/results/"+flux_real.representation()] << flux_real;
        ar["/simulation/results/"+flux_imag.representation()] << flux_imag;
        ar["/simulation/results/"+flux_real_squared.representation()] << flux_real_squared;
        ar["/simulation/results/"+flux_imag_squared.representation()] << flux_imag_squared;
        ar["/simulation/results/"+spin_corr.representation()] << spin_corr;
        ar["/parameters/T"] << temp_;
        ar["/parameters/BETA"] << beta_;
        ar["/parameters/SWEEPS"] << sweeps_;
        ar["/parameters/THERMALIZATION"] << thermalization_;
    }
    
    /////////////////////////////////////////////////////////////////////
    // Randomize zz-entries of matrix (likewise: all nonzero entries): //
    /////////////////////////////////////////////////////////////////////
    
    cx_mat randomize(cx_mat ham) 
    {    
        int N_ = size(ham)[0];
        int die, coord1, coord2;

        for (int j = 0 ; j < 3*N_/2; j++)
        {
            die = roll_die(length_);
            coord1 = v_[die]/N_;
            coord2 = v_[die]%N_;
        
            // Flip a coin:
            if (rng_() < 0.5)
            {
                ham(coord1, coord2) *= -1;
                ham(coord2, coord1) *= -1;
            }
        }

        return ham;
    }

    /*cx_mat bond_randomize(cx_mat ham) 
    {    
        int N_ = size(ham)[0];
        int die, coord1, coord2;
        double dJ = 0.8;

        for (int j = 0 ; j < N_/4; j++)
        {
            die = roll_die(length_);
            coord1 = v_[die]/N_;
            coord2 = v_[die]%N_;
        
            // Flip a coin:
            if (rng_() < 0.5)
            {
                ham(coord1, coord2) -= dJ;
                ham(coord2, coord1) = -ham(coord1, coord2);
            }
            else
            {
                ham(coord1, coord2) += dJ;
                ham(coord2, coord1) = -ham(coord1, coord2);
            }
        }

        return ham;
    }*/

    ////////////////////////////////////////////////////////////////
    
    protected:
    
    // Random int from the interval [0,max)
    int roll_die(int max) const
    {
        return static_cast<int>(max * rng_());
    }

private:
    typedef boost::mt19937 engine_type; // Mersenne twister
    typedef boost::uniform_real<> distribution_type;
    typedef boost::variate_generator<engine_type&, distribution_type> rng_type;
    engine_type eng_;
    distribution_type dist_;
    mutable rng_type rng_;

    size_t sweeps_;
    size_t thermalization_;

    // Everything here is described above:
    int np_;
    int me_;
    double T_min;
    double T_max; 
    double temp_;
    double beta_;

    cx_mat ham;
    size_t N_;

    std::vector<int> v_;
    int length_;

    vec eigval;

    double F;

    Mat<int> plaquettes;

    // ALPS Observables:

    alps::RealObservable energy_;
    alps::RealObservable e2_;
    //alps::RealObservable e4_;
    alps::RealObservable dE_db;
    //alps::RealObservable p_;
    alps::RealObservable flux_real;
    alps::RealObservable flux_imag;
    alps::RealObservable flux_real_squared;
    alps::RealObservable flux_imag_squared;
    alps::RealObservable spin_corr;

    int flip_rate;

    signed int sign;

    std::string dist;
    std::string filename_;
};
