//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//  Observable functions for Kitaev Quantum Monte Carlo
//  written by: Tim Eschmann, June 2016
//  Modified version: February 2019
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
   
#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <complex>
#include <armadillo>

using namespace arma;
   
///////////////////////////////////////////////////////////////
// Calculate free energy for a given Z2 configuration:
///////////////////////////////////////////////////////////////

double free_en(vec ev, double b)
{
    int p; // running index
    double fe = 0; // free energy
    
    // Calculating:
    for (p = 0; p < size(ev)[0]/2; p++)
    {
        fe -= (1/b)*logl(2*coshl(b*ev[p]/2));      
    }

    return fe;
       
}
    
///////////////////////////////////////////////////////////////
// Calculate internal energy of the Majorana fermion system:
///////////////////////////////////////////////////////////////

double en(vec ev, double b)
{
    int q; // running index;
    double en = 0; // internal energy
    
    for (q = 0; q < size(ev)[0]/2; q++)
    {
        en += ev[q]/2 * tanhl(-b * ev[q]/2);
    }
    
    return en;
}

///////////////////////////////////////////////////////////////
// Calculate derivative of E w.r.t. beta:
///////////////////////////////////////////////////////////////

double diffE(vec ev, double b)
{
    int qq; // running index
    double drv = 0;
    for (qq = 0; qq < size(ev)[0]/2; qq++)
    {
        //drv += ev[qq]*ev[qq]/4. * (1 - tanh(-beta_ * ev[qq]/2)*tanh(-beta_ * ev[qq]/2));
        drv += powl(ev[qq], 2)/4. / powl((coshl(-b * ev[qq]/2)),2);
    }

    return drv;
}
     
///////////////////////////////////////////////////////////////
// Calculate average flux per elementary plaquette:
///////////////////////////////////////////////////////////////

std::complex <double> flux(cx_mat ham, Mat<int> plaq)
{
    std::complex <double> flux;
    std::complex <double> av_flux = std::complex<double>(0.0, 0.0);
    int N = size(ham)[0];
    int M1 = size(plaq)[0];
    int M2 = size(plaq)[1];
    int coord1, coord2;
        
    for (int i = 0; i < M1; i++)
    {
        flux = std::complex<double>(1.0, 0.0);
        for (int j = 0; j < M2; j++)
        {
            coord1 = plaq(i,j) / N;
            coord2 = plaq(i,j) % N;
            
            // Plaquette operator: W_p = prod_<i,j> (-i*u_ij)
            flux *= (-ham(coord1, coord2)) / std::abs(ham(coord1, coord2));
        }
        av_flux += flux;
    }

    av_flux /= double(M1);

    return av_flux;
        
}

///////////////////////////////////////////////////////////////
// Calculate flux disorder ratio p:
///////////////////////////////////////////////////////////////

double get_p(cx_mat ham, Mat<int> plaq)
{
    std::complex <double> flux;
    double p;
    int N = size(ham)[0];
    int M1 = size(plaq)[0];
    int M2 = size(plaq)[1];
    int coord1, coord2;
        
    for (int i = 0; i < M1; i++)
    {
        flux = std::complex<double>(1.0, 0.0);
        for (int j = 0; j < M2; j++)
        {
            coord1 = plaq(i,j) / N;
            coord2 = plaq(i,j) % N;
            
            // Plaquette operator: W_p = prod_<i,j> (-i*u_ij)
            flux *= (-ham(coord1, coord2)) / std::abs(ham(coord1, coord2));
        }
        if ((M2 % 4 == 0) && (flux == std::complex<double>(-1, 0)))
            p += 1;
        else if ((M2 % 4 == 2) && (flux == std::complex<double>(1, 0)))
            p += 1;
        else if ((M2 % 2 != 0) && (flux == std::complex<double>(0, 1)))
            p += 1;
    }

    p /= double(M1);

    return p;
        
}

///////////////////////////////////////////////////////////////
// Give flux configurations as output:
///////////////////////////////////////////////////////////////
cx_vec flux_confs(cx_mat ham, Mat<int> plaq)
{
    int M1 = size(plaq)[0];
    int M2 = size(plaq)[1];
    int N = size(ham)[0];
    int coord1, coord2;
    cx_vec confs(M1);
    std::complex <double> flux;

    for (int i = 0; i < M1; i++)
    {
        flux = std::complex<double>(1.0, 0.0);
        for (int j = 0; j < M2; j++)
        {
            coord1 = plaq(i,j) / N;
            coord2 = plaq(i,j) % N;
            
            // Plaquette operator: W_p = prod_<i,j> (-i*u_ij)
            flux *= (-ham(coord1, coord2)) / std::abs(ham(coord1, coord2));
        }

        confs[i] = flux;
    }

    return confs;

} 

///////////////////////////////////////////////////////////////
// Calculate spin-spin correlation:
// Make sure only to include one subset of bonds
///////////////////////////////////////////////////////////////

double correlation(cx_mat ham, std::vector<int> v_, double b)
{
    double value = 0.;
    std::complex<double> av;
    vec eigval;
    cx_mat eigvec;
    int N = size(ham)[0];
    int coord1, coord2;
      
    eig_sym(eigval, eigvec, ham);
    //eigvec = normalise(eigvec);

    for (int j = 0; j < v_.size(); j++)
    {
        coord1 = v_[j]/N;
        coord2 = v_[j]%N;
        
        for (int i = size(eigval)[0]/2; i < size(eigval)[0]; i++)
        {
            av = conj(eigvec(coord1,i))*((-ham(coord1, coord2)) / std::abs(ham(coord1, coord2)))*eigvec(coord2, i);
            av += conj(eigvec(coord2,i))*((-ham(coord2, coord1)) / std::abs(ham(coord2, coord1)))*eigvec(coord1, i);
            value -= real(av)*tanhl(b * eigval[i]/2.);
        }
    }

    value *= 2/double(N);    
    return value;
        
}

/////////////////////////////////////////////////////////
// Set up simulation temperatures (w. different options):
/////////////////////////////////////////////////////////  

double calc_temp(double T_min, double T_max, int me, int np, std::string dist)
{
    int i;
    double T;
        
    // a) Read temperature distribution from file:
    if (dist == "external")
    {
        double temperatures[np];
        std::ifstream tempfile("temp.saved", std::ifstream::in);
        if(tempfile.good())
        {
            for (i = 0; i < np; i++)
            {
                tempfile >> temperatures[i];
            }
        }

        T = temperatures[me - 1];
    }

    // b) Linear temperature distribution:
    else if (dist == "lin")
    {
        T = T_min + (T_max - T_min)*(me - 1)/float(np);
    }
    // c) Logarithmic temperature distribution:*/    
    else if (dist == "log")
    {
        T = pow(10,log10(T_min)+((log10(T_max) - log10(T_min))*(me - 1)/double(np)));
    }
    // d) Double-logarithmic temperature distribution:
    else if (dist == "double_log")
    {
        T = pow(10,log10(T_min)+((log10(T_max) - log10(T_min))*pow(10, -(np - (me - 1))/double(np))));
    }

    return T;


}



//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//  Ensemble optimization functions 
//  -> check if MC replica has been at T_min or T_max latest and assign
// "+1" or "-1" accordingly 
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Record if replica has been at lowest or highest T latest ...      ////////
/////////////////////////////////////////////////////////////////////////////

int check_sign_i(int i, int np, int s_i, int s_i_plus_1)
{
    int sign;
    
    if (i != 1 && (i+1 != (np - 1)))
    {
        if (s_i == 1 && s_i_plus_1 == 0) 
            sign = 0;
        else if (s_i == 0 && s_i_plus_1 == 1)
            sign = 1;
        else if (s_i == 1 && s_i_plus_1 == -1)
            sign = -1;
        else if (s_i == -1 && s_i_plus_1 == 1)
            sign = 1;
        else if (s_i == 0 && s_i_plus_1 == -1)
            sign = -1;
        else if (s_i == -1 && s_i_plus_1 == 0)
            sign = 0;     
        else if (s_i == 0 && s_i_plus_1 == 0)
            sign = 0;
        else if (s_i == 1 && s_i_plus_1 == 1)
            sign = 1;
        else if (s_i == -1 && s_i_plus_1 == -1)
            sign = -1;
    }
    else if (i == 1)
        sign = 1;
    else if (i+1 == np - 1)
        sign = -1;
    else
        sign = 0;

    return sign;

}

int check_sign_i_plus_1(int i, int np, int s_i, int s_i_plus_1)
{
    int sign;
    
    if (i != 1 && (i+1 != (np - 1)))
    {
        if (s_i == 1 && s_i_plus_1 == 0)   
            sign = 1;
        else if (s_i == 0 && s_i_plus_1 == 1)
            sign = 0;
        else if (s_i == 1 && s_i_plus_1 == -1)
            sign = 1;
        else if (s_i == -1 && s_i_plus_1 == 1)
            sign = -1;
        else if (s_i == 0 && s_i_plus_1 == -1)
            sign = 0;
        else if (s_i == -1 && s_i_plus_1 == 0)
            sign = -1;     
        else if (s_i == 0 && s_i_plus_1 == 0)
            sign = 0;
        else if (s_i == 1 && s_i_plus_1 == 1)
            sign = 1;
        else if (s_i == -1 && s_i_plus_1 == -1)
            sign = -1;  
    }
    else if (i == 1)
        sign = 1;
    else if (i+1 == np - 1)
        sign = -1;
    else 
        sign = 0;

    return sign;
}



