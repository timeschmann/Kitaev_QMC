#include <armadillo>
#include <complex>

using namespace arma;

cx_mat def_matrix()
{
    cx_mat A(8,8, fill::zeros);
    A(0,1) = -std::complex<double>(0.0, 2/3.);
    A(1,0) = std::complex<double>(0.0, 2/3.);
    A(0,5) = -std::complex<double>(0.0, 2/3.); 
    A(5,0) = std::complex<double>(0.0, 2/3.);  
    A(1,2) = std::complex<double>(0.0, 2/3.);
    A(2,1) = -std::complex<double>(0.0, 2/3.);
    A(1,4) = std::complex<double>(0.0, 2/3.);
    A(4,1) = -std::complex<double>(0.0, 2/3.);
    A(2,3) = -std::complex<double>(0.0, 2/3.);
    A(3,2) = +std::complex<double>(0.0, 2/3.);
    A(2,7) = -std::complex<double>(0.0, 2/3.);
    A(7,2) = std::complex<double>(0.0, 2/3.);
    A(3,4) = std::complex<double>(0.0, 2/3.);
    A(4,3) = -std::complex<double>(0.0, 2/3.);
    A(3,6) = std::complex<double>(0.0, 2/3.);
    A(6,3) = -std::complex<double>(0.0, 2/3.);
    A(4,5) = -std::complex<double>(0.0, 2/3.);
    A(5,4) = std::complex<double>(0.0, 2/3.);
    A(5,6) = std::complex<double>(0.0, 2/3.);
    A(6,5) = -std::complex<double>(0.0, 2/3.);
    A(6,7) = -std::complex<double>(0.0, 2/3.);
    A(7,6) = std::complex<double>(0.0, 2/3.);
    return A;
}

std::vector<int> non_zeros()
{
    std::vector<int> nz;
    int coo;
    coo = 8*0 + 5;
    nz.push_back(coo);
    coo = 8*1 + 4;
    nz.push_back(coo);
    coo = 8*2 + 7;
    nz.push_back(coo);
    coo = 8*3 + 6;
    nz.push_back(coo);
    return nz;
}

Mat<int> create_plaquettes()
{
   Mat<int> P(1,6);
   P(0, 0) = 8*0+1;
   P(0, 1) = 8*1+2;
   P(0, 2) = 8*2+7;
   P(0, 3) = 8*7+6;
   P(0, 4) = 8*6+5;
   P(0, 5) = 8*5+1;
   return P;
}
