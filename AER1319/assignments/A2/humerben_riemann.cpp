#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cstdlib>

using namespace std;

void exact_riemann(
                // input parameters
                const double &rhoL, const double &uL, const double &pL,             // left solution state
                const double &rhoR, const double &uR, const double &pR,             // right solution state
                const double &gamma,                                                // constant values parametrizing the solution
                int &pattern,
                int &num_iter, int &max_iter, const double & conv_tol,              // variables for iterative solver
                double &rhoML, double &rhoMR, double &uM, double &pM,
                double &vsL, double &vhL, double &vtL,
                double &vsR, double &vhR, double &vtR
) {

    // set up variables to describe the 

    // set up variables for iterative solver
    double convergence_tolerance=1e-5;

    // set up intermediate variables
    double MSL, MSR;                    // shock mach numbers
    double pSL, pSR;
    double dpSL_duS, dpSR_duS;


    // calculate sound speeds for the left and right states
    double aL = sqrt(gamma*pL/rhoL);
    double aR = sqrt(gamma*pR/rhoR);

    // calculate reimann invariants
    double QL = uL - (2) / (gamma - 1) * aL;
    double QR = uR - (2) / (gamma - 1) * aL;
    double PL = uL + (2) / (gamma - 1) * aL;
    double PR = uR + (2) / (gamma - 1) * aR;

    // get an estimate of the intermediate velocity
    double Z = aR/aL * pow((rhoL/rhoR), (gamma - 1) / (2 * gamma));         // parameter for the initial estimate for newton's method
    double uS = (PL*Z + QR) / (1 + Z);                                     // initial estimate for u for newton's method

    // check to see if iteration is required, or if we are in the RCR pattern, which has an exacct solution
    if (uL <= uS && uS <= uR){
        // get intermediate pressure p*
        double pS = pL * pow(((gamma-1) / 2 * (PL-QR) / (aL*(1+Z))), (2*gamma)/(gamma-1));
        
        // get intermediate sound speeds aL* and aR*
        double aSL = aL * pow((pS / pL), (gamma-1)/(2*gamma));
        double aSR = aR * pow((pS / pR), (gamma-1)/(2*gamma));

        // get intermediate densities rhoL* and rhoR*
        double rhoSL = (gamma * pS) / (pow(aSL, 2));
        double rhoSR = (gamma * pS) / (pow(aSR, 2));

        // set variable for the wave pattern (RCR=1)
        int pattern=1;

        // set variables for wave speeds associated with this pattern
            // left rarefaction wave
        double uHL = uL - aL;           // head
        double uTL = uS - aSL;         // tail
            // right rarefaction wave
        double uHR = uR + aL;           // head
        double uTR = uS + aSR;         // tail
            // shock waves
        double ushkL = 0.0;               // left shock
        double ushkR = 0.0;               // right shock

    }
    
    // if the previous condition wasn't triggered, then iteration is required to determine the solution
    while (1) {             // set up loop, exit condition will be applied later in the iteration process
        // set up iteration counter
        num_iter = num_iter + 1;

        // evaluate pSL for the wave left of the contact discontinuity
            // start with shock case
        if (uL >= uS){
            MSL = (gamma + 1)/4 * (uS - uL)/(aL) - sqrt(1 + pow((gamma+1)/4 * (uS-uL)/aL, 2));
            pSL = pL + (gamma*pL)/(aL) * (uS-uL) * MSL;
            dpSL_duS = 2 * (gamma*pL)/(aL) * (pow(MSL,3))/(1+pow(MSL,2));
        }
            // now rarefaction case
        else{
            aSL = aL - (gamma - 1)/(2) * (uS-uL);
            pSL=pL * pow(aSL/aL, (2*gamma/(gamma-1)));
            dpSL_duS = -gamma*(pSL/aSL);
        }

        // evaluate for the wave right of the contact discontinuity
            // start with shock wave
        if (uS >= uR){
            MSR = (gamma+1)/4 * (uS-uR)/aR + sqrt(1 + pow((gamma+1)/4 * (uS-uR)/aR, 2));
            pSR = pR + (gamma*pR/aR)*(uS-uR) * MSR;
            dpSR_duS = 2 * (gamma * pR)/aR * (pow(MSR,3))/(1+pow(MSR,2));
        }
            // now rarefaction wave
        else{
            aSR = aR + (gamma-1)/2 * (uS - uR);
            pSR = pR * pow((aSR/aR), 2*gamma / (gamma-1));
            dpSR_duS = gamma * pSR / aSR;
        }

        // test convergence with the values from the evaluation above
            // break if the tolerance is low enough
        if (fabs(1 - pSL/pSR) < conv_tol) break;
            // throw error if the number of iterations has exceeded the maximum number of iterations
        if (num_iter > max_iter){
            // Error message taken from solver example written by Prof. CPT Groth
            cout << "\n ERROR: convergence problem in Riemann solver: " 
            << "n = " << num_iter
            << " tol = "
            << fabs(1.00-pSL/pSR);
            break;
        }

        // update value of u* to reach convergence
        uS = uS - (pSL - pSR) / (dpSL_duS - dpSR_duS);

    }


    // pressure should be equal after iteration, so take the average
    double pS = 0.5 * (pSR + pSL);

    // calculate head, tail, and shock velocities where applicable
        // begin with left moving wave
    if (uL >= uS){      // shock
            // density
        aSL = aL*sqrt(((gamma+1.00) + (gamma-1.00)*pS/pL) / ((gamma+1.00) + (gamma-1.00)*pL/pS) );
        rhoSL = (gamma * pS) / pow(aSL, 2);
            // shock speed
        ushkL = uL + MSL * aL;
            // rarefaction head and tail speeds
        uHL = 0;
        uTL = 0.0;
    }
    else{               // rarefaction
            // density
        rhoSL = (gamma * pS) / pow(aSL, 2);
            // shock speed
        ushkL = 0.0;
            // rarefaction head and tail speeds

    }

    

}




int main(void){
    // rhoL = 4.00*1.00;
    // uL = 0.00;
    // pL = 10.00*101.00*1000.00;

    // rhoR = 1.00;
    // uR = 0.00;
    // pR = 101.00*1000.00;
    // gamma = 1.40;

    // rhoL = 1.00;
    // uL = 0.00;
    // pL = 101.00*1000.00;

    // rhoR = 4.00*1.00;
    // uR = 0.00;
    // pR = 10.00*101.00*1000.00;
    // gamma = 1.40;

    // rhoL = 5.00*1.00;
    // uL = -78.00;
    // pL = 6.50*101.00*1000.00;

    // rhoR = 1.60*1.00;
    // uR = 120.00;
    // pR = 2.00*101.00*1000.00;
    // gamma = 1.40;

    rhoL = 1.60*1.00;
    uL = -120.00;
    pL = 2.00*101.00*1000.00;

    rhoR = 5.00*1.00;
    uR = 78.00;
    pR = 6.50*101.00*1000.00;
    gamma = 1.40;


}