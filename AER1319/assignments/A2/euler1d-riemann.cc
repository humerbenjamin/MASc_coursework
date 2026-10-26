
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cstdlib>

using namespace std;

/**********************************************************
 * Routine: Riemann (Exact Riemann Solution to Riemann    *
 *                   Problem)                             *
 *                                                        *
 * This routine uses a Newton method iterative            *
 * procedure to obtain the exact solution to the          *
 * Riemann problem for the 1D Euler equations,            *
 * returning the wave pattern, the intermediate solution  *
 * variables, as well as the wave speeds for the left     *
 * and right waves.  See Gottlieb and Groth (1988).       *
 *                                                        *
 **********************************************************/
void Riemann(const double &rhoL, const double &uL, const double &pL,
	     const double &rhoR, const double &uR, const double &pR,
	     const double &gamma,
	     int &pattern,
	     double &rhoML, double &rhoMR, double &uM, double &pM,
	     double &vsL, double &vhL, double &vtL,
	     double &vsR, double &vhR, double &vtR) {

  int number_of_iterations, max_iterations;

  double TOLER=0.0000001;
  
  double aL, aR, RIL, RIR, Z;
  double aML, aMR;
  double MsL, pML, dpMLduM, MsR, pMR, dpMRduM;

  /* Determine the left and right state sound speeds. */

  aL = sqrt(gamma*pL/rhoL);
  aR = sqrt(gamma*pR/rhoR);

  /* Compute the left and right state Riemann invariants. */

  RIL=uL+2.00*aL/(gamma-1.00);
  RIR=uR-2.00*aR/(gamma-1.00);

  /* Check for vacuum state. */
  
  /* Make an initial estimate of the intermediate flow
     velocity to begin the Newton-Raphson iterative solution
     procedure.  The initial guess for the velocity is made
     based on isentropic flow theory. */

  Z = (aR/aL)*pow((pL/pR), (gamma-1.00)/(2.00*gamma));
  uM = (RIL*Z+RIR)/(1.00+Z);

  /* In the case that two rarefaction waves are present,
     then an exact solution has been found and the iterative
     procedure is not required.  Check for this. */

  if ( uM >= uL && uM <= uR ) {
      aML = aL-0.50*(gamma-1.00)*(uM-uL);
      aMR = aR+0.50*(gamma-1.00)*(uM-uR);
      pM = pL*pow((aML/aL), 2.00*gamma/(gamma-1.00));
      //pM = pR*pow((aMR/aR), 2.00*gamma/(gamma-1.00));
      rhoML = gamma*pM/(aML*aML);
      rhoMR = gamma*pM/(aMR*aMR);
      vhL = uL-aL;
      vtL = uM-aML;
      vsL = 0.00;
      vhR = uR+aR;
      vtR = uM+aMR;
      vsR = 0.00;
      pattern = 1;
  } /* end if */

  /* Perform the Newton-Raphson iterative procedure and solve for
     the velocity, uM, in the intermediate state.  During this iterative
     process the pressure in the intermediate state is also found. */

  number_of_iterations = 0;
  max_iterations = 10;
   
  while (1) {
    /* Update the iteration counter. */
    number_of_iterations = number_of_iterations + 1;
 
    /* Evaluate pML for left wave. */
    if ( uM < uL ) {
      MsL = (gamma+1.00)*(uM-uL)/(4.00*aL);
      MsL = MsL-sqrt(1.00+MsL*MsL);
      pML = pL*(1.00+gamma*(uM-uL)*MsL/aL);
      dpMLduM = 2.00*gamma*pL*(MsL*MsL*MsL)/(aL*(1.00+MsL*MsL));
    } else {
      aML = aL-0.50*(gamma-1.00)*(uM-uL);
      pML = pL*pow((aML/aL), 2.00*gamma/(gamma-1.00));
      dpMLduM = -gamma*pML/aML;
    } /* end if */
    
    /* Evaluate pMR for right wave. */
    if ( uM > uR ) {
      MsR = (gamma+1.00)*(uM-uR)/(4.00*aR);
      MsR = MsR+sqrt(1.00+MsR*MsR);
      pMR = pR*(1.00+gamma*(uM-uR)*MsR/aR);
      dpMRduM = 2.00*gamma*pR*(MsR*MsR*MsR)/(aR*(1.00+MsR*MsR));
    } else {
      aMR = aR+0.50*(gamma-1.00)*(uM-uR);
      pMR = pR*pow((aMR/aR), 2.00*gamma/(gamma-1.00));
      dpMRduM = gamma*pMR/aMR;
    } /* end if */
    
    /* Check for convergence (i.e., pML=pMR). */
    
    if ( fabs(1.00-pML/pMR) <= TOLER) break;
    if ( number_of_iterations > max_iterations) {
      cout << "\n ERROR: convergence problem in Riemann solver: " 
      	   << "n = " << number_of_iterations
      	   << " tol = "
      	   << fabs(1.00-pML/pMR);
      break;
    } /* endif */ ;
    
    /* Compute next estimate for the intermediate
       velocity, uM. */
    
    uM = uM-(pML-pMR)/(dpMLduM-dpMRduM);
  } /* endwhile */
  
  /* Iterations complete.
     Return the complete solution to the Riemann problem. */
  
  pM = 0.50*(pML+pMR); // pML & pML should be equal, take average.

  if ( uM < uL ) {
    aML = aL*sqrt(((gamma+1.00)+(gamma-1.00)*pM/pL) /
		  ((gamma+1.00)+(gamma-1.00)*pL/pM) );
    rhoML = gamma*pM/(aML*aML);
    vsL = uL+MsL*aL;
    vhL = 0.00;
    vtL = 0.00;
  } else {
    rhoML = gamma*pM/(aML*aML);
    vhL = uL-aL;
    vtL = uM-aML;
    vsL = 0.00;
  } /* endif */
  if ( uM > uR ) {
    aMR = aR*sqrt(((gamma+1.00)+(gamma-1.00)*pM/pR) /
		  ((gamma+1.00)+(gamma-1.00)*pR/pM) );
    rhoMR = gamma*pM/(aMR*aMR);
    vsR = uR+MsR*aR;
    vhR = 0.00;
    vtR = 0.00;
  } else {
    rhoMR = gamma*pM/(aMR*aMR);
    vhR = uR+aR;
    vtR = uM+aMR;
    vsR = 0.00;
  } /* end if */

  if ( uM >= uL && uM <= uR ) {
    pattern = 1;
  } else if ( uM < uL && uM <= uR ) {
    pattern = 2;
  } else if ( uM >= uL && uM > uR ) {
    pattern = 3;
  } else {
    pattern = 4;  
  } /* end if */

}

/**********************************************************
 * Main program - call routine Riemann                    *
 **********************************************************/
int main(int num_arg, char *arg_ptr[]) {

  int i, pattern;

  double rhoL, uL, pL,
         rhoR, uR, pR, gamma;

  double rhoML, rhoMR, uM, pM;
  double vsL, vhL, vtL,
         vsR, vhR, vtR;

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

  Riemann(rhoL, uL, pL,
	  rhoR, uR, pR,
	  gamma, pattern,
	  rhoML, rhoMR, uM, pM,
	  vsL, vhL, vtL,
	  vsR, vhR, vtR);

  cout << "\n " << pattern << "\n "
       << rhoML << " "
       << rhoMR << " "
       << uM << " "
       << pM << "\n ";
  cout << vsL << " "
       << vhL << " "
       << vtL << "\n ";
  cout << vsR << " "
       << vhR << " "
       << vtR << "\n ";
  
  return (0);

}

