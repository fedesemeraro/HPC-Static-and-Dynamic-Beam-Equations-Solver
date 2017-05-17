/*
To fill required matrices for tasks 1, 2 and 3 (serial)
*/

#ifndef FILLMATRICESSERIAL_H	
#define FILLMATRICESSERIAL_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
using namespace std;


void FillK(double* K, int ldK, int nuk, double A, double E, double l, double I)
{
	/*
	K is a ldK * nuk symmetric banded matrix.

	The stiffness matrix K is assembled by noticing that the banded matrix minus the elements
	corresponding to the boundary degrees of freedom is basically a 5 x 3 block that
	repeats itself every 3 columns. This is more efficient than assembling the global matrix from
	the elements (O(n) instead of O(n^2)); an argument could be made that the approach used here is not as
	general, but the same argument could then be made against using a banded matrix instead of a general
	symmetric one.
	*/
	for (int j = 1; j <= nuk; j += 3) {
		// Using column major for lapack
		// Use K[(i-1) + ldK * (j-1)] to access K(i,j)
		K[(1 - 1) + ldK * (j - 1)] = 0;
		K[(2 - 1) + ldK * (j - 1)] = -A * E / l;
		K[(3 - 1) + ldK * (j - 1)] = 0;
		K[(4 - 1) + ldK * (j - 1)] = 0;
		K[(5 - 1) + ldK * (j - 1)] = 2 * A * E / l;

		K[(1 - 1) + ldK * (j + 1 - 1)] = 0;
		K[(2 - 1) + ldK * (j + 1 - 1)] = -12 * E * I / pow(l, 3);
		K[(3 - 1) + ldK * (j + 1 - 1)] = -6 * E * I / pow(l, 2);
		K[(4 - 1) + ldK * (j + 1 - 1)] = 0;
		K[(5 - 1) + ldK * (j + 1 - 1)] = 2 * 12 * E * I / pow(l, 3);

		K[(1 - 1) + ldK * (j + 2 - 1)] = 6 * E * I / pow(l, 2);
		K[(2 - 1) + ldK * (j + 2 - 1)] = 2 * E * I / l;
		K[(3 - 1) + ldK * (j + 2 - 1)] = 0;
		K[(4 - 1) + ldK * (j + 2 - 1)] = 0;
		K[(5 - 1) + ldK * (j + 2 - 1)] = 2 * 4 * E * I / l;
	}
}

void FillF(double* F, int nuk, double l, double q_y, double F_y)
{
	/*
	The applied forces vector F is assembled. It consists of:
	- A constant distributed load q_y in the vertical direction down
	- A point load F_y at the middle node in the vertical direction down
	*/

	// Distributed load:
	for (int j = 1; j <= nuk; j += 3) {
		F[j - 1] = 0;
		F[j + 1 - 1] = q_y * l;
		F[j + 2 - 1] = 0;
	}

	// Concentrated load:
	F[nuk / 2] += F_y;    // remember that nuk/2 is int division and F's domain is [0, nuk)
}

void FillM(double* M, int ldM, int nuk, double rho, double A, double l)
{
	/*
	M is a ldM by nuk symmetric banded matrix

	The mass matrix M is assembled by noticing that the banded symmetric matrix minus the elements
	corresponding to the boundary degrees of freedom is basically a 1 x 3 block that repeats itself
	every 3 columns
	*/
	for (int j = 1; j <= nuk; j += 3) {
		// Using column major for lapack
		// Use M[(i-1) + ldM * (j-1)] to access M(i,j)
		M[(1 - 1) + ldM * (j - 1)] = rho * A * l;
		M[(1 - 1) + ldM * (j + 1 - 1)] = rho * A * l;
		M[(1 - 1) + ldM * (j + 2 - 1)] = (1.0 / 12) * rho * A * pow(l, 3);
	}
}

#endif