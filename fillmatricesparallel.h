/*
To fill required matrices for tasks 4 and 5 (parallel)
*/

#ifndef FILLMATRICESPARALLEL_H	
#define FILLMATRICESPARALLEL_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
using namespace std;


void ParFillK(double* K, int ldK, int nuk_start, int nuk_end, double A, double E, double l, double I)
{
	/*
	For use in parallel.
	The stiffness matrix K (in banded symmetric form) is assembled between columns nuk_start and nuk_end (both included)
	Use the fact that K is basically a 5 by 3 block that repeats itself every 3 columns.
	*/
	const int col_offset = nuk_start - 1;    // so that nuk_start - col_offset = 1, (nuk_start + 1) - col_offset = 2, and so on.

	for (int j = nuk_start; j <= nuk_end; ++j) {
		// Using column major for lapack
		// Use K[(i-1) + ldK * (j-1)] to access element K(i,j)
		if (j % 3 == 1) {     // e.g. if j = 1 or 4, etc
			K[(1 - 1) + ldK * (j - col_offset - 1)] = 0;
			K[(2 - 1) + ldK * (j - col_offset - 1)] = -A * E / l;
			K[(3 - 1) + ldK * (j - col_offset - 1)] = 0;
			K[(4 - 1) + ldK * (j - col_offset - 1)] = 0;
			K[(5 - 1) + ldK * (j - col_offset - 1)] = 2 * A * E / l;
		}
		if (j % 3 == 2) {     // e.g. if j = 2
			K[(1 - 1) + ldK * (j - col_offset - 1)] = 0;
			K[(2 - 1) + ldK * (j - col_offset - 1)] = -12 * E * I / pow(l, 3);
			K[(3 - 1) + ldK * (j - col_offset - 1)] = -6 * E * I / pow(l, 2);
			K[(4 - 1) + ldK * (j - col_offset - 1)] = 0;
			K[(5 - 1) + ldK * (j - col_offset - 1)] = 2 * 12 * E * I / pow(l, 3);
		}
		if (j % 3 == 0) {     // e.g. if j = 3
			K[(1 - 1) + ldK * (j - col_offset - 1)] = 6 * E * I / pow(l, 2);
			K[(2 - 1) + ldK * (j - col_offset - 1)] = 2 * E * I / l;
			K[(3 - 1) + ldK * (j - col_offset - 1)] = 0;
			K[(4 - 1) + ldK * (j - col_offset - 1)] = 0;
			K[(5 - 1) + ldK * (j - col_offset - 1)] = 2 * 4 * E * I / l;
		}
	}
}

void ParFillFtask4(double* F, int nuk_start, int nuk_end, int nuk_midpoint, int rank, double l, double q_y, double F_y)
{
	/*
	The local part of the applied forces vector F is assembled. It consists of:
	- A constant distributed load q_y in the vertical direction down
	- A point load F_y at the middle node (taking care of how the 2 processes are split) in the vertical direction down
	*/

	// Remember that nuk_start is 1 for rank = 0 and that nuk_end is included
	const int row_offset = nuk_start - 1;    // so that nuk_start - row_offset = 1, (nuk_start + 1) - row_offset = 2, and so on.

	// Distributed load:
	for (int j = nuk_start; j <= nuk_end; ++j) {
		if (j % 3 == 1) {     // e.g. if j = 1 or 4, etc
			F[j - row_offset - 1] = 0;
		}
		if (j % 3 == 2) {     // e.g. if j = 2
			F[j - row_offset - 1] = q_y * l;
		}
		if (j % 3 == 0) {     // e.g. if j = 3
			F[j - row_offset - 1] = 0;
		}
	}

	// Concentrated load at midpoint (remember that the three middle nodes are shared):
	if (rank == 0) {
		F[nuk_midpoint - 1] += F_y;
	}
	else if (rank == 1) {
		F[5 - 1] += F_y;
	}
}

void ParFillF(double* F, int nuk_start, int nuk_end, int rank, int rank_midpoint, int relpos_midpoint, double l, double q_y, double F_y)
{
	/*
	The local part of the applied forces vector F is assembled. It consists of:
	- A constant distributed load q_y in the vertical direction down
	- A point load F_y at the middle node in the vertical direction down
	*/

	// Remember that nuk_start is 1 for rank = 0 and that nuk_end is included
	const int row_offset = nuk_start - 1;    // so that nuk_start - row_offset = 1, (nuk_start + 1) - row_offset = 2, and so on.

	// Distributed load:
	for (int j = nuk_start; j <= nuk_end; ++j) {
		if (j % 3 == 1) {     // e.g. if j = 1 or 4, etc
			F[j - row_offset - 1] = 0;
		}
		if (j % 3 == 2) {     // e.g. if j = 2
			F[j - row_offset - 1] = q_y * l;
		}
		if (j % 3 == 0) {     // e.g. if j = 3
			F[j - row_offset - 1] = 0;
		}
	}

	// Concentrated load at midpoint:
	if (rank == rank_midpoint) {
		// relpos_midpoint is the midpoint relative (=> no row_offset) position inside (1 would be the first column in rank_midpoint):
		F[relpos_midpoint - 1] += F_y;
	}
}

void ParFillM(double* M, int ldM, int nuk_start, int nuk_end, double rho, double A, double E, double l)
{
	/*
	For use in parallel.
	The mass matrix M (in banded symmetric form) is assembled between columns nuk_start and nuk_end (both included)
	Use the fact that M is basically a 1 by 3 block that repeats itself every 3 columns.
	*/
	const int col_offset = nuk_start - 1;    // so that nuk_start - col_offset = 1, (nuk_start + 1) - col_offset = 2, and so on.

	for (int j = nuk_start; j <= nuk_end; ++j) {
		// Using column major for lapack
		// Use M[(i-1) + ldM * (j-1)] to access M(i,j)
		if (j % 3 == 1) {     // e.g. if j = 1 or 4, etc
			M[(1 - 1) + ldM * (j - col_offset - 1)] = rho * A * l;
		}
		if (j % 3 == 2) {     // e.g. if j = 2
			M[(1 - 1) + ldM * (j - col_offset - 1)] = rho * A * l;
		}
		if (j % 3 == 0) {     // e.g. if j = 3
			M[(1 - 1) + ldM * (j - col_offset - 1)] = (1.0 / 12) * rho * A * pow(l, 3);
		}
	}
}

void Par_make_K_into_Keff(double* K, int ldK, double* M, int ldM, int nuk_start, int nuk_end, double dt, double beta)
{
	/*
	For use in parallel. The effective stiffness matrix Keff (in banded symmetric form) is
	created between columns nuk_start and nuk_end (both included) by making K into Keff
		Keff = K + (1 / (beta * pow(dt,2))) * M
	ON ENTRY: K is the local part of the global stiffness matrix between nuk_start and nuk_end
	ON EXIT: K is the local part of the global effective stiffness matrix between nuk_start and nuk_end
	Remember that Keff is basically a 5 by 3 block that repeats itself every 3 columns and M consists of just its main diagonal
	*/
	const double dt2 = pow(dt, 2);
	const int col_offset = nuk_start - 1;    // so that nuk_start - col_offset = 1, (nuk_start + 1) - col_offset = 2, and so on.

	for (int j = nuk_start; j <= nuk_end; ++j) {
		// Using column major for lapack
		// Use K[(i-1) + ldK * (j-1)] to access element K(i,j) and M[(i-1) + ldM * (j-1)] to access M(i,j)
		K[(5 - 1) + ldK * (j - col_offset - 1)] += 1.0 / (beta * dt2) * M[(1 - 1) + ldM*(j - col_offset - 1)];
	}
}

#endif