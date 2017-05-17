/*
Solvers for tasks 4 and 5 (parallel)
*/

#ifndef SOLVERSPARALLEL_H	
#define SOLVERSPARALLEL_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
using namespace std;

#include "cblas.h"
#include "mpi.h"

#define F77NAME(x) x##_
extern "C" {
	void F77NAME(pdpbtrf)(const char& UPLO, const int& N, const int& BW, double* A, const int& JA,
		const int* DESCA, double* AF, const int& LAF, double* WORK, const int& LWORK, int* INFO);
	void F77NAME(pdpbtrs)(const char& UPLO, const int& N, const int& BW, const int& NRHS, double* A, const int& JA,
		const int* DESCA, double* B, const int& IB, const int* DESCB, double* AF, const int& LAF,
		double* WORK, const int& LWORK, int* INFO);
	void Cblacs_pinfo(int*, int*);
	void Cblacs_get(int, int, int*);
	void Cblacs_gridinit(int*, char*, int, int);
	void Cblacs_gridinfo(int, int*, int*, int*, int*);
	void Cblacs_gridexit(int);
	void Cblacs_exit(int);
}


// FOR TASK 4:
void ParSolveDynamicEqExpl(double* deflection_midpoint, int N_t, double dt, int N_l, int nuk_loc, int nuk_midpoint, int DOFs_per_node, double* M, int ldM, double* K, int ldK, double* Ftotal)
{
	/*
	Solves dynamic beam equation using an explicit scheme (central differences method):
		M * u_{n+1} = dt2 * F_n - dt2 * K * u_n + M * (2 * u_n - u_{n-1}).
	- Initial conditions: u_0 = u_{-1} = 0.
	- The loading is linear between 0 and T_l (which corresponds to N_l), constant after then.
	The computation is split into two processes, which share the three middle nodes.
	*/

	int rank;                                    // identifier of current process: 0 or 1
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);        // Obtain rank of current process

	// Warning for N_l < 1:
	if (N_l < 1)
		cout << "Error: the loading time chosen would correspond to instantaneous loading." << endl;

	// Initialise variables:
	double*     F = new double[nuk_loc];         // F at n
	double* Fcopy = new double[nuk_loc];         // It is needed later because of how cblas_dsbmv works
	double*  Minv = new double[ldM * nuk_loc];   // Inverse of M
	double*     u = new double[nuk_loc];         // u at n
	double*    up = new double[nuk_loc];         // u previous: u at (n - 1)
	double*   RHS = new double[nuk_loc];         // RHS of the equation to solve
	const double dt2 = pow(dt, 2);               // because I don't like typing pow(dt,2)

												 // Set initial conditions:
	fill_n(F, nuk_loc, 0);                       // F at n = 0: initialise to 0
	fill_n(u, nuk_loc, 0);                       // u at n = 0: initialise to 0
	fill_n(up, nuk_loc, 0);                      // u at n = -1: initialise to 0

	// Compute Minv (easy since M is diagonal):
	for (int i = 0; i < ldM * nuk_loc; ++i)
		Minv[i] = 1 / M[i];

	/*
	Solve for u_{n+1} in each iteration n+1:
		u_{n+1} = Minv * (dt2 * F_n - dt2 * K * u_n + M * (2 * u_n - u_{n-1}))
	and record deflection of midpoint.
	*/
	// Record deflection of midpoint at n = 0 (use process 0):
	if (rank == 0)
		deflection_midpoint[0] = u[nuk_midpoint - 1];

	for (int n = 1; n < N_t; ++n) {
		// Increase load lineary while in [1, N_l], then constant at Ftotal:
		if (n <= N_l)
			cblas_daxpy(nuk_loc, 1.0 / N_l, Ftotal, 1, F, 1);   // F = 1 / N_l * Ftotal + F  (linear loading)

		/*
		Build
			RHS = dt2 * F_n - dt2 * K * u_n + M * (2 * u_n - u_{n-1})
		starting from u_{n-1} and solve
			u_{n+1} = Minv * RHS
		to get up = u_{n+1}. Then, exchange node information between processes and swap u and up for next iteration
		*/
		cblas_dcopy(nuk_loc, up, 1, RHS, 1);             // RHS = up
		cblas_dscal(nuk_loc, -1.0, RHS, 1);              // RHS = -1.0 * RHS
		cblas_daxpy(nuk_loc, 2.0, u, 1, RHS, 1);         // RHS = 2.0 * u + RHS
		cblas_dcopy(nuk_loc, F, 1, Fcopy, 1);            // Fcopy = F
		// Fcopy = 1.0 * M * RHS + dt2 * Fcopy  (this is why we need Fcopy):
		cblas_dsbmv(CblasColMajor, CblasUpper, nuk_loc, ldM - 1, 1.0, M, ldM, RHS, 1, dt2, Fcopy, 1);
		cblas_dcopy(nuk_loc, Fcopy, 1, RHS, 1);          // RHS = Fcopy
														 // RHS = -dt2 * K * u + RHS:
		cblas_dsbmv(CblasColMajor, CblasUpper, nuk_loc, ldK - 1, -dt2, K, ldK, u, 1, 1.0, RHS, 1);

		// up = 1.0 * Minv * RHS + 0.0 * up:
		cblas_dsbmv(CblasColMajor, CblasUpper, nuk_loc, ldM - 1, 1.0, Minv, ldM, RHS, 1, 0.0, up, 1);

		// Exchange values at nodes immediately adjacent to middle node between both processes:
		// Tags for MPI:
		const int  tag_left_nodes = 0;
		const int tag_right_nodes = 1;
		if (rank == 0) {
			MPI_Send(&up[(nuk_loc - 3 * DOFs_per_node + 1) - 1], 1 * DOFs_per_node, MPI_DOUBLE, 1, tag_left_nodes, MPI_COMM_WORLD);
			MPI_Recv(&up[(nuk_loc - 1 * DOFs_per_node + 1) - 1], 1 * DOFs_per_node, MPI_DOUBLE, 1, tag_right_nodes, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		else if (rank == 1) {
			MPI_Recv(&up[(1) - 1], 1 * DOFs_per_node, MPI_DOUBLE, 0, tag_left_nodes, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(&up[(2 * DOFs_per_node + 1) - 1], 1 * DOFs_per_node, MPI_DOUBLE, 0, tag_right_nodes, MPI_COMM_WORLD);
		}

		// Update u and up for this step:
		cblas_dswap(nuk_loc, up, 1, u, 1);               // swap up and u

		// Record midpoint deflection at this n (use process 0):
		if (rank == 0)
			deflection_midpoint[n] = u[nuk_midpoint - 1];
	}

	// Clean up heap
	delete[] F;
	delete[] Fcopy;
	delete[] Minv;
	delete[] u;
	delete[] up;
}

// FOR TASK 5:
void ParSolveDynamicEqImpl(double* deflection_midpoint, int rank_midpoint, int relpos_midpoint, int N_t, double dt, int N_l, int nuk, int nb, int nuk_loc, int nuk_start, int nuk_end, double* M, int ldM, double* K, int ldK, double* Ftotal)
{
	/*
	Solves dynamic beam equation using an implitic scheme (Newmark's method):
		Keff * u_{n+1} = F_{n+1} + M * (1/(beta * dt2) * u_n + 1/(beta * dt) * du_n + (1/(2.0 * beta) - 1) * ddu_n),
	where Keff = 1 / (beta * dt2) * M + K
		ddu_{n+1} = 1/(beta * dt2) * u_{n+1} - 1/(beta * dt2) * u_n - 1/(beta * dt) * du_n - (1/(2.0 * beta) - 1) * ddu_n
		du_{n+1} = du_n + dt * (1 - gamma) * ddu_n + gamma * dt * ddu_{n+1}
	- Initial conditions: u_0 = du_0 = ddu_0 = 0.
	- The loading is linear between 0 and T_l (which corresponds to N_l), constant after then.
	*/

	// Warning for N_l < 1:
	if (N_l < 1)
		cout << "Error: the loading time chosen would correspond to instantaneous loading." << endl;

	// Initialise local (i.e. corresponding part of global array in this process) variables:
	double*     F = new double[nb];              // F at n
	double* Fcopy = new double[nb];              // It is needed later because of how cblas_dsbmv works
	double*     u = new double[nb];              // displacement at n
	double*    du = new double[nb];              // velocity at n (dot u)
	double*   ddu = new double[nb];              // acceleration at n (dot dot u)
	double*    up = new double[nb];              // dispalacement at previous n: u at n - 1
	double*   dup = new double[nb];              // velocity at previous n: du at n - 1
	double*  ddup = new double[nb];              // acceleration at previous n: ddu at n - 1
	const double  beta = 1 / 4.0;                // for Newmark's method
	const double gamma = 1 / 2.0;                // for Newmark's method
	const double   dt2 = pow(dt, 2);             // because I don't like typing pow etc.
	int info;                                    // for ScaLAPACK routines

	// Make K into Keff, which is used in Newmark's method:
	Par_make_K_into_Keff(K, ldK, M, ldM, nuk_start, nuk_end, dt, beta);

	// Set initial conditions:
	fill_n(F, nuk_loc, 0);                            // F at n = 0: initialise to 0
	fill_n(up, nuk_loc, 0);                           // u at n = 0: initialise to 0
	fill_n(dup, nuk_loc, 0);                          // du at n = 0: initialise to 0
	fill_n(ddup, nuk_loc, 0);                         // ddu at n = 0: initialise to 0

	// Set up BLACS context:
	int mypnum;                     // uniquely identifies each process (0 to (nprcs - 1))
	int nprocs;                     // no. of processes available
	// Determine mypnum and nprocs:
	Cblacs_pinfo(&mypnum, &nprocs);
	int contxt;                     // In BLACS the context is like the communicator in MPI
	// Get default system context:
	Cblacs_get(0, 0, &contxt);      // First 0: ignored here; second 0: want it to return handle indicating default system context
	char layout = 'R';              // use row-major natural ordering to map processes to BLACS grid
	int   nprow = 1;                // no. of process rows in the process grid
	int   npcol = nprocs;           // no. of process columns in the process grid
	// Asign available processes into BLACS grid:
	/* This routine creates a nprow by npcol process grid. It uses the first nprow * npcol processes, and
	assigns them to the grid in a row- (used here) or column- major natural ordering.
	- On input, contxt is the system context used for creating the  BLACS context (obtained from Cblacs_get).
	- On output, contxt is the created BLACS context. */
	Cblacs_gridinit(&contxt, &layout, nprow, npcol);
	int myprow, mypcol;             // identificators for blacs row and column
	// Return information on the current grid:
	Cblacs_gridinfo(contxt, &nprow, &npcol, &myprow, &mypcol);

	// Create descriptor for matrix storage for ScaLAPACK:
	int desca[7];
	desca[0] = 501;          // descriptor type (501: 1-by-P grid)
	desca[1] = contxt;       // BLACS context handle
	desca[2] = nuk;          // size of the global array dimension being distributed
	desca[3] = nb;           // block size used to distribute the distributed dimension of the array
	desca[4] = 0;            // process row or column over which the first row or column of the array is distributed
	desca[5] = ldK;          // leading dimension of the local array
	desca[6] = 0;            // reserved, just type 0

	// Create descriptor for RHS storage for ScaLAPACK:
	int descb[7];
	descb[0] = 502;          // descriptor type (502: P-by-1 grid)
	descb[1] = contxt;       // BLACS context handle
	descb[2] = nuk;          // size of the global array dimension being distributed
	descb[3] = nb;           // block size used to distribute the distributed dimension of the array
	descb[4] = 0;            // process row or column over which the first row or column of the array is distributed
	descb[5] = nb;           // leading dimension of the local array
	descb[6] = 0;            // reserved, just type 0

	// Some options for pdpbtrf and pdpbtrs (ScaLAPACK):
	const int    bw = ldK - 1;            // no. of subdiagonals in U of K
	const int    ja = 1;                  // offset index in global array (col)
	const int    ib = 1;                  // offset index in global array (row)
	// const int laf = (nb + 2 * bw) * bw;         // size of user-input auxiliary fillin space af (from online documentation)
	const int   laf = 2 * (nb + 2 * bw) * bw;;    // WHAT IS GOING ON HERE??? (formula on previous line seems to be too small)
	double*      af = new double[laf];    // Auxiliary Fillin Space created during the factorization routine PDPBTRF. must be fed unaltered to PDPBTRS.
	const int lwork = bw * bw;            // size of work (from doc)
	double*    work = new double[lwork];  // temporary workspace for the pdpbtrf and pdpbtrs routines
	const int  nrhs = 1;                  // number of right hand sides

	/*
	Solve for u_{n+1}, then ddu_{n+1}, then du_{n+1} in each iteration n + 1.
	Record deflection of midpoint:
	*/
	// Computes the Cholesky factorization of the real symmetric positive definite band effective stiffness matrix K:
	F77NAME(pdpbtrf)('U', nuk, bw, K, ja, desca, af, laf, work, lwork, &info);
	if (info)
		cout << "Error in pdpbtrf: " << info << endl;

	// Record deflection of global midpoint at n = 0
	if (mypnum == rank_midpoint)
		deflection_midpoint[0] = up[relpos_midpoint - 1];

	for (int n = 1; n < N_t; ++n) {
		// Increase load lineary while in [1, N_l], then constant at Ftotal:
		if (n <= N_l)
			cblas_daxpy(nuk_loc, 1.0 / N_l, Ftotal, 1, F, 1);   // F = 1 / N_l * Ftotal + F  (linear loading)

		/*
		Build u to
			RHS = F + M * (1/(beta * dt2) * up + 1/(beta * dt) * dup + (1/(2.0 * beta) - 1) * ddup)
		starting from ddup and solve
			K * u_{n+1} = RHS = u
		to get u = u_{n+1}:
		*/
		cblas_dcopy(nuk_loc, ddup, 1, u, 1);                      // u = ddup
		cblas_dscal(nuk_loc, (1 / (2.0 * beta) - 1.0), u, 1);     // u = 1/(2.0 * beta) * u
		cblas_daxpy(nuk_loc, 1 / (beta * dt), dup, 1, u, 1);      // u = 1/(beta * dt) * dup + u
		cblas_daxpy(nuk_loc, 1 / (beta * dt2), up, 1, u, 1);      // u = 1/(beta * dt2) * up + u
		cblas_dcopy(nuk_loc, F, 1, Fcopy, 1);                     // Fcopy = F
		// Fcopy = 1.0 * M * u + 1.0 * Fcopy  (this is why we need Fcopy):
		cblas_dsbmv(CblasColMajor, CblasUpper, nuk_loc, ldM - 1, 1.0, M, ldM, u, 1, 1.0, Fcopy, 1);
		cblas_dcopy(nuk_loc, Fcopy, 1, u, 1);                     // u = Fcopy;

		// Solve linear equation using dpbtrs (LAPACK)
		// u = x, where x is the solution to Keff * x = u
		F77NAME(pdpbtrs)('U', nuk, bw, nrhs, K, ja, desca, u, ib, descb, af, laf, work, lwork, &info);
		if (info)
			cout << "Error in pdpbtrs: " << info << endl;

		/*
		Compute the acceleration:
			ddu = 1/(beta * dt2) * u - 1/(beta * dt2) * up - 1/(beta * dt) * dup - (1/(2.0 * beta) - 1) * ddup
		*/
		cblas_dcopy(nuk_loc, ddup, 1, ddu, 1);                       // ddu = ddup
		cblas_dscal(nuk_loc, -(1 / (2.0 * beta) - 1), ddu, 1);       // ddu = - (1/(2.0 * beta) - 1) * ddu
		cblas_daxpy(nuk_loc, -1 / (beta * dt), dup, 1, ddu, 1);      // ddu = -1/(beta * dt) * dup + ddu
		cblas_daxpy(nuk_loc, -1 / (beta * dt2), up, 1, ddu, 1);      // ddu = -1/(beta * dt2) * up + ddu
		cblas_daxpy(nuk_loc, 1 / (beta * dt2), u, 1, ddu, 1);        // ddu = 1/(beta * dt2) * u + ddu

		/*
		Compute the velocity:
			du = dup + dt* (1 - gamma) * ddup + gamma * dt * ddu
		*/
		cblas_dcopy(nuk_loc, ddu, 1, du, 1);                         // du = ddu
		cblas_dscal(nuk_loc, gamma * dt, du, 1);                     // du = gamma * dt * du
		cblas_daxpy(nuk_loc, dt* (1 - gamma), ddup, 1, du, 1);       // du = dt* (1 - gamma) * ddup + du
		cblas_daxpy(nuk_loc, 1.0, dup, 1, du, 1);                    // du = 1.0 * dup + du

		// Record deflection of global midpoint:
		if (mypnum == rank_midpoint)
			deflection_midpoint[n] = u[relpos_midpoint - 1];

		// Update previous displacement, velocity and acceleration for next step:
		cblas_dcopy(nuk_loc, u, 1, up, 1);                           // up = u
		cblas_dcopy(nuk_loc, du, 1, dup, 1);                         // dup = du
		cblas_dcopy(nuk_loc, ddu, 1, ddup, 1);                       // ddup = ddu
	}

	// Release BLACS contexts:
	Cblacs_gridexit(contxt);

	// Clean up heap
	delete[] F;
	delete[] Fcopy;
	delete[] u;
	delete[] du;
	delete[] ddu;
	delete[] up;
	delete[] dup;
	delete[] ddup;
	delete[] af;
	delete[] work;
}

#endif