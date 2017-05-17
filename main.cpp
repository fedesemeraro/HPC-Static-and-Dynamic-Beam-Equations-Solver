/*
 AE3-422 High-performance Computing
 Coursework

 Federico Semeraro
 CID: 00862704
 26/03/2017
*/

/*
	This program solves the equilibrium equation
		M udotdot + C udot + K u = F       (with no damping => C = 0)
of a beam fixed at both ends and loaded with a distributed and a concentrated load.
	The user can easily select which of the following cases he wants to execute by using
the makefile (e.g. type "makefile compile" to compile, "makefile task3" to run task 3 with
the values specified in the handout):
	- Task 1: solve static beam equation in serial
	- Task 2: solve dynamic beam equation in serial using an explicit time integration scheme (Central Differences Method)
	- Task 3: solve dynamic beam equation in serial using an implicit time integration scheme  (Newmark Method)
	- Task 4: solve dynamic beam equation in parallel using an explicit time integration scheme (Central Differences Method)
	- Task 5: solve dynamic beam equation in parallel using an implicit time integration scheme  (Newmark Method)

	The units of every physical quantity are in SI.
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
using namespace std;

// Header files for parallelisation:
#include "cblas.h"
#include "mpi.h"

// Header files for the tasks:
#include "fillmatricesserial.h"
#include "fillmatricesparallel.h"
#include "solversserial.h"
#include "solversparallel.h"
#include "printtofile.h"

int main(int argc, char* argv[]) {
	/*
	- The stiffness matrix K is stored as a banded symmetric matrix with 5 rows(including the main diagonal).
	- The zero boundary conditions at both ends are handled by focusing only on the remaining nuk unkown degrees of freedom.
	*/

	// Initialise MPI. Only use one communicator throughout, MPI_COMM_WORLD:
	int retval;
	retval = MPI_Init(&argc, &argv);
	if (retval != MPI_SUCCESS)
		cout << "An error occurred initialising MPI" << endl;
	int rank;                                 // identifier of current process: 0, 1, 2 ...
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);     // Obtain rank of current process
	int size;                                 // total no. of parallel processes running
	MPI_Comm_size(MPI_COMM_WORLD, &size);     // Obtain total no. of parallel processes running

	// Command line arguments (passed by makefile with values specified in the handout):
	double L, b, h, E, rho, T, T_l;
	int N_e, N_t, task;
	// Note that argv[0] is the name of the function
	// sscanf automatically validates the inputs:
	sscanf(argv[1], "%lf", &L);              // beam length (m)
	sscanf(argv[2], "%lf", &b);              // beam width (m)
	sscanf(argv[3], "%lf", &h);              // beam height (m)
	sscanf(argv[4], "%lf", &E);              // beam Young's modulus (Pa)
	sscanf(argv[5], "%lf", &rho);            // beam density (kg/m^3)
	sscanf(argv[6], "%d", &N_e);             // number of beam ELEMENTS, not nodes
	// Warning if N_e is not even (to make sure there is a node under the concentrated force:
	if (N_e % 2 != 0)
		cout << "Error: the number of elements, " << N_e << ", is not even" << endl;
	sscanf(argv[7], "%lf", &T);              // time (s, but doesn't matter here) goes from [0,T]
	sscanf(argv[8], "%d", &N_t);             // no. of time steps
	sscanf(argv[9], "%lf", &T_l);            // loading time (s): [0,T_l]
	sscanf(argv[10], "%d", &task);           // choose task (1, 2, 3, 4 , 5)

	// Other required values:
	const double        q_y = -1000;         // transverse distributed load (N/m), constant throughout
	const double        F_y = -1000;         // transverse distributed load (N) acting at the midpoint
	const int DOFs_per_node = 3;             // no. of DOFs per node
	const int           ldK = 5;             // leading dimension of K (no. of diagonals of banded symmetric matrix, including the main one)
	const int           ldM = 1;             // leading dimension of M (no. of diagonals of banded symmetric matrix, including the main one)

	// "Derived" values:
	const double      l = L / N_e;                           // element length (m)
	const double     dt = T / (N_t - 1);                     // time step (s)
	const int       N_l = 1 + round(T_l / dt);               // no. of time steps for loading
	const double      A = b * h;                             // beam cross-sectional area (m^2)
	const double      I = (1 / 12.0) * b * pow(h, 3);        // beam second moment of area (m^2)
    const int         N = ((N_e + 1) - 2);                   // no. of unkown nodes (i.e. nodes minus the two boundary ones)
    const int       nuk = DOFs_per_node * ((N_e + 1) - 2);   // no. of unkown DOFs


	// Perform the task chosen by the user:
	// The contents of each case are within brackets to give it scope, that way we can declare local variables inside it
	switch (task) {
		case 1 :
		{
			// Initialise arrays:
			double* K = new double[ldK * nuk];                // stiffness matrix of unkown DOFs only
			double* u = new double[nuk];                      // displacement vector (of unkown DOFs only)
			double* u_analytical = new double[(N_e + 1) - 2]; // displacement vector calculated analytically (only vert. displ.) (unkown DOFs only)
			double* F = new double[nuk];                      // vector of applied forces (on unkown DOFs only)

			// Fill the stiffness matrix K and vector of applied forces F:
			FillK(K, ldK, nuk, A, E, l, I);
			FillF(F, nuk, l, q_y, F_y);

			// Solve static equation and print results to file:
			SolveStaticEq(u, u_analytical, nuk, K, ldK, F, L, E, I, q_y, F_y, N, l);
			Print_results_task1_to_file(u, u_analytical, N, l);

			// Clean up heap
			delete[] K;
			delete[] u;
			delete[] u_analytical;
			delete[] F;
		}
			break;

		case  2 :
		{
			// Initialise arrays:
			double* K = new double[ldK * nuk];                // stiffness matrix of unkown DOFs only
			double* M = new double[ldM * nuk];                // mass matrix of unkown DOFs only
			double* u = new double[nuk];                      // displacement vector (of unkown DOFs only)
			double* F = new double[nuk];                      // vector of applied forces (on unkown DOFs only)
			double* deflection_midpoint = new double[N_t];    // vertical deflections of midpoint (m) at each moment in time

			// Fill the stiffness matrix K, vector of applied forces F and mass matrix M:
			FillK(K, ldK, nuk, A, E, l, I);
			FillF(F, nuk, l, q_y, F_y);
			FillM(M, ldM, nuk, rho, A, l);

			// Solve dynamic equation explitic and print results to file
			SolveDynamicEqExpl(deflection_midpoint, N_t, dt, N_l, nuk, M, ldM, K, ldK, F);
			Print_deflection_midpoint_to_file(deflection_midpoint, T, N_t, dt, T_l);

			// Clean up heap
			delete[] K;
			delete[] M;
			delete[] u;
			delete[] F;
			delete[] deflection_midpoint;
		}
			break;

		case  3:
		{
			// Initialise arrays:
			double* K = new double[ldK * nuk];                // stiffness matrix of unkown DOFs only
			double* M = new double[ldM * nuk];                // mass matrix of unkown DOFs only
			double* u = new double[nuk];                      // displacement vector (of unkown DOFs only)
			double* F = new double[nuk];                      // vector of applied forces (on unkown DOFs only)
			double* deflection_midpoint = new double[N_t];    // vertical deflections of midpoint (m) at each moment in time

			// Fill the stiffness matrix K, vector of applied forces F and mass matrix M:
			FillK(K, ldK, nuk, A, E, l, I);
			FillF(F, nuk, l, q_y, F_y);
			FillM(M, ldM, nuk, rho, A, l);

			// Solve dynamic equation implicit and print results to file
			SolveDynamicEqImpl(deflection_midpoint, N_t, dt, N_l, nuk, M, ldM, K, ldK, F);
			Print_deflection_midpoint_to_file(deflection_midpoint, T, N_t, dt, T_l);

			// Clean up heap
			delete[] K;
			delete[] M;
			delete[] u;
			delete[] F;
			delete[] deflection_midpoint;
		}
			break;

		case 4:
		{	
			// Warning if the number of processes is not exactly two:
			if (size != 2)
				cout << "Error: task4 only works with 2 processes, not " << size << endl;

			// Determine parameters for the parallelisation:
			const int   node_midpoint = N / 2 + 1;                               // node where the midpoint is
			const int    nuk_midpoint = nuk / 2 + 1;                             // unknown where the midpoint is (remember that nuk is always odd)
			const int         nuk_loc = (node_midpoint + 1) * DOFs_per_node;     // no. of local unknowns in current process (first and last included)
			const int           N_loc = node_midpoint + 1;                       // no. of local nodes in current process (first and last included)

			int nuk_start, nuk_end;      // unknown where the current process starts (count from 1) and ends (itself included)
			// Both arrays share the three middle nodes:
			if (rank == 0) {
				nuk_start = 1;
				nuk_end = nuk_start + nuk_loc - 1;
			}
			else if (rank == 1) {
				nuk_end = nuk;
				nuk_start = nuk_end - nuk_loc + 1;
			}
			
			// Initialise local arrays (e.g. K is the part of the global stiffness matrix allocated to the current process):
			double* K = new double[ldK * nuk_loc];                // stiffness matrix of local unkown DOFs only
			double* M = new double[ldM * nuk_loc];                // mass matrix of local unkown DOFs only
			double* u = new double[nuk_loc];                      // displacement vector (of local unkown DOFs only)
			double* F = new double[nuk_loc];                      // vector of applied forces (on local unkown DOFs only)
			double* deflection_midpoint = new double[N_t];        // vertical deflections of midpoint (m) at each moment in time

			// Fill the local stiffness matrix K, vector of applied forces F and mass matrix M:
			ParFillK(K, ldK, nuk_start, nuk_end, A, E, l, I);
			ParFillFtask4(F, nuk_start, nuk_end, nuk_midpoint, rank, l, q_y, F_y);
			ParFillM(M, ldM, nuk_start, nuk_end, rho, A, E, l);

			// Solve dynamic equation explicit:
			ParSolveDynamicEqExpl(deflection_midpoint, N_t, dt, N_l, nuk_loc, nuk_midpoint, DOFs_per_node, M, ldM, K, ldK, F);

			// Print results to file (using the process 0):
			if (rank == 0)
				Print_deflection_midpoint_to_file(deflection_midpoint, T, N_t, dt, T_l);

			// Clean up heap:
			delete[] K;
			delete[] M;
			delete[] u;
			delete[] F;
			delete[] deflection_midpoint;
		}
			break;

		case 5:
		{
			// Determine parameters for the parallelisation:
			const int              nb = ceil(double(nuk) / size);   // block size (no. of cols), necessary: size * nb >= nuk (block column, non-cyclic)
			int               nuk_loc = nb;                         // no. of local unknowns in processor (first and last included)
			// For last process, just use the actual columns it needs. Exceptional case when size = 1, it should remain what it is:
			if ((rank == size - 1) && (size != 1))  nuk_loc = nuk % nb;
			const int       nuk_start = 1 + rank * nb;              // unknown where the current process starts (starts counting at 1)
			const int         nuk_end = nuk_start + nuk_loc - 1;    // unknown where the current process end (itself included)
			const int    nuk_midpoint = nuk / 2 + 1;                // unknown where the midpoint is (remember that nuk is always odd)
			const int   rank_midpoint = (nuk_midpoint - 1) / nb;    // rank where the midpoint is
			const int relpos_midpoint = nuk_midpoint - rank_midpoint * nb;   // midpoint relative position inside the current process displacement vector (1 is the first column in rank_midpoint)

			// Initialise local arrays (e.g. K is the part of the global stiffness matrix allocated to the current process):
			double* K = new double[ldK * nb];                // stiffness matrix of local unkown DOFs only
			double* M = new double[ldM * nb];                // mass matrix of local unkown DOFs only
			double* u = new double[nb];                      // displacement vector (of local unkown DOFs only)
			double* F = new double[nb];                      // vector of applied forces (on local unkown DOFs only)
			double* deflection_midpoint = new double[N_t];   // vertical deflections of midpoint (m) at each moment in time

			// Fill the local stiffness matrix K, vector of applied forces F and mass matrix M:
			ParFillK(K, ldK, nuk_start, nuk_end, A, E, l, I);
			ParFillF(F, nuk_start, nuk_end, rank, rank_midpoint, relpos_midpoint, l, q_y, F_y);
			ParFillM(M, ldM, nuk_start, nuk_end, rho, A, E, l);

			// Solve dynamic equation implicit:
			ParSolveDynamicEqImpl(deflection_midpoint, rank_midpoint, relpos_midpoint, N_t, dt, N_l, nuk, nb, nuk_loc, nuk_start, nuk_end, M, ldM, K, ldK, F);

			// Print results to file (using the process where the global midpoint is):
			if (rank == rank_midpoint)
				Print_deflection_midpoint_to_file(deflection_midpoint, T, N_t, dt, T_l);

			// Clean up heap:
			delete[] K;
			delete[] M;
			delete[] u;
			delete[] F;
			delete[] deflection_midpoint;
		}
		break;
	}

	MPI_Finalize();
	return 0;
}
