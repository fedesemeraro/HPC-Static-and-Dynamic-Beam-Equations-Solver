/*
Solvers for tasks 1, 2 and 3 (serial)
*/

#ifndef SOLVERSSERIAL_H	
#define SOLVERSSERIAL_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
using namespace std;

#define F77NAME(x) x##_
extern "C" {
	void F77NAME(dpbsv) (const char& UPLO, const int& N, const int& KD, const int& NRHS,
		double* AB, const int& LDAB, double* B, const int& LDB, int* INFO);
	void F77NAME(dpbtrf)(const char& UPLO, const int& N, const int& KD, double* AB,
		const int& LDAB, const int* INFO);
	void F77NAME(dpbtrs)(const char& UPLO, const int& N, const int& KD, const int& NRHS,
		double* AB, const int& LDAB, double* B, const int& LDB, int* INFO);
}


// FOR TASK 1:
void SolveStaticEq(double* u, double* u_analytical, int nuk, double* K, int ldK, double* F, double L, double E, double I, double q_y, double F_y, int N, double l)
{
	/*
	Solves the static beam equation M u = F and outputs the solution u.
	Calculates the vertical displacement of the analytical solution u_analytical and outputs it.
	*/

	// Solve beam equation M u = F using dpbsv (LAPACK):
	cblas_dcopy(nuk, F, 1, u, 1);     // u = F (because of how dpbsv works)
	int info;
	// u = x, where x is the solution to K x = u; K is changed on exit
	F77NAME(dpbsv)('U', nuk, ldK - 1, 1, K, ldK, u, nuk, &info);
	if (info)
		cout << "Error in solve: " << info << endl;

	// Output analytical solution (only vertical displacements considered):
	// N is the number of nodes minus the 2 boundary nodes where we know the values:
	double x = l;      // length along node (m) (starts at first unkown node)
	// For nodes <= midpoint (N / 2 + 1):
	for (int i = 1; i <= N / 2 + 1; ++i) {
		// Distributed load:
		u_analytical[i - 1] = q_y * pow(x, 2) * pow((L - x), 2) / (24 * E * I);
		// Concentrated load:
		u_analytical[i - 1] += F_y * pow(x, 2) * (3 * L - 4 * x) / (48 * E *I);
		x += l;
	}
	// Use symmetry to fill remaining nodes:
	for (int i = N / 2 + 2; i <= N; ++i) {
		u_analytical[i - 1] = u_analytical[(N - i + 1) - 1];
	}
}

// FOR TASK 2:
void SolveDynamicEqExpl(double* deflection_midpoint, int N_t, double dt, int N_l, int nuk, double* M, int ldM, double* K, int ldK, double* Ftotal)
{
	/*
	Solves dynamic beam equation using an explicit scheme (central differences method):
		M * u_{n+1} = dt2 * F_n - dt2 * K * u_n + M * (2 * u_n - u_{n-1}).
	- Initial conditions: u_0 = u_{-1} = 0.
	- The loading is linear between 0 and T_l (which corresponds to N_l), constant after then.
	*/

	// Warning for N_l < 1:
	if (N_l < 1)
		cout << "Error: the loading time chosen would correspond to instantaneous loading." << endl;

	// Initialise variables:
	double*     F = new double[nuk];         // F at n
	double* Fcopy = new double[nuk];         // It is needed later because of how cblas_dsbmv works
	double*  Minv = new double[ldM * nuk];   // Inverse of M
	double*     u = new double[nuk];         // u at n
	double*    up = new double[nuk];         // u previous: u at (n - 1)
	double*   RHS = new double[nuk];         // RHS of the equation to solve
	const double dt2 = pow(dt, 2);           // because I don't like typing pow(dt,2)

	// Set initial conditions:
	fill_n(F, nuk, 0);                       // F at n = 0: initialise to 0
	fill_n(u, nuk, 0);                       // u at n = 0: initialise to 0
	fill_n(up, nuk, 0);                      // u at n = -1: initialise to 0

											 // Compute Minv (easy since M is diagonal):
	for (int i = 0; i < ldM * nuk; ++i)
		Minv[i] = 1 / M[i];

	/*
	Solve for u_{n+1} in each iteration n+1:
		u_{n+1} = Minv * (dt2 * F_n - dt2 * K * u_n + M * (2 * u_n - u_{n-1}))
	and record deflection of midpoint:
	*/
	// Record deflection of midpoint at n = 0:
	deflection_midpoint[0] = u[nuk / 2];       // remember that nuk/2 is int division and u's domain is [0, nuk)
	for (int n = 1; n < N_t; ++n) {
		// Increase load lineary while in [1, N_l], then constant at Ftotal:
		if (n <= N_l)
			cblas_daxpy(nuk, 1.0 / N_l, Ftotal, 1, F, 1);   // F = 1 / N_l * Ftotal + F  (linear loading)

		/*
		Build
			RHS = dt2 * F_n - dt2 * K * u_n + M * (2 * u_n - u_{n-1})
		starting from u_{n-1} and solve
			u_{n+1} = Minv * RHS
		to get up = u_{n+1}. Then, swap u and up for next iteration
		*/
		cblas_dcopy(nuk, up, 1, RHS, 1);             // RHS = up
		cblas_dscal(nuk, -1.0, RHS, 1);              // RHS = -1.0 * RHS
		cblas_daxpy(nuk, 2.0, u, 1, RHS, 1);         // RHS = 2.0 * u + RHS
		cblas_dcopy(nuk, F, 1, Fcopy, 1);            // Fcopy = F
		// Fcopy = 1.0 * M * RHS + dt2 * Fcopy  (this is why we need Fcopy):
		cblas_dsbmv(CblasColMajor, CblasUpper, nuk, ldM - 1, 1.0, M, ldM, RHS, 1, dt2, Fcopy, 1);
		cblas_dcopy(nuk, Fcopy, 1, RHS, 1);          // RHS = Fcopy
		// RHS = -dt2 * K * u + RHS:
		cblas_dsbmv(CblasColMajor, CblasUpper, nuk, ldK - 1, -dt2, K, ldK, u, 1, 1.0, RHS, 1);

		// up = 1.0 * Minv * RHS + 0.0 * up:
		cblas_dsbmv(CblasColMajor, CblasUpper, nuk, ldM - 1, 1.0, Minv, ldM, RHS, 1, 0.0, up, 1);

		// Update u and up for this step:
		cblas_dswap(nuk, up, 1, u, 1);               // swap up and u

		// Record midpoint deflection at this n
		deflection_midpoint[n] = u[nuk / 2];         // remember that nuk/2 is int division and u's domain is [0, nuk)
	}

	// Clean up heap
	delete[] F;
	delete[] Fcopy;
	delete[] Minv;
	delete[] u;
	delete[] up;
}

// FOR TASK 3:
void SolveDynamicEqImpl(double* deflection_midpoint, int N_t, double dt, int N_l, int nuk, double* M, int ldM, double* K, int ldK, double* Ftotal)
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

	// Initialise variables:
	double*     F = new double[nuk];              // F at n
	double* Fcopy = new double[nuk];              // It is needed later because of how cblas_dsbmv works
	double*  Keff = new double[ldK * nuk];        // for Newmark's method
	double*     u = new double[nuk];              // displacement at n
	double*    du = new double[nuk];              // velocity at n (dot u)
	double*   ddu = new double[nuk];              // acceleration at n (dot dot u)
	double*    up = new double[nuk];              // dispalacement at previous n: u at n - 1
	double*   dup = new double[nuk];              // velocity at previous n: du at n - 1
	double*  ddup = new double[nuk];              // acceleration at previous n: ddu at n - 1
	const double  beta = 1 / 4.0;                 // for Newmark's method
	const double gamma = 1 / 2.0;                 // for Newmark's method
	const double   dt2 = pow(dt, 2);              // because I don't like typing pow etc.
	int info;                                     // for LAPACK subroutines

												  // Set initial conditions:
	fill_n(F, nuk, 0);                            // F at n = 0: initialise to 0
	fill_n(up, nuk, 0);                           // u at n = 0: initialise to 0
	fill_n(dup, nuk, 0);                          // du at n = 0: initialise to 0
	fill_n(ddup, nuk, 0);                         // ddu at n = 0: initialise to 0

	/*
	Generate
		Keff = 1 / (beta * dt2) * M + K:
	*/
	cblas_dcopy(ldK*nuk, K, 1, Keff, 1);          // Keff = K
	// Use the fact that M consists of just one diagonal and perform addition element-wise:
	for (int j = 1; j <= nuk; ++j) {
		Keff[(5 - 1) + ldK * (j - 1)] += 1.0 / (beta * dt2) * M[(1 - 1) + ldM*(j - 1)];
	}

	/*
	Solve for u_{n+1}, then ddu_{n+1}, then du_{n+1} in each iteration n + 1.
	Record deflection of midpoint:
	*/
	// Computes the Cholesky factorization of the real symmetric positive definite band matrix Keff:
	F77NAME(dpbtrf)('U', nuk, ldK - 1, Keff, ldK, &info);
	if (info)
		cout << "Error in dpbtrf: " << info << endl;

	// Record deflection of midpoint at n = 0
	deflection_midpoint[0] = up[nuk / 2];         // remember that nuk/2 is int division and up's domain is [0, nuk)

	for (int n = 1; n < N_t; ++n) {
		// Increase load lineary while in [1, N_l], then constant at Ftotal:
		if (n <= N_l)
			cblas_daxpy(nuk, 1.0 / N_l, Ftotal, 1, F, 1);   // F = 1 / N_l * Ftotal + F  (linear loading)

		/*
		Build u to
			RHS = F + M * (1/(beta * dt2) * up + 1/(beta * dt) * dup + (1/(2.0 * beta) - 1) * ddup)
		starting from ddup and solve
			Keff * u_{n+1} = RHS = u
		to get u = u_{n+1}:
		*/
		cblas_dcopy(nuk, ddup, 1, u, 1);                        // u = ddup
		cblas_dscal(nuk, (1 / (2.0 * beta) - 1.0), u, 1);       // u = 1/(2.0 * beta) * u
		cblas_daxpy(nuk, 1 / (beta * dt), dup, 1, u, 1);        // u = 1/(beta * dt) * dup + u
		cblas_daxpy(nuk, 1 / (beta * dt2), up, 1, u, 1);        // u = 1/(beta * dt2) * up + u
		cblas_dcopy(nuk, F, 1, Fcopy, 1);                       // Fcopy = F
		// Fcopy = 1.0 * M * u + 1.0 * Fcopy  (this is why we need Fcopy):
		cblas_dsbmv(CblasColMajor, CblasUpper, nuk, ldM - 1, 1.0, M, ldM, u, 1, 1.0, Fcopy, 1);
		cblas_dcopy(nuk, Fcopy, 1, u, 1);                       // u = Fcopy;

		// Solve linear equation using dpbtrs (LAPACK)
		// u = x, where x is the solution to Keff * x = u
		F77NAME(dpbtrs)('U', nuk, ldK - 1, 1, Keff, ldK, u, nuk, &info);
		if (info)
			cout << "Error in dpbtrs: " << info << endl;

		/*
		Compute the acceleration:
			ddu = 1/(beta * dt2) * u - 1/(beta * dt2) * up - 1/(beta * dt) * dup - (1/(2.0 * beta) - 1) * ddup
		*/
		cblas_dcopy(nuk, ddup, 1, ddu, 1);                       // ddu = ddup
		cblas_dscal(nuk, -(1 / (2.0 * beta) - 1), ddu, 1);       // ddu = - (1/(2.0 * beta) - 1) * ddu
		cblas_daxpy(nuk, -1 / (beta * dt), dup, 1, ddu, 1);      // ddu = -1/(beta * dt) * dup + ddu
		cblas_daxpy(nuk, -1 / (beta * dt2), up, 1, ddu, 1);      // ddu = -1/(beta * dt2) * up + ddu
		cblas_daxpy(nuk, 1 / (beta * dt2), u, 1, ddu, 1);        // ddu = 1/(beta * dt2) * u + ddu

		/*
		Compute the velocity:
			du = dup + dt* (1 - gamma) * ddup + gamma * dt * ddu
		*/
		cblas_dcopy(nuk, ddu, 1, du, 1);                         // du = ddu
		cblas_dscal(nuk, gamma * dt, du, 1);                     // du = gamma * dt * du
		cblas_daxpy(nuk, dt* (1 - gamma), ddup, 1, du, 1);       // du = dt* (1 - gamma) * ddup + du
		cblas_daxpy(nuk, 1.0, dup, 1, du, 1);                    // du = 1.0 * dup + du

		// Record deflection of midpoint
		deflection_midpoint[n] = u[nuk / 2];       // remember that nuk/2 is int division and u's domain is [0, nuk)

		// Update previous displacement, velocity and acceleration for next step
		cblas_dcopy(nuk, u, 1, up, 1);           // up = u
		cblas_dcopy(nuk, du, 1, dup, 1);         // dup = du
		cblas_dcopy(nuk, ddu, 1, ddup, 1);       // ddup = ddu
	}

	// Clean up heap
	delete[] F;
	delete[] Fcopy;
	delete[] Keff;
	delete[] u;
	delete[] du;
	delete[] ddu;
	delete[] up;
	delete[] dup;
	delete[] ddup;
}

#endif