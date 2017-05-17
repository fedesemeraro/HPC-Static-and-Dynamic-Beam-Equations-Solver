/*
To print results to file
*/

#ifndef	PRINTTOFILE_H	
#define PRINTTOFILE_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
using namespace std;


void Print_results_task1_to_file(double* u, double* u_analytical, int N, double l)
{
	/*
	Prints to "static_solution_and_analytical.txt":
		- The number of nodes N + 2 in the first row
		- In the following rows
		- Column 1: x (distance along beam)
		- Column 2: calculated displacement (only vertical component)
		- Column 3: analytical solution (only vertical component)
	*/
	const int precision = 8;
	const int width = 15;
	// Open or create file (overwrite mode) in the current directory:
	ofstream f_out("static_solution_and_analytical.txt");
	// Test if we were able to open the file for writing:
	if (!f_out.good()) {
		cout << "Error: unable to open output file: static_solution_and_analytical.txt" << endl;
	}
	else {
		// Write N + 2 on the first row:
		f_out.precision(12);
		f_out.width(width);
		f_out << N + 2 << endl;

		// Write out the values of x, vertical part of calculated solution and vertical analytical solution in three columns:

		// First node:
		double x = 0;                     // node position
		f_out.precision(precision);       // precision: significant figures
		f_out.width(width);               // minimum space (occupied + blank)
		f_out << x;
		f_out.precision(precision);
		f_out.width(width);
		f_out << 0;
		f_out.precision(precision);
		f_out.width(width);
		f_out << 0 << endl;

		// Unknown nodes:
		x = l;                            // position of first unkown node
		for (int i = 1; i <= N; ++i) {
			f_out.precision(precision);
			f_out.width(width);
			f_out << x;

			f_out.precision(precision);
			f_out.width(width);
			f_out << u[(2 + 3 * (i - 1)) - 1];     // 2 + 3 * (i - 1) is the vertical displacement at unkown node i

			f_out.precision(precision);
			f_out.width(width);
			f_out << u_analytical[i - 1] << endl;

			x += l;
		}

		// Last node:
		f_out.precision(precision);       // precision: significant figures
		f_out.width(width);               // minimum space (occupied + blank)
		f_out << x;
		f_out.precision(precision);
		f_out.width(width);
		f_out << 0;
		f_out.precision(precision);
		f_out.width(width);
		f_out << 0 << endl;

		// Close file
		f_out.close();
		cout << "Written file static_solution_and_analytical.txt successfully" << endl;
	}
}

void Print_deflection_midpoint_to_file(double* deflection_midpoint, double T, int N_t, double dt, double T_l)
{
	/*
	Prints to "deflection_midpoint.txt":
		- T, N_t, T_l on first row
		- deflection_midpoint, time after that
	*/
	const int precision = 8;
	const int width = 15;

	// Open or create file (overwrite mode) in the current directory:
	ofstream f_out("deflection_midpoint.txt");

	// Test if we were able to open the file for writing:
	if (!f_out.good()) {
		cout << "Error: unable to open output file: deflection_midpoint.txt" << endl;
	}
	else {
		// Write T, N_t and T_l on the first row:
		f_out.precision(precision);       // precision: significant figures
		f_out.width(width);               // minimum space (occupied + blank)
		f_out << T;

		f_out.precision(12);
		f_out.width(width);
		f_out << N_t;

		f_out.precision(precision);
		f_out.width(width);
		f_out << T_l << endl;

		// Write out the values of delfection_midpoint and time in two columns:
		double time = 0;    // time at n = 0
		for (unsigned int n = 0; n < N_t; ++n) {
			f_out.precision(precision);
			f_out.width(width);
			f_out << deflection_midpoint[n];

			f_out.precision(precision);
			f_out.width(width);
			f_out << time << endl;

			time += dt;
		}
		// Close file
		f_out.close();
		cout << "Written file deflection_midpoint.txt successfully" << endl;
	}
}

#endif