#include "mpi.h"
#include <iostream>
using namespace std;

int main(int argc, char** argv)
{
	int debag = 0;

	// MPI initialization. 
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
	MPI_Comm_size(MPI_COMM_WORLD, &size); 

	// Creation of cartesian system

	MPI_Comm comm_2D;
	int const ndim = 2;       // Dimension of the Cartesian system
	int dims[ndim];           
	int periods[ndim];
	int coords[ndim];         //Process coordinates in Cartesian system
	dims[0] = sqrt(size);     //Works on 1,4 or 9 processes
	dims[1] = sqrt(size);
	periods[0] = 0;
	periods[1] = 0;
	MPI_Cart_create(MPI_COMM_WORLD, ndim, dims, periods, 0, &comm_2D);  
	MPI_Cart_coords(comm_2D, rank, ndim, coords);                      

	// Task
	int N = 240;               // N - problem size
	int m = N / sqrt(size);    // m - subproblem size

	double* T0 = new double[(m + 2) * (m + 2)]; 
	double* T1 = new double[(m + 2) * (m + 2)];  

	if ((rank == 0) && debag) {
		cout << "Parameter "
			<< " Size = " << size << "   dims[0] = " << coords[0] << "   dims[1] = " << coords[1]
			<< " N = " << N << "    m = " << m << endl;
	}

	MPI_Datatype Column, Row;
	MPI_Type_vector(m + 2, 1, m + 2, MPI_DOUBLE, &Column);
	MPI_Type_commit(&Column);
	MPI_Type_vector(1, m + 2, m + 2, MPI_DOUBLE, &Row);
	MPI_Type_commit(&Row);

	// Determining Shear Direction
	int left, right, lower, upper;
	MPI_Cart_shift(comm_2D, 1, 1, &left, &right);
	MPI_Cart_shift(comm_2D, 0, 1, &lower, &upper);
	//  CPU 2 (1,0)   CPU 3 (1,1)   
	//  CPU 0 (0,0)   CPU 1 (0,1)   

	if (debag) cout << "Rank = " << rank
		<< "   Coords[0] = " << coords[0] << "   Coords[1] = " << coords[1]
		<< "   Left = " << left << "   Right = " << right
		<< "   Lower = " << lower << "   Upper = " << upper << endl;

	for (int i = 0; i <= m + 1; i++)
		for (int j = 0; j <= m + 1; j++)
		{
			T0[i * (m + 2) + j] = 50;    // T0[i][j] = 50;
			T1[i * (m + 2) + j] = 50;    // T0[i][j] = 50;
		}

	// Setting boundary conditions
	for (int i = 0; i <= m + 1; i++)
	{
		if (coords[1] == 0)           T0[i * (m + 2)] = 10;             // T0[i][0]   =10;
		if (coords[1] == dims[0] - 1) T0[i * (m + 2) + m + 1] = 10;     // T0[i][m+1] =10;
	}
	for (int j = 0; j <= m + 1; j++)
	{
		if (coords[0] == 0) 	       T0[0 + j] = 10;                // T0[j][0]   =10;
		if (coords[0] == dims[0] - 1)  T0[(m + 1) * (m + 2) + j] = 10;  // T0[j][m+1] =10;
	}

	double alfa = 0.01;
	double h = 1.0 / (N + 1);
	double tau = 0.2 * (h * h * h * h) / (alfa * 2 * h * h);
	double ae = alfa * tau / (h * h);
	double aw = alfa * tau / (h * h);
	double an = alfa * tau / (h * h);
	double as = alfa * tau / (h * h);
	double ap = ae + aw + an + as;
	double time = 0;
	double time_fin = 2.0;

	// Calculations
	double time_s = MPI_Wtime();
	for (time = 0; time < time_fin; time += tau) {

		if ((rank == 0) && (debag)) cout << "time = " << time << endl;

		for (int i = 1; i < m + 1; i++)
			for (int j = 1; j < m + 1; j++)
				T1[i * (m + 2) + j] = (1.0 - ap) * T0[i * (m + 2) + j]
				+ ae * T0[(i + 1) * (m + 2) + j] + aw * T0[(i - 1) * (m + 2) + j]
				+ an * T0[i * (m + 2) + j + 1] + as * T0[i * (m + 2) + j - 1];

		for (int i = 1; i < m + 1; i++)
			for (int j = 1; j < m + 1; j++)
				T0[i * (m + 2) + j] = T1[i * (m + 2) + j];

		// Sending data to dummy cells
		MPI_Status status;
		int tag = 10;
		MPI_Send(&T0[1], 1, Column, left, tag, comm_2D);
		MPI_Recv(&T0[m + 1], 1, Column, right, tag, comm_2D, &status);

		MPI_Send(&T0[m], 1, Column, right, tag, comm_2D);
		MPI_Recv(&T0[0], 1, Column, left, tag, comm_2D, &status);

		MPI_Send(&T0[1 * (m + 2)], 1, Row, lower, tag, comm_2D);
		MPI_Recv(&T0[(m + 1) * (m + 2)], 1, Row, upper, tag, comm_2D, &status);

		MPI_Send(&T0[m * (m + 2)], 1, Row, upper, tag, comm_2D);
		MPI_Recv(&T0[0], 1, Row, lower, tag, comm_2D, &status);
	}
	double time_e = MPI_Wtime() - time_s;

	// Collecting the solution in master process
	double** TG = new double* [N + 2]; 
	for (int i = 0; i < N + 2; i++)
		TG[i] = new double[N + 2];

	int l = 0;
	double* t_send = new double[m * m];
	double* t_recv = new double[N * N];

	for (int i = 1; i < m + 1; i++)
		for (int j = 1; j < m + 1; j++) {
			t_send[l] = T0[i * (m + 2) + j];
			l = l + 1;
		}

	MPI_Gather(&t_send[0], m * m, MPI_DOUBLE,
		&t_recv[0], m * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		l = 0;
		for (int di = 0; di < dims[0]; di++)
			for (int dj = 0; dj < dims[1]; dj++)
				for (int i = 1; i < m + 1; i++)
					for (int j = 1; j < m + 1; j++)
					{
						TG[di * m + i][dj * m + j] = t_recv[l];
						l = l + 1;
					}
	}

	// Result
	if (rank == 0) {
		cout << " Rank = " << rank << endl;
		cout << " Size = " << size << endl;
		cout << " Time work  = " << time_e << endl;
		cout << "T0 = [1][1] = " << T0[1 * (m + 2) + 1] << endl;

		if (debag) {
			for (int i = 1; i < N + 1; i++) {
				for (int j = 1; j < N + 1; j++) {
					cout << int(TG[i][j]) << " ";
				}
				cout << endl;
			}
		}
	}
	MPI_Finalize();
	return 0;
}
