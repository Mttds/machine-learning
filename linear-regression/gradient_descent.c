/*
compile with: gcc -shared -o ./bin/gradient_descent.so -fPIC gradient_descent.c -lm -fopenmp
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/*
cost of f_wb
*/
double compute_cost(double *x, double *y, double w, double b, int m) {
    double total_cost = 0.0;

    for (int i = 0; i < m; i++) {
        double ith_f_wb_x = w * x[i] + b;
        double ith_cost = pow(ith_f_wb_x - y[i], 2);
        total_cost += ith_cost;
    }

    total_cost *= 1.0 / (2 * m);
    return total_cost;
}

/*
gradient function
*/
void compute_gradient(double *x, double *y, double w, double b, int m, double *dj_dw, double *dj_db) {
    *dj_dw = 0.0;
    *dj_db = 0.0;

    double dj_dw_local = 0.0, dj_db_local = 0.0;

    #pragma omp parallel for reduction(+:dj_dw_local, dj_db_local)
    for (int i = 0; i < m; i++) {
        double ith_f_wb = w * x[i] + b;
        double ith_dj_db = ith_f_wb - y[i];
        double ith_dj_dw = (ith_f_wb - y[i]) * x[i];
        dj_db_local += ith_dj_db;
        dj_dw_local += ith_dj_dw;
    }

    dj_db_local *= 1.0 / m;
    dj_dw_local *= 1.0 / m;
    *dj_dw = dj_dw_local;
    *dj_db = dj_db_local;
}

/*
gradient descent to find w and x (single feature x1)
*/
void gradient_descent(double *x, double *y, double w_in, double b_in, int m, double alpha, int num_iters, double *w_final, double *b_final) {
    double w = w_in;
    double b = b_in;

    double *J_history = (double *)malloc(num_iters * sizeof(double));
    double *w_history = (double *)malloc(num_iters * sizeof(double));

    // set the number of threads for OpenMP
    int procs = omp_get_num_procs();
    omp_set_num_threads(procs);

    for (int i = 0; i < num_iters; i++) {
        double dj_dw = 0.0, dj_db = 0.0;

        // Calculate gradients
        compute_gradient(x, y, w, b, m, &dj_dw, &dj_db);

        // Update Parameters using w, b, alpha, and gradient
        w -= alpha * dj_dw;
        b -= alpha * dj_db;

        // Save cost J at each iteration
        if (i < 1000000) { // Prevent resource exhaustion
            double cost = compute_cost(x, y, w, b, m);
            J_history[i] = cost;
        }

        // Print cost at intervals
        if (i % (int)ceil(num_iters / 10.0) == 0) {
            w_history[i] = w;
            printf("Iteration %4d: Cost %.2f\n", i, J_history[i]);
        }
    }

    *w_final = w;
    *b_final = b;

    // Clean up memory
    free(J_history);
    free(w_history);
}
