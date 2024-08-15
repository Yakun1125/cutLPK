#include "mps_lp.h"
#include "wrapper_lpk.h"
#include <time.h>

/*
main arguments are
num_cluster
lpk_version: no_fair strict_fair soft_fair
The others are parameters of cuPDLP
*/

int main(int argc, char** argv) {
    char* fname = "./iris.csv";
    int num_cluster = 2;
    int max_init = 1.5e7;
    int max_per_iter = 3e7;
    int warm_start = 2;
    int t_upper_bound = 2;
    double time_limit_iter = 180;
    char* solver = "gpu";

    double time_limit_all = 14400;
    double initial_pdlp_tol = 1e-6;
    double initial_time_limit = 180;
    double tolerance_per_iter = 1e-4;
    double ub_pdlp_tol = 1e-6;
    double cuts_vio_tol = 1e-4;
    double cuts_act_tol = 1e-4;
    double opt_gap = 1e-4;

    int random_seed = 12345;

    bool lb_ub_scheme = false;




    // Simple command-line argument parsing
    for (int i = 1; i < argc; i++) { // Start from 1 to skip the program name
        if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            fname = argv[++i]; // Increment 'i' to skip the filename argument
        }
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            num_cluster = atoi(argv[++i]); // Convert string to int
            t_upper_bound = num_cluster;
        }
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            max_init = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            max_per_iter = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            warm_start = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            t_upper_bound = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-tl_iter") == 0 && i + 1 < argc) {
            time_limit_iter = atof(argv[++i]);
            initial_time_limit = time_limit_iter;
        }
        else if (strcmp(argv[i], "-tl_all") == 0 && i + 1 < argc) {
            time_limit_all = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-tl_init") == 0 && i + 1 < argc) {
            initial_time_limit = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-solver") == 0 && i + 1 < argc) {
            solver = argv[++i];
        }
        else if (strcmp(argv[i], "-tol_init") == 0 && i + 1 < argc) {
            initial_pdlp_tol = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-tol_iter") == 0 && i + 1 < argc) {
            tolerance_per_iter = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-ub_tol") == 0 && i + 1 < argc) {
            ub_pdlp_tol = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-cuts_vio_tol") == 0 && i + 1 < argc) {
            cuts_vio_tol = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-cuts_act_tol") == 0 && i + 1 < argc) {
            cuts_act_tol = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-opt_gap") == 0 && i + 1 < argc) {
            opt_gap = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc) {
            random_seed = atoi(argv[++i]);
        }
        else {
            fprintf(stderr, "Unknown argument: %s, ignored\n", argv[i]);
            //fprintf(stderr, "Usage: %s [-f filename] [-c num_cluster] [-i max_init] [-p max_per_iter]\n", argv[0]);
            //return 1;
        }
    }
    printf("start_solving\n");



    clock_t start, end;
    double cpu_time_used;
    double wall_time_used;

    struct timespec wall_start, wall_end;

    // Get the start time
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    start = clock();
    solve_lpk(fname, num_cluster, max_init, max_per_iter, warm_start, t_upper_bound, initial_time_limit, time_limit_iter, time_limit_all, solver
        , initial_pdlp_tol, tolerance_per_iter, ub_pdlp_tol, cuts_vio_tol, cuts_act_tol, opt_gap, random_seed, lb_ub_scheme);
    end = clock();

    // Get the end time
    clock_gettime(CLOCK_MONOTONIC, &wall_end);

    // Calculate the elapsed time in seconds
    wall_time_used = wall_end.tv_sec - wall_start.tv_sec;
    wall_time_used += (wall_end.tv_nsec - wall_start.tv_nsec) / 1000000000.0;

    printf("Elapsed wall time: %f seconds\n", wall_time_used);

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CPU Time consumed: %f seconds\n", cpu_time_used);
    return 0;
}