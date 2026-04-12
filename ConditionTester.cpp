

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <string>
#include <cmath>
#include <vector>
#include <iterator>
#include "Eigen/Dense"
#include "Eigen/SVD"
#include <omp.h>
#include <ctime>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <functional>

// path matrix stored as flat vector in row-major order
// A[p][n] is stored at A[p * N + n]
// P = number of rows, N = number of columns

//############################################
//   HELPERS
//############################################

#include <chrono>

long long current_milliseconds() {
    using namespace std::chrono;

    return duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()
    ).count();
}

int current_minute() {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm* local = std::localtime(&t);
        return local->tm_min;
}

void write_result(int steps, int paths, double absPercentError, double kappa, double runtime) {
    std::ofstream file("results_1.csv", std::ios::app);
    file << steps << "," << paths << "," << absPercentError << "," << kappa << "," << runtime << "\n";
}

//############################################
//    REGRESSION STEP
//############################################

/*
TYPES
1 = Polynomial Deg 2
2 = Polynomial Deg 3
3 = Leandre Deg 2
4 = Leandre Deg 3
5 = Hermite Deg 2
5 = Hermite Deg 3
6 = Laguerre Deg 2
7 = Laguerre Deg 3
*/

std::vector<std::function<double(double)>> basisSet(int type) {
    using f = std::function<double(double)>;

    switch(type) {

        // Polynomial Deg 2
        case 1:
            return {
                [](double x){ return 1.0; },
                [](double x){ return x; },
                [](double x){ return x*x; },
                [](double x){ return 0.0; }
            };

        // Polynomial Deg 3
        case 2:
            return {
                [](double x){ return 1.0; },
                [](double x){ return x; },
                [](double x){ return x*x; },
                [](double x){ return x*x*x; }
            };

        // Legendre Deg 2
        case 3:
            return {
                [](double x){ return 1.0; },
                [](double x){ return x; },
                [](double x){ return (3*x*x - 1)/2.0; },
                [](double x){ return 0.0; }
            };

        // Legendre Deg 3
        case 4:
            return {
                [](double x){ return 1.0; },
                [](double x){ return x; },
                [](double x){ return (3*x*x - 1)/2.0; },
                [](double x){ return (5*x*x*x - 3*x)/2.0; }
            };

        // Hermite Deg 2
        case 5:
            return {
                [](double x){ return 1.0; },
                [](double x){ return x; },
                [](double x){ return x*x - 1.0; },
                [](double x){ return 0.0; }
            };

        // Hermite Deg 3
        case 6:
            return {
                [](double x){ return 1.0; },
                [](double x){ return x; },
                [](double x){ return x*x - 1.0; },
                [](double x){ return x*x*x - 3*x; }
            };

        // Laguerre Deg 2
        case 7:
            return {
                [](double x){ return std::exp(-x/2.0); },
                [](double x){ return std::exp(-x/2.0)*(1 - x); },
                [](double x){ return std::exp(-x/2.0)*(1 - 2*x + x*x/2.0); },
                [](double x){ return 0.0; }
            };

        // Laguerre Deg 3
        case 8:
            return {
                [](double x){ return std::exp(-x/2.0); },
                [](double x){ return std::exp(-x/2.0)*(1 - x); },
                [](double x){ return std::exp(-x/2.0)*(1 - 2*x + x*x*0.5); },
                [](double x){ return std::exp(-x/2.0)*(1 - 3*x + 1.5*x*x - x*x*x/6.0); }
            };

        default:
            throw std::invalid_argument("Invalid basis type");
    }
}

std::vector<long double> regress(const std::vector<double>& X, const std::vector<double>& Y, int regType) {
    const int n = X.size();

    auto basis = basisSet(regType);

    double B11 = 0.0;
    double B12 = 0.0;
    double B13 = 0.0;
    double B14 = 0.0;
    double B22 = 0.0;
    double B23 = 0.0;
    double B24 = 0.0;
    double B33 = 0.0;
    double B34 = 0.0;
    double B44 = 0.0;
    double S0 = 0.0;
    double S1 = 0.0;
    double S2 = 0.0;
    double S3 = 0.0;

    for (int i = 0; i < n; i++) {
        double x = X[i];
        double y = Y[i];
        double bas0 = basis[0](x);
        double bas1 = basis[1](x);
        double bas2 = basis[2](x);
        double bas3 = basis[3](x);
        B11 += bas0 * bas0;
        B12 += bas0 * bas1;
        B13 += bas0 * bas2;
        B14 += bas0 * bas3;
        B22 += bas1 * bas1;
        B23 += bas1 * bas2;
        B24 += bas1 * bas3;
        B33 += bas2 * bas2;
        B34 += bas2 * bas3;
        B44 += bas3 * bas3;
        S0 += bas0 * y;
        S1 += bas1 * y;
        S2 += bas2 * y;
        S3 += bas3 * y;
    }

    Eigen::Matrix<double, 4, 4> A;
    A << B11,   B12,  B13,  B14,
         B12,   B22,  B23,  B24,
         B13,   B23,  B33,  B34,
         B14,   B24,  B34,  B44;

    Eigen::Matrix<double, 4, 1> b(S0, S1, S2, S3);

    Eigen::Matrix<double, 4, 1> c = A.colPivHouseholderQr().solve(b);

    auto svd = A.jacobiSvd();
    auto s = svd.singularValues();
    // The 2-norm condition number is kappa(A) = sigma_max / sigma_min.
    // When the basis has fewer than 4 active terms (e.g. degree-2, where phi_4 = 0),
    // A is rank-deficient: one or more sigma_i = 0, making the naive ratio infinite.
    // The numerically meaningful analogue is the condition number of the non-trivial
    // subspace, i.e. kappa_eff = sigma_1 / sigma_r where r = effective rank.
    // Effective rank is determined by the Eckart-Young threshold: a singular value
    // sigma_i is treated as zero if sigma_i <= 4*eps*sigma_1, which bounds the error
    // introduced by floating-point rounding in the formation of A to O(eps*||A||_2).
    // Cast to long double only here so the ratio itself has full 80-bit precision.
    long double threshold = (long double)s(0) * 4.0L * std::numeric_limits<long double>::epsilon();
    long double sMin = 0.0L;
    for (int i = s.size() - 1; i >= 0; i--) {
        if ((long double)s(i) > threshold) { sMin = (long double)s(i); break; }
    }
    long double kappa = (sMin > 0.0L) ? (long double)s(0) / sMin : std::numeric_limits<long double>::infinity();

    return {c(0), c(1), c(2), c(3), kappa};
}

//################################################
//   GEOMETRIC BROWNIAN MOTION PATH SIMULATION
//################################################
//returns a matrix with N steps and 2*P paths (includes P antithetic)
std::vector<double> generatePricePathMatrix(
    int P, double So, double dt, int N, double r, double v
) {
    std::vector<double> paths(P * 2 * N);;

    double drift = (r - 0.5 * v * v) * dt;
    double vol = v * std::sqrt(dt);

    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> d(0.0, 1.0);

    for (int p = 0; p < P; ++p) {
        double last  = So;
        double lastA = So;
        for (int n = 0; n < N; ++n) {
            double z = d(gen);
            last  *= std::exp(drift + vol *  z);
            lastA *= std::exp(drift + vol * -z);
            paths[p * N + n] = last;
            paths[(P+p) * N + n] = lastA;
        }
    }

    return paths;
}

//P must be even
//returns P paths (includes P/2 antithetic)
std::vector<double> generatePricePathStep(
    int P, double So, double dt, double r, double v
) {
    std::vector<double> paths(P);

    double drift = (r - 0.5 * v * v) * dt;
    double vol = v * std::sqrt(dt);

    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> d(0.0, 1.0);

    P = P/2;

    for (int p = 0; p < P; ++p) {
        double z = d(gen);
        paths[p] = So * std::exp(drift + vol *  z);
        paths[P+p] = So * std::exp(drift + vol * -z);
    }

    return paths;
}

//################################################
//       PRICER
//################################################

std::vector<long double> priceAmericanPut(
    double So, double T, int N, int P, double r, double v, double K, int regType
) {
    N = (int)(T * N);
    double dt = T / N;
    std::vector<double> S = generatePricePathMatrix(P, So, dt, N, r, v);
    std::vector<std::vector<double>> C(P, std::vector<double>(N, 0.0));
    std::vector<int> itm_indices;
    std::vector<double> X;
    std::vector<double> Y;

    long double sumKappa = 0.0L;
    long double numKappa = 0.0L;

    double c_coeff;
    double b_coeff;
    double a_coeff;

    std::vector<int> itm_mask(P);
    std::vector<double> pv_arr(P);

    for (int p = 0; p<P; p++) {
        C[p][N-1] = fmax(K-S[(p*N)+N-1],0);
    }
    
    for (int n= N-2; n>=0; n--) {
        //get X and Y
        X.clear();
        Y.clear();
        itm_indices.clear();

        // Parallel scan: each path's pv lookup is independent
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < P; p++) {
            itm_mask[p] = 0;
            if (K - S[(p*N)+n] > 0) {
                double pv = 0.0;
                for (int future = n+1; future < N; future++) {
                    if (C[p][future] > 0) {
                        pv = C[p][future] * exp(-r * dt * (future - n));
                        break;
                    }
                }
                itm_mask[p] = 1;
                pv_arr[p]   = pv;
            }
        }
        // Serial compact — no race conditions, cache-friendly
        for (int p = 0; p < P; p++) {
            if (itm_mask[p]) {
                X.push_back(S[(p*N)+n]);
                Y.push_back(pv_arr[p]);
                itm_indices.push_back(p);
            }
        }

        // if it is optimal to exercise nowhere in this step, skip to next step
        if (X.size() == 0) {
            continue;
        }

        
        //here is the part where we determine E() function

        //if there are less that 3 datapoints, assume E() is mean of Y_filtered
        bool useReg = X.size() >2;
        if (useReg) {
            try {
                std::vector<long double> solution = regress(X,Y,regType);
                c_coeff = solution[0];
                b_coeff = solution[1];
                a_coeff = solution[2];
                sumKappa += solution[4];
                numKappa += 1;
            // ##################################################################################
            // NOTE THIS PART: THIS CAUSES INNACURACY IN THE DESIGN:
            // ##################################################################################
            } catch (const std::runtime_error&) {
                useReg = false;
            }
        }
        
        // Each path is independent — no write conflicts on C[p][n]
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)itm_indices.size(); i++) {
            int p = itm_indices[i];
            double intrinsic = fmax(K-S[(p*N)+n],0);
            double expectedContinuance;

            if (useReg) {
                expectedContinuance = c_coeff + (b_coeff * S[(p*N)+n]) + (a_coeff * S[(p*N)+n] * S[(p*N)+n]);
            } else {
                expectedContinuance = 0;
            }

            if (intrinsic > expectedContinuance) {
                C[p][n] = intrinsic;
            }
        }
    }
    double price = 0.0;
    #pragma omp parallel for reduction(+:price) schedule(static)
    for (int p=0; p<P; p++) {
        for (int n=0; n<N; n++) {
            if (C[p][n] > 0) {
                price += C[p][n] * exp(-r * dt * (n+1));
                break;
            }
        }
    }

    // Guard against numKappa == 0, which occurs when N = (int)(T * stepsPerYear) <= 1
    // (e.g. T=0.164, stepsPerYear=10 -> N=1), leaving the backward loop body unreachable.
    // In that case no regression was performed, so kappa is undefined; return 0 instead of NaN.
    long double avgKappa = (numKappa > 0.0L) ? sumKappa / numKappa : 0.0L;
    return {price/P, avgKappa};
}

int main() {
    /*
    TYPES
    1 = Polynomial Deg 2
    2 = Polynomial Deg 3
    3 = Leandre Deg 2
    4 = Leandre Deg 3
    5 = Hermite Deg 2
    5 = Hermite Deg 3
    6 = Laguerre Deg 2
    7 = Laguerre Deg 3
    */


    // OPTION PROPERTIES
    double So = 100;
    double T = 1;  
    double r = 0.04;
    double v = 0.3;
    double K = 100;
    double actualPrice =  5.8091;
    
    // HYPERPARAMS
    std::vector<int> Nsched = {1,5,25,50,100,150,250,500,750,1000};
    std::vector<int> Psched = {1000};
    int regType = 1;
    const size_t maxBytes = (size_t)4 * 1024 * 1024 * 1024; // 4 GB limit
    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (int i = 0; i < 40; i++) {
        for (int p = 0; p < (int)Psched.size(); p++) {
            for (int n = 0; n < (int)Nsched.size(); n++) {
                size_t N_actual = (size_t)(T * Nsched[n]);
                if (N_actual == 0 || (size_t)Psched[p] * N_actual * sizeof(double) > maxBytes) {
                    continue;
                }

                // ALGORITHM
                auto t0 = std::chrono::high_resolution_clock::now();
                std::vector<long double> output = priceAmericanPut(So, T, Nsched[n], Psched[p], r, v, K, regType);
                auto t1 = std::chrono::high_resolution_clock::now();
                double seconds = std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000;

                // RESULTS
                double APE = (std::abs(output[0]-actualPrice) / actualPrice) * 100;

                #pragma omp critical
                write_result(Nsched[n], Psched[p], APE, output[1], seconds);
        


            }

        }
    
    }
}