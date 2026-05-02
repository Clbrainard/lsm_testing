

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <string>
#include <cmath>
#include <vector>
#include <iterator>
#include <omp.h>
#include <ctime>
#include <chrono>
#include <thread>

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

void write_result(int steps, int paths, double price, double kappa, double runtime, double K, double dt) {
    std::ofstream file("euro/EuroConv.csv", std::ios::app);
    file << steps << "," << paths << "," << price << "," << kappa << "," << runtime << "," << K <<  "," << dt <<"\n";
}

std::vector<double> generatePricePathStep(
    int P, double So, double dt, double r, double v
) {
    std::vector<double> paths(2*P);

    double drift = (r - 0.5 * v * v) * dt;
    double vol = v * std::sqrt(dt);

    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> d(0.0, 1.0);

    for (int p = 0; p < P; ++p) {
        double z = d(gen);
        paths[p] = So * std::exp(drift + vol *  z);
        paths[P+p] = So * std::exp(drift + vol * -z);
    }

    return paths;
}

double priceEuropeanPut(
    double So, double T, int P, double r, double v, double K
) {
    std::vector<double> S = generatePricePathStep(P, So, T , r, v);
    P = P * 2;
    
    double sum = 0;
    for (int p =0; p<P; p++) {
        sum += fmax(K-S[p],0);
    }

    sum = sum / P;
    sum = sum * exp(-r * T);
    return sum;
}

int main() {
    /*
    TYPES
    1 = Polynomial Deg 2
    2 = Polynomial Deg 3
    3 = Legendre Deg 2
    4 = Legendre Deg 3
    5 = Hermite Deg 2
    6 = Hermite Deg 3
    7 = Laguerre Deg 2
    8 = Laguerre Deg 3
    */

    // current test is for 4 options, tested 50 times each with P=10,000, and T being decreased


    // OPTION PROPERTIES
    std::vector<std::vector<double>> cases = {
        {100,102.5,1,0.05,0.2,7.341820839041034},
        {100,97.5,1,0.05,0.2,4.98125593400984},
        {100,100,1,0.05,0.2,6.089769951939136},
        {100,105,1,0.05,0.2,8.73938852172418},
        {100,95,1,0.05,0.2,4.012655060686365}
    };

    int P = 10000;

    std::vector<int> Ns = {1};

    int regType = 1;

    int numTests = 100 * cases.size();
    int complete = 1;

    //Test cases
    for (int z = 0; z<cases.size(); z++) { 
        double So = cases[z][0];
        double T = cases[z][2];  
        double r = cases[z][3];
        double v = cases[z][4];
        double K = cases[z][1];

        double actualPrice = cases[z][5];
        for (int i = 0; i < 100; i++) {
            
            std::cout << "Running simulations: " << complete << "/" << numTests <<"\n";
            for (int k = 0; k<Ns.size(); k++) {
                    int N = Ns[k];
                    // ALGORITHM
                    auto t0 = std::chrono::high_resolution_clock::now();
                    double output = priceEuropeanPut(So, T, P, r, v, K);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double seconds = std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000;

                    // RESULTS
                    //double APE = (std::abs(output[0]-actualPrice) / actualPrice) * 100;

                    write_result(1, P, output, 0, seconds, K, T/N);
                    
                    
            }
            complete += 1;
        }
    }
}
