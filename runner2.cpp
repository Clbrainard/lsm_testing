#include <ql/quantlib.hpp>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <memory>
#ifdef _WIN32
#  include <windows.h>
#endif

using namespace QuantLib;

static std::mutex    io_mutex;
static std::ofstream outFile;

double priceAmericanPut(double S0, double K, double T, double r, double v,
                        int N, int P, unsigned int seed,
                        Date today, Date expiry)
{
    Calendar   calendar = NullCalendar();
    DayCounter dc       = Actual365Fixed();

    Handle<Quote> spot(ext::make_shared<SimpleQuote>(S0));
    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(today, r,   dc));
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(today, 0.0, dc));
    Handle<BlackVolTermStructure> volTS(
        ext::make_shared<BlackConstantVol>(today, calendar, v, dc));

    auto bsm = ext::make_shared<BlackScholesMertonProcess>(spot, qTS, rTS, volTS);

    auto payoff   = ext::make_shared<PlainVanillaPayoff>(Option::Put, K);
    auto exercise = ext::make_shared<AmericanExercise>(today, expiry);
    VanillaOption option(payoff, exercise);

    option.setPricingEngine(
        MakeMCAmericanEngine<PseudoRandom>(bsm)
            .withSteps(N)
            .withSamples(P)
            .withSeed(seed)
            .withPolynomOrder(2));

    return option.NPV();
}

static size_t availableRAM()
{
#ifdef _WIN32
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    GlobalMemoryStatusEx(&st);
    return (size_t)st.ullAvailPhys;
#else
    std::ifstream f("/proc/meminfo");
    std::string key; size_t val; std::string unit;
    while (f >> key >> val >> unit)
        if (key == "MemAvailable:") return val * 1024ULL;
    return (size_t)64ULL << 30;
#endif
}

int main()
{
    struct OptionCase { double S0, K, T, r, v, actualPrice; };

    std::vector<OptionCase> cases = {
        {100, 102.5, 1, 0.05, 0.2, 7.341820839041034},
        {100, 97.5,  1, 0.05, 0.2, 4.981255934009840},
        {100, 100,   1, 0.05, 0.2, 6.089769951939136},
        {100, 105,   1, 0.05, 0.2, 8.739388521724180},
        {100, 95,    1, 0.05, 0.2, 4.012655060686365}
    };

    std::vector<int> Ns = {3,10,100,1000,10000,100000,1000000};

    const int P         = 10000;
    const int numTrials = 100;  // seeds 1..100

    Date today(1, January, 2020);
    Settings::instance().evaluationDate() = today;

    // ---- Memory guard ------------------------------------------------
    // MCAmericanEngine allocates ~2 * N * P * sizeof(double) bytes per thread
    // (paths + calibration paths). Cap concurrent threads so peak usage
    // stays within 75% of available RAM.
    int maxN = *std::max_element(Ns.begin(), Ns.end());
    size_t memPerThread = (size_t)2 * maxN * P * sizeof(double);
    size_t avail        = availableRAM();
    int    byMem        = (int)((avail * 0.75) / (double)memPerThread);
    int    numThreads   = std::max(1, std::min(omp_get_max_threads(), byMem));
    omp_set_num_threads(numThreads);

    std::cout << "Available RAM : " << (avail >> 30) << " GB\n"
              << "Mem / thread  : " << (memPerThread >> 20) << " MB\n"
              << "Threads       : " << numThreads << "\n\n";

    // ---- Work items --------------------------------------------------
    // One flat list keeps all threads busy. Grouped so we can detect when
    // all 100 trials of an (option, N) pair are done and flush immediately.
    struct WorkItem {
        double S0, K, T, r, v, actualPrice;
        int    N, seed, groupId;
        double dt;
        Date   expiry;
    };

    const int numGroups = (int)(cases.size() * Ns.size()); // 25

    std::vector<WorkItem> work;
    work.reserve(numGroups * numTrials);

    for (int ci = 0; ci < (int)cases.size(); ci++) {
        const auto& opt = cases[ci];
        Date expiry = today + (int)std::round(opt.T * 365.0);
        for (int ni = 0; ni < (int)Ns.size(); ni++) {
            int    N   = Ns[ni];
            double dt  = opt.T / N;
            int    gid = ci * (int)Ns.size() + ni;
            for (int trial = 1; trial <= numTrials; trial++)
                work.push_back({opt.S0, opt.K, opt.T, opt.r, opt.v,
                                opt.actualPrice, N, trial, gid, dt, expiry});
        }
    }

    // ---- Per-group buffers -------------------------------------------
    // Non-copyable (contains mutex), allocated on heap via unique_ptr.
    struct GroupBuf {
        std::mutex          mtx;
        std::vector<double> prices;
        GroupBuf() { prices.reserve(numTrials); }
    };

    // Metadata snapshot used at flush time (same for every trial in a group)
    struct GroupMeta { int N; double dt, actualPrice, K; };
    std::vector<GroupMeta> meta(numGroups);
    for (const auto& w : work)
        meta[w.groupId] = {w.N, w.dt, w.actualPrice, w.K};

    std::vector<std::unique_ptr<GroupBuf>> groups(numGroups);
    for (auto& g : groups) g = std::make_unique<GroupBuf>();

    // ---- Output file -------------------------------------------------
    outFile.open("C213-results.csv");
    outFile << "N,P,simulatedPrice,actualPrice,K,dt\n";

    const int total = (int)work.size();
    std::atomic<int> done(0);
    std::atomic<int> groupsFlushed(0);

    // ---- Main parallel loop -----------------------------------------
    // Dynamic schedule: N values span 3..1,000,000 so items are NOT equally
    // expensive. Static would leave fast-N threads idle while slow-N threads
    // run. Dynamic lets threads grab the next item the moment they finish,
    // keeping all cores saturated. Each thread has its own QuantLib objects —
    // no shared mutable state. Per-group mutex guards the flush step.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < total; i++) {
        const WorkItem& w = work[i];

        double price = priceAmericanPut(
            w.S0, w.K, w.T, w.r, w.v,
            w.N, P, (unsigned int)w.seed, today, w.expiry);

        GroupBuf& g  = *groups[w.groupId];
        bool      flush = false;
        {
            std::lock_guard<std::mutex> gl(g.mtx);
            g.prices.push_back(price);
            flush = ((int)g.prices.size() == numTrials);
        }

        int d = ++done;

        if (flush) {
            // Exactly one thread reaches here per group (size hits numTrials once).
            // Dump all 100 results for this group in one locked write + flush.
            const GroupMeta& m = meta[w.groupId];
            std::lock_guard<std::mutex> fl(io_mutex);
            for (double p : g.prices)
                outFile << m.N << "," << P << "," << p << ","
                        << m.actualPrice << "," << m.K << "," << m.dt << "\n";
            outFile.flush();
            std::cout << "Flushed group " << ++groupsFlushed << "/" << numGroups
                      << "  N=" << m.N << "  K=" << m.K
                      << "  (" << d << "/" << total << " trials done)\n"
                      << std::flush;
        }
    }

    outFile.close();
    std::cout << "\nDone. Results written to C213-results.csv\n";
    return 0;
}
