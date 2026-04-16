#pragma once

#include <vector>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <limits>
#include <functional>
#include <numeric>
#include <algorithm>
#include "../renderer/Renderer.h"

// --- BenchmarkRunner ---
// measures CB recording time in ST (single-threaded, VK_SUBPASS_CONTENTS_INLINE)
// vs MT (2 workers, secondary CBs) mode.
//
// warmup phase discards samples to let the driver JIT any shader compilation and
// fill its internal caches. collecting before warmup is complete inflates ST variance
// with one-time costs unrelated to the recording path itself.
//
// the significance test is a simplified Welch criterion:
//   |mean_a - mean_b| > sigmaThreshold * sqrt(var_a/N + var_b/N)
// it does not assume equal variance between modes, which matters here because MT
// adds mutex/condvar contention that inflates variance relative to ST.

struct BenchmarkConfig {
    uint32_t warmupFrames  = 200;
    uint32_t collectFrames = 1000;
};

struct BenchmarkResult {
    double              meanMs   = 0.0;
    double              stddevMs = 0.0;
    double              minMs    = std::numeric_limits<double>::max();
    double              maxMs    = 0.0;
    double              p50Ms    = 0.0;
    double              p95Ms    = 0.0;
    double              p99Ms    = 0.0;
    // coefficient of variation: stddev/mean — unitless stability indicator.
    // CV > 0.3 suggests OS preemption events are dominating variance.
    double              cv       = 0.0;
    std::vector<double> samples;

    // GPU execution time measured via VkQueryPool timestamps (TOP_OF_PIPE → BOTTOM_OF_PIPE).
    // distinct from CPU recording time: GPU measures actual shader execution on the device.
    double              gpuMeanMs   = 0.0;
    double              gpuStddevMs = 0.0;
    std::vector<double> gpuSamples;
};

class BenchmarkRunner {
public:
    // pollEvents is called once per frame to prevent the OS from marking the window
    // as unresponsive during the benchmark loop. pass glfwPollEvents or equivalent.
    BenchmarkRunner(Renderer& renderer,
                    std::function<void()> pollEvents,
                    BenchmarkConfig cfg = {})
        : _renderer(renderer)
        , _pollEvents(std::move(pollEvents))
        , _cfg(cfg)
    {}

    BenchmarkResult Run(bool useMT) {
        _renderer.SetMultiThreaded(useMT);

        // warmup: samples discarded. driver reaches steady-state pipeline execution.
        for (uint32_t i = 0; i < _cfg.warmupFrames; ++i) {
            _pollEvents();
            _renderer.DrawFrame();
        }

        BenchmarkResult result;
        result.samples.reserve(_cfg.collectFrames);
        result.gpuSamples.reserve(_cfg.collectFrames);

        for (uint32_t i = 0; i < _cfg.collectFrames; ++i) {
            _pollEvents();
            _renderer.DrawFrame();
            double t = _renderer.GetLastRecordingTimeMs();
            result.samples.push_back(t);
            if (t < result.minMs) result.minMs = t;
            if (t > result.maxMs) result.maxMs = t;
            result.gpuSamples.push_back(_renderer.GetLastGpuTimeMs());
        }

        double sum = std::accumulate(result.samples.begin(), result.samples.end(), 0.0);
        result.meanMs = sum / result.samples.size();

        double sqSum = 0.0;
        for (double s : result.samples) {
            double d = s - result.meanMs;
            sqSum += d * d;
        }
        // population stddev over N samples — N-1 (Bessel) would be more correct for
        // a true estimator, but with N=1000 the difference is negligible (<0.05%).
        result.stddevMs = std::sqrt(sqSum / result.samples.size());

        // percentiles require a sorted copy — do not sort samples in-place to preserve
        // insertion order for any future time-series analysis.
        std::vector<double> sorted = result.samples;
        std::sort(sorted.begin(), sorted.end());
        const size_t N2 = sorted.size();
        result.p50Ms = sorted[static_cast<size_t>(N2 * 0.50)];
        result.p95Ms = sorted[static_cast<size_t>(N2 * 0.95)];
        result.p99Ms = sorted[static_cast<size_t>(N2 * 0.99)];
        result.cv    = (result.meanMs > 0.0) ? result.stddevMs / result.meanMs : 0.0;

        // GPU stats — skip the first sample (frame 0 may return 0.0 if query not yet available)
        if (result.gpuSamples.size() > 1) {
            auto gpuBegin = result.gpuSamples.begin() + 1;
            double gpuSum = std::accumulate(gpuBegin, result.gpuSamples.end(), 0.0);
            result.gpuMeanMs = gpuSum / (result.gpuSamples.size() - 1);
            double gpuSqSum = 0.0;
            for (auto it = gpuBegin; it != result.gpuSamples.end(); ++it) {
                double d = *it - result.gpuMeanMs;
                gpuSqSum += d * d;
            }
            result.gpuStddevMs = std::sqrt(gpuSqSum / (result.gpuSamples.size() - 1));
        }

        return result;
    }

    static void PrintComparison(const BenchmarkResult& st, const BenchmarkResult& mt) {
        const uint32_t N = (uint32_t)st.samples.size();
        double cpuDelta = mt.meanMs - st.meanMs;
        double gpuDelta = mt.gpuMeanMs - st.gpuMeanMs;

        // ratio CPU/GPU: how much longer the CPU recording takes vs the actual GPU execution.
        // ratio > 1 means CPU is the bottleneck for recording, not the GPU.
        double stRatio = (st.gpuMeanMs > 0.0) ? st.meanMs / st.gpuMeanMs : 0.0;
        double mtRatio = (mt.gpuMeanMs > 0.0) ? mt.meanMs / mt.gpuMeanMs : 0.0;

        std::printf("\n+------------------------------------------------------------------------+\n");
        std::printf("|                  M5 Benchmark -- CB Recording Time                    |\n");
        std::printf("|  N = %-5u frames per mode                                            |\n", N);
        std::printf("+----------+----------+----------+----------+----------+----------+------+\n");
        std::printf("| mode     | mean(ms) | p50      | p95      | p99      | max      |  CV  |\n");
        std::printf("+----------+----------+----------+----------+----------+----------+------+\n");
        std::printf("| ST  CPU  | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %.3f|\n",
            st.meanMs, st.p50Ms, st.p95Ms, st.p99Ms, st.maxMs, st.cv);
        std::printf("| MT  CPU  | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %.3f|\n",
            mt.meanMs, mt.p50Ms, mt.p95Ms, mt.p99Ms, mt.maxMs, mt.cv);
        std::printf("| CPU delta| %+8.4f | %-8s | %-8s | %-8s | %-8s | %-5s|\n",
            cpuDelta, "--", "--", "--", "--", "--");
        std::printf("+----------+----------+----------+----------+----------+----------+------+\n");
        std::printf("| ST  GPU  | %8.4f | %-8s | %-8s | %-8s | %-8s | %.3f|\n",
            st.gpuMeanMs, "--", "--", "--", "--", st.gpuStddevMs / (st.gpuMeanMs > 0.0 ? st.gpuMeanMs : 1.0));
        std::printf("| MT  GPU  | %8.4f | %-8s | %-8s | %-8s | %-8s | %.3f|\n",
            mt.gpuMeanMs, "--", "--", "--", "--", mt.gpuStddevMs / (mt.gpuMeanMs > 0.0 ? mt.gpuMeanMs : 1.0));
        std::printf("| GPU delta| %+8.4f | %-8s | %-8s | %-8s | %-8s | %-5s|\n",
            gpuDelta, "--", "--", "--", "--", "--");
        std::printf("+----------+----------+----------+----------+----------+----------+------+\n");
        std::printf("| CPU/GPU ratio   ST: %5.2fx        MT: %5.2fx                          |\n",
            stRatio, mtRatio);
        std::printf("+----------+----------+----------+----------+----------+----------+------+\n");

        bool sig = IsSignificant(st, mt);
        std::printf("| CPU significant: %-3s  (2-sigma Welch, N=%-5u)                       |\n",
            sig ? "yes" : "no", N);

        // GPU delta should be NOT significant: MT and ST submit identical GPU work.
        // if it comes back significant, the timer is capturing presentation stalls or
        // command processor overhead rather than pure shader execution.
        bool gpuSig = (st.gpuMeanMs > 0.0 && mt.gpuMeanMs > 0.0)
                   && IsSignificantGpu(st, mt);
        std::printf("| GPU significant: %-3s  (expected: no -- identical GPU workload)        |\n",
            gpuSig ? "yes" : "no");
        std::printf("+------------------------------------------------------------------------+\n\n");
    }

    // Welch criterion without a t-table: treat 2-sigma as the threshold.
    // for N=1000 this is conservative (true alpha ~0.05 at ~1.96 sigma),
    // meaning we may accept a null hypothesis that is actually false — but
    // false positives are the worse outcome here, so conservatism is correct.
    static bool IsSignificant(const BenchmarkResult& a,
                               const BenchmarkResult& b,
                               double sigmaThreshold = 2.0)
    {
        double N_a = (double)a.samples.size();
        double N_b = (double)b.samples.size();
        if (N_a < 2.0 || N_b < 2.0) return false;

        double var_a = a.stddevMs * a.stddevMs;
        double var_b = b.stddevMs * b.stddevMs;
        double se    = std::sqrt(var_a / N_a + var_b / N_b);
        if (se == 0.0) return false;

        double delta = std::abs(a.meanMs - b.meanMs);
        return delta > sigmaThreshold * se;
    }

    static bool IsSignificantGpu(const BenchmarkResult& a,
                                  const BenchmarkResult& b,
                                  double sigmaThreshold = 2.0)
    {
        double N_a = (double)a.gpuSamples.size();
        double N_b = (double)b.gpuSamples.size();
        if (N_a < 2.0 || N_b < 2.0) return false;

        double var_a = a.gpuStddevMs * a.gpuStddevMs;
        double var_b = b.gpuStddevMs * b.gpuStddevMs;
        double se    = std::sqrt(var_a / N_a + var_b / N_b);
        if (se == 0.0) return false;

        double delta = std::abs(a.gpuMeanMs - b.gpuMeanMs);
        return delta > sigmaThreshold * se;
    }

private:
    Renderer&             _renderer;
    std::function<void()> _pollEvents;
    BenchmarkConfig       _cfg;
};
