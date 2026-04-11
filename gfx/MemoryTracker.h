#pragma once
#include <vulkan/vulkan.h>
#include <atomic>
#include <cstdio>
#include <chrono>

// --- MemoryTracker ---
// lightweight singleton that intercepts every vkAllocateMemory/vkFreeMemory
// from GpuBuffer and GpuImage. atomic counters, safe for concurrent use.

class MemoryTracker {
public:
    static MemoryTracker& Get() {
        static MemoryTracker instance;
        return instance;
    }

    MemoryTracker(const MemoryTracker&) = delete;
    MemoryTracker& operator=(const MemoryTracker&) = delete;

    // called immediately after vkAllocateMemory
    // starts upload timer on first invocation
    void OnAllocate(VkDeviceSize size, const char* tag) {
        _totalBytes.fetch_add(size, std::memory_order_relaxed);
        _allocationCount.fetch_add(1, std::memory_order_relaxed);

        // CAS loop avoids a race between load(peak) and load(total)
        uint64_t observed, candidate;
        do {
            observed  = _peakBytes.load(std::memory_order_relaxed);
            candidate = _totalBytes.load(std::memory_order_relaxed);
            if (candidate <= observed) break;
        } while (!_peakBytes.compare_exchange_weak(observed, candidate,
                                                    std::memory_order_relaxed));

        if (!_uploadStarted.exchange(true, std::memory_order_relaxed))
            _uploadStart = std::chrono::high_resolution_clock::now();

#ifndef NDEBUG
        std::printf("[MemoryTracker] ALLOC  %s : %llu bytes  (total : %llu bytes, count : %llu)\n",
            tag,
            (unsigned long long)size,
            (unsigned long long)_totalBytes.load(),
            (unsigned long long)_allocationCount.load());
#else
        (void)tag;
#endif
    }

    // called immediately before vkFreeMemory
    void OnFree(VkDeviceSize size) {
        _totalBytes.fetch_sub(size, std::memory_order_relaxed);
        _freeCount.fetch_add(1, std::memory_order_relaxed);

#ifndef NDEBUG
        std::printf("[MemoryTracker] FREE               (total restant : %llu bytes)\n",
            (unsigned long long)_totalBytes.load());
#endif
    }

    // call once all init uploads are done. Records total upload duration
    void MarkUploadComplete() {
        if (_uploadStarted.load() && !_uploadCompleted.exchange(true)) {
            auto now = std::chrono::high_resolution_clock::now();
            _uploadDurationMs = std::chrono::duration<double, std::milli>(
                now - _uploadStart).count();
        }
    }

    uint64_t TotalAllocatedBytes()  const { return _totalBytes.load(); }
    uint64_t PeakAllocatedBytes()   const { return _peakBytes.load(); }
    uint64_t AllocationCount()      const { return _allocationCount.load(); }
    uint64_t FreeCount()            const { return _freeCount.load(); }
    double   UploadDurationMs()     const { return _uploadDurationMs; }

    void PrintReport() const {
        std::printf("\n+----------------------------------------------+\n");
        std::printf("|         MemoryTracker -- M4 Report            |\n");
        std::printf("+----------------------------------------------+\n");
        std::printf("|  Allocations totales  : %-6llu               |\n",
            (unsigned long long)_allocationCount.load());
        std::printf("|  Frees totaux         : %-6llu               |\n",
            (unsigned long long)_freeCount.load());
        std::printf("|  Memoire actuelle     : %-10llu bytes     |\n",
            (unsigned long long)_totalBytes.load());
        std::printf("|  Pic memoire          : %-10llu bytes     |\n",
            (unsigned long long)_peakBytes.load());
        if (_uploadCompleted.load())
            std::printf("|  Temps upload total   : %-8.2f ms           |\n",
                _uploadDurationMs);
        std::printf("+----------------------------------------------+\n\n");
    }

    void Reset() {
        _totalBytes.store(0);
        _peakBytes.store(0);
        _allocationCount.store(0);
        _freeCount.store(0);
        _uploadStarted.store(false);
        _uploadCompleted.store(false);
        _uploadDurationMs = 0.0;
    }

private:
    MemoryTracker() = default;

    std::atomic<uint64_t> _totalBytes{ 0 };
    std::atomic<uint64_t> _peakBytes{ 0 };
    std::atomic<uint64_t> _allocationCount{ 0 };
    std::atomic<uint64_t> _freeCount{ 0 };

    std::atomic<bool> _uploadStarted{ false };
    std::atomic<bool> _uploadCompleted{ false };
    std::chrono::high_resolution_clock::time_point _uploadStart;
    double _uploadDurationMs = 0.0;
};