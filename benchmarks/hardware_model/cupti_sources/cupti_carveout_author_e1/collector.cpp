// Host-only CUPTI activity metadata collector. No CUDA launches.
#include <cupti_activity.h>
#include <cupti_version.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <malloc.h>
#include <mutex>
#include <set>

namespace {
constexpr size_t buffer_bytes = 1024 * 1024;
std::mutex mutex;
FILE* output = nullptr;
std::set<uint8_t*> outstanding;
std::atomic<unsigned> callbacks{0};
unsigned errors = 0;
unsigned long long dropped_total = 0;
unsigned long long requested_buffers = 0;
unsigned long long completed_buffers = 0;
unsigned long long kernel_records = 0;
bool started = false;
bool registered = false;
bool enabled = false;

struct CallbackScope {
    CallbackScope() { ++callbacks; }
    ~CallbackScope() { --callbacks; }
};

void quote(const char* value) {
    std::fputc('"', output);
    if (value) {
        for (const unsigned char* p = reinterpret_cast<const unsigned char*>(value); *p; ++p) {
            if (*p == '"' || *p == '\\') {
                std::fputc('\\', output);
                std::fputc(*p, output);
            } else if (*p < 32 || *p >= 127) {
                std::fprintf(output, "\\u%04x", static_cast<unsigned>(*p));
            } else {
                std::fputc(*p, output);
            }
        }
    }
    std::fputc('"', output);
}

bool result(const char* name, CUptiResult code, bool end_of_buffer = false) {
    std::lock_guard<std::mutex> lock(mutex);
    const bool accepted = code == CUPTI_SUCCESS ||
        (end_of_buffer && code == CUPTI_ERROR_MAX_LIMIT_REACHED);
    if (!accepted) ++errors;
    if (output) {
        std::fprintf(output,
            "{\"type\":\"api\",\"name\":\"%s\",\"result\":%d,\"accepted\":%s}\n",
            name, static_cast<int>(code), accepted ? "true" : "false");
        std::fflush(output);
    }
    return accepted;
}

void internal_error(const char* message) {
    std::lock_guard<std::mutex> lock(mutex);
    ++errors;
    if (output) {
        std::fprintf(output, "{\"type\":\"error\",\"message\":");
        quote(message);
        std::fprintf(output, "}\n");
        std::fflush(output);
    }
}

void CUPTIAPI request_buffer(uint8_t** buffer, size_t* size, size_t* max_records) {
    CallbackScope scope;
    *buffer = nullptr;
    *size = 0;
    *max_records = 0;
    uint8_t* allocation = nullptr;
    try {
        allocation = static_cast<uint8_t*>(_aligned_malloc(buffer_bytes, 8));
        if (!allocation) {
            internal_error("Aligned host buffer allocation failed");
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        outstanding.insert(allocation);
        ++requested_buffers;
        *buffer = allocation;
        *size = buffer_bytes;
        if (output) std::fprintf(output,
            "{\"type\":\"buffer_requested\",\"pointer\":%llu,\"bytes\":%llu}\n",
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(allocation)),
            static_cast<unsigned long long>(buffer_bytes));
    } catch (...) {
        if (allocation) _aligned_free(allocation);
        internal_error("Exception in buffer request callback");
    }
}

void kernel_record(const CUpti_ActivityKernel12& k) {
    std::lock_guard<std::mutex> lock(mutex);
    ++kernel_records;
    std::fprintf(output, "{\"type\":\"kernel\",\"kind\":%d,\"name\":", static_cast<int>(k.kind));
    quote(k.name);
    std::fprintf(output,
        ",\"correlation_id\":%u,\"grid_id\":%lld,\"device_id\":%u,\"context_id\":%u,\"stream_id\":%u,"
        "\"grid\":[%d,%d,%d],\"block\":[%d,%d,%d],\"registers_per_thread\":%u,"
        "\"static_shared_bytes\":%d,\"dynamic_shared_bytes\":%d,\"local_bytes_per_thread\":%u,"
        "\"local_bytes_total\":%llu,\"carveout_requested\":%u,\"requested_percent\":%u,"
        "\"shared_memory_executed_bytes\":%u,\"cache_config_requested\":%u,\"cache_config_executed\":%u,"
        "\"start_ns\":%llu,\"end_ns\":%llu,\"completed_ns\":%llu,\"execution_model\":%u}\n",
        k.correlationId, static_cast<long long>(k.gridId), k.deviceId, k.contextId, k.streamId,
        k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
        static_cast<unsigned>(k.registersPerThread), k.staticSharedMemory, k.dynamicSharedMemory,
        k.localMemoryPerThread, static_cast<unsigned long long>(k.localMemoryTotal_v2),
        static_cast<unsigned>(k.isSharedMemoryCarveoutRequested),
        static_cast<unsigned>(k.sharedMemoryCarveoutRequested), k.sharedMemoryExecuted,
        static_cast<unsigned>(k.cacheConfig.config.requested),
        static_cast<unsigned>(k.cacheConfig.config.executed),
        static_cast<unsigned long long>(k.start), static_cast<unsigned long long>(k.end),
        static_cast<unsigned long long>(k.completed), k.executionModel);
}

void dropped(CUcontext context, uint32_t stream) {
    size_t count = 0;
    const CUptiResult code = cuptiActivityGetNumDroppedRecords(context, stream, &count);
    result("cuptiActivityGetNumDroppedRecords", code);
    std::lock_guard<std::mutex> lock(mutex);
    dropped_total += count;
    if (count) ++errors;
    if (output) std::fprintf(output,
        "{\"type\":\"dropped\",\"context\":%llu,\"stream\":%u,\"count\":%llu}\n",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(context)), stream,
        static_cast<unsigned long long>(count));
}

void CUPTIAPI complete_buffer(CUcontext context, uint32_t stream, uint8_t* buffer,
                              size_t size, size_t valid_size) {
    CallbackScope scope;
    bool owned = false;
    {
        std::lock_guard<std::mutex> lock(mutex);
        owned = outstanding.count(buffer) == 1;
    }
    if (!owned) {
        internal_error("Completion returned an unknown buffer");
        return;
    }
    try {
        if (size != buffer_bytes || valid_size > size) {
            internal_error("Invalid completed buffer dimensions");
        } else {
            CUpti_Activity* record = nullptr;
            for (;;) {
                const CUptiResult code = cuptiActivityGetNextRecord(buffer, valid_size, &record);
                result("cuptiActivityGetNextRecord", code, true);
                if (code == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
                if (code != CUPTI_SUCCESS) break;
                if (!record || record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                    internal_error("Unexpected activity kind in kernel-only collector");
                    break;
                }
                kernel_record(*reinterpret_cast<const CUpti_ActivityKernel12*>(record));
            }
        }
        dropped(context, stream);
    } catch (...) {
        internal_error("Exception in buffer completion callback");
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        outstanding.erase(buffer);
        ++completed_buffers;
        if (output) {
            std::fprintf(output,
                "{\"type\":\"buffer_completed\",\"pointer\":%llu,\"bytes\":%llu,\"valid_bytes\":%llu}\n",
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(buffer)),
                static_cast<unsigned long long>(size), static_cast<unsigned long long>(valid_size));
            std::fflush(output);
        }
    }
    // The completion contract relinquishes CUPTI ownership of this buffer.
    _aligned_free(buffer);
}
}  // namespace

extern "C" __declspec(dllexport) int collector_start(const wchar_t* path) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (started || !path) return 2;
        started = true;
        if (_wfopen_s(&output, path, L"wb") != 0 || !output) return 3;
        std::fprintf(output,
            "{\"type\":\"header\",\"header_cupti_api_version\":%u,\"cuda_header_version\":%u,"
            "\"kernel_record_version\":12,\"kernel_record_size\":%llu,\"enabled_kind\":%d,"
            "\"offset_requested_flag\":%llu,\"offset_requested_percent\":%llu,\"offset_shared_executed\":%llu}\n",
            CUPTI_API_VERSION, CUDA_VERSION, static_cast<unsigned long long>(sizeof(CUpti_ActivityKernel12)),
            static_cast<int>(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
            static_cast<unsigned long long>(offsetof(CUpti_ActivityKernel12, isSharedMemoryCarveoutRequested)),
            static_cast<unsigned long long>(offsetof(CUpti_ActivityKernel12, sharedMemoryCarveoutRequested)),
            static_cast<unsigned long long>(offsetof(CUpti_ActivityKernel12, sharedMemoryExecuted)));
    }
    uint32_t version = 0;
    if (!result("cuptiGetVersion", cuptiGetVersion(&version))) return 4;
    {
        std::lock_guard<std::mutex> lock(mutex);
        std::fprintf(output, "{\"type\":\"runtime_version\",\"value\":%u}\n", version);
    }
    if (version != CUPTI_API_VERSION) {
        internal_error("Loaded CUPTI version differs from compiled header");
        return 5;
    }
    if (!result("cuptiActivityRegisterCallbacks",
                cuptiActivityRegisterCallbacks(request_buffer, complete_buffer))) return 6;
    registered = true;
    if (!result("cuptiActivityEnable_CONCURRENT_KERNEL",
                cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL))) return 7;
    enabled = true;
    return 0;
}

extern "C" __declspec(dllexport) int collector_stop() {
    if (!started || !output) return 2;
    if (enabled) {
        result("cuptiActivityDisable_CONCURRENT_KERNEL",
               cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        enabled = false;
    }
    if (registered) {
        result("cuptiActivityFlushAll_FORCED", cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
        dropped(nullptr, 0);
    }
    // The caller has completed stream synchronization and the forced flush.
    result("cuptiFinalize", cuptiFinalize());
    std::lock_guard<std::mutex> lock(mutex);
    if (callbacks.load() != 0 || !outstanding.empty()) ++errors;
    if (std::ferror(output)) ++errors;
    std::fprintf(output,
        "{\"type\":\"summary\",\"errors\":%u,\"dropped_records\":%llu,"
        "\"requested_buffers\":%llu,\"completed_buffers\":%llu,\"outstanding_buffers\":%llu,"
        "\"active_callbacks\":%u,\"kernel_records\":%llu,\"safe_to_close\":%s}\n",
        errors, dropped_total, requested_buffers, completed_buffers,
        static_cast<unsigned long long>(outstanding.size()), callbacks.load(), kernel_records,
        callbacks.load() == 0 && outstanding.empty() ? "true" : "false");
    std::fflush(output);
    // A failed quiescence check leaves the file alive for any late callback.
    if (callbacks.load() != 0 || !outstanding.empty()) return 8;
    if (std::fclose(output) != 0) ++errors;
    output = nullptr;
    return errors == 0 ? 0 : 9;
}
