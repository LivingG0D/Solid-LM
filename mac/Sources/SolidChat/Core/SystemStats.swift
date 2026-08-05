import Foundation
import Observation
import Darwin

/// Live unified-memory pressure, sampled straight from the mach kernel.
///
/// `usedBytes` mirrors what Activity Monitor calls "Memory Used":
/// `(active + wired + compressed) * page size`. Total comes from
/// `ProcessInfo.physicalMemory`, which is the real installed RAM.
@Observable
@MainActor
final class SystemStats {

    /// Installed physical memory, in bytes.
    var totalBytes: Int64
    /// Active + wired + compressed, in bytes.
    var usedBytes: Int64 = 0
    /// `totalBytes - usedBytes`, clamped to zero.
    var freeBytes: Int64 = 0

    /// Seconds between samples.
    private static let interval: Duration = .seconds(2)

    @ObservationIgnored private var pump: Task<Void, Never>?

    init() {
        let total = Int64(clamping: ProcessInfo.processInfo.physicalMemory)
        totalBytes = total
        freeBytes = total
        refresh()
    }

    // MARK: - Lifecycle

    /// Starts (or restarts) the 2 second sampling loop. Safe to call repeatedly.
    func start() {
        guard pump == nil else { return }
        refresh()
        pump = Task { [weak self] in
            while !Task.isCancelled {
                do {
                    try await Task.sleep(for: SystemStats.interval)
                } catch {
                    return  // cancelled while sleeping
                }
                guard let self, !Task.isCancelled else { return }
                self.refresh()
            }
        }
    }

    /// Stops the sampling loop. Values keep their last sampled reading.
    func stop() {
        pump?.cancel()
        pump = nil
    }

    // MARK: - Sampling

    /// Takes one sample immediately.
    func refresh() {
        let total = Int64(clamping: ProcessInfo.processInfo.physicalMemory)
        if totalBytes != total { totalBytes = total }

        guard let sample = Self.sampleVM() else { return }

        let vm = sample.stats
        let page = sample.pageSize
        let pages = Int64(vm.active_count) + Int64(vm.wire_count) + Int64(vm.compressor_page_count)
        let used = min(pages * page, total)
        let free = max(total - used, 0)

        // Assign only on change: @Observable notifies on every write.
        if usedBytes != used { usedBytes = used }
        if freeBytes != free { freeBytes = free }
    }

    /// One `host_statistics64(HOST_VM_INFO64)` call plus the kernel page size those
    /// page counts are denominated in. Returns nil if the kernel refuses.
    ///
    /// The page size comes from `host_page_size()` rather than the `vm_kernel_page_size`
    /// global: both report the kernel's `PAGE_SIZE` (16384 on Apple silicon), but the
    /// global is an imported mutable C variable and Swift 6 strict concurrency rejects
    /// reading it.
    private static func sampleVM() -> (stats: vm_statistics64_data_t, pageSize: Int64)? {
        let host = mach_host_self()
        defer { mach_port_deallocate(mach_task_self_, host) }

        var pageSize: vm_size_t = 0
        guard host_page_size(host, &pageSize) == KERN_SUCCESS, pageSize > 0 else { return nil }

        var info = vm_statistics64_data_t()
        // HOST_VM_INFO64_COUNT — the C macro is sizeof(struct)/sizeof(integer_t).
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size
        )

        let result = withUnsafeMutablePointer(to: &info) { raw -> kern_return_t in
            raw.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { typed in
                host_statistics64(host, host_flavor_t(HOST_VM_INFO64), typed, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return (info, Int64(clamping: UInt64(pageSize)))
    }
}
