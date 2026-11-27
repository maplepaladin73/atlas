// SPDX-License-Identifier: AGPL-3.0-only

//! Frequency-ordered hot-row device cache for the bf16 Engram tables.
//!
//! Decode lookups are hash-scattered single-row reads; when the table is
//! NVMe-mmap'd they pay host/NVMe latency with only a handful of rows of
//! parallelism to hide it. The n-gram row frequency distribution is heavy
//! but steep (top ~500k rows/head cover ~93% of lookups on code corpora),
//! so a few GB of device-resident hot rows — verbatim copies, no quality
//! change — take most lookups off the slow path. Misses fall through to
//! the table pointer exactly as before, so this is purely a read
//! accelerator: output is bit-identical with the cache on or off.
//!
//! Sidecar format (little-endian, written by the offline builder):
//! magic u32 0x48305443, version u32, layers u32, heads u32, row_bytes u32,
//! k u32; then per (layer, head) in order: count u32, rowids u32[count]
//! (head-local, sorted ascending), row data bf16[count][row_bytes/2].
//!
//! `ATLAS_HY4_ENGRAM_HOTCACHE` is the sidecar path; unset or empty keeps
//! the uncached path. fp8 tables are already device-resident and never
//! use the cache.
//!
//! `ATLAS_HY4_ENGRAM_HOTCACHE_STATS=1` enables hit/miss counters in the
//! hot-cache probe kernels. Counters are cumulative per process, split into
//! decode, verify, and prefill probes, and copied back for a tracing line every
//! 64 Engram forward invocations per layer. This is a diagnostic mode: the
//! copyback synchronizes the stream, so timing measurements made with stats
//! enabled are observations of the instrumented engine, not production
//! performance.

use crate::layers::hy4::attention::memory::Owned;
use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) const PATH_ENV: &str = "ATLAS_HY4_ENGRAM_HOTCACHE";
pub(crate) const STATS_ENV: &str = "ATLAS_HY4_ENGRAM_HOTCACHE_STATS";
const MAGIC: u32 = 0x48305443;
const STATS_BYTES: usize = 48;
pub(crate) const STATS_KIND_DECODE: u32 = 0;
pub(crate) const STATS_KIND_VERIFY: u32 = 2;
pub(crate) const STATS_KIND_PREFILL: u32 = 4;
const REPORT_INTERVAL_LAUNCHES: usize = 64;

/// One layer's device-resident hot rows for the heads this rank owns:
/// per-head sorted sections of layer-wide row ids concatenated in
/// `keys`, matching row bytes in `rows`, and 17 section offsets in `offs`
/// (empty sections for heads the rank does not own).
pub(crate) struct HotCache {
    pub(crate) keys: DevicePtr,
    pub(crate) rows: DevicePtr,
    pub(crate) offs: DevicePtr,
    /// u64 counters: decode hit/miss, verify hit/miss, prefill hit/miss. Null
    /// unless stats collection was explicitly enabled at load time.
    pub(crate) stats: DevicePtr,
    layer_id: usize,
    report_ticks: AtomicUsize,
}

impl HotCache {
    /// Copy and report counters occasionally. The interval is counted in
    /// Engram invocations rather than probes because probe count depends on
    /// batch shape; the device counters themselves count every valid probe.
    pub(crate) fn maybe_report_stats(
        &self,
        gpu: &dyn GpuBackend,
        rank: usize,
        stream: u64,
    ) -> Result<()> {
        if self.stats.is_null() {
            return Ok(());
        }
        let tick = self.report_ticks.fetch_add(1, Ordering::Relaxed) + 1;
        if !tick.is_multiple_of(REPORT_INTERVAL_LAUNCHES) {
            return Ok(());
        }
        let mut bytes = [0u8; STATS_BYTES];
        gpu.copy_d2h_on_stream(self.stats, &mut bytes, stream)?;
        let counts: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let decode_rate = hit_rate(counts[0], counts[1]);
        let verify_rate = hit_rate(counts[2], counts[3]);
        let prefill_rate = hit_rate(counts[4], counts[5]);
        tracing::info!(
            "HY4_ENGRAM_HOTCACHE_STATS rank={} layer={} forward_calls={} decode_hits={} decode_misses={} decode_hit_rate={:.4} verify_hits={} verify_misses={} verify_hit_rate={:.4} prefill_hits={} prefill_misses={} prefill_hit_rate={:.4}",
            rank,
            self.layer_id,
            tick,
            counts[0],
            counts[1],
            decode_rate,
            counts[2],
            counts[3],
            verify_rate,
            counts[4],
            counts[5],
            prefill_rate,
        );
        Ok(())
    }
}

fn hit_rate(hits: u64, misses: u64) -> f64 {
    let total = hits + misses;
    if total == 0 {
        f64::NAN
    } else {
        hits as f64 / total as f64
    }
}

fn stats_enabled_from_env(value: Option<&str>) -> bool {
    matches!(value, Some("1" | "true"))
}

fn u32s(b: &[u8]) -> impl Iterator<Item = u32> + '_ {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
}

/// Load the sections for `layer_pos` (positional index among the model's
/// engram layers) covering `heads` and upload them. `head_offsets` are the
/// layer-wide row offsets per head (cumulative head sizes) so the probe
/// key matches the layer-wide ids the hash kernel emits.
pub(crate) fn load(
    mem: &mut Owned<'_>,
    layer_pos: usize,
    layer_id: usize,
    heads: std::ops::Range<usize>,
    head_offsets: &[usize],
    dim: usize,
    verify: Option<&dyn Fn(u32, &[u8]) -> bool>,
) -> Result<Option<HotCache>> {
    let path = match std::env::var(PATH_ENV) {
        Ok(p) if !p.is_empty() => p,
        _ => return Ok(None),
    };
    let file = std::fs::read(&path).with_context(|| format!("reading {path}"))?;
    ensure!(file.len() >= 24, "hotcache header truncated");
    let hdr: Vec<u32> = u32s(&file[..24]).collect();
    ensure!(
        hdr[0] == MAGIC && hdr[1] == 1 && hdr[4] as usize == dim * 2 && hdr[3] == 16,
        "hotcache header mismatch"
    );
    let (layers, nheads, row_bytes, k) = (
        hdr[2] as usize,
        hdr[3] as usize,
        hdr[4] as usize,
        hdr[5] as usize,
    );
    ensure!(
        layer_pos < layers && heads.end <= nheads,
        "hotcache layer/head range"
    );
    // Sections are variable-length (`count` rows each), so the only sound
    // way to reach one is walking the file in order — a fixed stride
    // misreads everything after a short section.
    let mut cursor = 24usize;
    let mut sections: std::collections::HashMap<(usize, usize), (usize, usize)> =
        std::collections::HashMap::new();
    for l in 0..layers {
        for h in 0..nheads {
            ensure!(cursor + 4 <= file.len(), "hotcache section truncated");
            let count = u32::from_le_bytes(file[cursor..cursor + 4].try_into().unwrap()) as usize;
            ensure!(count <= k, "hotcache count over k");
            let bytes = 4 + count * 4 + count * row_bytes;
            ensure!(
                cursor + bytes <= file.len(),
                "hotcache section data truncated"
            );
            sections.insert((l, h), (cursor, count));
            cursor += bytes;
        }
    }
    // Gather this rank's head sections in head order: layer-wide ids
    // (adding the constant head offset preserves the ascending sort).
    let mut keys: Vec<u32> = Vec::new();
    let mut rows: Vec<u8> = Vec::new();
    let mut offs = [0u32; 17];
    for h in 0..nheads {
        offs[h] = keys.len() as u32;
        if !heads.contains(&h) {
            continue;
        }
        let (at, count) = sections[&(layer_pos, h)];
        let ids_at = at + 4;
        let rows_at = ids_at + count * 4;
        let hoff = head_offsets[h] as u32;
        keys.extend(u32s(&file[ids_at..rows_at]).map(|r| r + hoff));
        rows.extend_from_slice(&file[rows_at..rows_at + count * row_bytes]);
        if let Some(verify) = verify {
            // Up to 8 evenly spaced rows per section: enough that a stale
            // sidecar cannot plausibly survive, cheap enough that the
            // startup cost is a handful of page faults.
            let samples = count.min(8);
            for i in 0..samples {
                let r = i * count / samples;
                let rowid = u32::from_le_bytes(
                    file[ids_at + r * 4..ids_at + r * 4 + 4].try_into().unwrap(),
                ) + hoff;
                let expect = &file[rows_at + r * row_bytes..rows_at + (r + 1) * row_bytes];
                ensure!(
                    verify(rowid, expect),
                    "engram hotcache row {rowid} does not match the checkpoint table — the sidecar is stale; rebuild it for this checkpoint"
                );
            }
        }
    }
    offs[nheads] = keys.len() as u32;
    let count = keys.len();
    if count == 0 {
        // No hot rows for this rank's heads: run uncached rather than fail
        // startup on the empty upload.
        return Ok(None);
    }
    let keys_b: Vec<u8> = keys.iter().flat_map(|v| v.to_le_bytes()).collect();
    let offs_b: Vec<u8> = offs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let dkeys = mem.upload(&keys_b)?;
    let drows = mem.upload(&rows)?;
    let doffs = mem.upload(&offs_b)?;
    let stats = if stats_enabled_from_env(std::env::var(STATS_ENV).ok().as_deref()) {
        mem.upload(&[0u8; STATS_BYTES])?
    } else {
        DevicePtr::NULL
    };
    tracing::info!(
        "engram hot-row cache: {count} rows ({:.2} GB) device-resident for layer {}",
        count as f64 * row_bytes as f64 / 1073741824.0,
        layer_pos
    );
    Ok(Some(HotCache {
        keys: dkeys,
        rows: drows,
        offs: doffs,
        stats,
        layer_id,
        report_ticks: AtomicUsize::new(0),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn stats_knob_is_explicitly_opt_in() {
        assert!(!stats_enabled_from_env(None));
        assert!(!stats_enabled_from_env(Some("0")));
        assert!(!stats_enabled_from_env(Some("yes")));
        assert!(stats_enabled_from_env(Some("1")));
        assert!(stats_enabled_from_env(Some("true")));
    }

    #[test]
    fn stats_reporting_is_rate_limited() {
        let gpu = MockGpuBackend::new();
        let stats = gpu.alloc(STATS_BYTES).unwrap();
        let cache = HotCache {
            keys: DevicePtr::NULL,
            rows: DevicePtr::NULL,
            offs: DevicePtr::NULL,
            stats,
            layer_id: 2,
            report_ticks: AtomicUsize::new(0),
        };
        for _ in 0..REPORT_INTERVAL_LAUNCHES - 1 {
            cache.maybe_report_stats(&gpu, 0, 0).unwrap();
        }
        assert_eq!(gpu.sync_count(), 0);
        cache.maybe_report_stats(&gpu, 0, 0).unwrap();
        assert_eq!(gpu.sync_count(), 1);
        gpu.free(stats).unwrap();
    }

    #[test]
    fn empty_hit_rate_is_nan_not_zero() {
        assert!(hit_rate(0, 0).is_nan());
        assert_eq!(hit_rate(3, 1), 0.75);
    }
}
