// SPDX-License-Identifier: AGPL-3.0-only

//! Exact-row NVMe staging for HY4 Engram tables.
//!
//! The mmap/page-cache path faults whole host pages for randomly distributed
//! 192-byte rows. HY4 row ids are a pure function of token ids, so the host
//! can instead resolve the exact rows, fault them into pinned GPU-addressable
//! slots with O_DIRECT reads, and rewrite the lookup ids to slot indices.
//! This module is the thin HY4 adapter over `spark_storage::NgramRowCache`.

use super::Hy4Engram;
use anyhow::{Result, ensure};
#[cfg(feature = "cuda")]

/// Opt-in exact-row cache. Off preserves the existing mmap/device-resident
/// table behavior bit-for-bit.
pub(crate) const ENABLE_ENV: &str = "ATLAS_HY4_ENGRAM_ROW_CACHE";
/// Slots per engram layer per rank. Each bf16 row is 192 B, so the default
/// costs about 48 MiB per layer (192 MiB per rank across four layers).
pub(crate) const SLOTS_ENV: &str = "ATLAS_HY4_ENGRAM_ROW_CACHE_SLOTS";
pub(crate) const DEFAULT_SLOTS: usize = 262_144;

pub(crate) fn enabled_from_env(value: Option<&str>) -> bool {
    matches!(value, Some("1") | Some("true"))
}

pub(crate) fn enabled() -> bool {
    enabled_from_env(std::env::var(ENABLE_ENV).ok().as_deref())
}

pub(crate) fn slots_from_env(value: Option<&str>) -> Result<usize> {
    match value {
        None => Ok(DEFAULT_SLOTS),
        Some(v) => {
            let n = v.parse::<usize>()?;
            ensure!(n > 0, "{SLOTS_ENV} must be positive");
            Ok(n)
        }
    }
}

pub(crate) fn slots() -> Result<usize> {
    slots_from_env(std::env::var(SLOTS_ENV).ok().as_deref())
}

#[cfg(feature = "cuda")]
pub(crate) fn open_table(
    fp8_table: bool,
    d: &spark_runtime::weights::DeferredTensor,
    name: &str,
    dtype: spark_runtime::weights::WeightDtype,
    shape: &[usize],
    start: usize,
    rows_total: usize,
    row_stride: usize,
) -> Result<Option<(u64, spark_storage::NgramRowCache, usize)>> {
    if fp8_table || !enabled() {
        return Ok(None);
    }
    ensure!(
        d.shape == shape && d.dtype == dtype,
        "Hy4 Engram deferred table contract: {name}"
    );
    let slots = slots()?;
    let cache = spark_storage::NgramRowCache::open_at(
        &d.path,
        d.offset + start as u64,
        None,
        rows_total as u64,
        row_stride,
        slots,
    )
    .map_err(|e| anyhow::anyhow!("Hy4 Engram exact-row cache {name}: {e:#}"))?;
    let table = cache.table_dev_va()?;
    Ok(Some((table, cache, slots)))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn open_table(
    _fp8_table: bool,
    _d: &spark_runtime::weights::DeferredTensor,
    _name: &str,
    _dtype: spark_runtime::weights::WeightDtype,
    _shape: &[usize],
    _start: usize,
    _rows_total: usize,
    _row_stride: usize,
) -> Result<Option<(u64, (), usize)>> {
    Ok(None)
}

/// A resolved batch's pin guard. Pins are NOT released at drop: the lookup
/// kernel runs asynchronously after the launch, so a slot dropped at launch
/// time could be evicted and overwritten before the kernel reads it. Instead
/// the batch stays pinned until the NEXT resolve on the same cache, which
/// first synchronizes the event recorded after the previous lookup launch —
/// stream-order proof that kernel completed — and only then ends the batch.
/// This keeps at most one batch of pins outstanding per cache and preserves
/// the single-batch pin model of `NgramRowCache`.
#[cfg(feature = "cuda")]
pub(crate) struct RowBatchGuard<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
}

#[cfg(not(feature = "cuda"))]
pub(crate) struct RowBatchGuard<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
}

impl Hy4Engram<'_> {
    /// Resolve every valid local row id in a row-major `[rows, total_heads]`
    /// hash matrix to its exact row-cache slot, in place. Returns the pin
    /// guard that must live until after the lookup launch. Without the cache
    /// the matrix is left unchanged and `None` is returned.
    #[cfg(feature = "cuda")]
    pub(crate) fn resolve_hash_matrix<'a>(
        &'a self,
        matrix: &mut [i32],
    ) -> Result<Option<RowBatchGuard<'a>>> {
        let Some(cache) = &self.rowcache else {
            return Ok(None);
        };
        let total_heads = self.layout.sizes.len();
        ensure!(
            matrix.len().is_multiple_of(total_heads),
            "Hy4 Engram row-cache hash matrix geometry"
        );
        let rows_out = matrix.len() / total_heads;
        let mut ids = Vec::new();
        let mut positions = Vec::new();
        for r in 0..rows_out {
            for local_head in 0..self.heads.len() {
                let head = self.heads.start + local_head;
                let pos = r * total_heads + head;
                let row = matrix[pos];
                if row < 0 {
                    continue;
                }
                let row = row as usize;
                ensure!(
                    row >= self.rows.start && row < self.rows.end,
                    "Hy4 Engram hash row {row} outside this rank's slice {:?}",
                    self.rows
                );
                ids.push((row - self.rows.start) as u64);
                positions.push(pos);
            }
        }
        let mut cache = cache
            .lock()
            .map_err(|_| anyhow::anyhow!("Hy4 Engram row cache mutex poisoned"))?;
        // Release the PREVIOUS batch's pins, but only after the event recorded
        // after its lookup launch proves the kernel finished reading the slots.
        // First resolve on this cache has no outstanding batch.
        {
            let mut pending = self
                .rowcache_event
                .lock()
                .map_err(|_| anyhow::anyhow!("Hy4 Engram row-cache event mutex poisoned"))?;
            if let Some(event) = pending.take() {
                self.mem.gpu.event_synchronize(event)?;
                self.mem.gpu.destroy_event(event)?;
                cache.end_batch();
            }
        }
        let mut slots = Vec::with_capacity(ids.len());
        cache.resolve(&ids, &mut slots)?;
        ensure!(
            slots.len() == positions.len(),
            "Hy4 Engram row-cache slot count"
        );
        for (pos, slot) in positions.into_iter().zip(slots) {
            matrix[pos] = slot as i32;
        }
        Ok(Some(RowBatchGuard {
            _marker: std::marker::PhantomData,
        }))
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn resolve_hash_matrix<'a>(
        &'a self,
        _matrix: &mut [i32],
    ) -> Result<Option<RowBatchGuard<'a>>> {
        Ok(None)
    }

    /// Bounds passed to the lookup kernels. Cached lookups use slot ids in
    /// `[0, slots)`; uncached lookups use row ids in this rank's table slice.
    pub(crate) fn lookup_bounds(&self) -> (u32, u32) {
        if self.rowcache_slots > 0 {
            (0, self.rowcache_slots as u32)
        } else {
            (self.rows.start as u32, self.rows.len() as u32)
        }
    }

    /// Record the stream event that completes when the just-launched lookup
    /// has finished reading the pinned slots. The next resolve synchronizes it
    /// before releasing that batch's pins. No-op without the cache.
    #[cfg(feature = "cuda")]
    pub(crate) fn record_lookup_event(&self, stream: u64) -> Result<()> {
        if self.rowcache_slots == 0 {
            return Ok(());
        }
        let event = self.mem.gpu.create_event()?;
        self.mem.gpu.record_event(event, stream)?;
        let mut pending = self
            .rowcache_event
            .lock()
            .map_err(|_| anyhow::anyhow!("Hy4 Engram row-cache event mutex poisoned"))?;
        // The pin batch is released one resolve later; an event still pending
        // here means a lookup launch was missed, which would leak pins — the
        // release ordering is the correctness invariant, so fail loudly.
        ensure!(
            pending.is_none(),
            "Hy4 Engram row-cache lookup event already pending"
        );
        *pending = Some(event);
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn record_lookup_event(&self, _stream: u64) -> Result<()> {
        Ok(())
    }

    /// Host-side HY4 n-gram hashes for one row. `context` is the oldest-first
    /// suffix of at most the two preceding tokens; missing prefix positions are
    /// absent, which is exactly the dead-id/padding contract of `HashLayout`.
    pub(crate) fn host_hashes(&self, context: &[u32], token: u32) -> Result<Vec<i32>> {
        ensure!(context.len() <= 2, "Hy4 Engram host hash context width");
        let history: Vec<i64> = context.iter().map(|v| *v as i64).collect();
        self.layout.hash_validated(&history, token as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_cache_is_explicit_opt_in() {
        assert!(!enabled_from_env(None));
        assert!(!enabled_from_env(Some("0")));
        assert!(!enabled_from_env(Some("yes")));
        assert!(enabled_from_env(Some("1")));
        assert!(enabled_from_env(Some("true")));
    }

    #[test]
    fn row_cache_slots_default_and_parse() {
        assert_eq!(slots_from_env(None).unwrap(), DEFAULT_SLOTS);
        assert_eq!(slots_from_env(Some("1024")).unwrap(), 1024);
        assert!(slots_from_env(Some("0")).is_err());
        assert!(slots_from_env(Some("bad")).is_err());
    }
}
