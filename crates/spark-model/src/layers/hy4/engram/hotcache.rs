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

use crate::layers::hy4::attention::memory::Owned;
use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

pub(crate) const PATH_ENV: &str = "ATLAS_HY4_ENGRAM_HOTCACHE";
const MAGIC: u32 = 0x48305443;

/// One layer's device-resident hot rows for the heads this rank owns:
/// per-head sorted sections of layer-wide row ids concatenated in
/// `keys`, matching row bytes in `rows`, and 17 section offsets in `offs`
/// (empty sections for heads the rank does not own).
pub(crate) struct HotCache {
    pub(crate) keys: DevicePtr,
    pub(crate) rows: DevicePtr,
    pub(crate) offs: DevicePtr,
}

fn u32s(b: &[u8]) -> impl Iterator<Item = u32> + '_ {
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()))
}

/// Load the sections for `layer_pos` (positional index among the model's
/// engram layers) covering `heads` and upload them. `head_offsets` are the
/// layer-wide row offsets per head (cumulative head sizes) so the probe
/// key matches the layer-wide ids the hash kernel emits.
pub(crate) fn load(
    mem: &mut Owned<'_>,
    layer_pos: usize,
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
    let (layers, nheads, row_bytes, k) =
        (hdr[2] as usize, hdr[3] as usize, hdr[4] as usize, hdr[5] as usize);
    ensure!(layer_pos < layers && heads.end <= nheads, "hotcache layer/head range");
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
            ensure!(cursor + bytes <= file.len(), "hotcache section data truncated");
            sections.insert((l, h), (cursor, count));
            cursor += bytes;
        }
    }
    // Gather this rank's head sections in head order: layer-wide ids
    // (adding the constant head offset preserves the ascending sort).
    let mut keys: Vec<u32> = Vec::new();
    let mut rows: Vec<u8> = Vec::new();
    let mut offs = vec![0u32; 17];
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
                let expect =
                    &file[rows_at + r * row_bytes..rows_at + (r + 1) * row_bytes];
                ensure!(
                    verify(rowid, expect),
                    "engram hotcache row {rowid} does not match the checkpoint table —                      the sidecar is stale; rebuild it for this checkpoint"
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
    tracing::info!(
        "engram hot-row cache: {count} rows ({:.2} GB) device-resident for layer {}",
        count as f64 * row_bytes as f64 / 1073741824.0,
        layer_pos
    );
    Ok(Some(HotCache {
        keys: dkeys,
        rows: drows,
        offs: doffs,
    }))
}
