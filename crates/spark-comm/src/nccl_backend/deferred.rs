// SPDX-License-Identifier: AGPL-3.0-only

//! Deferred-completion all-reduce: the reduce is enqueued on an internal
//! stream ordered after the caller's compute stream by events, but the
//! compute stream is not joined to it, so kernels enqueued in between (the
//! hy4 shared-expert MLP) overlap the transfer. The caller joins just before
//! the result is consumed (`all_reduce_wait_inner`). Same reduction, same
//! order, same numbers — only the schedule changes. The trait methods live
//! in `comm_impl.rs` (a trait impl cannot span files); they delegate here.

use super::NcclBackend;
use crate::nccl;
use anyhow::Result;
use std::sync::atomic::Ordering;

impl NcclBackend {
    /// [`crate::CommBackend::all_reduce_async_deferred`]'s body.
    pub(super) fn all_reduce_deferred_inner(
        &self,
        ptr: u64,
        bytes: usize,
        compute_stream: u64,
    ) -> Result<bool> {
        // Only the two-rank paths have an internal stream to defer onto: the
        // one-shot channel is a device sequence on the caller's stream and the
        // chain needs world > 2, so both take `false` and the caller falls
        // back to `all_reduce_async`.
        if self.world_size == 2 && self.add_kernel.load(Ordering::Relaxed) != 0 {
            if self.try_rdma_all_reduce_deferred(ptr, bytes, compute_stream)? {
                return Ok(true);
            }
            // The NCCL send/recv pair already runs on the comm stream; only
            // the final compute-stream join is deferred.
            nccl::record_event(self.compute_done_event, compute_stream)?;
            nccl::stream_wait_event(self.comm_stream, self.compute_done_event)?;
            self.all_reduce_2rank(ptr, bytes, self.comm_stream)?;
            nccl::record_event(self.comm_done_event, self.comm_stream)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// [`crate::CommBackend::all_reduce_async_wait`]'s body.
    pub(super) fn all_reduce_wait_inner(&self, compute_stream: u64) -> Result<()> {
        // Whichever internal stream the deferred op took, its completion is
        // on one of these events; the other is either never recorded or
        // already complete, so waiting on it is a no-op.
        nccl::stream_wait_event(compute_stream, self.comm_done_event)?;
        #[cfg(atlas_rdma_verbs)]
        if let Some(rdma) = &self.rdma {
            rdma.exchange_wait(compute_stream)?;
        }
        Ok(())
    }
}
