// SPDX-License-Identifier: AGPL-3.0-only

//! Request helpers used by model-specific admission fences.

use super::inference_types::InferenceRequest;

impl InferenceRequest {
    /// Whether this request carries a constrained-decoding grammar.
    pub fn has_grammar_spec(&self) -> bool {
        match self {
            InferenceRequest::Blocking { grammar_spec, .. }
            | InferenceRequest::Streaming { grammar_spec, .. } => grammar_spec.is_some(),
        }
    }

    /// Force native serial decode for a request that cannot enter a model's
    /// speculative collective.
    pub(crate) fn disable_mtp_for_fallback(&mut self) {
        match self {
            InferenceRequest::Blocking { disable_mtp, .. }
            | InferenceRequest::Streaming { disable_mtp, .. } => *disable_mtp = true,
        }
    }
}

/// `ATLAS_DISABLE_MTP_FOR_TOOLS` gate (default on). Resolves the env value;
/// a caller must pass the result to `tools_gate_disable_mtp`. "0" / "false"
/// turns the gate off; anything else (including absent) leaves it on.
pub(crate) fn disable_mtp_for_tools_gate_on() -> bool {
    !matches!(
        std::env::var("ATLAS_DISABLE_MTP_FOR_TOOLS").as_deref(),
        Ok("0") | Ok("false")
    )
}

/// Whether MTP must stay off for this tool-active request (F60, 2026-04-27).
/// Client-supplied `disable_mtp` always wins. Otherwise the gate decides:
/// tools active + gate on -> serial decode.
pub(crate) fn tools_gate_disable_mtp(
    tools_active: bool,
    client_disable_mtp: bool,
    gate_on: bool,
) -> bool {
    if client_disable_mtp {
        return true;
    }
    tools_active && gate_on
}

#[cfg(test)]
mod tests {
    use super::tools_gate_disable_mtp;

    #[test]
    fn client_request_always_forces_serial() {
        assert!(tools_gate_disable_mtp(false, true, true));
        assert!(tools_gate_disable_mtp(true, true, true));
        assert!(tools_gate_disable_mtp(true, true, false));
    }

    #[test]
    fn non_tool_requests_keep_client_choice() {
        assert!(!tools_gate_disable_mtp(false, false, true));
        assert!(!tools_gate_disable_mtp(false, false, false));
    }

    #[test]
    fn tools_active_gate_on_forces_serial() {
        assert!(tools_gate_disable_mtp(true, false, true));
    }

    #[test]
    fn tools_active_gate_off_keep_mtp() {
        assert!(!tools_gate_disable_mtp(true, false, false));
    }
}
