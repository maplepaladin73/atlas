RULES: list[tuple[str, str, str]] = [
    ("kernels/gb10/hy4-flash/", "P0", "hy4-flash"),
    ("crates/spark-model/src/model/hy4", "P0", "hy4-flash"),
    ("crates/spark-model/src/layers/hy4", "P0", "hy4-flash"),
    ("crates/spark-model/src/weight_loader/hy4", "P0", "hy4-flash"),
    ("crates/spark-model/src/factory/hy4", "P0", "hy4-flash"),
    ("crates/spark-model/examples/hy4_dfly_offline", "P0", "hy4-flash"),
    ("crates/atlas-core/src/config/hy4", "P0", "hy4-flash"),
    ("crates/atlas-core/src/config/parsers/hy4", "P0", "hy4-flash"),
    ("crates/atlas-core/src/config/fixtures/hy4", "P0", "hy4-flash"),
    ("crates/atlas-core/src/hy4_reference", "P0", "hy4-flash"),
    ("crates/atlas-kernels/tests/target_resolution/hy4", "P0", "hy4-flash"),
    ("crates/spark-server/src/tokenizer/hy4", "P0", "hy4-flash"),
    ("crates/spark-server/src/tool_parser/hy4", "P0", "hy4-flash"),
    ("crates/spark-server/src/grammar/tests/hy4", "P0", "hy4-flash"),
    ("crates/spark-server/src/main_modules/serve_phases/preflight/hy4", "P0", "hy4-flash"),
    ("jinja-templates/hy4", "P0", "hy4-flash"),
    ("docs/HY4_TWO_SPARK.md", "P0", "hy4-flash"),
    ("kernels/gb10/ling-3.2-air/", "P0", "ling-3.2-air"),
    ("crates/spark-model/src/model/bailing_c1", "P0", "ling-3.2-air"),
    ("crates/spark-model/src/weight_loader/bailing", "P0", "ling-3.2-air"),
    ("crates/spark-model/src/layers/ling_mtp", "P0", "ling-3.2-air"),
    ("crates/atlas-core/src/config/parsers/bailing", "P0", "ling-3.2-air"),
    ("jinja-templates/ling-3.2-air", "P0", "ling-3.2-air"),
    ("scripts_debug/ling_e2e", "P0", "ling-3.2-air"),
    ("crates/spark-model/src/factory", "P1", "shared spark-model factory"),
    ("crates/spark-model/src/weight_loader/", "P1", "shared weight loading"),
    ("crates/spark-model/src/preflight", "P1", "shared preflight"),
    ("crates/spark-model/src/layers/", "P1", "shared layers/ops"),
    ("crates/spark-model/src/model/", "P1", "shared model backends"),
    ("crates/spark-runtime/", "P1", "shared runtime"),
    ("crates/atlas-kernels/", "P1", "shared kernel registry/build"),
    ("crates/spark-server/src/api/", "P1", "shared HTTP API"),
    ("crates/spark-server/src/scheduler", "P1", "shared scheduler"),
    ("crates/spark-server/src/scheduling_policy", "P1", "shared scheduling policy"),
    ("crates/spark-server/src/grammar/", "P1", "shared grammar"),
    ("crates/spark-server/src/tool_parser/", "P1", "shared tool parsing"),
    ("crates/spark-server/src/tokenizer/", "P1", "shared tokenizer"),
    ("crates/atlas-core/src/", "P1", "shared atlas-core"),
    ("jinja-templates/", "P1", "shared chat templates"),
    ("kernels/gb10/common/", "P1", "shared gb10 kernels"),
    ("kernels/gb10/HARDWARE.toml", "P1", "shared gb10 hardware registry"),
    (".github/workflows/", "P1", "CI"),
]


def classify(path: str) -> dict[str, str]:
    normalized = path.lstrip("./")
    for prefix, tier, area in RULES:
        if normalized.startswith(prefix):
            return {"tier": tier, "area": area}
    return {"tier": "P2", "area": "other"}


def render_rules() -> str:
    lines = []
    for prefix, tier, area in RULES:
        lines.append(f"- `{prefix}` -> {tier} ({area})")
    return "\n".join(lines)
