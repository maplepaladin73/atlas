import pytest
from mistralai import agents

from main import AREA_PREFIXES, agent
from priority_map import classify


@pytest.fixture(autouse=True)
def dummy_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # A dummy key satisfies the SDK's connector assembly; every test overrides the
    # model with a ScriptedModel, so no real API call ever happens.
    monkeypatch.setenv("MISTRAL_API_KEY", "test-only-not-a-real-key")


def test_classify_p0_model_areas() -> None:
    assert classify("kernels/gb10/hy4-flash/nvfp4/hy4_mla.cu") == {"tier": "P0", "area": "hy4-flash"}
    assert classify("crates/spark-model/src/model/hy4/mod.rs") == {"tier": "P0", "area": "hy4-flash"}
    assert classify("crates/atlas-core/src/config/parsers/hy4.rs") == {"tier": "P0", "area": "hy4-flash"}
    assert classify("kernels/gb10/ling-3.2-air/nvfp4/ling_mla_rope.cuh") == {
        "tier": "P0",
        "area": "ling-3.2-air",
    }
    assert classify("crates/spark-model/src/model/bailing_c1/mla.rs") == {
        "tier": "P0",
        "area": "ling-3.2-air",
    }
    assert classify("jinja-templates/ling-3.2-air.jinja") == {"tier": "P0", "area": "ling-3.2-air"}


def test_classify_p0_wins_over_shared_prefixes() -> None:
    assert classify("crates/spark-model/src/weight_loader/hy4.rs")["tier"] == "P0"
    assert classify("crates/spark-model/src/layers/hy4/ffn.rs")["tier"] == "P0"
    assert classify("crates/atlas-core/src/config/parsers/bailing.rs")["tier"] == "P0"


def test_classify_p1_shared_and_p2_fallback() -> None:
    assert classify("crates/spark-model/src/factory.rs")["tier"] == "P1"
    assert classify("crates/spark-server/src/api/chat/sampling_setup.rs")["tier"] == "P1"
    assert classify("README.md") == {"tier": "P2", "area": "other"}
    assert classify("./kernels/gb10/hy4-flash/MODEL.toml")["tier"] == "P0"


def test_area_prefixes_cover_both_p0_areas() -> None:
    assert any(p.startswith("kernels/gb10/hy4-flash") for p in AREA_PREFIXES["hy4-flash"])
    assert any(p.startswith("kernels/gb10/ling-3.2-air") for p in AREA_PREFIXES["ling-3.2-air"])
    assert all(classify(p + "x")["tier"] == "P0" for prefixes in AREA_PREFIXES.values() for p in prefixes)


async def test_classify_files_tool_runs_in_turn() -> None:
    model = agents.testing.ScriptedModel(
        [
            agents.testing.run_typescript(
                'return await tools.client.classify_files({paths: ['
                '"kernels/gb10/hy4-flash/nvfp4/hy4_mla.cu", '
                '"kernels/gb10/ling-3.2-air/nvfp4/ling_mtp_eh.cu", '
                '"crates/spark-model/src/factory.rs", "README.md"]});'
            ),
            agents.testing.reply("classified 4 paths"),
        ]
    )
    state = await agent.override(model=model).run("classify these paths")
    assert isinstance(state, agents.Completed), state
    assert state.text == "classified 4 paths"
    assert [call.name for call in state.tool_calls] == ["client.classify_files"]
    assert state.tool_calls[0].arguments == {
        "paths": [
            "kernels/gb10/hy4-flash/nvfp4/hy4_mla.cu",
            "kernels/gb10/ling-3.2-air/nvfp4/ling_mtp_eh.cu",
            "crates/spark-model/src/factory.rs",
            "README.md",
        ]
    }
    model.assert_consumed()


async def test_now_iso_tool_returns_iso_timestamp() -> None:
    model = agents.testing.ScriptedModel(
        [
            agents.testing.run_typescript("return await tools.client.now_iso();"),
            agents.testing.reply("got the time"),
        ]
    )
    state = await agent.override(model=model).run("what time is it?")
    assert isinstance(state, agents.Completed), state
    assert [call.name for call in state.tool_calls] == ["client.now_iso"]
    model.assert_consumed()
