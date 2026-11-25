import pytest
from mistralai import agents

from main import agent
from priority_map import classify


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_CLIENT_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)


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


async def test_missing_token_surfaces_as_tool_error_and_turn_continues() -> None:
    model = agents.testing.ScriptedModel(
        [
            agents.testing.run_typescript(
                'return await tools.client.list_commits({repo: "maplepaladin73/atlas"});'
            ),
            agents.testing.reply("GITHUB_TOKEN is missing, so the scan cannot start."),
        ]
    )
    state = await agent.override(model=model).run("list recent commits")
    assert isinstance(state, agents.Completed), state
    assert state.text == "GITHUB_TOKEN is missing, so the scan cannot start."
    assert [call.name for call in state.tool_calls] == ["client.list_commits"]
    model.assert_consumed()
