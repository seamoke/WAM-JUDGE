from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robotwin_critic.two_stage_rft.online_iteration import (
    commit_collect,
    complete_update,
    initialize_state,
    load_state,
    prepare_collect,
    read_jsonl,
    select_online_winners,
)
from robotwin_critic.two_stage_rft.stage_updated_model import stage_model


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class OnlineIterationTest(unittest.TestCase):
    def test_collect_only_capacity_never_creates_ready_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts = root / "contexts.jsonl"
            model = root / "model"
            model.mkdir()
            contexts.write_text('{"context_id":"q"}\n', encoding="utf-8")
            state_path = root / "state.json"
            initialize_state(state_path, contexts, model, base_seed=1)
            collect = root / "collect"
            prepare_collect(state_path, collect, workers=1, q_per_worker=1)
            scored = collect / "scored.jsonl"
            scored.write_text(
                json.dumps({
                    "candidate_id": "c",
                    "context_id": "q@e0",
                    "process_score": 8.0,
                    "process_critic": {"numeric_parsed": True},
                    "action_critic": {"accepted": True, "action_score": 0.9},
                }) + "\n",
                encoding="utf-8",
            )
            manifest = root / "split.json"
            manifest.write_text("{}\n", encoding="utf-8")
            summary = commit_collect(
                state_path,
                collect,
                scored,
                root / "pending.jsonl",
                root / "buffers",
                capacity=0,
                min_action_score=0.5,
                min_process_score=0.0,
                split_manifest=manifest,
            )
            self.assertIsNone(load_state(state_path)["ready_buffer"])
            self.assertEqual(summary["pending_after_commit"], 1)

    def test_prepare_shards_global_q_and_wraps_without_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts = root / "contexts.jsonl"
            write_jsonl(
                contexts,
                [{"context_id": f"q{index}", "text": "task"} for index in range(3)],
            )
            state_path = root / "state.json"
            initialize_state(state_path, contexts, root / "model", base_seed=7)
            manifest = prepare_collect(
                state_path, root / "collect", workers=2, q_per_worker=2
            )
            rows = read_jsonl(root / "collect/contexts_worker_00.jsonl")
            rows += read_jsonl(root / "collect/contexts_worker_01.jsonl")
            self.assertEqual(manifest["global_q"], 4)
            self.assertEqual(len({row["context_id"] for row in rows}), 4)
            self.assertEqual(manifest["next_context_epoch"], 1)
            self.assertEqual(manifest["next_context_index"], 1)

    def test_dual_selection_is_one_winner_per_context(self) -> None:
        rows = []
        for context in ("a", "b"):
            for candidate, process in enumerate((4.0, 8.0)):
                rows.append(
                    {
                        "context_id": context,
                        "candidate_id": f"{context}/{candidate}",
                        "process_score": process,
                        "process_critic": {"numeric_parsed": True},
                        "action_critic": {
                            "accepted": True,
                            "action_score": 0.9,
                        },
                    }
                )
        selected, summary = select_online_winners(
            rows,
            min_action_score=0.5,
            min_process_score=5.0,
            split_manifest_sha256="split",
            collect_index=0,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual({row["process_score"] for row in selected}, {8.0})
        self.assertEqual(summary["process_rejected"], 2)

    def test_zero_max_per_context_keeps_all_dual_accepted_candidates(self) -> None:
        rows = []
        for context in ("a", "b"):
            for candidate, process in enumerate((6.0, 8.0)):
                rows.append(
                    {
                        "context_id": context,
                        "candidate_id": f"{context}/{candidate}",
                        "process_score": process,
                        "process_critic": {"numeric_parsed": True},
                        "action_critic": {"accepted": True, "action_score": 0.9},
                    }
                )
        selected, summary = select_online_winners(
            rows,
            min_action_score=0.5,
            min_process_score=5.0,
            split_manifest_sha256="split",
            collect_index=0,
            max_per_context=0,
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(summary["eligible_candidates"], 4)
        self.assertEqual(summary["max_per_context"], 0)
        self.assertEqual(
            [row["rft_selection"]["rank_within_context"] for row in selected],
            [0, 1, 0, 1],
        )

    def test_commit_fills_buffer_and_complete_advances_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts = root / "contexts.jsonl"
            split = root / "split.json"
            split.write_text("{}\n", encoding="utf-8")
            write_jsonl(
                contexts,
                [{"context_id": f"q{index}", "text": "task"} for index in range(2)],
            )
            state_path = root / "state.json"
            initialize_state(state_path, contexts, root / "model0", base_seed=7)
            collect = root / "collect"
            prepare_collect(state_path, collect, workers=1, q_per_worker=2)
            scheduled = read_jsonl(collect / "contexts_worker_00.jsonl")
            scored = []
            for row in scheduled:
                scored.append(
                    {
                        **row,
                        "candidate_id": f"{row['context_id']}/0",
                        "process_score": 9.0,
                        "process_critic": {"numeric_parsed": True},
                        "action_critic": {
                            "accepted": True,
                            "action_score": 0.9,
                        },
                    }
                )
            scored_path = collect / "scored.jsonl"
            write_jsonl(scored_path, scored)
            summary = commit_collect(
                state_path,
                collect,
                scored_path,
                root / "pending.jsonl",
                root / "buffers",
                capacity=2,
                min_action_score=0.5,
                min_process_score=5.0,
                split_manifest=split,
            )
            self.assertTrue(Path(summary["ready_buffer"]).is_file())
            state = complete_update(state_path, root / "model1")
            self.assertEqual(state["update_index"], 1)
            self.assertIsNone(state["ready_buffer"])
            self.assertEqual(state["consumed_total"], 2)

    def test_stage_model_replaces_only_transformer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            updated = root / "checkpoint/transformer"
            (base / "transformer").mkdir(parents=True)
            (base / "vae").mkdir()
            updated.mkdir(parents=True)
            (base / "transformer/config.json").write_text("{}")
            (updated / "config.json").write_text("{}")
            output = root / "staged"
            stage_model(base, updated, output)
            self.assertEqual((output / "transformer").resolve(), updated.resolve())
            self.assertEqual((output / "vae").resolve(), (base / "vae").resolve())
            updated2 = root / "checkpoint2/transformer"
            updated2.mkdir(parents=True)
            (updated2 / "config.json").write_text("{}")
            output2 = root / "staged2"
            stage_model(output, updated2, output2)
            self.assertEqual((output2 / "transformer").resolve(), updated2.resolve())
            self.assertEqual((output2 / "vae").resolve(), (base / "vae").resolve())
            self.assertFalse((output2 / "online_rft_model.json").is_symlink())

    def test_stage_model_can_materialize_updated_transformer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            updated = root / "checkpoint/transformer"
            (base / "transformer").mkdir(parents=True)
            (base / "vae").mkdir()
            updated.mkdir(parents=True)
            (base / "transformer/config.json").write_text("{}")
            (updated / "config.json").write_text("{}")
            (updated / "weights.bin").write_bytes(b"weights")

            output = root / "staged"
            manifest = stage_model(
                base,
                updated,
                output,
                move_transformer=True,
            )

            self.assertFalse(updated.exists())
            self.assertFalse((output / "transformer").is_symlink())
            self.assertEqual((output / "transformer/weights.bin").read_bytes(), b"weights")
            self.assertEqual(manifest["transformer_storage"], "materialized")

    def test_stage_model_can_copy_updated_transformer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            updated = root / "checkpoint/transformer"
            (base / "transformer").mkdir(parents=True)
            (base / "vae").mkdir()
            updated.mkdir(parents=True)
            (base / "transformer/config.json").write_text("{}")
            (updated / "config.json").write_text("{}")
            (updated / "weights.bin").write_bytes(b"weights")

            output = root / "staged"
            manifest = stage_model(base, updated, output, copy_transformer=True)

            self.assertTrue((updated / "weights.bin").is_file())
            self.assertFalse((output / "transformer").is_symlink())
            self.assertEqual((output / "transformer/weights.bin").read_bytes(), b"weights")
            self.assertEqual(manifest["transformer_storage"], "copied")


if __name__ == "__main__":
    unittest.main()
