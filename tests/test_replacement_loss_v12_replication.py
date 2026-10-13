from __future__ import annotations

import unittest

import replacement_loss_v12_replication as audit


class ReplacementLossV12ReplicationTests(unittest.TestCase):
    def test_candidate_content_excludes_runtime_locations_but_binds_runtime_versions(self) -> None:
        freeze = {
            "protocol_id": "v11",
            "state_slice": "sandbox",
            "protocol_slice": "v11",
            "packet_sha256": "packet",
            "visible_fixture_sha256": "visible",
            "visible_selection_only": True,
            "selected_layers": [0, 7],
            "visible_layer_variance": [0.2, 0.4],
            "visible_input_rms": {"0": 1.0, "7": 1.2},
            "adaptation_stress_coordinate": {"0:0": 1},
            "layer_dimensions": {"0": [4, 4, True]},
            "adaptation_fixture_sha256": ["adapt"],
            "adaptation_artifacts": [{"sha256": "capture"}],
            "candidate_count": 1,
            "candidates": [{"file_sha256": "candidate", "state_sha256": "state"}],
            "causal_fit_updates": 100,
            "source_revision": "revision",
            "runtime": {
                "python": "3.12.13",
                "python_executable": "/different/one/python",
                "platform": "macOS-arm64",
                "host_id_sha256": "host-one",
                "packages": {"torch": "2.6.0"},
                "dependency_lock_sha256": "lock",
                "torch_num_threads": 1,
                "deterministic_algorithms": True,
            },
        }
        selection = {
            "model_id": "org/model",
            "revision": "commit",
            "resolved_revision": "commit",
            "source_artifact_tree_sha256": "tree",
            "config_sha256": "config",
            "weights_sha256": "weights",
            "weight_files": [{"path": "weights.bin", "sha256": "weights", "bytes": 10}],
            "local_path": "/different/model/path",
        }
        first = audit.stable_candidate_content(freeze, selection)
        freeze["runtime"]["python_executable"] = "/different/two/python"
        freeze["runtime"]["host_id_sha256"] = "host-two"
        selection["local_path"] = "/another/model/path"
        self.assertEqual(first, audit.stable_candidate_content(freeze, selection))
        freeze["runtime"]["packages"] = {"torch": "2.6.1"}
        self.assertNotEqual(first, audit.stable_candidate_content(freeze, selection))

    def test_candidate_content_binds_candidate_state_digest(self) -> None:
        freeze = {
            "protocol_id": "v11", "state_slice": "sandbox", "protocol_slice": "v11",
            "packet_sha256": "packet", "visible_fixture_sha256": "visible",
            "visible_selection_only": True, "selected_layers": [0, 7],
            "visible_layer_variance": [], "visible_input_rms": {},
            "adaptation_stress_coordinate": {}, "layer_dimensions": {},
            "adaptation_fixture_sha256": [], "adaptation_artifacts": [],
            "candidate_count": 1, "candidates": [{"state_sha256": "before"}],
            "causal_fit_updates": 100, "source_revision": "revision",
            "runtime": {"python": "3.12.13", "platform": "macOS-arm64", "packages": {},
                        "dependency_lock_sha256": "lock", "torch_num_threads": 1,
                        "deterministic_algorithms": True},
        }
        selection = {"model_id": "org/model", "revision": "commit", "resolved_revision": "commit",
                     "source_artifact_tree_sha256": "tree", "config_sha256": "config",
                     "weights_sha256": "weights", "weight_files": []}
        first = audit.stable_candidate_content(freeze, selection)
        freeze["candidates"][0]["state_sha256"] = "after"
        self.assertNotEqual(first, audit.stable_candidate_content(freeze, selection))

    def test_normalized_challenge_and_prediction_identities_ignore_v11_parent_paths(self) -> None:
        challenge_a = {"protocol_id": "v11", "state_slice": "sandbox", "protocol_slice": "v11",
                       "packet_sha256": "packet", "candidate_set_sha256": "path-a",
                       "challenge_generation_started_after_candidate_freeze": True,
                       "entropy": [{"entropy_sha256": "entropy"}], "challenge_rows": [{"prompt_sha256": "prompt"}],
                       "created_at_utc": "time-a", "challenge_manifest_sha256": "old-a"}
        challenge_b = {**challenge_a, "candidate_set_sha256": "path-b", "created_at_utc": "time-b", "challenge_manifest_sha256": "old-b"}
        normalized_a = audit.stable_challenge_content(challenge_a, "candidate-content")
        normalized_b = audit.stable_challenge_content(challenge_b, "candidate-content")
        self.assertEqual(normalized_a, normalized_b)
        prediction_a = {"protocol_id": "v11", "state_slice": "sandbox", "protocol_slice": "v11",
                        "packet_sha256": "packet", "candidate_set_sha256": "path-a",
                        "challenge_manifest_sha256": "old-a", "predictions": [{"sha256": "predictions"}],
                        "original_model_outcomes_opened": False, "created_at_utc": "time-a",
                        "prediction_manifest_sha256": "pred-a"}
        prediction_b = {**prediction_a, "candidate_set_sha256": "path-b",
                        "challenge_manifest_sha256": "old-b", "created_at_utc": "time-b",
                        "prediction_manifest_sha256": "pred-b"}
        self.assertEqual(
            audit.stable_prediction_content(prediction_a, "candidate-content", normalized_a),
            audit.stable_prediction_content(prediction_b, "candidate-content", normalized_b),
        )


if __name__ == "__main__":
    unittest.main()
