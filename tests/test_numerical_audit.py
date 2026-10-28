import torch

from selfplay_opsd.config import load_config
from selfplay_opsd.core import EpisodeRecord, ProposalRecord, Status, TokenTurn
from selfplay_opsd.numerical_audit import assess, error_statistics
from selfplay_opsd.policy import HFPolicy
from selfplay_opsd.storage import rng_state


def test_error_statistics_counts_clip_and_signed_errors():
    report = error_statistics([0, -0.3, 0.1], 0.2)
    assert report["tokens"] == 3
    assert report["outside_clip"] == 1
    assert report["mean_signed"] < 0


def test_counterfactual_measures_all_objectives_without_updates(tmp_path):
    cfg = load_config(study_id="audit", hardware_profile="cpu_debug", output_root=tmp_path)
    policy = HFPolicy(cfg)
    prompt, targets = [1, 7, 8], [9, 10]
    turn = TokenTurn(
        prompt_ids=prompt,
        target_ids=targets,
        old_logps=[v + 0.3 for v in policy.score(prompt, targets).tolist()],
        teacher_logps=[-2.0, -3.0],
        history=[],
        raw_text="PRIVATE",
        action="finish",
        parse_status="valid",
    )
    episodes = []
    for index in range(4):
        episodes.append(
            EpisodeRecord(
                episode_id=str(index),
                task_id="task",
                seed=index,
                policy_version=policy.version("student"),
                turns=[turn],
                reward=float(index == 1),
                public_pass=False,
                hidden_pass=False,
                termination="finish",
                status=Status.STUDENT_FAILURE,
                final_source="PRIVATE",
                final_source_hash="hash",
                test_calls=0,
                seconds=0,
            ).model_dump(mode="json")
        )
    with policy.active("proposer"):
        proposal_turn = turn.model_copy(
            update={"old_logps": policy.score(prompt, targets).tolist()}
        )
        proposals = [
            ProposalRecord(
                proposal_id=str(i),
                parent_id=str(i // 4),
                slot=i % 4,
                seed=i,
                policy_version=policy.version("proposer"),
                turn=proposal_turn,
                valid=i == 0,
                task_id="task" if i == 0 else None,
                reason="PRIVATE",
            ).model_dump(mode="json")
            for i in range(16)
        ]
    proposals = [row | {"reward": 0.5 if row["valid"] else -1.0} for row in proposals]
    before = {name: value.clone() for name, value in policy.model.state_dict().items()}
    rng = rng_state()
    report = assess(
        policy, cfg, episodes, proposals, [{"edit": {"prompt": prompt, "targets": targets}}]
    )
    assert set(report["methods"]) == set("ABCD")
    assert report["mixed_groups"] == 1
    assert report["opsd_gate_abs_delta_max"] > 0
    assert report["methods"]["D"]["student_gradients"]["combined"]["relative_l2"] > 0
    assert "PRIVATE" not in str(report)
    assert torch.equal(rng["torch"], rng_state()["torch"])
    assert all(
        torch.equal(before[name], value) for name, value in policy.model.state_dict().items()
    )
    assert not any(policy.optimizers[role].state for role in ("student", "proposer"))
    policy.close()
