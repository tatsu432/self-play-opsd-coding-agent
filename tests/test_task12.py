from pathlib import Path

from selfplay_opsd.config import load_config


def test_corrected_study_preserves_scientific_settings_and_binds_new_data():
    old = load_config(Path("configs/task10-online.yaml"))
    source = load_config(Path("configs/task12-source.yaml"))
    corrected = load_config(Path("configs/task12-corrected.yaml"))
    old_science = old.scientific_settings(common=True)
    new_science = source.scientific_settings(common=True)
    old_science.pop("study_id")
    new_science.pop("study_id")
    assert old_science == new_science
    assert source.study_id == corrected.study_id != old.study_id
    assert source.data_root == old.data_root == Path("data/protocol-v2")
    assert corrected.data_root == Path("data/protocol-v3")
    assert source.checkpoint_every == corrected.checkpoint_every == old.checkpoint_every
