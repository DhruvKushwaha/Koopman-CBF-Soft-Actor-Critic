import pytest
from pathlib import Path
pytest.importorskip("safe_control_gym")

_CONFIGS = Path(__file__).parent.parent / "configs"

def test_cartpole_1000_step_training_does_not_crash(tmp_path):
    from robust_koopman_cbf_rl.train.train_sac_kcbf import main as train_sac
    from robust_koopman_cbf_rl.train.train_koopman import main as train_kp
    env_cfg = str(_CONFIGS / "env_cartpole_stab.yaml")
    k_cfg = str(_CONFIGS / "koopman.yaml")
    model_path = tmp_path / "k.npz"
    train_kp(env_cfg, k_cfg, str(model_path))
    train_sac(env_cfg, str(_CONFIGS / "sac_kcbf.yaml"),
              str(model_path), str(tmp_path / "logs"), total_steps=1000)
