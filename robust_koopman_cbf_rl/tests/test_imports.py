def test_top_level_imports():
    import robust_koopman_cbf_rl
    import robust_koopman_cbf_rl.envs
    import robust_koopman_cbf_rl.cbf
    import robust_koopman_cbf_rl.koopman
    import robust_koopman_cbf_rl.agents
    import robust_koopman_cbf_rl.baselines
    import robust_koopman_cbf_rl.train
    import robust_koopman_cbf_rl.utils
    import robust_koopman_cbf_rl.plots
    assert robust_koopman_cbf_rl.__version__ == "0.1.0"
