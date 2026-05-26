import numpy as np


def test_cartpole_position_barrier_values_and_coeffs():
    from robust_koopman_cbf_rl.cbf.cartpole_barriers import CartPolePositionBarrier
    bar = CartPolePositionBarrier(x_max=2.2)
    raw = np.array([1.0, 0.0, 0.05, 0.0])  # [x, x_dot, theta, theta_dot]
    assert np.isclose(bar.value(raw, info={}), 2.2 - 1.0)
    label = bar.label(raw, info={})
    assert label == "safe"
    # Lifted CBF coeffs target z = [y, rbf(y)] with dim_y=4, n_rbf=3 -> z_dim=7
    c, d = bar.lifted_barrier_coeffs(z_dim=7, dim_y=4)
    assert c.shape == (7,)
    assert c[0] == -1.0  # -x part
    assert np.allclose(c[1:4], 0.0)  # x_dot, theta, theta_dot do not appear
    assert np.allclose(c[4:], 0.0)   # rbf part zeroed
    assert np.isclose(d, 2.2)


def test_cartpole_angle_barrier():
    from robust_koopman_cbf_rl.cbf.cartpole_barriers import CartPoleAngleBarrier
    # Default theta_max=0.14 rad is conservative: SCG constraint bound is 0.16 rad
    bar = CartPoleAngleBarrier(theta_max=0.14)
    raw = np.array([0.0, 0.0, 0.10, 0.0])
    v = bar.value(raw, info={})
    assert np.isclose(v, 0.14 - 0.10)
    c, d = bar.lifted_barrier_coeffs(z_dim=7, dim_y=4)
    assert c.shape == (7,)


def test_quadrotor_altitude_barrier():
    from robust_koopman_cbf_rl.cbf.quadrotor_barriers import Quadrotor2DAltitudeBarrier
    # z_max=1.8 is conservative: obs_space ceiling is z_threshold=2.0, giving 0.2m margin
    # z_min=0.2 is conservative: ground plane is GROUND_PLANE_Z=-0.05, giving 0.25m clearance
    bar = Quadrotor2DAltitudeBarrier(z_min=0.2, z_max=1.8)
    raw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # [x, xdot, z, zdot, theta, thetadot]
    assert bar.value(raw, info={}) > 0
    c, d = bar.lifted_barrier_coeffs(z_dim=10, dim_y=6, state_index=2)
    assert c.shape == (10,)


def test_velocity_norm_barrier_quadratic_lift():
    from robust_koopman_cbf_rl.cbf.velocity_barriers import VelocityNormBarrier
    bar = VelocityNormBarrier(v_max=2.0, vel_indices=[0, 1])
    raw = np.array([1.5, 0.5, 0.0])
    v = bar.value(raw, info={"velocity": 1.5})
    # h(y) = v_max^2 - sum(v_i^2) = 4.0 - (2.25 + 0.25) = 1.5
    assert np.isclose(v, 4.0 - (1.5**2 + 0.5**2))
    c, d, lift_extra = bar.lifted_barrier_coeffs(z_dim=8, dim_y=3, n_rbf=3)
    assert c.shape == (8,)
    assert lift_extra is not None
