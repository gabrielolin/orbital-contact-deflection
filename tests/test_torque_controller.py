import numpy as np

from contact_deflection.control import PDControllerConfig, TorqueController


def controller() -> TorqueController:
    return TorqueController(PDControllerConfig([2.0, 3.0], [0.5, 0.5], [4.0, 5.0]))


def test_zero_error_gives_zero_torque() -> None:
    np.testing.assert_array_equal(
        controller().compute([1, 2], [0, 0], [1, 2], [0, 0]), [0, 0]
    )


def test_feedback_sign_and_clipping() -> None:
    torque = controller().compute([0, 0], [0, 0], [10, -10], [0, 0])
    np.testing.assert_array_equal(torque, [4, -5])
