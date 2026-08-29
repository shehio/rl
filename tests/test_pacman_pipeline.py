"""Regression tests for the Ms. Pacman policy-gradient pipeline.

Each test here pins a bug that made MsPacman training silently meaningless
while every existing test stayed green. Unlike the other test modules these
imports are deliberately NOT wrapped in a skip-on-ImportError guard: CI
installs the full requirements.txt, so a failure to import is a real failure.
"""

import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "atari", "algorithms"))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "atari", "algorithms", "pg", "scripts"
    ),
)

from pg.hyperparameters import HyperParameters
from pg.pacman.multi_action_agent import MultiActionAgent
from pg.pacman.cnn_torch_multiaction import CNNMultiAction
from pg.pacman.preprocess_pacman import preprocess_pacman_frame_color_aware
from pg_trainer import PolicyGradientTrainer


def _frame(with_blob: bool) -> np.ndarray:
    """A 210x160x3 RGB frame, optionally with a bright yellow square in it."""
    frame = np.zeros((210, 160, 3), dtype=np.uint8)
    if with_blob:
        frame[50:150, 50:110] = (255, 255, 0)
    return frame


class TestPreviousFrameIsRecorded:
    """`previous_frame` was initialised to None and never assigned, so the
    pacman preprocessor took its `previous_frame is None` branch on every
    single step and returned np.zeros_like(...). The CNN was trained on an
    all-zero 80x80x7 tensor for the entire run: the policy could not depend on
    the game state at all."""

    def test_second_step_is_not_all_zero(self):
        trainer = PolicyGradientTrainer("pacman", {})
        game = SimpleNamespace(observation=_frame(with_blob=False))

        first = trainer.preprocess_state(game)
        assert not np.any(first), "first step has no predecessor, so zeros is correct"

        game.observation = _frame(with_blob=True)
        second = trainer.preprocess_state(game)
        assert np.any(second), (
            "the second step must see a real frame difference; all-zero here "
            "means previous_frame was never recorded"
        )

    def test_episode_boundary_clears_the_frame(self):
        trainer = PolicyGradientTrainer("pacman", {})
        game = SimpleNamespace(observation=_frame(with_blob=False))
        trainer.preprocess_state(game)
        assert trainer.previous_frame is not None

        trainer.post_episode(
            SimpleNamespace(episode_number=1, reward_sum=0.0, running_reward=0.0),
            agent=None,
        )
        assert trainer.previous_frame is None, "must not difference across episodes"


class TestDiscountedReturn:
    """pg/agent.py zeroes the running return on any non-zero reward, which its
    own comment marks "pong specific!" — in Pong a point ends the rally. That
    line was copied into the MsPacman agent, where every pellet scores, so the
    return collapsed to the immediate reward and no action was ever credited
    for anything beyond the current step."""

    def test_return_accumulates_across_scoring_steps(self):
        agent = MultiActionAgent(
            policy_network=None,
            hyperparams=HyperParameters(
                learning_rate=1e-4,
                decay_rate=0.99,
                gamma=0.99,
                batch_size=10,
                save_interval=100,
            ),
        )
        # Three consecutive pellets. True returns are 29.70 > 19.90 > 10.00,
        # so after normalisation the ordering must survive. With the Pong reset
        # every entry is 10.0, std is 0, normalisation is skipped, and all three
        # come back equal.
        rewards = np.array([10.0, 10.0, 10.0])
        out = agent._MultiActionAgent__discount_and_normalize_rewards(rewards, 0.99)

        assert (
            out[0] > out[1] > out[2]
        ), f"earlier actions must carry more discounted return, got {out}"


class TestFeatureLayout:
    """preprocess_pacman_frame_color_aware stacks its 7 feature masks with
    axis=-1 and ravels, so the flat vector is channels-LAST. The CNN reshaped
    it straight to (N, 7, 80, 80) — channels-first — which reinterprets
    interleaved pixels as whole channel planes and scrambles the conv input."""

    def test_flat_features_are_channels_last(self):
        frame = _frame(with_blob=True)
        flat = preprocess_pacman_frame_color_aware(frame)
        assert flat.size == 80 * 80 * 7

        resized = cv2.resize(frame, (80, 80), interpolation=cv2.INTER_AREA)
        expected_gray = (
            cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        )

        # Channel 6 is the grayscale plane appended last by the preprocessor.
        channels_last = flat.reshape(80, 80, 7).transpose(2, 0, 1)
        assert np.allclose(
            channels_last[6], expected_gray
        ), "the ravelled vector is channels-last"

        channels_first = flat.reshape(7, 80, 80)
        assert not np.allclose(channels_first[6], expected_gray), (
            "the old channels-first reshape must NOT recover the plane — if it "
            "does, this test no longer proves anything"
        )

    def test_cnn_unpacks_the_flat_vector_into_the_right_planes(self):
        """The reshape inside forward_pass must recover each feature mask as a
        whole channel plane. Channel c is filled with the constant c, so a
        correct unpack gives seven constant planes and the old channels-first
        reshape gives seven identically-striped ones."""
        net = CNNMultiAction(7, 16, 9, "unused", "ALE/MsPacman-v5")
        planes = np.stack(
            [np.full((80, 80), float(c), dtype=np.float32) for c in range(7)], axis=-1
        )
        flat = planes.ravel()  # channels-last, exactly as the preprocessor emits

        captured = {}
        original_forward = net.forward

        def spy(x):
            captured["x"] = x.detach().cpu().numpy()
            return original_forward(x)

        net.forward = spy
        net.forward_pass(flat)

        got = captured["x"]
        assert got.shape == (1, 7, 80, 80)
        for c in range(7):
            assert np.allclose(got[0, c], float(c)), (
                f"channel {c} is not the constant plane {c} — the flat vector "
                f"was unpacked in the wrong axis order"
            )
