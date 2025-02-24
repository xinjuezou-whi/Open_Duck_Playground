"""Runs training and evaluation loop for the Z-Bot."""

import argparse

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.open_duck_mini_v2 import (
    joystick,
    open_duck_mini_v2_constants as constants,
)
from ml_collections import config_dict


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=450,
        action_repeat=1,
        action_scale=0.5,
        dof_vel_scale=0.05,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            action_min_delay=0,  # env steps
            action_max_delay=3,  # env steps
            imu_min_delay=0,  # env steps
            imu_max_delay=3,  # env steps
            scales=config_dict.create(
                hip_pos=0.03,  # rad #for each hip joint
                kfe_pos=0.05,  # kfe=Knee Pitch
                ffe_pos=0.08,  # ffe=Ankle pitch
                # faa_pos=0.03, #ffa=Ankle Roll #FIXME!
                joint_vel=2.5,  # rad/s # Was 1.5
                gravity=0.1,
                linvel=0.1,
                gyro=0.2,  # angvel. # was 0.2
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Tracking related rewards.
                tracking_lin_vel=2.5,
                tracking_ang_vel=1.5,
                # Base related rewards.
                lin_vel_z=0.0,
                ang_vel_xy=0.0,
                orientation=-0.5,
                base_height=0.0,
                base_y_swing=0.0,
                # Energy related rewards.
                torques=-1.0e-3,
                action_rate=-0.75,  # was -1.5
                energy=0.0,
                # Feet related rewards.
                feet_clearance=0.0,
                feet_air_time=0.0,
                feet_slip=0.0,
                feet_height=0.0,
                # feet_phase=0.0,
                # Other rewards.
                stand_still=-0.2,  # was -1.0 TODO try to relax this a bit ?
                alive=20.0,
                termination=0.0,
                imitation=1.0,
                # Pose related rewards.
                joint_deviation_knee=0.0,  # -0.1
                joint_deviation_hip=0.0,  # -0.25
                dof_pos_limits=0.0,  # -1.0
                pose=0.0,  # -1.0
            ),
            tracking_sigma=0.01,  # was working at 0.01
            max_foot_height=0.03,  # 0.1,
            base_height_target=0.15,  # 0.5,
        ),
        push_config=config_dict.create(
            enable=True,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        lin_vel_x=[-0.1, 0.15],
        lin_vel_y=[-0.2, 0.2],
        ang_vel_yaw=[-0.5, 0.5],  # [-1.0, 1.0]
    )


class OpenDuckMiniV2Runner(BaseRunner):

    def __init__(self, args):
        super().__init__(args)
        self.env_config = joystick.default_config()
        self.env = joystick.Joystick(task=args.task)
        self.randomizer = randomize.domain_randomize

    # TODO
    @classmethod
    def get_root_body(cls) -> str:
        return constants.ROOT_BODY


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Duck Mini Runner Script")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to save the checkpoints",
    )
    parser.add_argument("--task", type=str, default="flat_terrain", help="Task to run")
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with minimal parameters"
    )
    # parser.add_argument("--save-model", action="store_true", help="Save model after training")
    # parser.add_argument("--load-model", action="store_true", help="Load existing model instead of training")
    # parser.add_argument("--seed", type=int, default=1, help="Random seed")
    # parser.add_argument("--num-episodes", type=int, default=2, help="Number of evaluation episodes")
    # parser.add_argument("--episode-length", type=int, default=3000, help="Length of each episode")
    # parser.add_argument("--x-vel", type=float, default=1.0, help="X velocity command")
    # parser.add_argument("--y-vel", type=float, default=0.0, help="Y velocity command")
    # parser.add_argument("--yaw-vel", type=float, default=0.0, help="Yaw velocity command")
    args = parser.parse_args()

    runner = OpenDuckMiniV2Runner(args)

    runner.train()


if __name__ == "__main__":
    main()
