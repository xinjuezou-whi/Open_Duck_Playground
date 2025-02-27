import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from etils import epath
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
# from playground.open_duck_mini_v2 import constants
from playground.open_duck_mini_v2 import base




class MjInfer():
    def __init__(self, model_path:str, reference_data:str, onnx_model_path:str):
        self.model = mujoco.MjModel.from_xml_string(
            epath.Path(model_path).read_text(), assets=base.get_assets()
        )

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.5

        self.PRM = PolyReferenceMotion(reference_data)


        NUM_DOFS = self.model.nu

        self.joint_names = [
            self.model.jnt(k).name for k in range(1, self.model.njnt)
        ]  # all the joint (including the backlash joints)
        self.actuator_names = [
            self.model.actuator(k).name for k in range(0, self.model.nu)
        ]  # will be useful to get only the actuators we care about

        self.backlash_joint_names = [
            j for j in self.joint_names if j not in self.actuator_names
        ]  # only the dummy backlash joint
        self.actual_joint_ids = [
            self.get_joint_id_from_name(n) for n in self.actuator_names
        ]
        self.backlash_joint_ids=[
            self.get_joint_id_from_name(n) for n in self.backlash_joint_names
        ]
        self.all_joint_ids = [self.get_joint_id_from_name(n) for n in self.joint_names]

        self._floating_base_add = self.model.jnt_dofadr[
            np.where(self.model.jnt_type == 0)
        ][
            0
        ]  # Assuming there is only one floating base! the jnt_type==0 is a floating joint. 3 is a hinge

        self.all_joint_no_backlash_ids=[idx for idx in self.all_joint_ids if idx not in self.backlash_joint_ids]+list(range(self._floating_base_add,self._floating_base_add+7))


        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)

        self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.COMMANDS_RANGE_X = [-0.1, 0.15]
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        self.COMMANDS_RANGE_THETA = [-0.5, 0.5] # [-1.0, 1.0]

        self.last_action = np.zeros(NUM_DOFS)
        self.last_last_action = np.zeros(NUM_DOFS)
        self.last_last_last_action = np.zeros(NUM_DOFS)
        self.commands = [0.0, 0.0, 0.0]
        self.decimation = 10

        self.init_pos = np.array(
            self.get_all_joints_qpos(self.model.keyframe("home"))
        )  # pose of all the joints (no floating base)
        self.default_actuator = self.model.keyframe(
            "home"
        ).ctrl  # ctrl of all the actual joints (no floating base and no backlash)


        #orientation
        # data.qpos[3 : 3 + 4] = [1, 0, 0.0, 0]
        # data.qpos[7:] = init_pos
        self.data.qpos[:]=self.model.keyframe("home").qpos
        self.data.ctrl[:] = self.default_actuator

        self.gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        self.gyro_dimensions = 3
        self.linvel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "local_linvel")
        self.linvel_dimensions = 3


        self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")

        self.imitation_i = 0
        self.saved_obs = []

    def get_actual_joints_idx(self) -> np.ndarray:
        """Return the all the idx of actual joints"""
        addr = np.array(
            [self.model.jnt_qposadr[idx] for idx in self.actual_joint_ids]
        )
        return addr


    def get_actuator_id_from_name(self, name: str) -> int:
        """Return the id of a specified actuator"""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def get_joint_id_from_name(self, name: str) -> int:
        """Return the id of a specified joint"""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)


    def get_all_joints_idx(self) -> np.ndarray:
        """Return the all the idx of all joints"""
        addr = np.array([self.model.jnt_qposadr[idx] for idx in self.all_joint_ids])
        return addr

    def get_all_joints_qpos(self, data: mujoco.MjData) -> np.ndarray:
        """Return the all the qpos of all joints"""
        return data.qpos[self.get_all_joints_idx()]

    def get_actual_joints_qpos(self, data: mujoco.MjData) -> np.ndarray:
        """Return the all the qpos of actual joints"""
        return data.qpos[self.get_actual_joints_idx()]

    def set_actual_joints_qpos(self, qpos: np.ndarray, data: mujoco.MjData) -> np.ndarray:
        """Set the qpos only for the actual joints (omit the backlash joint)"""
        return data.qpos.at[self.get_actual_joints_idx()].set(qpos)

    def get_actual_joints_qpvel(self, data: mujoco.MjData) -> np.ndarray:
        """Return the all the qvel of actual joints"""
        return data.qvel[self.get_actual_joints_idx()]


    def get_sensor(self, data, name, dimensions):
        i = self.model.sensor_name2id(name)
        return data.sensordata[i : i + dimensions]


    def get_gyro(self, data):
        return data.sensordata[self.gyro_id : self.gyro_id + self.gyro_dimensions]


    def get_linvel(self, data):
        return data.sensordata[self.linvel_id : self.linvel_id + self.linvel_dimensions]


    def get_gravity(self, data):
        return data.site_xmat[self.imu_site_id].reshape((3, 3)).T @ np.array([0, 0, -1])


    def check_contact(self, data, body1_name, body2_name):
        body1_id = data.body(body1_name).id
        body2_id = data.body(body2_name).id

        for i in range(data.ncon):
            try:
                contact = data.contact[i]
            except Exception as e:
                return False

            if (
                self.model.geom_bodyid[contact.geom1] == body1_id
                and self.model.geom_bodyid[contact.geom2] == body2_id
            ) or (
                self.model.geom_bodyid[contact.geom1] == body2_id
                and self.model.geom_bodyid[contact.geom2] == body1_id
            ):
                return True

        return False


    def get_feet_contacts(self, data):
        left_contact = self.check_contact(data, "foot_assembly", "floor")
        right_contact = self.check_contact(data,  "foot_assembly_2", "floor")
        return left_contact, right_contact

    def get_obs(self,
        data, last_action, command  # , qvel_history, qpos_error_history, gravity_history
    ):
        gravity = self.get_gravity(data)
        joint_angles = self.get_actual_joints_qpos(data)
        joint_vel = self.get_actual_joints_qpvel(data)

        contacts = self.get_feet_contacts(data)

        ref = self.PRM.get_reference_motion(*command, self.imitation_i)

        obs = np.concatenate(
            [
                gravity,
                command,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                last_action,
                self.last_last_action,
                self.last_last_last_action,
                contacts,
                ref,
            ]
        )

        return obs

    def key_callback(self,keycode):
        print(f"key: {keycode}")
        lin_vel_x = 0
        lin_vel_y = 0
        ang_vel = 0
        if keycode==265: #arrow up
            lin_vel_x = self.COMMANDS_RANGE_X[1]
        if keycode==264: #arrow down
            lin_vel_x = self.COMMANDS_RANGE_X[0]
        if keycode==263: #arrow left
            lin_vel_y = self.COMMANDS_RANGE_Y[1]
        if keycode==262: #arrow right
            lin_vel_y = self.COMMANDS_RANGE_Y[0]
        if keycode==81: #a
            ang_vel = self.COMMANDS_RANGE_THETA[1]
        if keycode==69: #e
            ang_vel = self.COMMANDS_RANGE_THETA[0]

        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y
        self.commands[2] = ang_vel



    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False, key_callback=self.key_callback
            ) as viewer:
                counter = 0
                while True:

                    step_start = time.time()

                    mujoco.mj_step(self.model, self.data)

                    counter += 1

                    if counter % self.decimation == 0:
                        self.imitation_i += 1
                        self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period
                        obs = self.get_obs(
                            self.data,
                            self.last_action,
                            self.commands,
                        )
                        self.saved_obs.append(obs)
                        action = self.policy.infer(obs)

                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()

                        action = self.default_actuator + action * self.action_scale
                        self.data.ctrl = action.copy()

                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    # parser.add_argument("-k", action="store_true", default=False)
    parser.add_argument("--reference_data", type=str, default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl")
    parser.add_argument("--model_path", type=str, default="playground/open_duck_mini_v2/xmls/scene_mjx_flat_terrain.xml")

    args = parser.parse_args()

    mjinfer=MjInfer(args.model_path, args.reference_data, args.onnx_model_path)
    mjinfer.run()
