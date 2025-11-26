# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import logging
import math
import warnings
from dataclasses import dataclass

# Suppress do_mpc warnings BEFORE importing - warnings trigger at import time
# We don't use the ONNX sysid or opcua features, and the SyntaxWarning is a bug in do_mpc
warnings.filterwarnings("ignore", message="The ONNX feature is not available")
warnings.filterwarnings("ignore", message="The opcua feature is not available")
warnings.filterwarnings("ignore", message='"is" with a literal', category=SyntaxWarning)

import casadi  # noqa: E402
import do_mpc  # noqa: E402
import numpy as np  # noqa: E402
from alpasim_grpc.v0 import common_pb2, controller_pb2  # noqa: E402
from alpasim_utils import trajectory  # noqa: E402


class VehicleModel:
    """
    The VehicleModel class uses a planar dynamic bicycle model to simulate the vehicle dynamics
    and relative pose over an advance call. The state is:
    [
               inertial x:     x position of the rig origin relative to the inertial frame,
               inertial y:     y position of the rig origin relative to the inertial frame,
               inertial yaw:   yaw angle of the rig origin relative to the inertial frame,
               body x-vel:     x component of the cg velocity (relative to inertial frame),
                               resolved in the rig frame,
               body y-vel:     y component of the cg velocity (relative to inertial frame),
                               resolved in the rig frame,
               yaw rate:       yaw rate of the rig in the rig frame,
               steering_angle: steering angle of the front wheel,
               acceleration:   time derivative of the longitudinal velocity
    ]
    Note: this state has the velocity measured at the body center of gravity (cg), whereas
    the interface (trajectories, grpc-provided states) use the rig frame velocity.
    Note: lateral and longitudinal dynamics are decoupled
    """

    @dataclass
    class Parameters:
        """
        Vehicle model parameters. Default values from Ford Fusion
        """

        mass: float = 2014.4  # Mass [kg]
        inertia: float = 3414.2  # Moment of inertia around z-axis [kg*m^2]
        l_rig_to_cg: float = 1.59  # Distance from rear wheel to CoG [m]
        wheelbase: float = 2.85  # Wheelbase [m]
        front_cornering_stiffness: float = 93534.5  # Front cornering stiffness [N/rad]
        rear_cornering_stiffness: float = 176162.1  # Rear cornering stiffness [N/rad]
        steering_time_constant: float = (
            0.1  # Time constant for steering angle response [s]
        )
        acceleration_time_constant: float = (
            0.1  # Time constant for longitudinal accel response [s]
        )
        kinematic_threshold_speed: float = (
            5.0  # Speed threshold below which a kinematic model is used [m/s]
        )

    def __init__(
        self,
        initial_velocity: np.ndarray,
        initial_yaw_rate: float,
    ):
        """
        Initializes the vehicle model with the initial velocity and yaw rate.
        Args:
            initial_velocity: Initial velocity in the rig frame [vx, vy] [m/s]
            initial_yaw_rate: Initial yaw rate in the rig frame [rad/s]
        """
        self._parameters = self.Parameters()
        if initial_velocity[0] > 0.25:
            # approximate kinematic steering angle
            initial_steering_angle = math.atan(
                initial_yaw_rate / initial_velocity[0] * self._parameters.wheelbase
            )
        else:
            initial_steering_angle = 0.0
        self._state = np.array(
            [
                0.0,
                0.0,
                0.0,
                initial_velocity[0],
                initial_velocity[1],
                initial_yaw_rate,
                initial_steering_angle,
                0.0,
            ]
        )
        # Store the last computed accelerations (derivatives of velocity states)
        # [d_v_cg_x (longitudinal accel), d_v_cg_y (lateral accel), d_yaw_rate (angular accel)]
        self._accelerations = np.array([0.0, 0.0, 0.0])

    @property
    def parameters(self) -> "VehicleModel.Parameters":
        """Getter for the vehicle parameters."""
        return self._parameters

    @property
    def state(self) -> np.ndarray:
        """Getter for the vehicle state."""
        return self._state

    @property
    def front_steering_angle(self) -> float:
        """Getter for the front steering angle."""
        return self._state[6]

    @property
    def accelerations(self) -> np.ndarray:
        """
        Getter for the last computed accelerations in the CG frame.
        Returns:
            Array of [d_v_cg_x, d_v_cg_y, d_yaw_rate] where:
            - d_v_cg_x: longitudinal acceleration at CG [m/s²]
            - d_v_cg_y: lateral acceleration at CG [m/s²]
            - d_yaw_rate: angular acceleration (yaw) [rad/s²]
        """
        return self._accelerations

    def reset_origin(self) -> None:
        """Reset the state to the origin (x, y, yaw) = 0."""
        self._state[:3] = 0.0

    def set_velocity(self, v_cg_x: float, v_cg_y: float) -> None:
        """
        Set the velocity of the CG in the rig/body frame.
        Args:
            v_cg_x: Longitudinal velocity of the CG in the rig frame [m/s]
            v_cg_y: Lateral velocity of the CG in the rig frame [m/s]
        """
        self._state[3] = v_cg_x
        self._state[4] = v_cg_y

    def advance(self, u: np.ndarray, dt: float) -> trajectory.QVec:
        """
        Advances the vehicle model by dt seconds using a 2nd order Runge-Kutta method.
        Args:
            u: Control input [steering angle, longitudinal acceleration]
            dt: Time step in seconds
        Returns:
            The current pose. Note that, if reset_origin is called prior to calling this
            method, the returned value will be the relative position in the rig frame after
            dt seconds (pose_rig_at_t0_to_rig_at_t1)
        """
        DT_STEP_MAX = 0.01

        logging.debug("state: %s, u: %s, dt: %s", self._state, u, dt)
        total_time = 0.0
        while total_time < dt:
            if dt - total_time > DT_STEP_MAX:
                step_dt = DT_STEP_MAX
            else:
                step_dt = dt - total_time

            total_time += step_dt

            # 2nd order Runge-Kutta method for numerical integration
            k1 = step_dt * self._derivs(self._state, u)
            k2 = step_dt * self._derivs(self._state + k1 / 2.0, u)
            self._state += k2
            self._state[3] = max(0.0, self._state[3])  # Ensure non-negative velocity

        # Store the final accelerations (derivatives at the end state)
        # derivs returns [dx, dy, dyaw, d_v_x, d_v_y, d_yaw_rate, d_steering, d_accel]
        final_derivs = self._derivs(self._state, u)
        # d_v_x is the longitudinal acceleration (state[7]), d_v_y and d_yaw_rate are computed
        self._accelerations = np.array(
            [final_derivs[3], final_derivs[4], final_derivs[5]]
        )

        logging.debug("state (after prop): %s", self._state)
        return trajectory.QVec(
            vec3=np.array([self._state[0], self._state[1], 0]),
            quat=np.array(
                [0, 0, math.sin(self._state[2] / 2), math.cos(self._state[2] / 2)]
            ),
        )

    def _derivs(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:

        # Grab states
        yaw_angle = state[2]
        v_x = state[3]
        v_y = state[4]
        yaw_rate = state[5]
        front_steering_angle = state[6]
        longitudinal_acceleration = state[7]

        use_kinematic_model = v_x < self._parameters.kinematic_threshold_speed
        if use_kinematic_model:
            # Differential equations

            # Note: when using the linearized kinematic model, the dynamics should pull the
            # system to the no-slip conditions:
            # yaw_rate = v_x_cg * front_steering_angle / (l_f + l_r)
            # v_cg_y = v_cg_x * front_steering_angle * l_r / (l_f + l_r)
            steady_state_v_y = (
                v_x
                * front_steering_angle
                * self._parameters.l_rig_to_cg
                / self._parameters.wheelbase
            )
            steady_state_yaw_rate = (
                v_x * front_steering_angle / self._parameters.wheelbase
            )
            GAIN = 10.0
            v_y_rig = 0.0  # No slip at rear
            d_v_y = GAIN * (steady_state_v_y - v_y)
            d_yaw_rate = GAIN * (steady_state_yaw_rate - yaw_rate)
        else:
            # Equations of motion for single-track bicycle model
            kinetic_mass = self._parameters.mass * v_x
            kinetic_inertia = self._parameters.inertia * v_x
            lf = self._parameters.wheelbase - self._parameters.l_rig_to_cg
            lf_caf = lf * self._parameters.front_cornering_stiffness
            lr_car = (
                self._parameters.l_rig_to_cg * self._parameters.rear_cornering_stiffness
            )
            lf_sq_caf = lf * lf_caf
            lr_sq_car = self._parameters.l_rig_to_cg * lr_car
            a_00 = (
                -2
                * (
                    self._parameters.front_cornering_stiffness
                    + self._parameters.rear_cornering_stiffness
                )
                / kinetic_mass
            )
            a_01 = -v_x - 2 * (lf_caf - lr_car) / kinetic_mass
            a_10 = -2 * (lf_caf - lr_car) / kinetic_inertia
            a_11 = -2 * (lf_sq_caf + lr_sq_car) / kinetic_inertia

            b_00 = (
                2 * self._parameters.front_cornering_stiffness / self._parameters.mass
            )
            b_10 = 2 * lf_caf / self._parameters.inertia

            v_y_rig = (
                v_y - state[5] * self._parameters.l_rig_to_cg
            )  # v_y in the rig frame, accounting for yaw rate

            d_v_y = a_00 * v_y + a_01 * yaw_rate + b_00 * front_steering_angle
            d_yaw_rate = a_10 * v_y + a_11 * yaw_rate + b_10 * front_steering_angle

        front_steering_angle_cmd = u[0][0]
        longitudinal_acceleration_cmd = u[1][0]

        return np.array(
            [
                v_x * math.cos(yaw_angle) - v_y_rig * math.sin(yaw_angle),
                v_x * math.sin(yaw_angle) + v_y_rig * math.cos(yaw_angle),
                yaw_rate,
                longitudinal_acceleration,
                d_v_y,
                d_yaw_rate,
                (front_steering_angle_cmd - front_steering_angle)
                / self._parameters.steering_time_constant,
                (longitudinal_acceleration_cmd - longitudinal_acceleration)
                / self._parameters.acceleration_time_constant,
            ]
        )


class System:
    DT_MPC = 0.1  # MPC time step in seconds

    @dataclass
    class MPCGains:
        """
        These default gains are provisional. They were chosen to provide a stable
        step response for the default vehicle model, but have not been tuned to
        better match the in-car performance.

        See https://www.do-mpc.com/en/latest/api/do_mpc.controller.MPC.html#set-objective
        and https://www.do-mpc.com/en/latest/api/do_mpc.controller.MPC.html#set-rterm
        for more information about the cost function.
        """

        long_position_weight: float = (
            2.0  # lagrange term / meyer term penalty on x position error
        )
        lat_position_weight: float = (
            1.0  # lagrange term / meyer term penalty on y position error
        )
        heading_weight: float = (
            1.0  # lagrange term / meyer term penalty on heading error
        )
        acceleration_weight: float = (
            0.1  # lagrange term / meyer term penalty on acceleration state
        )
        rel_front_steering_angle_weight: float = (
            5.0  # regularization term on the commanded steering angle changes
        )
        rel_acceleration_weight: float = (
            1.0  # regularization term on the commanded acceleration changes
        )
        idx_start_penalty: int = 10  # Tracking costs are ignored up to this index

    def __init__(self, log_file: str, initial_state_grpc: common_pb2.StateAtTime):
        self._timestamp_us = initial_state_grpc.timestamp_us
        self._reference_trajectory = None
        self._trajectory = trajectory.Trajectory.create_empty()
        self._trajectory.update_absolute(
            initial_state_grpc.timestamp_us,
            trajectory.QVec.from_grpc_pose(initial_state_grpc.pose),
        )

        self._vehicle_model = VehicleModel(
            initial_velocity=np.array(
                [
                    initial_state_grpc.state.linear_velocity.x,
                    initial_state_grpc.state.linear_velocity.y,
                ]
            ),
            initial_yaw_rate=initial_state_grpc.state.angular_velocity.z,
        )
        self._gains = self.MPCGains()

        velocity_cg = self._dynamic_state_to_cg_velocity(initial_state_grpc.state)
        self._x0 = np.array(
            [
                0.0,  # Initial x position [m]
                0.0,  # Initial y position [m]
                0.0,  # Initial yaw angle [rad]
                velocity_cg[0],  # Initial x-velocity of cg [m/s]
                velocity_cg[1],  # Initial y-velocity of cg [m/s]
                initial_state_grpc.state.angular_velocity.z,  # Initial yaw rate [rad/s]
                self._vehicle_model.front_steering_angle,  # Initial steering angle [rad]
                0.0,  # Initial acceleration [m/s^2, not used in the MPC]
            ]
        ).reshape(-1, 1)

        self._model = System._build_model(self._vehicle_model.parameters)
        self._setup_mpc()

        self._log_file_handle = open(log_file, "w", encoding="utf-8")
        self._log_header()
        # Initialize attributes that will be set during stepping.
        # Done to satisfy the type checking gods.
        self._first_reference_pose_rig: trajectory.QVec = trajectory.QVec(
            vec3=np.array([0, 0, 0]), quat=np.array([0, 0, 0, 1])
        )
        self.control_input: np.ndarray = np.array([[0.0], [0.0]])

    def _dynamic_state_to_cg_velocity(
        self, dynamic_state: common_pb2.DynamicState
    ) -> np.ndarray:
        return np.array(
            [
                dynamic_state.linear_velocity.x,
                dynamic_state.linear_velocity.y
                + self._vehicle_model.parameters.l_rig_to_cg
                * dynamic_state.angular_velocity.z,
            ]
        )

    @staticmethod
    def _build_model(params: VehicleModel.Parameters) -> do_mpc.model.Model:
        # The model used is a planar dynamic bicycle model

        model = do_mpc.model.Model("continuous", "SX")

        model.set_variable(var_type="_x", var_name="x_rig_inertial", shape=(1, 1))
        model.set_variable(var_type="_x", var_name="y_rig_inertial", shape=(1, 1))
        yaw_angle = model.set_variable(
            var_type="_x", var_name="yaw_angle", shape=(1, 1)
        )  # yaw angle
        v_cg_x = model.set_variable(var_type="_x", var_name="v_cg_x", shape=(1, 1))
        v_cg_y = model.set_variable(var_type="_x", var_name="v_cg_y", shape=(1, 1))
        yaw_rate = model.set_variable(
            var_type="_x", var_name="yaw_rate", shape=(1, 1)
        )  # yaw rate
        front_steering_angle = model.set_variable(
            var_type="_x", var_name="front_steering_angle", shape=(1, 1)
        )  # steering angle
        acceleration = model.set_variable(
            var_type="_x", var_name="acceleration", shape=(1, 1)
        )  # longitudinal acceleration

        # Input struct (optimization variables):
        front_steering_angle_cmd = model.set_variable(
            var_type="_u", var_name="front_steering_angle_cmd"
        )  # Steering angle
        acceleration_cmd = model.set_variable(
            var_type="_u", var_name="acceleration_cmd"
        )  # Acceleration command

        EPS_SPEED = 1.0e-2
        kinetic_mass = params.mass * casadi.fmax(v_cg_x, EPS_SPEED)
        kinetic_inertia = params.inertia * casadi.fmax(v_cg_x, EPS_SPEED)
        lf = params.wheelbase - params.l_rig_to_cg
        lf_caf = lf * params.front_cornering_stiffness
        lr_car = params.l_rig_to_cg * params.rear_cornering_stiffness
        lf_sq_caf = lf * lf_caf
        lr_sq_car = params.l_rig_to_cg * lr_car
        a_00 = (
            -2
            * (params.front_cornering_stiffness + params.rear_cornering_stiffness)
            / kinetic_mass
        )
        a_01 = -v_cg_x - 2 * (lf_caf - lr_car) / kinetic_mass
        a_10 = -2 * (lf_caf - lr_car) / kinetic_inertia
        a_11 = -2 * (lf_sq_caf + lr_sq_car) / kinetic_inertia

        b_00 = 2 * params.front_cornering_stiffness / params.mass
        b_10 = 2 * lf_caf / params.inertia

        # Differential equations
        # Note: when using the kinematic model, the dynamics should pull the system
        # to the no-slip conditions, and linearized:
        # yaw_rate = v_x_cg * front_steering_angle / (l_f + l_r)
        # v_cg_y = v_cg_x * front_steering_angle * l_r / (l_f + l_r)

        use_kinematic_model = casadi.lt(v_cg_x, params.kinematic_threshold_speed)

        model.set_rhs(
            "x_rig_inertial",
            casadi.if_else(
                use_kinematic_model,
                v_cg_x * casadi.cos(yaw_angle),
                v_cg_x * casadi.cos(yaw_angle)
                - (v_cg_y - params.l_rig_to_cg * yaw_rate) * casadi.sin(yaw_angle),
            ),
        )
        model.set_rhs(
            "y_rig_inertial",
            casadi.if_else(
                use_kinematic_model,
                v_cg_x * casadi.sin(yaw_angle),
                v_cg_x * casadi.sin(yaw_angle)
                + (v_cg_y - params.l_rig_to_cg * yaw_rate) * casadi.cos(yaw_angle),
            ),
        )
        model.set_rhs("yaw_angle", yaw_rate)
        model.set_rhs("v_cg_x", acceleration)

        yaw_rate_kinematic = v_cg_x * front_steering_angle / params.wheelbase
        v_cg_y_kinematic = (
            v_cg_x * front_steering_angle * params.l_rig_to_cg / params.wheelbase
        )

        GAIN = 10.0  # gain to bring states to kinematic model consistency

        model.set_rhs(
            "v_cg_y",
            casadi.if_else(
                use_kinematic_model,
                GAIN * (v_cg_y_kinematic - v_cg_y),
                a_00 * v_cg_y + a_01 * yaw_rate + b_00 * front_steering_angle,
            ),
        )
        model.set_rhs(
            "yaw_rate",
            casadi.if_else(
                use_kinematic_model,
                GAIN * (yaw_rate_kinematic - yaw_rate),
                a_10 * v_cg_y + a_11 * yaw_rate + b_10 * front_steering_angle,
            ),
        )
        model.set_rhs(
            "front_steering_angle",
            (front_steering_angle_cmd - front_steering_angle)
            / params.steering_time_constant,
        )
        model.set_rhs(
            "acceleration",
            (acceleration_cmd - acceleration) / params.acceleration_time_constant,
        )

        # Trajectory reference
        model.set_variable(var_type="_tvp", var_name="x_ref")
        model.set_variable(var_type="_tvp", var_name="y_ref")
        model.set_variable(var_type="_tvp", var_name="heading_ref")
        model.set_variable(var_type="_tvp", var_name="tracking_enabled")

        # Build the model
        model.setup()
        return model

    def _setup_mpc(self) -> None:
        self.mpc = do_mpc.controller.MPC(self._model)

        # Set MPC settings
        self.mpc.settings.n_horizon = 20
        self.mpc.settings.t_step = 0.1
        self.mpc.settings.n_robust = 0
        self.mpc.settings.open_loop = 0
        self.mpc.settings.state_discretization = "collocation"
        self.mpc.settings.collocation_type = "radau"
        self.mpc.settings.collocation_deg = 2
        self.mpc.settings.collocation_ni = 1
        self.mpc.settings.store_full_solution = False
        self.mpc.settings.nlpsol_opts = {"ipopt.max_iter": 30}
        self.mpc.settings.supress_ipopt_output()

        # Set up the cost function
        term = self._model.tvp["tracking_enabled"] * (
            self._gains.long_position_weight
            * (self._model.x["x_rig_inertial"] - self._model.tvp["x_ref"]) ** 2
            + self._gains.lat_position_weight
            * (self._model.x["y_rig_inertial"] - self._model.tvp["y_ref"]) ** 2
            + self._gains.acceleration_weight * (self._model.x["acceleration"] ** 2)
            + self._gains.heading_weight
            * casadi.atan2(
                casadi.sin(self._model.x["yaw_angle"] - self._model.tvp["heading_ref"]),
                casadi.cos(self._model.x["yaw_angle"] - self._model.tvp["heading_ref"]),
            )
            ** 2
        )
        self.mpc.set_objective(mterm=term, lterm=term)

        self.mpc.set_rterm(
            front_steering_angle_cmd=self._gains.rel_front_steering_angle_weight,
            acceleration_cmd=self._gains.rel_acceleration_weight,
        )

        self.mpc.set_tvp_fun(self._tvp_fun)

        # Provide state bounds
        self.mpc.bounds["lower", "_x", "x_rig_inertial"] = -500
        self.mpc.bounds["upper", "_x", "x_rig_inertial"] = 500
        self.mpc.bounds["lower", "_x", "y_rig_inertial"] = -20
        self.mpc.bounds["upper", "_x", "y_rig_inertial"] = 20
        self.mpc.bounds["lower", "_x", "yaw_angle"] = -0.78
        self.mpc.bounds["upper", "_x", "yaw_angle"] = 0.78
        self.mpc.bounds["lower", "_x", "v_cg_x"] = 0.0
        self.mpc.bounds["upper", "_x", "v_cg_x"] = 35
        self.mpc.bounds["lower", "_x", "v_cg_y"] = -10
        self.mpc.bounds["upper", "_x", "v_cg_y"] = 10
        self.mpc.bounds["lower", "_x", "yaw_rate"] = -3
        self.mpc.bounds["upper", "_x", "yaw_rate"] = 3
        self.mpc.bounds["lower", "_x", "front_steering_angle"] = -math.pi / 4
        self.mpc.bounds["upper", "_x", "front_steering_angle"] = math.pi / 4

        # Provide input bounds
        self.mpc.bounds["lower", "_u", "front_steering_angle_cmd"] = -2
        self.mpc.bounds["upper", "_u", "front_steering_angle_cmd"] = 2
        self.mpc.bounds["lower", "_u", "acceleration_cmd"] = -9.0
        self.mpc.bounds["upper", "_u", "acceleration_cmd"] = 6.0

        # Final setup
        self.mpc.setup()
        self.mpc.x0 = self._x0
        self.mpc.set_initial_guess()

    def _tvp_fun(self, _t_now: float) -> dict:
        # Notes:
        # - _t_now ignored in favor of self._timestamp_us
        # - This function is called once during MPC setup (before the reference trajectory is set)

        DT_MPC_US = int(self.DT_MPC * 1e6)
        tvp_template = self.mpc.get_tvp_template()

        # Account for temporal inconsistency between the command and current state by
        # remapping the quantities from the inertial frame "dropped" at the trajectory
        # validity timestamp into the current vehicle frame (rig frame).
        if self._reference_trajectory is not None:
            pose_local_to_rig_at_reference_start_time = (
                self._trajectory.interpolate_pose(
                    self._reference_trajectory.timestamps_us[0]
                )
            )
            pose_local_to_rig_now = self._trajectory.last_pose
            pose_rig_now_to_rig_at_trajectory_time = (
                pose_local_to_rig_now.inverse()
                @ pose_local_to_rig_at_reference_start_time
            )
        else:
            pose_rig_now_to_rig_at_trajectory_time = trajectory.QVec(
                vec3=np.array([0, 0, 0]), quat=np.array([0, 0, 0, 1])
            )

        for k in range(self.mpc.settings.n_horizon + 1):
            tk_us = self._timestamp_us + k * DT_MPC_US
            if self._reference_trajectory is not None:
                ref_pose_in_rig = (
                    pose_rig_now_to_rig_at_trajectory_time
                    @ self._reference_trajectory.interpolate_pose(tk_us)
                )
                ref_heading = ref_pose_in_rig.yaw
                if k == 0:
                    self._first_reference_pose_rig = ref_pose_in_rig
            else:
                ref_pose_in_rig = trajectory.QVec(
                    vec3=np.array([0, 0, 0]), quat=np.array([0, 0, 0, 1])
                )
                ref_heading = 0.0

            tvp_template["_tvp", k, "x_ref"] = ref_pose_in_rig.vec3[0]
            tvp_template["_tvp", k, "y_ref"] = ref_pose_in_rig.vec3[1]
            tvp_template["_tvp", k, "heading_ref"] = ref_heading
            tvp_template["_tvp", k, "tracking_enabled"] = (
                1.0 if k >= self._gains.idx_start_penalty else 0.0
            )
        return tvp_template

    def run_controller_and_vehicle_model(
        self, request: controller_pb2.RunControllerAndVehicleModelRequest
    ) -> controller_pb2.RunControllerAndVehicleModelResponse:
        """
        Runs the controller and vehicle model for the given request.
        Args:
            request: The request containing the current state, planned trajectory, and future time.
        Returns:
            A response containing the current pose and dynamic state in the local frame.
        """

        logging.debug(
            "run_controller_and_vehicle_model: %s: %s -> %s",
            request.session_uuid,
            request.state.timestamp_us,
            request.future_time_us,
        )

        # Input sanity checks
        if request.state.timestamp_us != self._timestamp_us:
            raise ValueError(
                f"Timestamp mismatch: expected {self._timestamp_us}, got {request.state.timestamp_us}"
            )
        if len(request.planned_trajectory_in_rig.poses) == 0:
            raise ValueError("Planned trajectory is empty")
        if request.future_time_us <= request.state.timestamp_us:
            raise ValueError(
                f"future_time_us ({request.future_time_us}) must be greater than "
                f"current timestamp ({request.state.timestamp_us})"
            )

        # Update the pose (corrected for ground constraints) for the current timestamp
        if request.state.timestamp_us != self._trajectory.timestamps_us[-1]:
            raise ValueError(
                f"Timestamp mismatch: expected {self._trajectory.timestamps_us[-1]}, got {request.state.timestamp_us}"
            )
        logging.debug(
            "overriding pose at timestamp %s with %s",
            request.state.timestamp_us,
            request.state.pose,
        )
        self._trajectory.poses.vec3[-1] = trajectory.QVec.from_grpc_pose(
            request.state.pose
        ).vec3
        self._trajectory.poses.quat[-1] = trajectory.QVec.from_grpc_pose(
            request.state.pose
        ).quat

        # Store the reference trajectory
        self._reference_trajectory = trajectory.Trajectory.from_grpc(
            request.planned_trajectory_in_rig
        )

        # Setup and execute the MPC
        # Reinitialize the state
        if request.coerce_dynamic_state:
            velocity_cg = self._dynamic_state_to_cg_velocity(request.state.state)
            self._vehicle_model.set_velocity(velocity_cg[0], velocity_cg[1])

        # Choose the number of steps such that:
        # - we do at least one step
        # - all the steps are DT_MPC seconds long, except possibly the last one
        # - the last step can be shorter or slightly longer than DT_MPC seconds
        dt_request_us = request.future_time_us - self._timestamp_us
        dt_mpc_us = int(1e6 * self.DT_MPC)
        n_steps = dt_request_us // dt_mpc_us
        if (dt_request_us % dt_mpc_us) / dt_mpc_us > 0.1:
            n_steps += 1
        n_steps = max(1, n_steps)

        for i in range(n_steps):
            if i == n_steps - 1:
                dt_us = request.future_time_us - self._timestamp_us
            else:
                dt_us = int(1e6 * self.DT_MPC)
            self._step(dt_us)

        current_pose_local_to_rig = self._trajectory.last_pose.to_grpc_pose_at_time(
            self._timestamp_us
        )

        # Build dynamic state with velocities and accelerations in rig frame
        dynamic_state = self._build_dynamic_state_in_rig_frame()

        return controller_pb2.RunControllerAndVehicleModelResponse(
            pose_local_to_rig=current_pose_local_to_rig,
            pose_local_to_rig_estimated=current_pose_local_to_rig,
            dynamic_state=dynamic_state,
            dynamic_state_estimated=dynamic_state,
        )

    def _build_dynamic_state_in_rig_frame(self) -> common_pb2.DynamicState:
        """
        Builds the DynamicState message with velocities and accelerations
        converted from CG frame to rig frame.

        The rig frame is offset from the CG by l_rig_to_cg along the x-axis.
        The transformation accounts for:
        - Velocity: v_rig = v_cg + omega × r_cg_to_rig
        - Acceleration: a_rig = a_cg + alpha × r_cg_to_rig + omega × (omega × r_cg_to_rig)

        Where r_cg_to_rig = [-l_rig_to_cg, 0, 0] (CG is ahead of rig origin).
        """
        l_rig_to_cg = self._vehicle_model.parameters.l_rig_to_cg
        state = self._vehicle_model.state
        accels = self._vehicle_model.accelerations

        # Extract CG-frame quantities
        v_cg_x = state[3]  # Longitudinal velocity at CG
        v_cg_y = state[4]  # Lateral velocity at CG
        yaw_rate = state[5]  # Angular velocity (yaw rate)

        a_cg_x = accels[0]  # Longitudinal acceleration at CG (= state[7])
        a_cg_y = accels[1]  # Lateral acceleration at CG
        d_yaw_rate = accels[2]  # Angular acceleration

        # Convert velocity from CG to rig frame:
        # v_rig = v_cg + omega × r_cg_to_rig
        # omega × [-l, 0, 0] = [0, 0, yaw_rate] × [-l, 0, 0] = [0, -yaw_rate * l, 0]
        v_rig_x = v_cg_x
        v_rig_y = v_cg_y - yaw_rate * l_rig_to_cg

        # Convert acceleration from CG to rig frame:
        # a_rig = a_cg + alpha × r_cg_to_rig + omega × (omega × r_cg_to_rig)
        # alpha × [-l, 0, 0] = [0, 0, d_yaw] × [-l, 0, 0] = [0, -d_yaw * l, 0]
        # omega × (omega × r) = omega × [0, -yaw_rate * l, 0]
        #                     = [0, 0, yaw_rate] × [0, -yaw_rate * l, 0]
        #                     = [yaw_rate² * l, 0, 0]
        a_rig_x = a_cg_x + yaw_rate * yaw_rate * l_rig_to_cg
        a_rig_y = a_cg_y - d_yaw_rate * l_rig_to_cg

        return common_pb2.DynamicState(
            linear_velocity=common_pb2.Vec3(x=v_rig_x, y=v_rig_y, z=0.0),
            angular_velocity=common_pb2.Vec3(x=0.0, y=0.0, z=yaw_rate),
            linear_acceleration=common_pb2.Vec3(x=a_rig_x, y=a_rig_y, z=0.0),
            angular_acceleration=common_pb2.Vec3(x=0.0, y=0.0, z=d_yaw_rate),
        )

    def _step(self, dt_us: int) -> None:
        # Reset the integrated positional states (x, y, psi)
        # This is equivalent to resetting the vehicle model to the origin so we can
        # compute the relative pose over the propagation time
        self._vehicle_model.reset_origin()

        # Reset the state based on the vehicle model
        self._x0 = self._vehicle_model.state.copy().reshape(-1, 1)

        self.control_input = self.mpc.make_step(self._x0)

        # Advance the vehicle model and update the trajectory history
        pose_rig_t0_to_rig_t1 = self._vehicle_model.advance(
            self.control_input, dt_us * 1e-6
        )
        self._timestamp_us += dt_us

        logging.debug(
            "pose_rig_t0_to_rig_t1: %s, %s",
            pose_rig_t0_to_rig_t1.vec3,
            pose_rig_t0_to_rig_t1.quat,
        )
        self._trajectory.update_relative(self._timestamp_us, pose_rig_t0_to_rig_t1)
        logging.debug(
            "current pose local to rig: %s, %s",
            self._trajectory.last_pose.vec3,
            self._trajectory.last_pose.quat,
        )

        self._log()

    def _log_header(self) -> None:
        self._log_file_handle.write("timestamp_us,")
        self._log_file_handle.write("x,y,z,")
        self._log_file_handle.write("qx,qy,qz,qw,")
        self._log_file_handle.write("vx,vy,wz,")
        self._log_file_handle.write("u_steering_angle,")
        self._log_file_handle.write("u_longitudinal_actuation,")
        self._log_file_handle.write("ref_traj_0_x,ref_traj_0_y,")
        self._log_file_handle.write("front_steering_angle,")
        self._log_file_handle.write("acceleration,")
        self._log_file_handle.write("x_ref_0,y_ref_0,")
        self._log_file_handle.write("yaw_ref_0\n")

    def _log(self) -> None:
        self._log_file_handle.write(f"{self._timestamp_us},")  # 0
        for i in range(3):
            self._log_file_handle.write(f"{self._trajectory.last_pose.vec3[i]},")  # 1-3
        for i in range(4):
            self._log_file_handle.write(f"{self._trajectory.last_pose.quat[i]},")  # 4-7
        for i in range(3):
            self._log_file_handle.write(f"{self._vehicle_model.state[i + 3]},")  # 8-10
        for i in range(2):
            self._log_file_handle.write(f"{self.control_input[i][0]},")  # 11-12
        for i in range(2):
            self._log_file_handle.write(
                f"{self._reference_trajectory.poses[0].vec3[i]},"
            )  # 13-14
        self._log_file_handle.write(
            f"{self._vehicle_model.front_steering_angle},"
        )  # 15
        self._log_file_handle.write(f"{self._vehicle_model.state[7]},")  # 16
        for i in range(2):
            self._log_file_handle.write(
                f"{self._first_reference_pose_rig.vec3[i]},"
            )  # 17-18
        self._log_file_handle.write(f"{self._first_reference_pose_rig.yaw}\n")  # 19
