# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import logging

from alpasim_controller.system import System
from alpasim_grpc.v0 import common_pb2, controller_pb2


class SystemManager:
    """
    The SystemManager class manages multiple vehicle dynamics and control systems.
    """

    def __init__(self, log_dir):
        self._log_dir = log_dir
        self._systems = {}  # session uuid to System mapping

    def close_session(self, request: controller_pb2.VDCSessionCloseRequest):
        if request.session_uuid in self._systems:
            logging.info(f"Closing session: {request.session_uuid}")
            del self._systems[request.session_uuid]
            return common_pb2.Empty()
        raise KeyError(f"Session {request.session_uuid} does not exist")

    def _create_system(self, session_uuid: str, state: common_pb2.StateAtTime):
        logging.info(f"Creating system for session_uuid: {session_uuid}")
        self._systems[session_uuid] = System(
            f"{self._log_dir}/alpasim_controller_{session_uuid}.csv", state
        )

    def run_controller_and_vehicle_model(
        self, request: controller_pb2.RunControllerAndVehicleModelRequest
    ):
        logging.debug(
            f"run_controller_and_vehicle called for session_uuid: {request.session_uuid}"
        )
        if request.session_uuid not in self._systems:
            self._create_system(request.session_uuid, request.state)
        system = self._systems[request.session_uuid]
        return system.run_controller_and_vehicle_model(request)
