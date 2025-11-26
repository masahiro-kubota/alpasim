# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Context management for AlpasimWizard."""

from __future__ import annotations

import glob
import logging
import os
import socket
import subprocess
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import yaml

from .compatibility import CompatibilityMatrix
from .s3.sceneset import SceneIdAndUuid
from .schema import AlpasimConfig
from .utils import nre_image_to_nre_version

logger = logging.getLogger(__name__)


def _filesystem_search(
    scene_ids: list[str] | None,
    compatible_nre_versions: set[str],
    scene_cache: str,
) -> list[SceneIdAndUuid]:
    """Search filesystem for scene artifacts.

    It looks for all usdz files under {scene_cache}/**/*.usdz and returns the artifacts that are in
    scene_ids list and are compatible with the given compatible_nre_versions.

    Args:
        scene_ids: List of scene ids to search for. If None, all artifacts under the scene cache
            path will be considered.
        compatible_nre_versions: Set of compatible NRE versions.
        scene_cache: Path to the scene cache.

    Returns:
        List of matching artifacts (SceneIdAndUuid).
    """
    matching_artifacts: list[SceneIdAndUuid] = []
    mismatched_version_artifacts: list[str] = []

    if scene_ids is None:
        scene_ids = []
        find_all_scenes = True
    else:
        find_all_scenes = False

    for usdz in glob.glob(f"{scene_cache}/**/*.usdz", recursive=True):
        with zipfile.ZipFile(usdz, "r") as zf:
            with zf.open("metadata.yaml") as f:
                yaml_dict = yaml.safe_load(f)
                if yaml_dict["scene_id"] not in scene_ids and not find_all_scenes:
                    continue

                if yaml_dict["version_string"] in compatible_nre_versions:
                    matching_artifacts.append(
                        SceneIdAndUuid(yaml_dict["scene_id"], yaml_dict["uuid"])
                    )
                else:
                    mismatched_version_artifacts.append(
                        yaml_dict["scene_id"] + "@" + yaml_dict["version_string"]
                    )

    artifact_counts = Counter([a.scene_id for a in matching_artifacts])
    duplicate_artifacts = [
        scene_id for scene_id, count in artifact_counts.items() if count > 1
    ]
    if duplicate_artifacts:
        raise RuntimeError(f"Duplicate artifacts found: {duplicate_artifacts}")

    missing_artifacts = sorted(
        list(set(scene_ids) - {a.scene_id for a in matching_artifacts})
    )
    if missing_artifacts:
        message = f"Failed to find all artifacts with required NRE versions, missing: {missing_artifacts}."
        if mismatched_version_artifacts:
            message += f" Some artifacts found with mismatched NRE versions: {mismatched_version_artifacts}."
        raise FileNotFoundError(message)

    return matching_artifacts


def fetch_artifacts(cfg: AlpasimConfig) -> tuple[list[SceneIdAndUuid], str | None]:
    """Fetch artifacts from database or filesystem."""

    compatibility_matrix = CompatibilityMatrix.from_config(
        cfg.scenes.artifact_compatibility_matrix
    )

    # Create scene cache directory if needed
    Path(cfg.scenes.database.scene_cache).mkdir(parents=True, exist_ok=True)

    # Determine NRE version
    if cfg.scenes.database.nre_version_string is not None:
        nre_version_string = cfg.scenes.database.nre_version_string
    else:
        assert cfg.services.sensorsim is not None  # For mypy
        nre_version_string = nre_image_to_nre_version(cfg.services.sensorsim.image)
    compatible_nre_versions = compatibility_matrix.lookup(nre_version_string)

    # Handle scene sources
    if cfg.scenes.source == "local":
        sceneset_dir_relative_path = Path(cfg.scenes.local.directory)
        search_directory = (
            cfg.scenes.database.scene_cache + "/" + str(sceneset_dir_relative_path)
        )
        if not os.path.isdir(search_directory):
            raise FileNotFoundError(
                f"Local directory {search_directory} not found. "
                "Note: cfg.scenes.local.directory should be relative to the nre-artifacts directory."
            )
        artifact_list = _filesystem_search(
            cfg.scenes.scene_ids, compatible_nre_versions, search_directory
        )
    else:
        raise ValueError(f"Unknown scene source: {cfg.scenes.source}")

    # Return both artifact list and sceneset path
    return artifact_list, (
        str(sceneset_dir_relative_path) if sceneset_dir_relative_path else None
    )


def detect_gpus() -> int:
    """Detect number of GPUs on the system."""
    try:
        num_gpus = int(
            subprocess.check_output(
                "nvidia-smi -i 0 --query-gpu=count --format=csv,noheader",
                shell=True,
            )
        )
        logger.debug(f"Found {num_gpus} GPUs on system.")
    except subprocess.CalledProcessError as exc:
        logger.warning(
            f"Failed to determine GPU count via 'nvidia-smi' (code {exc.returncode}). "
            "Defaulting to 0 GPUs."
        )
        return 0
    return num_gpus


def setup_directories(cfg: AlpasimConfig) -> None:
    """Create necessary directories and symlinks."""
    log_dir = Path(cfg.wizard.log_dir)

    logger.debug(f"Creating log directory at path: {log_dir}")

    # Create subdirectories
    for subdir in ("asl", "metrics", "txt-logs", "controller"):
        subdir_path = log_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True, mode=0o777)
        os.chmod(subdir_path, 0o777)


@dataclass
class WizardContext:
    """Unified context for all wizard operations.

    Combines configuration access with runtime state,
    eliminating the need for a separate GlobalContext.
    """

    cfg: AlpasimConfig
    port_assigner: Iterator[int]

    # Expensive operations (only loaded when needed for actual execution)
    artifact_list: list[SceneIdAndUuid] = field(default_factory=list)
    num_gpus: int = 0
    sceneset_path: str | None = None

    def get_num_gpus(self) -> int:
        """Get GPU count with fallback to 0."""
        return self.num_gpus if self.num_gpus is not None else 0

    def get_artifacts(self) -> list[SceneIdAndUuid]:
        """Get artifacts with fallback to empty list."""
        return self.artifact_list if self.artifact_list is not None else []

    @property
    def all_services_to_run(self) -> list[str]:
        """Get all services that should be run."""
        result = []
        if self.cfg.wizard.run_sim_services:
            result.extend(self.cfg.wizard.run_sim_services)
        if self.cfg.wizard.run_eval_services:
            result.extend(self.cfg.wizard.run_eval_services)
        if self.cfg.wizard.run_aggregation_services:
            result.extend(self.cfg.wizard.run_aggregation_services)
        return result

    @staticmethod
    def create(cfg: AlpasimConfig) -> WizardContext:
        """Build context."""

        # Always set these basic attributes
        artifact_list, sceneset_path = fetch_artifacts(cfg)
        context = WizardContext(
            cfg=cfg,
            port_assigner=create_port_assigner(cfg.wizard.baseport),
            artifact_list=artifact_list,
            num_gpus=detect_gpus(),
            sceneset_path=sceneset_path,
        )

        setup_directories(cfg)

        return context


def _is_port_open(port: int) -> bool:
    """Check if a port is available (not in use)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def _find_next_open_port(start_port: int, max_search: int = 1000) -> int:
    """Find the next available port starting from start_port.

    Args:
        start_port: Port number to start searching from
        max_search: Maximum number of ports to check

    Returns:
        The next available port number

    Raises:
        RuntimeError: If no available port is found within the search range
    """
    for port in range(start_port, start_port + max_search):
        if _is_port_open(port):
            logger.debug(f"Found available port: {port}")
            return port
    raise RuntimeError(
        f"Could not find an available port in range {start_port}-{start_port + max_search - 1}"
    )


def create_port_assigner(baseport: int) -> Iterator[int]:
    """Create an iterator over port numbers starting from the first available port >= baseport."""

    def port_assigner() -> Iterator[int]:
        ports_assigned = 0
        max_ports = 1000
        # Start from the first available port >= baseport
        next_port = _find_next_open_port(baseport)

        while ports_assigned < max_ports:
            yield next_port
            ports_assigned += 1
            # Find the next available port after the one we just yielded
            next_port = _find_next_open_port(next_port + 1)

        raise AssertionError(
            f"Handed out {max_ports} different port numbers - something's fishy."
        )

    return port_assigner()
