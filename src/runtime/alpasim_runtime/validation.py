# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Pre-flight validation for simulation runs."""

import logging

import alpasim_runtime
from alpasim_grpc.v0.common_pb2 import Empty, VersionId
from alpasim_runtime.config import (
    NetworkSimulatorConfig,
    ScenarioConfig,
    SimulatorConfig,
)
from alpasim_runtime.endpoints import get_service_endpoints

import grpc

logger = logging.getLogger(__name__)


async def gather_versions_from_addresses(
    network_config: NetworkSimulatorConfig,
    timeout_s: int = 30,
) -> None:
    """Probe each endpoint for version info before spawning workers."""
    versions: dict[str, VersionId] = {}
    endpoint_stubs = get_service_endpoints(network_config)

    for svc_name, (stub_class, addresses) in endpoint_stubs.items():
        if not addresses:
            continue
        channel = grpc.aio.insecure_channel(addresses[0])
        try:
            stub = stub_class(channel)
            if svc_name == "trafficsim":
                # trafficsim returns version via get_metadata instead of get_version
                metadata = await stub.get_metadata(
                    Empty(), wait_for_ready=True, timeout=timeout_s
                )
                versions[svc_name] = metadata.version_id
            else:
                versions[svc_name] = await stub.get_version(
                    Empty(), wait_for_ready=True, timeout=timeout_s
                )
        finally:
            await channel.close()

    # Log versions
    runtime_version = alpasim_runtime.VERSION_MESSAGE
    for svc_name, version in versions.items():
        logger.info("%s: %s", svc_name, version)
    logger.info("runtime: %s", runtime_version)


async def validate_scenarios(config: SimulatorConfig) -> None:
    """
    Validate all scenarios before building job list.

    Uses lightweight probes to check scene availability without creating full pools.
    This ensures we fail fast in the parent if any scenario is invalid.
    """
    error_messages: list[str] = []

    # driver and controller return wildcard (work with any scene), no need to probe
    service_endpoints = get_service_endpoints(
        config.network, services=["sensorsim", "physics", "trafficsim"]
    )

    for svc_name, (stub_class, addresses) in service_endpoints.items():
        if not addresses:
            continue

        errors = await _probe_scenario_compatibility(
            stub_class,
            addresses[0],
            config.user.scenarios,
            timeout_s=config.user.endpoints.startup_timeout_s,
            use_metadata=(svc_name == "trafficsim"),
        )
        error_messages.extend(errors)

    if error_messages:
        raise AssertionError("\n".join(error_messages))


async def _probe_scenario_compatibility(
    stub_class: type,
    address: str,
    scenarios: list[ScenarioConfig],
    timeout_s: int = 30,
    use_metadata: bool = False,
) -> list[str]:
    """Probe a service address to validate scenario compatibility without creating pools.

    Args:
        use_metadata: If True, use get_metadata().supported_map_ids instead of
            get_available_scenes().scene_ids (for trafficsim compatibility).
    """
    incompatibilities = []

    channel = grpc.aio.insecure_channel(address)
    try:
        stub = stub_class(channel)
        if use_metadata:
            # trafficsim uses get_metadata with supported_map_ids
            response = await stub.get_metadata(
                Empty(), wait_for_ready=True, timeout=timeout_s
            )
            # trafficsim returns map_ids without the clipgt- prefix, so we add it
            available_scenes = set(
                f"clipgt-{map_id}" for map_id in response.supported_map_ids
            )
        else:
            response = await stub.get_available_scenes(
                Empty(), wait_for_ready=True, timeout=timeout_s
            )
            available_scenes = set(response.scene_ids)

        for scenario in scenarios:
            if (
                scenario.scene_id not in available_scenes
                and "*" not in available_scenes
            ):
                incompatibilities.append(
                    f"Scene {scenario.scene_id} not available at {address}. "
                    f"Available: {sorted(available_scenes)}"
                )
    finally:
        await channel.close()

    return incompatibilities
