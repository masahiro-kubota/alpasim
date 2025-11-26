# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import argparse
import asyncio
import dataclasses
import glob
import logging
import multiprocessing as mp
import os
import pathlib
import pprint
import sys
import traceback
from functools import partial
from typing import Any

import numpy as np
import polars as pl
from alpasim_grpc.v0.logging_pb2 import ActorPoses
from alpasim_utils.artifact import Artifact
from alpasim_utils.logs import async_read_pb_log
from alpasim_utils.qvec import QVec
from alpasim_utils.trajectory import Trajectory
from omegaconf import OmegaConf
from tqdm import tqdm

from eval.aggregation import processing
from eval.data import (
    RAABB,
    ActorPolygons,
    Cameras,
    DriverResponses,
    EvaluationResultContainer,
    RenderableTrajectory,
    Routes,
    SimulationResult,
)
from eval.kratos_utils import get_metadata
from eval.schema import EvalConfig
from eval.scorers import create_scorer_group
from eval.video import render_and_save_video_for_eval_container

logger = logging.getLogger("alpasim_eval")

TRACKER_FILE_NAME = "_complete"


async def load_simulation_results(
    eval_result_container: EvaluationResultContainer,
    cfg: EvalConfig,
    artifacts: dict[str, Artifact],
) -> EvaluationResultContainer:
    # Mapping from actor id to trajectory
    actor_trajectories: dict[str, RenderableTrajectory] = {}
    cameras: Cameras = Cameras()
    next_drive_request_time_query_us = None
    routes: Routes = Routes()
    driver_estimated_trajectory: Trajectory = Trajectory.create_empty()

    # Read first message only to get metadata
    async for message in async_read_pb_log(eval_result_container.file_path):
        assert (
            message.WhichOneof("log_entry") == "rollout_metadata"
        ), "First message must be rollout metadata"

        session_metadata = message.rollout_metadata.session_metadata
        for actor_aabb in message.rollout_metadata.actor_definitions.actor_aabb:
            actor_id = actor_aabb.actor_id
            actor_trajectories[actor_id] = RenderableTrajectory.create_empty_with_bbox(
                RAABB.from_grpc(actor_aabb.aabb, cfg.vehicle)
            )
        ego_coords_rig_to_aabb_center = QVec.from_grpc_pose(
            message.rollout_metadata.transform_ego_coords_rig_to_aabb
        )
        ego_recorded_ground_truth_trajectory = RenderableTrajectory.from_grpc_with_aabb(
            message.rollout_metadata.ego_rig_recorded_ground_truth_trajectory,
            actor_trajectories["EGO"].raabb,
        ).transform(ego_coords_rig_to_aabb_center, is_relative=True)
        break

    # Start new loop for remaining messages
    driver_responses: DriverResponses = DriverResponses(
        ego_coords_rig_to_aabb_center=ego_coords_rig_to_aabb_center,
        ego_trajectory_local=actor_trajectories["EGO"],
    )

    async for message in async_read_pb_log(eval_result_container.file_path):
        if message.WhichOneof("log_entry") == "actor_poses":
            # actor_poses = message.actor_poses
            poses_message: ActorPoses = message.actor_poses
            timestamp_us = poses_message.timestamp_us
            for pose in poses_message.actor_poses:
                actor_trajectories[pose.actor_id].update_absolute(
                    timestamp_us, QVec.from_grpc_pose(pose.actor_pose)
                )
        elif message.WhichOneof("log_entry") == "driver_request":
            next_drive_request_time_query_us = message.driver_request.time_query_us
            next_drive_request_time_us = message.driver_request.time_now_us
        elif message.WhichOneof("log_entry") == "driver_return":
            driver_responses.add_drive_response(
                message.driver_return,
                now_time_us=next_drive_request_time_us,
                query_time_us=next_drive_request_time_query_us,  # type: ignore
            )
        elif message.WhichOneof("log_entry") == "driver_camera_image":
            cameras.add_camera_image(message.driver_camera_image.camera_image)
        elif message.WhichOneof("log_entry") == "route_request":
            routes.add_route(message.route_request.route)  # type: ignore
        elif message.WhichOneof("log_entry") == "driver_ego_trajectory":
            driver_estimated_trajectory = driver_estimated_trajectory.append(
                Trajectory.from_grpc(message.driver_ego_trajectory.trajectory),
            )
        elif message.WhichOneof("log_entry") == "available_cameras_return":
            for available_camera in message.available_cameras_return.available_cameras:
                cameras.add_calibration(available_camera)

    driver_estimated_trajectory = RenderableTrajectory.from_trajectory(
        driver_estimated_trajectory,
        actor_trajectories["EGO"].raabb,
    ).transform(ego_coords_rig_to_aabb_center, is_relative=True)
    routes.convert_routes_to_global_frame(
        # TODO: Remove the noise in the route (which is w.r.t the noise ego pos)
        # ego_trajectory=driver_estimated_trajectory,
        ego_trajectory=actor_trajectories["EGO"],
        ego_coords_rig_to_aabb_center=ego_coords_rig_to_aabb_center,
    )
    eval_result_container.sim_result = SimulationResult(
        session_metadata=session_metadata,
        ego_coords_rig_to_aabb_center=ego_coords_rig_to_aabb_center,
        ego_recorded_ground_truth_trajectory=ego_recorded_ground_truth_trajectory,
        actor_trajectories=actor_trajectories,
        driver_estimated_trajectory=driver_estimated_trajectory,
        driver_responses=driver_responses,
        vec_map=artifacts[session_metadata.scene_id].map,
        actor_polygons=ActorPolygons.from_actor_trajectories(actor_trajectories),
        cameras=cameras,
        routes=routes,
    )
    return eval_result_container


def df_from_evaluation_result_container(
    eval_result_container: EvaluationResultContainer,
    run_metadata: dict[str, Any],
) -> processing.UnprocessedMetricsDFs:
    """Convert the evaluation result container to a dataframe.

    Args:
        eval_result_container: The evaluation result container.
        run_metadata: The run metadata (load with `kratos_utils.get_metadata`).

    Returns:
        A dataframe with all the metrics and metadata (long format).
    """
    clipgt_id, batch_id, rollout_id = (
        eval_result_container.get_clipgt_batch_and_rollout_id()
    )

    for mr in eval_result_container.metric_results:
        mr.values = np.array(mr.values, dtype=np.float64)

    dictionaries = [
        {
            **dataclasses.asdict(mr),
            "clipgt_id": clipgt_id,
            "batch_id": batch_id,
            "rollout_id": rollout_id,
            "run_uuid": run_metadata["run_uuid"],
            "run_name": run_metadata["run_name"],
        }
        for mr in eval_result_container.metric_results
    ]

    return processing.UnprocessedMetricsDFs(
        pl.concat([pl.DataFrame(d) for d in dictionaries])
    )


def process_evaluation_result_container(
    eval_result_container: EvaluationResultContainer,
    cfg: EvalConfig,
    args: argparse.Namespace,
    artifacts: dict[str, Artifact],
    run_metadata: dict[str, Any],
) -> pl.DataFrame:
    scorers = create_scorer_group(cfg)
    evaluation_result_container = asyncio.run(
        load_simulation_results(eval_result_container, cfg, artifacts)
    )

    evaluation_result_container = scorers.run(evaluation_result_container)

    unprocessed_metrics = df_from_evaluation_result_container(
        evaluation_result_container, run_metadata
    )

    processed_metrics_dfs = unprocessed_metrics.process()

    if cfg.video.render_video:
        os.makedirs(os.path.join(args.output_dir, "videos"), exist_ok=True)
        try:
            render_and_save_video_for_eval_container(
                evaluation_result_container,
                processed_metrics_dfs,
                args.output_dir,
                cfg,
            )
        except Exception as e:
            logger.error("Error rendering video: %s", e)
            logger.error("Stacktrace: %s", traceback.format_exc())
    else:
        logger.info("Skipping video rendering as it is disabled in the config.")

    # Throw away the aggregated data as we'll need to aggragate again over all
    # workers.
    return processed_metrics_dfs.unprocessed_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Perform KPIs evaluation of rclog files."
    )
    parser.add_argument("--asl_search_glob", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--trajdata_cache_dir", type=str)
    parser.add_argument("--usdz_glob", type=str)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )

    assert args.config_path.endswith(".yaml")

    # Check if metrics_unprocessed.parquet already exists
    metrics_file = os.path.join(args.output_dir, "metrics_unprocessed.parquet")
    if os.path.exists(metrics_file):
        logger.info(f"Eval already completed: {metrics_file} exists, skipping eval...")
        logger.info("To force re-evaluation, delete the existing eval directory.")
        return 0

    # Needed for matpltolib to work in multiprocessing
    mp.set_start_method("forkserver")

    config_untyped = OmegaConf.load(args.config_path)
    cfg: EvalConfig = OmegaConf.merge(EvalConfig, config_untyped)
    #
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Config details:")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg, resolve=True)))

    artifacts = Artifact.discover_from_glob(args.usdz_glob)
    files = glob.glob(args.asl_search_glob, recursive=True)
    run_metadata = get_metadata(pathlib.Path(args.config_path).parent)

    filtered_files = [
        fname
        for fname in files
        if (pathlib.Path(fname).parent / TRACKER_FILE_NAME).exists()
    ]
    if not filtered_files:
        raise ValueError(
            f"No files found in {args.asl_search_glob}. "
            "This case should be handled by the wizard by not dispatching KPIs at all."
        )
    if len(filtered_files) != len(files):
        raise ValueError(
            f"Only {len(filtered_files)} out of {len(files)} files have been found to be complete."
        )

    evaluation_result_containers = [
        EvaluationResultContainer(file_path=fname) for fname in filtered_files
    ]

    num_workers = min(
        mp.cpu_count(),
        cfg.num_processes,
        len(evaluation_result_containers),
    )

    logger.info(
        "Using %d workers: %d CPUs available, config asks for %d",
        num_workers,
        mp.cpu_count(),
        cfg.num_processes,
    )

    logger.info(
        "Processing evaluation %d result containers...",
        len(evaluation_result_containers),
    )

    # Use partial to fix the arguments that are the same for all calls
    process_func = partial(
        process_evaluation_result_container,
        cfg=cfg,
        args=args,
        run_metadata=run_metadata,
        artifacts=artifacts,
    )

    if num_workers > 1:
        logger.info("Processing evaluation result containers in multi-process mode.")
        with mp.Pool(
            processes=num_workers,
        ) as pool:
            # Use imap_unordered for potentially better performance if render times vary
            # Wrap with list() to ensure all tasks complete before moving on
            df_list = list(
                tqdm(
                    pool.imap_unordered(process_func, evaluation_result_containers),
                    total=len(evaluation_result_containers),
                    desc="Processing evaluation result containers",
                )
            )
    else:  # num_workers == 1
        logger.info("Processing evaluation result containers in single-process mode.")
        df_list = [
            process_func(evaluation_result_container)
            for evaluation_result_container in tqdm(
                evaluation_result_containers,
                desc="Processing evaluation result containers",
            )
        ]

    df = pl.concat(df_list)

    logger.info(f"Writing metrics df to {args.output_dir}/metrics_unprocessed.parquet")
    df.write_parquet(os.path.join(args.output_dir, "metrics_unprocessed.parquet"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
