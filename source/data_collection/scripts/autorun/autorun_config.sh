#!/usr/bin/env bash

PROJECT_ROOT="/home/agxi/RealityLab/genie_sim"
DATA_COLLECTION_DIR="${PROJECT_ROOT}/source/data_collection"

CONDA_SH="/home/agxi/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="issac"
ROS_ENV_BASHRC="${DATA_COLLECTION_DIR}/ros2_env.bashrc"
ASSETS_ENV_BASHRC="${DATA_COLLECTION_DIR}/assets.bashrc"

TASK_TEMPLATE="tasks/diy/meta_task/galbot_meta_pick_place_V2.json"
TASK_NAME_OVERRIDE=""

PRE_SERVER_SLEEP_SECONDS=60
PRE_CLIENT_SLEEP_SECONDS=60

RECORDING_DIR="${DATA_COLLECTION_DIR}/recording_data"
ARCHIVE_ROOT="/home/agxi/Datasets/galbot_sim/raw"
ARCHIVE_PREFIX="autorun"

STATE_DIR="${DATA_COLLECTION_DIR}/scripts/autorun/state"
LOG_ROOT="${DATA_COLLECTION_DIR}/logs/autorun"

SERVER_CMD=(
    python scripts/data_collector_server.py
    --enable_physics
    --enable_curobo
    --publish_ros
    --headless
)

CLIENT_CMD=(
    python scripts/run_data_collection.py
    --task_template "${TASK_TEMPLATE}"
    --use_recording
)
