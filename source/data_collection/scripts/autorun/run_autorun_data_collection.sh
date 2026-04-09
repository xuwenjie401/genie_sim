#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/autorun_config.sh"

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${LOG_ROOT}/${RUN_TIMESTAMP}"
RUNNER_LOG="${RUN_LOG_DIR}/autorun.log"

RUNNER_PID_FILE="${STATE_DIR}/runner.pid"
RUNNER_BOOT_ID_FILE="${STATE_DIR}/runner.boot_id"
SERVER_PID_FILE="${STATE_DIR}/server.pid"
CLIENT_PID_FILE="${STATE_DIR}/client.pid"

CLEANED_UP=0
SERVER_PID=""
CLIENT_PID=""

mkdir -p "${STATE_DIR}" "${RUN_LOG_DIR}"
ln -sfn "${RUN_LOG_DIR}" "${LOG_ROOT}/latest"

log() {
    local level="$1"
    shift
    printf '%s [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${level}" "$*" | tee -a "${RUNNER_LOG}"
}

die() {
    log "ERROR" "$*"
    exit 1
}

read_pid_file() {
    local pid_file="$1"
    if [[ -f "${pid_file}" ]]; then
        tr -d '[:space:]' < "${pid_file}"
    fi
}

read_text_file() {
    local text_file="$1"
    if [[ -f "${text_file}" ]]; then
        tr -d '[:space:]' < "${text_file}"
    fi
}

write_pid_file() {
    local pid_file="$1"
    local pid_value="$2"
    printf '%s\n' "${pid_value}" > "${pid_file}"
}

write_text_file() {
    local text_file="$1"
    local text_value="$2"
    printf '%s\n' "${text_value}" > "${text_file}"
}

is_pid_alive() {
    local pid_value="${1:-}"
    [[ -n "${pid_value}" ]] && kill -0 "${pid_value}" 2>/dev/null
}

current_boot_id() {
    if [[ -r /proc/sys/kernel/random/boot_id ]]; then
        tr -d '[:space:]' < /proc/sys/kernel/random/boot_id
    fi
}

process_cmdline() {
    local pid_value="$1"
    if [[ -r "/proc/${pid_value}/cmdline" ]]; then
        tr '\0' ' ' < "/proc/${pid_value}/cmdline"
    fi
}

is_runner_process() {
    local pid_value="$1"
    local cmdline
    cmdline="$(process_cmdline "${pid_value}")"
    [[ -n "${cmdline}" && "${cmdline}" == *"run_autorun_data_collection.sh"* ]]
}

is_server_process() {
    local pid_value="$1"
    local cmdline
    cmdline="$(process_cmdline "${pid_value}")"
    [[ -n "${cmdline}" && "${cmdline}" == *"data_collector_server.py"* ]]
}

is_client_process() {
    local pid_value="$1"
    local cmdline
    cmdline="$(process_cmdline "${pid_value}")"
    [[ -n "${cmdline}" && "${cmdline}" == *"run_data_collection.py"* ]]
}

clear_all_state() {
    rm -f "${RUNNER_PID_FILE}" "${RUNNER_BOOT_ID_FILE}" "${SERVER_PID_FILE}" "${CLIENT_PID_FILE}"
}

clear_runner_state() {
    rm -f "${RUNNER_PID_FILE}" "${RUNNER_BOOT_ID_FILE}"
}

remove_stale_pid_file() {
    local pid_file="$1"
    local pid_value
    pid_value="$(read_pid_file "${pid_file}")"
    if [[ -n "${pid_value}" ]] && ! is_pid_alive "${pid_value}"; then
        rm -f "${pid_file}"
    fi
}

task_name() {
    if [[ -n "${TASK_NAME_OVERRIDE}" ]]; then
        printf '%s\n' "${TASK_NAME_OVERRIDE}"
        return
    fi
    basename "${TASK_TEMPLATE}" .json
}

sanitize_name() {
    printf '%s' "$1" | sed 's/[^A-Za-z0-9_-]/_/g'
}

wait_for_pid_exit() {
    local pid_value="$1"
    local timeout_seconds="$2"
    local waited=0
    while is_pid_alive "${pid_value}"; do
        if (( waited >= timeout_seconds )); then
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 0
}

archive_existing_recordings() {
    mkdir -p "${RECORDING_DIR}" "${ARCHIVE_ROOT}"
    if ! find "${RECORDING_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
        log "INFO" "recording_data is empty, skip archive"
        return
    fi

    local archive_dir="${ARCHIVE_ROOT}/${ARCHIVE_PREFIX}_$(sanitize_name "$(task_name)")_$(date +%m%d_%H%M)"
    mkdir -p "${archive_dir}"
    find "${RECORDING_DIR}" -mindepth 1 -maxdepth 1 -exec mv -t "${archive_dir}" -- {} +
    log "INFO" "archived existing recording_data to ${archive_dir}"
}

start_server() {
    log "INFO" "starting server"
    (
        set -Eeuo pipefail
        cd "${DATA_COLLECTION_DIR}"
        set +u
        source "${CONDA_SH}"
        conda activate "${CONDA_ENV_NAME}"
        source "${ASSETS_ENV_BASHRC}"
        source "${ROS_ENV_BASHRC}"
        set -u
        exec "${SERVER_CMD[@]}"
    ) >> "${RUN_LOG_DIR}/server.log" 2>&1 &
    SERVER_PID=$!
    write_pid_file "${SERVER_PID_FILE}" "${SERVER_PID}"
    log "INFO" "server pid=${SERVER_PID}, log=${RUN_LOG_DIR}/server.log"
}

start_client() {
    log "INFO" "starting client"
    (
        set -Eeuo pipefail
        cd "${DATA_COLLECTION_DIR}"
        set +u
        source "${CONDA_SH}"
        conda activate "${CONDA_ENV_NAME}"
        source "${ASSETS_ENV_BASHRC}"
        source "${ROS_ENV_BASHRC}"
        set -u
        exec "${CLIENT_CMD[@]}"
    ) >> "${RUN_LOG_DIR}/client.log" 2>&1 &
    CLIENT_PID=$!
    write_pid_file "${CLIENT_PID_FILE}" "${CLIENT_PID}"
    log "INFO" "client pid=${CLIENT_PID}, log=${RUN_LOG_DIR}/client.log"
}

stop_client() {
    local pid_value
    pid_value="$(read_pid_file "${CLIENT_PID_FILE}")"
    if ! is_pid_alive "${pid_value}"; then
        rm -f "${CLIENT_PID_FILE}"
        return
    fi

    log "INFO" "sending SIGINT to client pid=${pid_value}"
    kill -INT "${pid_value}" 2>/dev/null || true
    if wait_for_pid_exit "${pid_value}" 300; then
        log "INFO" "client exited cleanly"
        rm -f "${CLIENT_PID_FILE}"
        return
    fi

    log "WARN" "client did not exit after SIGINT, sending SIGTERM"
    kill -TERM "${pid_value}" 2>/dev/null || true
    if wait_for_pid_exit "${pid_value}" 30; then
        log "INFO" "client exited after SIGTERM"
        rm -f "${CLIENT_PID_FILE}"
        return
    fi

    log "WARN" "client still alive, sending SIGKILL"
    kill -KILL "${pid_value}" 2>/dev/null || true
    wait_for_pid_exit "${pid_value}" 10 || true
    rm -f "${CLIENT_PID_FILE}"
}

stop_server() {
    local pid_value
    pid_value="$(read_pid_file "${SERVER_PID_FILE}")"
    if ! is_pid_alive "${pid_value}"; then
        rm -f "${SERVER_PID_FILE}"
        return
    fi

    log "INFO" "waiting up to 30s for server pid=${pid_value} to exit"
    if wait_for_pid_exit "${pid_value}" 30; then
        log "INFO" "server exited cleanly"
        rm -f "${SERVER_PID_FILE}"
        return
    fi

    log "WARN" "server is still alive, sending SIGTERM"
    kill -TERM "${pid_value}" 2>/dev/null || true
    if wait_for_pid_exit "${pid_value}" 30; then
        log "INFO" "server exited after SIGTERM"
        rm -f "${SERVER_PID_FILE}"
        return
    fi

    log "WARN" "server still alive, sending SIGKILL"
    kill -KILL "${pid_value}" 2>/dev/null || true
    wait_for_pid_exit "${pid_value}" 10 || true
    rm -f "${SERVER_PID_FILE}"
}

cleanup() {
    local exit_code="$?"
    trap - EXIT INT TERM
    if (( CLEANED_UP )); then
        return
    fi
    CLEANED_UP=1
    stop_client
    stop_server
    rm -f "${RUNNER_PID_FILE}" "${RUNNER_BOOT_ID_FILE}"
    log "INFO" "autorun finished with exit code ${exit_code}"
}

handle_stop_signal() {
    log "INFO" "received stop signal"
    exit 0
}

trap cleanup EXIT
trap handle_stop_signal INT TERM

remove_stale_pid_file "${RUNNER_PID_FILE}"
remove_stale_pid_file "${SERVER_PID_FILE}"
remove_stale_pid_file "${CLIENT_PID_FILE}"

runner_pid="$(read_pid_file "${RUNNER_PID_FILE}")"
runner_boot_id="$(read_text_file "${RUNNER_BOOT_ID_FILE}")"
boot_id="$(current_boot_id)"

if [[ -n "${runner_pid}" ]]; then
    if [[ -n "${runner_boot_id}" && -n "${boot_id}" && "${runner_boot_id}" != "${boot_id}" ]]; then
        log "INFO" "found stale autorun state from previous boot, clearing it"
        clear_all_state
    elif is_pid_alive "${runner_pid}" && is_runner_process "${runner_pid}"; then
        die "autorun is already running with pid=${runner_pid}"
    else
        log "INFO" "found stale or reused runner pid=${runner_pid}, clearing stale autorun state"
        clear_runner_state
    fi
fi

server_pid_state="$(read_pid_file "${SERVER_PID_FILE}")"
client_pid_state="$(read_pid_file "${CLIENT_PID_FILE}")"

if is_pid_alive "${server_pid_state}" && is_server_process "${server_pid_state}"; then
    die "autorun server is already running with pid=${server_pid_state}; stop it before starting a new run"
fi

if is_pid_alive "${client_pid_state}" && is_client_process "${client_pid_state}"; then
    die "autorun client is already running with pid=${client_pid_state}; stop it before starting a new run"
fi

[[ -f "${CONDA_SH}" ]] || die "conda init script not found: ${CONDA_SH}"
[[ -f "${ASSETS_ENV_BASHRC}" ]] || die "assets env script not found: ${ASSETS_ENV_BASHRC}"
[[ -f "${ROS_ENV_BASHRC}" ]] || die "ROS env script not found: ${ROS_ENV_BASHRC}"
[[ -f "${DATA_COLLECTION_DIR}/${TASK_TEMPLATE}" || -f "${TASK_TEMPLATE}" ]] || die "task template not found: ${TASK_TEMPLATE}"

write_pid_file "${RUNNER_PID_FILE}" "$$"
write_text_file "${RUNNER_BOOT_ID_FILE}" "${boot_id}"

log "INFO" "run log directory: ${RUN_LOG_DIR}"
log "INFO" "task template: ${TASK_TEMPLATE}"
log "INFO" "conda env: ${CONDA_ENV_NAME}"
log "INFO" "server delay: ${PRE_SERVER_SLEEP_SECONDS}s, client delay: ${PRE_CLIENT_SLEEP_SECONDS}s"

archive_existing_recordings

log "INFO" "sleeping ${PRE_SERVER_SLEEP_SECONDS}s before starting server"
sleep "${PRE_SERVER_SLEEP_SECONDS}"
start_server
sleep 2
is_pid_alive "${SERVER_PID}" || die "server exited immediately, see ${RUN_LOG_DIR}/server.log"

log "INFO" "sleeping ${PRE_CLIENT_SLEEP_SECONDS}s before starting client"
sleep "${PRE_CLIENT_SLEEP_SECONDS}"
start_client
sleep 2
is_pid_alive "${CLIENT_PID}" || die "client exited immediately, see ${RUN_LOG_DIR}/client.log"

log "INFO" "client is running, waiting for completion"
set +e
wait "${CLIENT_PID}"
CLIENT_EXIT_CODE="$?"
set -e
log "INFO" "client exited with code ${CLIENT_EXIT_CODE}"
exit "${CLIENT_EXIT_CODE}"
