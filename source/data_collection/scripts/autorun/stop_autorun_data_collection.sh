#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/autorun_config.sh"

RUNNER_PID_FILE="${STATE_DIR}/runner.pid"
RUNNER_BOOT_ID_FILE="${STATE_DIR}/runner.boot_id"
SERVER_PID_FILE="${STATE_DIR}/server.pid"
CLIENT_PID_FILE="${STATE_DIR}/client.pid"

mkdir -p "${STATE_DIR}"

log() {
    printf '%s [INFO] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

read_pid_file() {
    local pid_file="$1"
    if [[ -f "${pid_file}" ]]; then
        tr -d '[:space:]' < "${pid_file}"
    fi
}

is_pid_alive() {
    local pid_value="${1:-}"
    [[ -n "${pid_value}" ]] && kill -0 "${pid_value}" 2>/dev/null
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

runner_pid="$(read_pid_file "${RUNNER_PID_FILE}")"
if is_pid_alive "${runner_pid}"; then
    log "sending SIGTERM to autorun runner pid=${runner_pid}"
    kill -TERM "${runner_pid}" 2>/dev/null || true
    if wait_for_pid_exit "${runner_pid}" 360; then
        log "autorun runner exited cleanly"
        exit 0
    fi
    log "runner did not exit in time, stopping child processes directly"
fi

client_pid="$(read_pid_file "${CLIENT_PID_FILE}")"
if is_pid_alive "${client_pid}"; then
    log "sending SIGINT to client pid=${client_pid}"
    kill -INT "${client_pid}" 2>/dev/null || true
    if ! wait_for_pid_exit "${client_pid}" 60; then
        log "client still alive, sending SIGTERM"
        kill -TERM "${client_pid}" 2>/dev/null || true
        wait_for_pid_exit "${client_pid}" 15 || true
    fi
fi

server_pid="$(read_pid_file "${SERVER_PID_FILE}")"
if is_pid_alive "${server_pid}"; then
    log "sending SIGTERM to server pid=${server_pid}"
    kill -TERM "${server_pid}" 2>/dev/null || true
    if ! wait_for_pid_exit "${server_pid}" 30; then
        log "server still alive, sending SIGKILL"
        kill -KILL "${server_pid}" 2>/dev/null || true
        wait_for_pid_exit "${server_pid}" 10 || true
    fi
fi

if is_pid_alive "${runner_pid}"; then
    log "runner still alive, sending SIGKILL"
    kill -KILL "${runner_pid}" 2>/dev/null || true
    wait_for_pid_exit "${runner_pid}" 10 || true
fi

rm -f "${RUNNER_PID_FILE}" "${RUNNER_BOOT_ID_FILE}" "${SERVER_PID_FILE}" "${CLIENT_PID_FILE}"
log "autorun stop sequence finished"
