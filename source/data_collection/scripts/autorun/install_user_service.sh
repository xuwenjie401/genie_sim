#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="genie-sim-autorun.service"
USER_SERVICE_DIR="${HOME}/.config/systemd/user"

mkdir -p "${USER_SERVICE_DIR}"
cp "${SCRIPT_DIR}/${SERVICE_NAME}" "${USER_SERVICE_DIR}/${SERVICE_NAME}"

systemctl --user daemon-reload

if [[ "${1:-}" == "--now" ]]; then
    systemctl --user enable --now "${SERVICE_NAME}"
else
    systemctl --user enable "${SERVICE_NAME}"
fi

cat <<EOF
Installed ${SERVICE_NAME} to ${USER_SERVICE_DIR}.

Useful commands:
  systemctl --user start ${SERVICE_NAME}
  systemctl --user stop ${SERVICE_NAME}
  systemctl --user status ${SERVICE_NAME}
  journalctl --user -u ${SERVICE_NAME} -f

To make the user service start at boot before login:
  sudo loginctl enable-linger $(whoami)
EOF
