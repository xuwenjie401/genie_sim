# Autorun Data Collection

Files in this directory:

- `autorun_config.sh`: edit the task path and startup delays here.
- `run_autorun_data_collection.sh`: main autorun entrypoint.
- `stop_autorun_data_collection.sh`: graceful stop helper.
- `genie-sim-autorun.service`: user `systemd` service file.
- `install_user_service.sh`: installs the service into `~/.config/systemd/user/`.

Current defaults:

- task template: `tasks/diy/meta_task/galbot_meta_pick_place_V1.json`
- sleep before server: `60s`
- sleep before client: `60s`
- archive target: `/home/agxi/Datasets/galbot_sim/raw`

Behavior on each start:

1. Archive any existing content under `recording_data/` into
   `autorun_<task-name>_<mmdd>_<HHMM>/`.
2. Sleep `60s`.
3. Start the server with the same Conda and ROS environment as the manual workflow.
4. Sleep `60s`.
5. Start the client with the configured `--task_template`.

Recommended installation:

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection/scripts/autorun
./install_user_service.sh
sudo loginctl enable-linger agxi
```

Manual control:

```bash
systemctl --user start genie-sim-autorun.service
systemctl --user stop genie-sim-autorun.service
systemctl --user status genie-sim-autorun.service
journalctl --user -u genie-sim-autorun.service -f
```
