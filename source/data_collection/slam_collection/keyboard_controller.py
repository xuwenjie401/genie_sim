"""Isaac GUI keyboard controls for Galbot omni-base teleop."""

from __future__ import annotations

import threading

from slam_collection.base_motion import BaseCommand


class KeyboardBaseController:
    """Track WASD/QE state from the active Isaac window."""

    def __init__(self) -> None:
        self._pressed = set()
        self._actions = []
        self._quit_requested = False
        self._subscription = None
        self._keyboard = None
        self._iface = None
        self._lock = threading.Lock()

    def start(self) -> None:
        import carb.input
        import omni.appwindow

        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            raise RuntimeError("Isaac app window is not available; keyboard teleop requires GUI mode")

        self._keyboard = app_window.get_keyboard()
        self._iface = carb.input.acquire_input_interface()
        key_enum = carb.input.KeyboardInput
        action_keys = {
            key_enum.B: "record_start",
            key_enum.N: "record_stop",
            key_enum.H: "empty_move",
            key_enum.J: "move_grasp",
            key_enum.K: "move_place",
            key_enum.U: "gripper_open",
            key_enum.I: "gripper_close",
            key_enum.L: "lift",
            key_enum.O: "reset_arm",
        }

        def on_key(event, *_args):
            with self._lock:
                if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                    is_new_press = event.input not in self._pressed
                    self._pressed.add(event.input)
                    if event.input in (key_enum.ESCAPE, key_enum.X):
                        self._quit_requested = True
                    if is_new_press and event.input in action_keys:
                        self._actions.append(action_keys[event.input])
                elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                    self._pressed.discard(event.input)
            return True

        self._subscription = self._iface.subscribe_to_keyboard_events(self._keyboard, on_key)

    def stop(self) -> None:
        if self._subscription is None or self._iface is None or self._keyboard is None:
            return
        try:
            self._iface.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
        finally:
            self._subscription = None
            self._keyboard = None
            self._iface = None

    def command(self) -> BaseCommand:
        import carb.input

        key = carb.input.KeyboardInput
        with self._lock:
            pressed = set(self._pressed)
            quit_requested = self._quit_requested

        forward = 0.0
        if key.W in pressed:
            forward += 1.0
        if key.S in pressed:
            forward -= 1.0

        strafe = 0.0
        if key.A in pressed:
            strafe += 1.0
        if key.D in pressed:
            strafe -= 1.0

        yaw = 0.0
        if key.Q in pressed:
            yaw += 1.0
        if key.E in pressed:
            yaw -= 1.0

        vertical = 0.0
        if key.R in pressed:
            vertical += 1.0
        if key.F in pressed:
            vertical -= 1.0

        return BaseCommand(
            forward=forward,
            strafe=strafe,
            yaw=yaw,
            vertical=vertical,
            brake=key.SPACE in pressed,
            quit=quit_requested,
        )

    def consume_actions(self) -> list[str]:
        with self._lock:
            actions = list(self._actions)
            self._actions.clear()
        return actions
