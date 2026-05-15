"""Websocket client for pi0/openpi-style policy servers."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any
import json
import time

import msgpack
import numpy as np
import websockets.sync.client


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _packb(value: Any) -> bytes:
    return msgpack.packb(value, default=_pack_array)


def _unpackb(value: bytes) -> Any:
    return msgpack.unpackb(value, object_hook=_unpack_array)


class _WebsocketPolicyClient:
    """Minimal local client matching geniesim/openpi websocket policy protocol."""

    def __init__(self, host: str, port: int):
        self.uri = f"ws://{host}:{port}"
        self.ws = None
        self.server_metadata = None

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.ws is None:
            self.ws = websockets.sync.client.connect(self.uri, compression=None, max_size=None)
            self.server_metadata = _unpackb(self.ws.recv())
        self.ws.send(_packb(payload))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return _unpackb(response)

    def close(self) -> None:
        if self.ws is not None:
            self.ws.close()
            self.ws = None


class Pi0PolicyClient:
    def __init__(self, host: str, port: int, timeout_sec: float, log_path: str | Path):
        self.client = _WebsocketPolicyClient(host=host, port=port)
        self.timeout_sec = timeout_sec
        self.action_buffer: deque[np.ndarray] = deque()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("a", encoding="utf-8")
        self.infer_count = 0

    def close(self) -> None:
        self.client.close()
        if not self._log_file.closed:
            self._log_file.close()

    def reset(self) -> None:
        self.action_buffer.clear()

    def act(self, payload: dict[str, Any], step: int) -> np.ndarray:
        if not self.action_buffer:
            self._infer(payload, step)
        return np.asarray(self.action_buffer.popleft(), dtype=np.float32)

    def _infer(self, payload: dict[str, Any], step: int) -> None:
        start = time.time()
        last_error = None
        while True:
            try:
                result = self.client.infer(payload)
                actions = _dict_get(result, "actions", [])
                for action in actions:
                    self.action_buffer.append(np.asarray(action, dtype=np.float32))
                self._write_log(step=step, payload=payload, result=result, latency_sec=time.time() - start)
                self.infer_count += 1
                if not self.action_buffer:
                    raise RuntimeError("Policy server returned no actions")
                return
            except Exception as exc:  # noqa: BLE001 - policy server failures are retried.
                last_error = exc
                if time.time() - start > self.timeout_sec:
                    raise TimeoutError(f"Policy inference timed out after {self.timeout_sec}s: {last_error}") from exc
                time.sleep(1.0)

    def _write_log(self, step: int, payload: dict[str, Any], result: dict[str, Any], latency_sec: float) -> None:
        state = np.asarray(payload.get("state", []), dtype=np.float32)
        actions = np.asarray(_dict_get(result, "actions", []), dtype=np.float32)
        images = payload.get("images", {})
        event = {
            "time": time.time(),
            "step": step,
            "latency_sec": latency_sec,
            "prompt": payload.get("prompt", ""),
            "state": state.tolist(),
            "image_shapes": {key: list(np.asarray(value).shape) for key, value in images.items()},
            "actions": actions.tolist(),
        }
        self._log_file.write(json.dumps(event) + "\n")
        self._log_file.flush()


def _dict_get(value: dict[Any, Any], key: str, default: Any = None) -> Any:
    return value.get(key, value.get(key.encode("utf-8"), default))
