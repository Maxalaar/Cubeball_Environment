import atexit
import json
import socket
import subprocess

MAJOR_VERSION = "0"
MINOR_VERSION = "7"


class CubeballConnection:
    # A small, self-contained client for the training wire protocol implemented by
    # addons/godot_rl_agents/agent_synchronizer.gd in the Godot project: JSON messages,
    # each prefixed by its length as 4 little-endian bytes. Every episode (including
    # the first) is a "reset" message carrying a config payload — Godot doesn't build
    # anything until it receives one, so there's exactly one way this ever talks to it.
    def __init__(
        self,
        env_path: str,
        port: int,
        show_window: bool,
        action_repeat: int,
        speedup: float,
        debug_logs: bool = False,
        seed: int = 0,
    ):
        launch_command = [
            env_path,
            f"--port={port}",
            f"--env_seed={seed}",
            f"--action_repeat={action_repeat}",
            f"--speedup={speedup}",
            f"--debug_logs={'true' if debug_logs else 'false'}",
        ]
        if not show_window:
            launch_command += ["--disable-render-loop", "--headless"]

        self.process = subprocess.Popen(launch_command, start_new_session=True)
        atexit.register(self.close)

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("127.0.0.1", port))
        server_socket.listen(1)
        server_socket.settimeout(60)
        self.connection, _ = server_socket.accept()

        self._send({"type": "handshake", "major_version": MAJOR_VERSION, "minor_version": MINOR_VERSION})

    def reset(self, config: dict) -> dict:
        self._send({"type": "reset", "config": config})
        return self._receive()

    def step(self, actions: dict) -> dict:
        self._send({"type": "action", "action": actions})
        return self._receive()

    def close(self) -> None:
        self._send({"type": "close"})
        self.connection.close()
        atexit.unregister(self.close)

    def _send(self, message: dict) -> None:
        payload = json.dumps(message).encode()
        self.connection.sendall(len(payload).to_bytes(4, "little") + payload)

    def _receive(self) -> dict:
        length = int.from_bytes(self._receive_exactly(4), "little")
        return json.loads(self._receive_exactly(length))

    def _receive_exactly(self, size: int) -> bytes:
        buffer = bytearray()
        while len(buffer) < size:
            buffer.extend(self.connection.recv(size - len(buffer)))
        return bytes(buffer)


def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port
