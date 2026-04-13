"""Foot-pedal keyboard-event listener via Linux evdev.

Reusable background listener that reads key-down events from a LinTx USB
foot pedal (or any HID keyboard) and dispatches them to a callback.

Unlike pynput or terminal stdin, this path reads /dev/input/event* directly,
so it works in fully headless SSH / tmux / systemd sessions with no TTY,
no X server, and no window focus. Any process that can open the event node
receives key events regardless of who "owns" the foreground.

Defaults target the two LinTx pedals on zhaobo-4090-1 (serial-based
``/dev/input/by-id`` symlinks, stable across USB re-plug). Override via the
``devices`` argument to adapt to other machines.

Requirements:
    - ``pip install evdev``
    - Read permission on the event device. Managed via udev rule:
      ``GROUP="plugdev", MODE="0660"`` on the LinTx vendor/product.
"""
from __future__ import annotations

import logging
import os
import select
import threading
from collections.abc import Callable

log = logging.getLogger(__name__)

# Stable symlinks for the LinTx pedals on zhaobo-4090-1.
# Pedal A is remapped via udev (KEYBOARD_KEY_7002c=r) to emit KEY_R;
# Pedal B stays on KEY_SPACE. Both devices expose input interface 1 as the keyboard HID.
DEFAULT_PEDAL_DEVICES: tuple[str, ...] = (
    "/dev/input/by-id/usb-LinTx_LinTx_Keyboard_BE1072C8-if01-event-kbd",
    "/dev/input/by-id/usb-LinTx_LinTx_Keyboard_BE136B2F-if01-event-kbd",
)


class PedalListener:
    """Background-thread listener that reports foot-pedal key-downs via callback.

    Usage::

        def on_pedal(key: str) -> None:
            if key == "r": ...
            elif key == "space": ...

        listener = PedalListener(on_pedal)
        if listener.start():
            ...  # runs in a daemon thread until stop() or process exit

    The callback receives the lowercase key name ("r", "space", ...). Only
    key-down events are delivered. Exceptions raised inside the callback are
    logged but do not kill the listener thread, since dropping pedal events
    silently is worse than a visible traceback during a recording session.
    """

    def __init__(
        self,
        on_press: Callable[[str], None],
        devices: tuple[str, ...] | list[str] | None = None,
        key_map: dict[int, str] | None = None,
    ) -> None:
        self._on_press = on_press
        self._device_paths: tuple[str, ...] = (
            tuple(devices) if devices is not None else DEFAULT_PEDAL_DEVICES
        )
        # Import deferred to avoid hard dependency at module import time.
        from evdev import ecodes

        self._key_map: dict[int, str] = key_map or {
            ecodes.KEY_R: "r",
            ecodes.KEY_SPACE: "space",
        }
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._devices: list = []

    def start(self) -> bool:
        """Open configured devices and start the background thread.

        Returns True if at least one device was opened and the thread started,
        False if no devices are present or readable (the caller can treat the
        listener as a no-op in that case).
        """
        from evdev import InputDevice

        for path in self._device_paths:
            if not os.path.exists(path):
                log.info("Pedal device not found, skipping: %s", path)
                continue
            if not os.access(path, os.R_OK):
                log.warning(
                    "Pedal device not readable (check udev rule for plugdev:0660): %s",
                    path,
                )
                continue
            self._devices.append(InputDevice(path))
            log.info("Pedal listener attached: %s", path)

        if not self._devices:
            log.info("No pedals available; PedalListener.start() is a no-op")
            return False

        self._thread = threading.Thread(
            target=self._run, name="pedal-listener", daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Request the listener thread to exit and close devices."""
        self._stop_event.set()
        for dev in self._devices:
            dev.close()
        self._devices = []

    def _run(self) -> None:
        from evdev import ecodes

        fd_to_dev = {dev.fd: dev for dev in self._devices}
        while not self._stop_event.is_set():
            readable, _, _ = select.select(list(fd_to_dev), [], [], 0.2)
            for fd in readable:
                for event in fd_to_dev[fd].read():
                    if event.type != ecodes.EV_KEY or event.value != 1:
                        continue
                    key_name = self._key_map.get(event.code)
                    if key_name is None:
                        continue
                    try:
                        self._on_press(key_name)
                    except Exception:
                        log.exception(
                            "PedalListener callback raised on key '%s'; "
                            "listener continues",
                            key_name,
                        )
