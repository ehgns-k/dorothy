"""
Minimal, robust driver for the Feetech STS3215 servos of the SO-101 arm.

Replaces the (very heavy) lerobot dependency: the game only ever needs raw
tick positions, so we talk to the bus directly through `scservo_sdk`
(pip package `feetech-servo-sdk`). All reads are retried; transient serial
glitches never surface as crashes.
"""

import threading
import time
from typing import Dict, Iterable, Optional

import scservo_sdk as scs

# STS3215 control table (SRAM area - nothing here touches EEPROM).
ADDR_TORQUE_ENABLE = 40
ADDR_ACCELERATION = 41
ADDR_GOAL_POSITION = 42
ADDR_TORQUE_LIMIT = 48
ADDR_OVERLOAD_TORQUE = 36       # % of max: output above this for Protection_Time trips the servo
ADDR_PROTECTION_TIME = 35       # x10 ms
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_LOAD = 60
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMPERATURE = 63
ADDR_MOVING = 66

ERRBIT_OVERLOAD = 32

READ_RETRIES = 4


class ServoError(RuntimeError):
    pass


def _decode_sign_magnitude(v: int, sign_bit: int = 15) -> int:
    if v & (1 << sign_bit):
        return -(v & ~(1 << sign_bit))
    return v


class FeetechBus:
    def __init__(self, port: str, motors: Dict[str, int], baudrate: int = 1_000_000):
        self.port_name = port
        self.motors = dict(motors)
        self.baudrate = baudrate
        self._lock = threading.RLock()
        self._port = scs.PortHandler(port)
        self._ph = scs.PacketHandler(0)  # STS series = protocol "end" 0
        self._connected = False
        self._locked_out = False   # after go_limp(): refuse all motion writes

    # ------------------------------------------------------------------ #
    def go_limp(self) -> None:
        """Torque off on every motor and refuse further position writes (so
        a still-running motion loop can't re-energise the arm)."""
        self._locked_out = True
        with self._lock:
            if not self._connected:
                return
            for name, mid in self.motors.items():
                for _ in range(3):
                    res, _ = self._ph.write1ByteTxRx(self._port, mid, ADDR_TORQUE_ENABLE, 0)
                    if res == scs.COMM_SUCCESS:
                        break

    def connect(self) -> None:
        with self._lock:
            if not self._port.openPort():
                raise ServoError(
                    f"Cannot open {self.port_name}. Is the arm plugged in and is "
                    "your user in the 'dialout' group?")
            if not self._port.setBaudRate(self.baudrate):
                raise ServoError(f"Cannot set baudrate {self.baudrate}")
            self._connected = True
        missing = [n for n, i in self.motors.items() if not self.ping(i)]
        if missing:
            self.disconnect()
            raise ServoError(f"No response from motors: {missing}. "
                             "Check the arm's power supply and cable.")

    def disconnect(self, disable_torque: bool = False) -> None:
        with self._lock:
            if not self._connected:
                return
            if disable_torque:
                try:
                    self.set_torque(False)
                except Exception:
                    pass
            try:
                self._port.closePort()
            finally:
                self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def ping(self, motor_id: int) -> bool:
        with self._lock:
            for _ in range(3):
                _, res, _ = self._ph.ping(self._port, motor_id)
                if res == scs.COMM_SUCCESS:
                    return True
                time.sleep(0.02)
        return False

    # ------------------------------------------------------------------ #
    def _ids(self, names: Optional[Iterable[str]]) -> Dict[str, int]:
        if names is None:
            return self.motors
        if isinstance(names, str):
            names = [names]
        return {n: self.motors[n] for n in names}

    def _write(self, name: str, addr: int, value: int, nbytes: int) -> int:
        """Write with retries. Returns the servo's error byte."""
        mid = self.motors[name]
        fn = self._ph.write1ByteTxRx if nbytes == 1 else self._ph.write2ByteTxRx
        last = None
        with self._lock:
            for _ in range(READ_RETRIES):
                res, err = fn(self._port, mid, addr, int(value))
                if res == scs.COMM_SUCCESS:
                    return err
                last = self._ph.getTxRxResult(res)
                time.sleep(0.005)
        raise ServoError(f"write {name}@{addr} failed: {last}")

    def set_torque(self, enabled: bool, names: Optional[Iterable[str]] = None) -> None:
        if enabled and self._locked_out:
            raise ServoError("arm is limp (shutting down)")
        for n in self._ids(names):
            self._write(n, ADDR_TORQUE_ENABLE, 1 if enabled else 0, 1)

    def read_register(self, name: str, addr: int, nbytes: int = 1) -> int:
        """Read one control-table register of one motor (no sign decoding)."""
        return self._sync_read(addr, nbytes, [name])[name]

    def set_torque_limit(self, name: str, limit: int) -> None:
        self._write(name, ADDR_TORQUE_LIMIT, max(0, min(1000, int(limit))), 2)

    def write_position(self, name: str, ticks: int) -> int:
        """Single-motor goal write. Returns the status error byte so the
        caller can react to an Overload latch."""
        if self._locked_out:
            raise ServoError("arm is limp (shutting down)")
        return self._write(name, ADDR_GOAL_POSITION, max(0, min(4095, int(ticks))), 2)

    def sync_write_positions(self, goals: Dict[str, int]) -> None:
        if self._locked_out:
            raise ServoError("arm is limp (shutting down)")
        with self._lock:
            gw = scs.GroupSyncWrite(self._port, self._ph, ADDR_GOAL_POSITION, 2)
            for n, v in goals.items():
                v = max(0, min(4095, int(v)))
                gw.addParam(self.motors[n], [scs.SCS_LOBYTE(v), scs.SCS_HIBYTE(v)])
            res = gw.txPacket()
        if res != scs.COMM_SUCCESS:
            raise ServoError(f"sync write failed: {self._ph.getTxRxResult(res)}")

    def _sync_read(self, addr: int, nbytes: int, names=None) -> Dict[str, int]:
        ids = self._ids(names)
        last = None
        for _ in range(READ_RETRIES):
            with self._lock:
                gr = scs.GroupSyncRead(self._port, self._ph, addr, nbytes)
                for mid in ids.values():
                    gr.addParam(mid)
                res = gr.txRxPacket()
                if res == scs.COMM_SUCCESS and all(
                        gr.isAvailable(mid, addr, nbytes) for mid in ids.values()):
                    return {n: gr.getData(mid, addr, nbytes) for n, mid in ids.items()}
                last = self._ph.getTxRxResult(res)
                self._port.clearPort()
            time.sleep(0.01)
        raise ServoError(f"sync read @{addr} failed: {last}")

    def read_positions(self, names=None) -> Dict[str, int]:
        return {n: _decode_sign_magnitude(v)
                for n, v in self._sync_read(ADDR_PRESENT_POSITION, 2, names).items()}

    def read_position_load(self, name: str):
        """(present_position, present_load) of one motor.
        Load is signed, in 0.1 % of max torque (sign = direction)."""
        pos = self._sync_read(ADDR_PRESENT_POSITION, 2, [name])[name]
        load = self._sync_read(ADDR_PRESENT_LOAD, 2, [name])[name]
        return _decode_sign_magnitude(pos), _decode_sign_magnitude(load, 10)

    def read_temperatures(self, names=None) -> Dict[str, int]:
        return self._sync_read(ADDR_PRESENT_TEMPERATURE, 1, names)

    def read_voltage(self, name: str = "shoulder_pan") -> float:
        return self._sync_read(ADDR_PRESENT_VOLTAGE, 1, [name])[name] / 10.0
