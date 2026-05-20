"""pick_ui_node — debugging UI for cup-pick integration testing.

Subscribes to /digital_twin/boxes (MarkerArray published by point_cloud_node),
extracts per-cup IDs and top-world positions, and provides a minimal tkinter
window to select a cup and trigger a real robot pick.

Pick action: POSTs the selected cup's TOP-centre world position to the YARR
robot skill API (POST /api/robot/skill/pick). The HTTP call runs on a daemon
thread so the UI never blocks; the outcome is surfaced in the status line and
the ROS console.

Coordinate / field mapping:
  * /digital_twin/boxes is published in the `world` frame, which
    world_origin_node aligns to the robot base — i.e. world == base_link.
    The API expects base_link metres, so positions are sent as-is.
  * The `box_top` marker is the cup TOP centre (the grasp target). The API
    `cup_bottom_z` field is the cup BOTTOM (server adds a grip offset down),
    which is NOT what we have. We send the raw gripper Z via the `z` field
    instead. `ori` is omitted → server defaults the gripper to point down.

Prerequisites (tkinter):
  sudo apt install python3-tk

Usage:
  ros2 run depth_digital_twin pick_ui_node
  ros2 run depth_digital_twin pick_ui_node --ros-args \\
      -p boxes_topic:=/digital_twin/boxes_exo \\
      -p trigger_scan_service:=/point_cloud_node/trigger_scan \\
      -p pick_api_url:=https://yarr-api.simplyimg.com/api/robot/skill/pick
"""
from __future__ import annotations

import json
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from tkinter import ttk

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray


class PickUiNode(Node):
    def __init__(self) -> None:
        super().__init__('pick_ui_node')
        self.declare_parameter('boxes_topic', '/digital_twin/boxes')
        self.declare_parameter('trigger_scan_service',
                               '/point_cloud_node/trigger_scan')
        self.declare_parameter(
            'pick_api_url',
            'https://yarr-api.simplyimg.com/api/robot/skill/pick')
        self.declare_parameter('pick_api_timeout_s', 15.0)

        boxes_topic = str(self.get_parameter('boxes_topic').value)
        trigger_svc = str(self.get_parameter('trigger_scan_service').value)
        self._pick_api_url = str(self.get_parameter('pick_api_url').value)
        self._pick_timeout = float(
            self.get_parameter('pick_api_timeout_s').value)

        self._cups: dict[int, dict] = {}  # tid → {'pos': (x,y,z), 'label': str, 'locked': bool}
        self._lock = threading.Lock()

        # Async message from a pick worker thread → surfaced by the UI loop.
        self._ui_msg_lock = threading.Lock()
        self._ui_msg: tuple[str, str] | None = None
        self._pick_inflight = False

        self.create_subscription(MarkerArray, boxes_topic, self._on_boxes, 10)
        self._scan_client = self.create_client(Trigger, trigger_svc)

        self.get_logger().info(
            f'pick_ui_node ready  boxes={boxes_topic}  scan_svc={trigger_svc}  '
            f'pick_api={self._pick_api_url}')

    # ── ROS callbacks ──────────────────────────────────────────────────────

    def _on_boxes(self, msg: MarkerArray) -> None:
        with self._lock:
            for m in msg.markers:
                if m.action == Marker.DELETEALL:
                    self._cups.clear()
                    continue
                if m.action == Marker.DELETE:
                    if m.ns in ('box_top', 'boxes'):
                        self._cups.pop(m.id, None)
                    continue
                # box_top sphere = top-centre world position (the pick target)
                if m.ns == 'box_top':
                    self._cups.setdefault(m.id, {})['pos'] = (
                        float(m.pose.position.x),
                        float(m.pose.position.y),
                        float(m.pose.position.z),
                    )
                # box_labels text = full label string (includes [L] if locked)
                elif m.ns == 'box_labels':
                    self._cups.setdefault(m.id, {})['label'] = m.text
                    self._cups[m.id]['locked'] = m.text.startswith('[L]')

    # ── Public API (called from UI thread) ────────────────────────────────

    def get_cups(self) -> dict[int, dict]:
        with self._lock:
            return {tid: dict(v) for tid, v in self._cups.items()}

    def set_ui_message(self, text: str, colour: str) -> None:
        """Thread-safe: hand a status message to the UI loop (worker → UI)."""
        with self._ui_msg_lock:
            self._ui_msg = (text, colour)

    def pop_ui_message(self) -> tuple[str, str] | None:
        with self._ui_msg_lock:
            msg, self._ui_msg = self._ui_msg, None
        return msg

    def pick(self, tid: int) -> None:
        if self._pick_inflight:
            self.get_logger().warn('[PICK] a pick is already in progress')
            self.set_ui_message('⚠ pick already in progress', 'orange')
            return
        cups = self.get_cups()
        if tid not in cups:
            self.get_logger().warn(f'[PICK] cup #{tid} not in current detections')
            self.set_ui_message(f'⚠ #{tid} no longer detected', 'orange')
            return
        cup = cups[tid]
        pos = cup.get('pos', (0.0, 0.0, 0.0))
        locked = cup.get('locked', False)
        # box_top = cup TOP centre in `world` (== base_link). Send as raw
        # gripper Z (`z`), NOT cup_bottom_z (which is the cup's bottom).
        payload = {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])}
        self.get_logger().info(
            f'[PICK] cup #{tid} locked={locked} '
            f'top_world=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) '
            f'→ POST {self._pick_api_url} {payload}')
        self._pick_inflight = True
        threading.Thread(
            target=self._pick_worker, args=(tid, payload), daemon=True).start()

    def _pick_worker(self, tid: int, payload: dict) -> None:
        """Daemon-thread HTTP POST to the robot skill-pick API."""
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self._pick_api_url, data=data, method='POST',
                headers={'Content-Type': 'application/json',
                         'Accept': 'application/json'})
            with urllib.request.urlopen(
                    req, timeout=self._pick_timeout) as resp:
                body = resp.read().decode('utf-8', 'replace')
            result = json.loads(body) if body.strip() else {}
            success = bool(result.get('success', False))
            skill = result.get('skill', '')
            detail = result.get('detail', '')
            if success:
                self.get_logger().info(
                    f'[PICK] #{tid} OK skill={skill!r} detail={detail!r}')
                self.set_ui_message(f'✓ PICK #{tid} done', '#2a7a2a')
            else:
                self.get_logger().warn(
                    f'[PICK] #{tid} API success=false '
                    f'skill={skill!r} detail={detail!r}')
                self.set_ui_message(
                    f'✗ PICK #{tid} failed: {detail or "success=false"}',
                    '#cc0000')
        except urllib.error.HTTPError as e:
            msg = e.read().decode('utf-8', 'replace')[:200]
            self.get_logger().error(f'[PICK] #{tid} HTTP {e.code}: {msg}')
            self.set_ui_message(f'✗ PICK #{tid} HTTP {e.code}', '#cc0000')
        except urllib.error.URLError as e:
            self.get_logger().error(
                f'[PICK] #{tid} request failed: {e.reason}')
            self.set_ui_message(f'✗ PICK #{tid} API unreachable', '#cc0000')
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'[PICK] #{tid} error: {e!r}')
            self.set_ui_message(f'✗ PICK #{tid} error', '#cc0000')
        finally:
            self._pick_inflight = False

    def trigger_scan(self) -> None:
        if not self._scan_client.wait_for_service(timeout_sec=0.3):
            self.get_logger().warn('[RESCAN] trigger_scan service not available')
            return
        self._scan_client.call_async(Trigger.Request())
        self.get_logger().info('[RESCAN] trigger_scan called')


# ── tkinter UI ─────────────────────────────────────────────────────────────

class PickUI:
    _REFRESH_MS = 300

    def __init__(self, node: PickUiNode) -> None:
        self.node = node
        self._cup_ids: list[int] = []
        # Sticky status: a held message overrides the generic detection line
        # until it expires (so async pick results stay readable).
        self._held: tuple[str, str] | None = None
        self._hold_until: float = 0.0

        self.root = tk.Tk()
        self.root.title('Cup Pick UI  [debug]')
        self.root.resizable(False, False)

        frm = ttk.Frame(self.root, padding=12)
        frm.grid(row=0, column=0, sticky='nsew')

        # ── Header ────────────────────────────────────────────────────────
        ttk.Label(frm, text='Detected cups', font=('Helvetica', 12, 'bold')).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 6))

        # ── Cup list ──────────────────────────────────────────────────────
        self.listbox = tk.Listbox(
            frm, width=60, height=10, font=('Courier', 10),
            selectmode=tk.SINGLE, activestyle='dotbox')
        self.listbox.grid(row=1, column=0, columnspan=2, sticky='ew')

        sb = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=self.listbox.yview)
        sb.grid(row=1, column=2, sticky='ns')
        self.listbox.config(yscrollcommand=sb.set)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_frm = ttk.Frame(frm)
        btn_frm.grid(row=2, column=0, columnspan=3, pady=(8, 0), sticky='ew')
        btn_frm.columnconfigure(0, weight=1)
        btn_frm.columnconfigure(1, weight=1)

        self.pick_btn = ttk.Button(
            btn_frm, text='▶  Pick selected',
            command=self._on_pick)
        self.pick_btn.grid(row=0, column=0, padx=(0, 4), sticky='ew')

        self.scan_btn = ttk.Button(
            btn_frm, text='⟳  Re-scan',
            command=self._on_rescan)
        self.scan_btn.grid(row=0, column=1, padx=(4, 0), sticky='ew')

        # ── Status ────────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value='waiting for detections…')
        self.status_lbl = ttk.Label(
            frm, textvariable=self.status_var, foreground='gray',
            font=('Helvetica', 9))
        self.status_lbl.grid(row=3, column=0, columnspan=3, sticky='w', pady=(6, 0))

        self.root.after(self._REFRESH_MS, self._refresh)

    # ── Internal ──────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        cups = self.node.get_cups()

        # Preserve current selection across refresh
        sel = self.listbox.curselection()
        prev_tid = self._cup_ids[sel[0]] if sel else None

        self.listbox.delete(0, tk.END)
        self._cup_ids = sorted(cups.keys())

        for tid in self._cup_ids:
            cup = cups[tid]
            pos = cup.get('pos', (0.0, 0.0, 0.0))
            locked = cup.get('locked', False)
            lock_tag = '[L]' if locked else '   '
            self.listbox.insert(
                tk.END,
                f'{lock_tag} #{tid:3d}  '
                f'({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})')

        # Restore selection
        if prev_tid is not None and prev_tid in cups:
            idx = self._cup_ids.index(prev_tid)
            self.listbox.selection_set(idx)
            self.listbox.see(idx)

        # Async pick result from the worker thread takes over the status line.
        msg = self.node.pop_ui_message()
        if msg is not None:
            self._held = msg
            self._hold_until = time.monotonic() + 6.0

        n = len(cups)
        n_locked = sum(1 for c in cups.values() if c.get('locked'))
        if self._held is not None and time.monotonic() < self._hold_until:
            self.status_var.set(self._held[0])
            self.status_lbl.config(foreground=self._held[1])
        elif n:
            self.status_var.set(f'{n} cup(s) detected  |  {n_locked} locked')
            self.status_lbl.config(foreground='#2a7a2a')
        else:
            self.status_var.set('no cups detected')
            self.status_lbl.config(foreground='gray')

        self.root.after(self._REFRESH_MS, self._refresh)

    def _on_pick(self) -> None:
        sel = self.listbox.curselection()
        if not sel:
            self._hold_status('⚠ select a cup first', 'orange', 3.0)
            return
        tid = self._cup_ids[sel[0]]
        cups = self.node.get_cups()
        pos = cups.get(tid, {}).get('pos', (0.0, 0.0, 0.0))
        locked = cups.get(tid, {}).get('locked', False)
        # Hold past the request timeout; the worker's result resets the hold.
        self._hold_status(
            f'→ requesting PICK #{tid}  '
            f'({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  locked={locked}',
            '#0055cc', self.node._pick_timeout + 2.0)
        self.node.pick(tid)

    def _on_rescan(self) -> None:
        self._hold_status('⟳ re-scan triggered', '#884400', 3.0)
        self.node.trigger_scan()

    def _hold_status(self, text: str, colour: str, secs: float) -> None:
        """Show `text` and keep it for `secs` (overrides the detection line)."""
        self._held = (text, colour)
        self._hold_until = time.monotonic() + secs
        self.status_var.set(text)
        self.status_lbl.config(foreground=colour)

    def run(self) -> None:
        self.root.mainloop()


# ── Entry point ────────────────────────────────────────────────────────────

def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PickUiNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    ui = PickUI(node)
    try:
        ui.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
