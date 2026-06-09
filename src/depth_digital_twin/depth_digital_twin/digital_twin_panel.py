"""digital_twin_panel — one integrated Tkinter window for the dual-camera
(exo + hand) digital twin. Replaces the old per-node ArUco redetect popup
(world_origin_control).

Layout
------
  ┌───────────────────────────────────────────────┐
  │            ArUco Re-detect All (hand, exo)     │  ← row of buttons,
  │              [ ArUco Exo ] [ ArUco Hand ]      │     fixed size, centred
  ├──────────────┬──────────────┬──────────────────┤
  │   EXO RGB    │   EXO Depth  │     EXO 3D       │  ← 2 rows × 3 cols image
  ├──────────────┼──────────────┼──────────────────┤     grid, uniform 33%,
  │   HAND RGB   │  HAND Depth  │     HAND 3D      │     fits on resize
  └──────────────┴──────────────┴──────────────────┘

Each image cell scales its latest frame to FIT (letter-boxed, never cropped)
the cell's current size; the three columns stay a uniform third of the width
and the two image rows split the remaining height evenly, so a window resize
re-flows everything without distorting aspect ratios.

ROS runs on a background thread (rclpy spin); Tk owns the main thread. Image
callbacks stash the latest cv frame under a lock; a Tk `after` loop redraws.
Redetect buttons enqueue a service name that the ROS thread calls async.
"""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk

import cv2
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
import rclpy
from cv_bridge import CvBridge
from PIL import Image, ImageTk
from rclpy.node import Node
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import Image as RosImage
from std_srvs.srv import Trigger


# (row, col) → (title, kind). kind: 'color' | 'depth' | 'debug'.
_CELLS = [
    ('exo', 'color', 'EXO RGB'),
    ('exo', 'depth', 'EXO Depth'),
    ('exo', 'debug', 'EXO 3D'),
    ('hand', 'color', 'HAND RGB'),
    ('hand', 'depth', 'HAND Depth'),
    ('hand', 'debug', 'HAND 3D'),
]


class DigitalTwinPanel(Node):
    def __init__(self) -> None:
        super().__init__('digital_twin_panel')
        gp = self.declare_parameter
        # Per-camera image topics (color, depth, "3D" debug overlay).
        gp('exo_color_topic', '/camera_exo/color/image_raw')
        gp('exo_depth_topic', '/camera_exo/aligned_depth_to_color/image_raw')
        gp('exo_debug_topic', '/digital_twin/detection_debug_exo')
        gp('hand_color_topic', '/camera_hand/color/image_raw')
        gp('hand_depth_topic', '/camera_hand/aligned_depth_to_color/image_raw')
        gp('hand_debug_topic', '/digital_twin/detection_debug_hand')
        # Redetect services (Trigger) for each camera's world_origin node.
        gp('exo_redetect_srv', '/world_origin_node_exo/redetect')
        gp('hand_redetect_srv', '/world_origin_node_hand/redetect')
        gp('fusion_node_name', 'cup_fusion_node')   # for checkboxes + tuning
        gp('tuning_save_dir', '')                   # '' → package share config dir

        def P(n):
            return str(self.get_parameter(n).value)

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._frames: dict[tuple, np.ndarray] = {}     # (cam, kind) -> BGR uint8
        self._req_q: queue.Queue = queue.Queue()       # redetect requests

        topics = {
            ('exo', 'color'): P('exo_color_topic'),
            ('exo', 'depth'): P('exo_depth_topic'),
            ('exo', 'debug'): P('exo_debug_topic'),
            ('hand', 'color'): P('hand_color_topic'),
            ('hand', 'depth'): P('hand_depth_topic'),
            ('hand', 'debug'): P('hand_debug_topic'),
        }
        for (cam, kind), topic in topics.items():
            self.create_subscription(
                RosImage, topic,
                lambda msg, c=cam, k=kind: self._on_image(msg, c, k), 1)

        self._srv = {
            'exo': self.create_client(Trigger, P('exo_redetect_srv')),
            'hand': self.create_client(Trigger, P('hand_redetect_srv')),
        }
        # Parameter clients (checkboxes + tuning sliders) — one per target node.
        self._param_q: queue.Queue = queue.Queue()
        self._param_clis: dict = {}
        self.fusion_node = P('fusion_node_name')
        # Scan&Lock clear: Trigger service on the fusion node (~/clear_scan).
        self._clear_q: queue.Queue = queue.Queue()
        self._clear_scan_cli = self.create_client(
            Trigger, '/' + self.fusion_node.strip('/') + '/clear_scan')
        # Default save dir = the REAL config dir. params.yaml in the install
        # share is symlink-installed to the source tree, so realpath() lands the
        # saved snapshots next to the source config (not the install copy).
        _cfg = os.path.join(
            get_package_share_directory('depth_digital_twin'), 'config')
        self.save_dir = P('tuning_save_dir') or os.path.dirname(
            os.path.realpath(os.path.join(_cfg, 'params.yaml')))
        # Drain redetect + param requests from the Tk thread on the ROS thread.
        self.create_timer(0.1, self._drain_requests)
        self.get_logger().info('digital_twin_panel ready')

    # ---- ROS side ----------------------------------------------------
    def _on_image(self, msg: RosImage, cam: str, kind: str) -> None:
        try:
            if kind == 'depth':
                d = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                bgr = self._colorize_depth(d)
            else:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'{cam}/{kind} convert failed: {e}',
                                   throttle_duration_sec=5.0)
            return
        with self._lock:
            self._frames[(cam, kind)] = bgr

    @staticmethod
    def _colorize_depth(d: np.ndarray) -> np.ndarray:
        d = np.nan_to_num(d).astype(np.float32)
        valid = d[d > 0]
        if valid.size:
            lo, hi = float(valid.min()), float(np.percentile(valid, 98))
        else:
            lo, hi = 0.0, 1.0
        norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        u8 = (norm * 255).astype(np.uint8)
        col = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        col[d <= 0] = (0, 0, 0)
        return col

    def request_redetect(self, who: str) -> None:
        self._req_q.put(who)

    def _client_for(self, node_name: str):
        cli = self._param_clis.get(node_name)
        if cli is None:
            cli = self.create_client(
                SetParameters, '/' + node_name.strip('/') + '/set_parameters')
            self._param_clis[node_name] = cli
        return cli

    def request_param(self, node_name: str, name: str, value,
                      ptype: str = 'double') -> None:
        self._param_q.put((node_name, name, value, ptype))

    def request_clear_scan(self) -> None:
        self._clear_q.put(True)

    def save_yaml(self, sections: dict) -> str:
        """sections: {node_name: {param: value}} → config/params_<ts>.yaml."""
        os.makedirs(self.save_dir, exist_ok=True)
        ts = datetime.now().strftime('%y%m%d_%H%M%S')
        path = os.path.join(self.save_dir, f'params_{ts}.yaml')
        doc = {n: {'ros__parameters': p} for n, p in sections.items()}
        with open(path, 'w') as f:
            yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)
        self.get_logger().info(f'saved tuning → {path}')
        return path

    def _drain_requests(self) -> None:
        while not self._param_q.empty():
            node_name, name, value, ptype = self._param_q.get()
            cli = self._client_for(node_name)
            if not cli.service_is_ready():
                self.get_logger().warn(f'{node_name} param service not ready')
                continue
            pv = ParameterValue()
            if ptype == 'bool':
                pv.type = ParameterType.PARAMETER_BOOL
                pv.bool_value = bool(value)
            elif ptype == 'integer':
                pv.type = ParameterType.PARAMETER_INTEGER
                pv.integer_value = int(value)
            else:
                pv.type = ParameterType.PARAMETER_DOUBLE
                pv.double_value = float(value)
            req = SetParameters.Request()
            req.parameters = [ParameterMsg(name=name, value=pv)]
            cli.call_async(req)
        while not self._req_q.empty():
            who = self._req_q.get()
            cams = ('exo', 'hand') if who == 'all' else (who,)
            for cam in cams:
                cli = self._srv[cam]
                if not cli.service_is_ready():
                    self.get_logger().warn(
                        f'{cam} redetect service not available yet')
                    continue
                fut = cli.call_async(Trigger.Request())
                fut.add_done_callback(
                    lambda f, c=cam: self.get_logger().info(
                        f'{c} redetect: '
                        f'{getattr(f.result(), "message", "?")}'))
        while not self._clear_q.empty():
            self._clear_q.get()
            if not self._clear_scan_cli.service_is_ready():
                self.get_logger().warn('clear_scan service not available yet')
                continue
            fut = self._clear_scan_cli.call_async(Trigger.Request())
            fut.add_done_callback(
                lambda f: self.get_logger().info(
                    f'clear_scan: {getattr(f.result(), "message", "?")}'))

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._frames)


class _Tooltip:
    """Hover tooltip for a widget."""

    def __init__(self, widget, text: str) -> None:
        self.widget, self.text, self.tip = widget, text, None
        widget.bind('<Enter>', self._show)
        widget.bind('<Leave>', self._hide)

    def _show(self, _e=None) -> None:
        if self.tip:
            return
        x = self.widget.winfo_rootx() + 24
        y = self.widget.winfo_rooty() + 22
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f'+{x}+{y}')
        tk.Label(self.tip, text=self.text, bg='#ffffe0', fg='#000',
                 justify='left', relief='solid', borderwidth=1,
                 font=('TkDefaultFont', 9), wraplength=340).pack()

    def _hide(self, _e=None) -> None:
        if self.tip:
            self.tip.destroy()
            self.tip = None


_F = 'cup_fusion_node'
_PE = 'point_cloud_node_exo'
_PH = 'point_cloud_node_hand'
# (display, node, param, kind, min, max, resolution, default, tooltip)
_TUNING = [
    ('merge_dist', _F, 'merge_dist_m', 'double', 0.0, 0.1, 0.001, 0.035,
     'STAGE-0 거친 클러스터 거리(m). ↓ 덜 합침'),
    ('max_cup_footprint', _F, 'max_cup_footprint_m', 'double', 0.0, 0.2, 0.005,
     0.11, '이보다 큰 박스는 거부(과병합/노이즈 차단). ↓ 더 엄격'),
    ('premerge_iou', _F, 'premerge_iou', 'double', 0.0, 1.0, 0.01, 0.62,
     '1차 병합 AABB-IoU 임계. ↑ 더 엄격(같은컵만 합침)'),
    ('premerge_dxy', _F, 'premerge_dxy_m', 'double', 0.0, 0.1, 0.001, 0.018,
     '1차 병합 중심 XY 게이트(m)'),
    ('premerge_dz', _F, 'premerge_dz_m', 'double', 0.0, 0.1, 0.001, 0.022,
     '1차 병합 중심 Z 게이트(m). 수직스택 오병합 차단'),
    ('postmerge_dxy', _F, 'postmerge_dxy_m', 'double', 0.0, 0.1, 0.001, 0.015,
     'fit 후 재병합 XY(m). ↑ 같은 컵 두 fit 더 합침'),
    ('postmerge_dz', _F, 'postmerge_dz_m', 'double', 0.0, 0.1, 0.001, 0.015,
     'fit 후 재병합 Z(m)'),
    ('assoc_gate_xy', _F, 'assoc_gate_xy_m', 'double', 0.0, 0.1, 0.001, 0.035,
     '트랙 연관 XY 타원게이트(m). ↑ hand+exo 더 합침/↓ 더 분리'),
    ('assoc_gate_z', _F, 'assoc_gate_z_m', 'double', 0.0, 0.1, 0.001, 0.018,
     '트랙 연관 Z 게이트(m). 층 간격보다 작게'),
    ('max_age', _F, 'max_age_s', 'double', 0.0, 3.0, 0.1, 1.5,
     '카메라 클라우드 유지 시간(s). ↑ 느린 exo 점 안 사라짐'),
    ('min_hits', _F, 'min_hits', 'integer', 1, 10, 1, 3,
     '렌더 전 연속 매칭 횟수. ↑ 깜빡임↓ 등장지연↑'),
    ('keepalive', _F, 'keepalive_ticks', 'integer', 1, 30, 1, 8,
     '트랙 유지 이벤트 수(검출 끊겨도 coast)'),
    ('w_hand_base', _F, 'w_hand_base', 'double', 0.0, 1.0, 0.05, 0.6,
     'hand 뷰 가중(병합 점 수 비율)'),
    ('w_exo_base', _F, 'w_exo_base', 'double', 0.0, 1.0, 0.05, 0.4,
     'exo 뷰 가중. ↓ exo 떨림 영향↓'),
    ('kf_gate', _F, 'kf_gate_mahalanobis', 'double', 0.0, 20.0, 0.5, 9.0,
     'KF 스파이크 거부. ↓ 강하게 거부(튀는 값 차단)'),
    ('kf_meas_xy', _F, 'kf_meas_std_xy_m', 'double', 0.0, 0.05, 0.001, 0.005,
     '측정 노이즈 XY(m). ↑ 더 평활(측정 덜 신뢰)'),
    ('kf_meas_z', _F, 'kf_meas_std_z_m', 'double', 0.0, 0.05, 0.001, 0.010,
     '측정 노이즈 Z(m)'),
    ('kf_proc_xy', _F, 'kf_process_std_xy_m', 'double', 0.0, 0.02, 0.0005, 0.002,
     '프로세스 노이즈 XY(m). ↓ 추정 더 단단'),
    ('kf_proc_z', _F, 'kf_process_std_z_m', 'double', 0.0, 0.02, 0.0005, 0.004,
     '프로세스 노이즈 Z(m)'),
    ('cup_resid_max', _F, 'cup_fit_residual_max', 'double', 0.0, 0.1, 0.002, 0.02,
     'cone fit 허용 잔차(m). ↑ OBB로 덜 빠짐'),
    ('exo depth_grad', _PE, 'depth_gradient_max_m', 'double', 0.0, 0.1, 0.005,
     0.06, 'exo 깊이경사 필터(m). ↑ 먼 점 살림(노이즈↑)/↓ 깨끗'),
    ('hand depth_grad', _PH, 'depth_gradient_max_m', 'double', 0.0, 0.1, 0.005,
     0.06, 'hand 깊이경사 필터(m)'),
]


class PanelUI:
    """Tkinter front-end driven by the ROS node above."""

    def __init__(self, node: DigitalTwinPanel) -> None:
        self.node = node
        self.root = tk.Tk()
        self.root.title('Digital Twin — Exo / Hand')
        self.root.geometry('1280x800')
        self.root.configure(bg='#1e1e1e')
        self._photo: dict[tuple, ImageTk.PhotoImage] = {}
        self.labels: dict[tuple, tk.Label] = {}

        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True)
        home = tk.Frame(nb, bg='#1e1e1e')
        tuning = tk.Frame(nb, bg='#1e1e1e')
        nb.add(home, text='  Home  ')
        nb.add(tuning, text='  Tuning  ')
        self._build_home(home)
        self._build_tuning(tuning)

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self._closing = False
        self._tick()

    # ---- HOME tab: ArUco buttons + debug checkboxes + image grid ----
    def _build_home(self, parent: tk.Frame) -> None:
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        bar = tk.Frame(parent, bg='#1e1e1e')
        bar.grid(row=0, column=0, sticky='ew', pady=6)
        bar.grid_columnconfigure(0, weight=1)
        bar.grid_columnconfigure(99, weight=1)
        BW = {'width': 24, 'height': 1}
        tk.Button(bar, text='ArUco Re-detect All (hand, exo)', **BW,
                  command=lambda: self.node.request_redetect('all')
                  ).grid(row=0, column=1, columnspan=2, padx=4, pady=2)
        tk.Button(bar, text='ArUco Exo', **BW,
                  command=lambda: self.node.request_redetect('exo')
                  ).grid(row=1, column=1, padx=4, pady=2)
        tk.Button(bar, text='ArUco Hand', **BW,
                  command=lambda: self.node.request_redetect('hand')
                  ).grid(row=1, column=2, padx=4, pady=2)

        dbg = tk.Frame(bar, bg='#1e1e1e')
        dbg.grid(row=2, column=1, columnspan=2, pady=(8, 2))
        tk.Label(dbg, text='Debug plot:', bg='#1e1e1e', fg='#9cdcfe').pack(
            side='left', padx=(0, 6))
        self._dbg_vars: dict[str, tk.BooleanVar] = {}
        for text, pname, default in (
                ('1 Detections', 'dbg_detections', False),
                ('2 Pre-merge', 'dbg_premerge', False),
                ('3 Fit-1', 'dbg_fit1', False),
                ('4 Fit-2 (final)', 'dbg_fit2', True)):
            var = tk.BooleanVar(value=default)
            self._dbg_vars[pname] = var
            tk.Checkbutton(
                dbg, text=text, variable=var, bg='#1e1e1e', fg='#dddddd',
                selectcolor='#333333', activebackground='#1e1e1e',
                activeforeground='#ffffff',
                command=lambda p=pname, v=var: self.node.request_param(
                    self.node.fusion_node, p, v.get(), 'bool')
            ).pack(side='left', padx=4)

        # Scan & Lock: multi-view accumulate-at-waypoints. Checkbox toggles the
        # cup_fusion scan_lock_active param; Clear Lock calls ~/clear_scan.
        scan = tk.Frame(bar, bg='#1e1e1e')
        scan.grid(row=3, column=1, columnspan=2, pady=(4, 2))
        self._scan_var = tk.BooleanVar(value=False)
        scan_cb = tk.Checkbutton(
            scan, text='Scan & Lock', variable=self._scan_var, bg='#1e1e1e',
            fg='#ffd479', selectcolor='#333333', activebackground='#1e1e1e',
            activeforeground='#ffffff',
            command=lambda: self.node.request_param(
                self.node.fusion_node, 'scan_lock_active',
                self._scan_var.get(), 'bool'))
        scan_cb.pack(side='left', padx=4)
        _Tooltip(scan_cb,
                 'scan_lock_active\n'
                 '체크: 스캔 웨이포인트에서 멀티뷰 누적 후 안정된 lock 발행.\n'
                 '해제: 캡처만 일시정지(lock은 유지).\n'
                 '완전 초기화는 [Clear Lock] 또는 skill_manager.')

        def _clear_lock():
            self._scan_var.set(False)
            self.node.request_clear_scan()
        clr_btn = tk.Button(scan, text='Clear Lock', width=10,
                            command=_clear_lock)
        clr_btn.pack(side='left', padx=8)
        _Tooltip(clr_btn, 'cup_fusion_node/clear_scan (Trigger)\n'
                          '누적·lock 폐기 → 라이브 검출 복귀.')

        grid = tk.Frame(parent, bg='#1e1e1e')
        grid.grid(row=1, column=0, sticky='nsew')
        for c in range(3):
            grid.grid_columnconfigure(c, weight=1, uniform='cells')
        for r in range(2):
            grid.grid_rowconfigure(r, weight=1, uniform='cells')
        for i, (cam, kind, title) in enumerate(_CELLS):
            r, c = divmod(i, 3)
            cell = tk.Frame(grid, bg='#101010', bd=1, relief='solid')
            cell.grid(row=r, column=c, sticky='nsew', padx=2, pady=2)
            cell.grid_rowconfigure(1, weight=1)
            cell.grid_columnconfigure(0, weight=1)
            tk.Label(cell, text=title, fg='#9cdcfe', bg='#101010',
                     font=('TkDefaultFont', 9, 'bold')
                     ).grid(row=0, column=0, sticky='ew')
            lbl = tk.Label(cell, bg='#101010')
            lbl.grid(row=1, column=0, sticky='nsew')
            self.labels[(cam, kind)] = lbl

    # ---- TUNING tab: live sliders + tooltips + save ----
    def _build_tuning(self, parent: tk.Frame) -> None:
        top = tk.Frame(parent, bg='#1e1e1e')
        top.pack(fill='x', pady=4)
        tk.Button(top, text='💾  Save YAML', command=self._save_yaml,
                  bg='#2d4', fg='#000').pack(side='left', padx=8)
        self._save_lbl = tk.Label(top, text='(hover a name for help)',
                                  bg='#1e1e1e', fg='#9cdcfe')
        self._save_lbl.pack(side='left', padx=6)

        canvas = tk.Canvas(parent, bg='#1e1e1e', highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        inner = tk.Frame(canvas, bg='#1e1e1e')
        inner.bind('<Configure>',
                   lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

        self._sliders: dict[tuple, tuple] = {}   # (node, param) -> (var, kind)
        for spec in _TUNING:
            self._add_slider(inner, spec)

    def _add_slider(self, parent, spec) -> None:
        display, node, param, kind, lo, hi, res, default, tip = spec
        row = tk.Frame(parent, bg='#1e1e1e')
        row.pack(fill='x', padx=10, pady=1)
        name = tk.Label(row, text=display, width=18, anchor='w',
                        bg='#1e1e1e', fg='#dddddd')
        name.pack(side='left')
        _Tooltip(name, f'{param}\n{tip}')
        val = tk.Label(row, width=8, anchor='e', bg='#1e1e1e', fg='#9cdcfe')
        var = tk.DoubleVar(value=default)

        def on_change(v, p=param, n=node, k=kind, vl=val):
            x = float(v)
            vl.configure(text=(f'{int(round(x))}' if k == 'integer'
                               else f'{x:.3f}'))
            self.node.request_param(n, p,
                                    int(round(x)) if k == 'integer' else x, k)

        sc = tk.Scale(row, from_=lo, to=hi, resolution=res, orient='horizontal',
                      length=420, variable=var, showvalue=False,
                      command=on_change, bg='#1e1e1e', fg='#ddd',
                      troughcolor='#333', highlightthickness=0, sliderlength=16)
        sc.pack(side='left', fill='x', expand=True, padx=6)
        val.configure(text=(f'{int(default)}' if kind == 'integer'
                            else f'{default:.3f}'))
        val.pack(side='left')
        self._sliders[(node, param)] = (var, kind)

    def _save_yaml(self) -> None:
        sections: dict = {}
        for (node, param), (var, kind) in self._sliders.items():
            v = int(round(var.get())) if kind == 'integer' else round(var.get(), 4)
            sections.setdefault(node, {})[param] = v
        for p, var in self._dbg_vars.items():
            sections.setdefault(self.node.fusion_node, {})[p] = bool(var.get())
        try:
            path = self.node.save_yaml(sections)
            self._save_lbl.configure(text=f'saved: {os.path.basename(path)}')
        except Exception as e:   # noqa: BLE001
            self._save_lbl.configure(text=f'save failed: {e}')

    def _tick(self) -> None:
        if self._closing:
            return
        frames = self.node.snapshot()
        for (cam, kind), lbl in self.labels.items():
            bgr = frames.get((cam, kind))
            w = max(lbl.winfo_width(), 1)
            h = max(lbl.winfo_height(), 1)
            if bgr is None or w < 8 or h < 8:
                continue
            photo = self._fit(bgr, w, h)
            self._photo[(cam, kind)] = photo   # keep ref alive
            lbl.configure(image=photo)
        self.root.after(40, self._tick)        # ~25 Hz redraw

    @staticmethod
    def _fit(bgr: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
        ih, iw = bgr.shape[:2]
        scale = min(w / iw, h / ih)            # fit inside, never crop
        nw, nh = max(int(iw * scale), 1), max(int(ih * scale), 1)
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _on_close(self) -> None:
        self._closing = True
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DigitalTwinPanel()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        PanelUI(node).run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
