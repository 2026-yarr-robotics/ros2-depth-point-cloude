"""recorder_node — record exo + hand RGB-D and the M0609 trajectory.

Starts recording IMMEDIATELY on launch into a fresh auto-numbered folder
(``output_root/0001``…). One shared timeline: a fixed-rate timer grabs the
LATEST sample of every stream and writes them as a single step, so exo /
hand / robot all share the same step index.

Robot data:
  • joint_states : subscribed from /dsr01/joint_states  [rad]
                   — RViz visualisation only.
  • EE pose      : DSR API call ``get_current_posx()`` (same as the
                   Calibration_Tutorial samples) → [x,y,z,a,b,c] mm/deg
                   ZYZ, consistent with the hand-eye T_gripper2camera.npy
                   used in Phase 2. Polled in a background thread.

If the robot is not connected / the DSR API cannot be imported or called,
the recorder still records the cameras (and joints if present) and warns
once — i.e. "just record images" as requested.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Any

import rclpy
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState

from recode_sequence.seq_io import CameraMeta, SequenceWriter


class _EePoller(threading.Thread):
    """Poll DSR ``get_current_posx()`` in the background; cache latest.

    Mirrors the Calibration_Tutorial usage: create a node, set DR_init,
    import DSR_ROBOT2, then call get_current_posx() (the wrapper handles
    its own spinning, so this node must NOT be added to an executor).
    Fully fault-tolerant: any failure → ``available=False`` and the
    recorder simply records images only.
    """

    def __init__(self, robot_id: str, robot_model: str, imp_path: str,
                 poll_hz: float, logger) -> None:
        super().__init__(daemon=True)
        self._robot_id = robot_id
        self._robot_model = robot_model
        self._imp_path = imp_path
        self._period = 1.0 / max(poll_hz, 1.0)
        self._log = logger
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._value: list[float] | None = None
        self.available = False
        self.reason = 'initialising'

    @property
    def value(self) -> list[float] | None:
        with self._lock:
            return None if self._value is None else list(self._value)

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        try:
            if self._imp_path and self._imp_path not in sys.path:
                sys.path.insert(0, self._imp_path)
            import DR_init  # noqa: WPS433
            DR_init.__dsr__id = self._robot_id
            DR_init.__dsr__model = self._robot_model
            api_node = rclpy.create_node(
                'recorder_dsr_api', namespace=self._robot_id)
            DR_init.__dsr__node = api_node
            from DSR_ROBOT2 import get_current_posx  # noqa: WPS433
        except Exception as e:  # noqa: BLE001
            self.reason = f'DSR API unavailable ({e})'
            self._log.warn(
                f'EE pose disabled — {self.reason}. Recording images'
                ' (+ joints if present) only.')
            return

        fails = 0
        while not self._stop_evt.is_set():
            try:
                posx = get_current_posx()
                pos = posx[0] if posx else None
                if pos is not None and len(pos) >= 6:
                    with self._lock:
                        self._value = [float(v) for v in pos[:6]]
                    if not self.available:
                        self._log.info('EE pose via get_current_posx OK')
                    self.available = True
                    self.reason = 'ok'
                    fails = 0
            except Exception as e:  # noqa: BLE001
                fails += 1
                self.available = False
                self.reason = f'get_current_posx failed ({e})'
                if fails == 1:
                    self._log.warn(
                        f'{self.reason} — recording images only;'
                        ' will keep retrying.')
                # Back off when the robot is clearly absent.
                if fails >= 5:
                    self._stop_evt.wait(2.0)
            self._stop_evt.wait(self._period)


class RecorderNode(Node):
    def __init__(self) -> None:
        super().__init__('recorder_node')
        gp = self.declare_parameter
        gp('output_root', '/home/eunwoosong/Projects/record_sequence')
        gp('record_rate_hz', 30.0)
        gp('exo_color_topic', '/exo/exo/color/image_raw')
        gp('exo_depth_topic', '/exo/exo/aligned_depth_to_color/image_raw')
        gp('exo_info_topic', '/exo/exo/color/camera_info')
        gp('hand_color_topic', '/hand/hand/color/image_raw')
        gp('hand_depth_topic', '/hand/hand/aligned_depth_to_color/image_raw')
        gp('hand_info_topic', '/hand/hand/color/camera_info')
        gp('joint_states_topic', '/dsr01/joint_states')
        # EE pose via DSR API (get_current_posx), polled in a thread.
        gp('robot_id', 'dsr01')
        gp('robot_model', 'm0609')
        gp('dsr_imp_path',
           '/home/eunwoosong/ros2_ws/src/doosan-robot2/dsr_common2/imp')
        gp('ee_poll_hz', 30.0)
        gp('exo_serial', '')
        gp('hand_serial', '')
        gp('require_exo', True)
        gp('max_duration_s', 0.0)
        # Disk writing runs on background workers (PNG encode is slow);
        # the timer only snapshots + enqueues so it can hit 30 Hz.
        gp('write_workers', 4)
        # Persist meta.json + trajectory.pkl every N steps so an abrupt
        # kill (not a clean Ctrl-C) still leaves a readable sequence.
        gp('checkpoint_every', 90)
        # Wait this long after the node starts (≈ after the camera nodes
        # come up, since record.launch.py starts them together) before
        # the FIRST step is written — lets both D435i fully warm up so
        # all 4 streams are present from step 0 (otherwise the first few
        # hand_depth frames can be missing).
        gp('start_delay_s', 20.0)

        def P(n: str) -> Any:
            return self.get_parameter(n).value

        self.bridge = CvBridge()
        self.require_exo = bool(P('require_exo'))
        self.max_duration_s = float(P('max_duration_s'))
        self._ckpt_every = max(int(P('checkpoint_every')), 1)
        self._start_delay = max(float(P('start_delay_s')), 0.0)
        self._rec_start: float | None = None   # wall time of first step
        self._warmup_log_at = 0.0

        self._exo_rgb = self._exo_depth = None
        self._hand_rgb = self._hand_depth = None
        self._joint_names: list[str] | None = None
        self._joint_pos: list[float] | None = None
        self._warned_ee = False
        self._started = False
        self._t_start = time.time()

        robot_meta = {
            'model': str(P('robot_model')),
            'joint_units': 'radian',
            'ee_units': 'mm,deg',
            'ee_euler': 'ZYZ',
            'ee_source': 'dsr_api:get_current_posx',
            'joint_states_topic': str(P('joint_states_topic')),
        }
        self.writer = SequenceWriter(
            str(P('output_root')), float(P('record_rate_hz')), robot_meta)
        for view, serial in (('exo', str(P('exo_serial'))),
                             ('hand', str(P('hand_serial')))):
            self.writer.set_camera(view, CameraMeta(serial=serial))

        # Background disk-writer pool: timer enqueues (path, img, is_depth),
        # workers PNG-encode + write so the 30 Hz timeline never blocks.
        self._wq: queue.Queue = queue.Queue()
        self._n_workers = max(int(P('write_workers')), 1)
        self._workers = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(self._n_workers)]
        for w in self._workers:
            w.start()
        self._backlog_warned = False

        # Background EE poller (graceful no-op if robot/API absent).
        self.ee = _EePoller(
            str(P('robot_id')), str(P('robot_model')),
            str(P('dsr_imp_path')), float(P('ee_poll_hz')),
            self.get_logger())
        self.ee.start()

        img_qos = qos_profile_sensor_data
        self.create_subscription(
            Image, str(P('exo_color_topic')),
            lambda m: self._set_img('_exo_rgb', m, 'bgr8'), img_qos)
        self.create_subscription(
            Image, str(P('exo_depth_topic')),
            lambda m: self._set_img('_exo_depth', m, 'passthrough'), img_qos)
        self.create_subscription(
            Image, str(P('hand_color_topic')),
            lambda m: self._set_img('_hand_rgb', m, 'bgr8'), img_qos)
        self.create_subscription(
            Image, str(P('hand_depth_topic')),
            lambda m: self._set_img('_hand_depth', m, 'passthrough'), img_qos)
        self.create_subscription(
            CameraInfo, str(P('exo_info_topic')),
            lambda m: self._set_info('exo', m), img_qos)
        self.create_subscription(
            CameraInfo, str(P('hand_info_topic')),
            lambda m: self._set_info('hand', m), img_qos)
        self.create_subscription(
            JointState, str(P('joint_states_topic')), self._on_joints, 10)

        rate = max(float(P('record_rate_hz')), 1.0)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'recorder_node: writing → {self.writer.root}  '
            f'@ {rate:.1f} Hz. Recording starts on first exo frame.')

    # -- subscription handlers ---------------------------------------------
    def _set_img(self, attr: str, msg: Image, enc: str) -> None:
        try:
            setattr(self, attr, self.bridge.imgmsg_to_cv2(
                msg, desired_encoding=enc))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'{attr}: cv_bridge {e}', once=True)

    def _set_info(self, view: str, msg: CameraInfo) -> None:
        cm = self.writer.cameras.get(view) or CameraMeta()
        if cm.width:
            return
        cm.width = int(msg.width)
        cm.height = int(msg.height)
        cm.K = [float(v) for v in msg.k]
        cm.dist = [float(v) for v in msg.d]
        cm.frame_id = msg.header.frame_id
        self.writer.set_camera(view, cm)

    def _on_joints(self, msg: JointState) -> None:
        self._joint_names = list(msg.name)
        self._joint_pos = list(msg.position)

    # -- background disk writer --------------------------------------------
    def _worker(self) -> None:
        while True:
            job = self._wq.get()
            if job is None:          # shutdown sentinel
                self._wq.task_done()
                return
            path, img, is_depth = job
            try:
                SequenceWriter.save_image(path, img, is_depth)
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f'write {path}: {e}', once=True)
            finally:
                self._wq.task_done()

    # -- timeline tick ------------------------------------------------------
    def _tick(self) -> None:
        # Warm-up: subscriptions keep filling caches + CameraInfo is still
        # captured into the writer meta, but no step is written yet.
        elapsed = time.time() - self._t_start
        if elapsed < self._start_delay:
            if time.time() >= self._warmup_log_at:
                self._warmup_log_at = time.time() + 5.0
                self.get_logger().info(
                    f'warming up cameras — recording starts in '
                    f'{self._start_delay - elapsed:.0f}s')
            return
        if self.require_exo and (self._exo_rgb is None
                                 or self._exo_depth is None):
            return
        if not self._started:
            self._started = True
            self._rec_start = time.time()
            self.get_logger().info('▶ recording started')

        ee_tcp = self.ee.value  # None if robot/API unavailable
        if (not self._warned_ee and ee_tcp is None
                and (time.time() - self._t_start) > 3.0):
            self._warned_ee = True
            self.get_logger().warn(
                f'EE pose not available ({self.ee.reason}) — recording '
                'images + joints only.')

        # Snapshot the latest frames (copy: the subscription buffer may be
        # recycled before a worker encodes it).
        def _cp(a):
            return None if a is None else a.copy()

        step, jobs = self.writer.add_step(
            t_wall=time.time(),
            exo_rgb=_cp(self._exo_rgb), exo_depth=_cp(self._exo_depth),
            hand_rgb=_cp(self._hand_rgb), hand_depth=_cp(self._hand_depth),
            joint_names=self._joint_names, joint_pos=self._joint_pos,
            ee_tcp=ee_tcp, ee_flange=None)
        for j in jobs:
            self._wq.put(j)

        backlog = self._wq.qsize()
        if backlog > 8 * self._n_workers and not self._backlog_warned:
            self._backlog_warned = True
            self.get_logger().warn(
                f'disk write backlog={backlog} — storage cannot keep up; '
                'lower record_rate_hz or write to a faster disk.')
        elif backlog <= self._n_workers:
            self._backlog_warned = False
        if step > 0 and step % self._ckpt_every == 0:
            self.writer.checkpoint()      # crash/kill-safe snapshot
        if step % 30 == 0:
            self.get_logger().info(f'  step {step}… (backlog {backlog})')

        if (self.max_duration_s > 0.0 and self._rec_start is not None
                and (time.time() - self._rec_start) > self.max_duration_s):
            self.get_logger().info('max_duration_s reached — stopping.')
            raise KeyboardInterrupt

    def finalize(self) -> None:
        self.ee.stop()
        pending = self._wq.qsize()
        if pending:
            self.get_logger().info(
                f'flushing {pending} queued frames to disk…')
        self._wq.join()                       # wait for all writes
        for _ in self._workers:               # stop workers
            self._wq.put(None)
        for w in self._workers:
            w.join(timeout=5.0)
        self.writer.close()
        self.get_logger().info(
            f'✔ sequence {self.writer.seq_id} closed: '
            f'{self.writer.n_steps} steps → {self.writer.root}')


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RecorderNode()
    # MultiThreadedExecutor: camera callbacks keep flowing while the
    # record timer runs (no single-thread serialisation stall).
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
