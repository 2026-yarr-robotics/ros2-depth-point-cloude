"""world_origin_control — Reset / Redetect ArUco control panel.

Launched automatically by digital_twin.launch.py (control_panel:=true).
Can also be run standalone:
  ros2 run depth_digital_twin world_origin_control

Calls /world_origin_node/redetect (std_srvs/Trigger) on button press.
"""
from __future__ import annotations

import threading
import tkinter as tk
import tkinter.font as tkfont

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

_SERVICE = '/world_origin_node/redetect'


class _RosThread(threading.Thread):
    """Spin a ROS node in a background thread."""
    def __init__(self, node: Node) -> None:
        super().__init__(daemon=True)
        self._node = node
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(node)

    def run(self) -> None:
        try:
            self._exec.spin()
        except Exception:
            pass

    def shutdown(self) -> None:
        self._exec.shutdown(timeout_sec=0)


class _ControlNode(Node):
    def __init__(self) -> None:
        super().__init__('world_origin_control')
        self._client = self.create_client(Trigger, _SERVICE)

    def call_redetect_async(self, callback) -> None:
        """Non-blocking service call. Calls callback(ok, message) when done."""
        if not self._client.wait_for_service(timeout_sec=0.5):
            callback(False, f'Service {_SERVICE!r} not available')
            return
        fut = self._client.call_async(Trigger.Request())

        def _done(future):
            try:
                r = future.result()
                callback(r.success, r.message)
            except Exception as e:
                callback(False, str(e))

        fut.add_done_callback(_done)


class ControlPanel:
    def __init__(self, node: _ControlNode) -> None:
        self._node = node

        self._root = tk.Tk()
        self._root.title('World Origin Control')
        self._root.resizable(False, False)
        self._root.attributes('-topmost', True)

        bold = tkfont.Font(weight='bold', size=11)
        normal = tkfont.Font(size=10)

        frame = tk.Frame(self._root, padx=12, pady=10)
        frame.pack()

        btn_frame = tk.Frame(frame)
        btn_frame.pack()

        tk.Button(
            btn_frame, text='Reset', width=10,
            font=bold, bg='#e07b39', fg='white',
            activebackground='#c05a20',
            command=self._on_redetect,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            btn_frame, text='Redetect ArUco', width=15,
            font=bold, bg='#3a8ed4', fg='white',
            activebackground='#2a6eb0',
            command=self._on_redetect,
        ).pack(side=tk.LEFT)

        self._status_var = tk.StringVar(value='Ready')
        tk.Label(frame, textvariable=self._status_var,
                 font=normal, fg='#444444', pady=4).pack(fill=tk.X)

    def _on_redetect(self) -> None:
        self._status_var.set('Requesting…')
        self._root.update_idletasks()

        def _cb(ok: bool, msg: str) -> None:
            status = f'{"OK" if ok else "FAIL"}: {msg}'
            # Schedule GUI update on the main thread
            self._root.after(0, lambda: self._status_var.set(status))

        self._node.call_redetect_async(_cb)

    def run(self) -> None:
        self._root.mainloop()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = _ControlNode()
    ros_thread = _RosThread(node)
    ros_thread.start()

    panel = ControlPanel(node)
    try:
        panel.run()  # blocks until window is closed
    finally:
        ros_thread.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
