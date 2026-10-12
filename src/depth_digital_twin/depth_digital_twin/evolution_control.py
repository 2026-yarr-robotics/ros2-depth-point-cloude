"""evolution_control — tiny Tk panel for the evolution demo.

One row of buttons drives `evolution_demo_node`:
  • ◀ Prev          — previous detection method
  • Next method ▶   — advance to the next method (the GIF-recording trigger)
  • ⟳ ArUco re-detect — re-lock the exo world origin if it drifted
A large label mirrors the node's `~/status` ("[k/5] <title> — <ko>").

Standalone:
  ros2 run depth_digital_twin evolution_control
"""
from __future__ import annotations

import threading
import tkinter as tk
import tkinter.font as tkfont

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


class _RosThread(threading.Thread):
    def __init__(self, node: Node) -> None:
        super().__init__(daemon=True)
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(node)

    def run(self) -> None:
        try:
            self._exec.spin()
        except Exception:  # noqa: BLE001
            pass

    def shutdown(self) -> None:
        self._exec.shutdown(timeout_sec=0)


class _CtrlNode(Node):
    def __init__(self) -> None:
        super().__init__('evolution_control')
        self.declare_parameter('demo_node', '/evolution_demo_node')
        self.declare_parameter('exo_redetect_srv',
                               '/world_origin_node_exo/redetect')
        demo = str(self.get_parameter('demo_node').value).rstrip('/')
        self._cli = {
            'next': self.create_client(Trigger, f'{demo}/next_method'),
            'prev': self.create_client(Trigger, f'{demo}/prev_method'),
            'redetect': self.create_client(
                Trigger, str(self.get_parameter('exo_redetect_srv').value)),
        }
        self.status_text = 'connecting…'
        self.create_subscription(String, f'{demo}/status', self._on_status, 5)

    def _on_status(self, msg: String) -> None:
        self.status_text = msg.data

    def call(self, name: str, cb) -> None:
        cli = self._cli[name]
        if not cli.wait_for_service(timeout_sec=0.5):
            cb(False, f'{name}: service unavailable')
            return
        fut = cli.call_async(Trigger.Request())
        fut.add_done_callback(
            lambda f: cb(*((f.result().success, f.result().message)
                           if f.result() else (False, 'no response'))))


class ControlPanel:
    def __init__(self, node: _CtrlNode) -> None:
        self._node = node
        self._root = tk.Tk()
        self._root.title('Evolution Demo — Method Switch')
        self._root.resizable(False, False)
        self._root.attributes('-topmost', True)
        self._root.configure(bg='#1e1e1e')

        big = tkfont.Font(weight='bold', size=13)
        small = tkfont.Font(size=10)
        frame = tk.Frame(self._root, padx=16, pady=14, bg='#1e1e1e')
        frame.pack()

        self._method = tk.StringVar(value='—')
        tk.Label(frame, textvariable=self._method, font=big, fg='#9cdcfe',
                 bg='#1e1e1e', wraplength=360, justify='center'
                 ).pack(fill=tk.X, pady=(0, 12))

        btns = tk.Frame(frame, bg='#1e1e1e')
        btns.pack()
        tk.Button(btns, text='◀ Prev', width=9, font=small,
                  command=lambda: self._call('prev')).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text='Next method ▶', width=14, font=big,
                  bg='#3aa657', fg='white', activebackground='#2a7e42',
                  command=lambda: self._call('next')).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text='⟳ ArUco', width=9, font=small,
                  command=lambda: self._call('redetect')
                  ).pack(side=tk.LEFT, padx=4)

        self._status = tk.StringVar(value='Ready')
        tk.Label(frame, textvariable=self._status, font=small, fg='#888',
                 bg='#1e1e1e', pady=4).pack(fill=tk.X)
        self._poll()

    def _set_status(self, ok: bool, msg: str) -> None:
        self._root.after(
            0, lambda: self._status.set(f'{"OK" if ok else "FAIL"}: {msg}'))

    def _call(self, name: str) -> None:
        self._status.set(f'{name}…')
        self._node.call(name, self._set_status)

    def _poll(self) -> None:
        self._method.set(self._node.status_text)
        self._root.after(150, self._poll)

    def run(self) -> None:
        self._root.mainloop()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = _CtrlNode()
    rt = _RosThread(node)
    rt.start()
    panel = ControlPanel(node)
    try:
        panel.run()
    finally:
        rt.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
