"""playback_control — Tk control panel for sequence_player_node.

Buttons / fields (matching the spec):
  • Stop      — pause playback
  • Resume    — continue from the current step
  • Replay    — restart from step 0
  • [step] Apply — jump to that step and PAUSE (then press Resume to run)

Live status ("RUNNING 123/2000") is mirrored from ~/state.

Standalone:
  ros2 run recode_sequence playback_control
"""
from __future__ import annotations

import threading
import tkinter as tk
import tkinter.font as tkfont

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger

_NS = '/sequence_player_node'


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
        super().__init__('playback_control')
        self.declare_parameter('player_ns', _NS)
        ns = str(self.get_parameter('player_ns').value)
        self._cli = {
            'stop': self.create_client(Trigger, f'{ns}/stop'),
            'resume': self.create_client(Trigger, f'{ns}/resume'),
            'replay': self.create_client(Trigger, f'{ns}/replay'),
        }
        self._goto_pub = self.create_publisher(Int32, f'{ns}/goto_step', 10)
        self.state_text = 'connecting…'
        self.create_subscription(
            String, f'{ns}/state', self._on_state, 5)

    def _on_state(self, msg: String) -> None:
        self.state_text = msg.data

    def call(self, name: str, cb) -> None:
        cli = self._cli[name]
        if not cli.wait_for_service(timeout_sec=0.5):
            cb(False, f'{name}: service unavailable')
            return
        fut = cli.call_async(Trigger.Request())
        fut.add_done_callback(
            lambda f: cb(*( (f.result().success, f.result().message)
                            if f.result() else (False, 'no response'))))

    def goto(self, step: int) -> None:
        m = Int32()
        m.data = int(step)
        self._goto_pub.publish(m)


class ControlPanel:
    def __init__(self, node: _CtrlNode) -> None:
        self._node = node
        self._root = tk.Tk()
        self._root.title('Sequence Playback Control')
        self._root.resizable(False, False)
        self._root.attributes('-topmost', True)

        bold = tkfont.Font(weight='bold', size=11)
        normal = tkfont.Font(size=10)
        frame = tk.Frame(self._root, padx=14, pady=12)
        frame.pack()

        btns = tk.Frame(frame)
        btns.pack()
        tk.Button(btns, text='Stop', width=9, font=bold,
                  bg='#e07b39', fg='white', activebackground='#c05a20',
                  command=lambda: self._call('stop')).pack(side=tk.LEFT,
                                                           padx=4)
        tk.Button(btns, text='Resume', width=9, font=bold,
                  bg='#3aa657', fg='white', activebackground='#2a7e42',
                  command=lambda: self._call('resume')).pack(side=tk.LEFT,
                                                             padx=4)
        tk.Button(btns, text='Replay', width=9, font=bold,
                  bg='#3a8ed4', fg='white', activebackground='#2a6eb0',
                  command=lambda: self._call('replay')).pack(side=tk.LEFT,
                                                             padx=4)

        goto = tk.Frame(frame)
        goto.pack(pady=(10, 0))
        tk.Label(goto, text='Step:', font=normal).pack(side=tk.LEFT)
        self._step_var = tk.StringVar(value='0')
        tk.Entry(goto, textvariable=self._step_var, width=8,
                 font=normal).pack(side=tk.LEFT, padx=6)
        tk.Button(goto, text='Apply', width=8, font=bold,
                  command=self._on_apply).pack(side=tk.LEFT)

        self._status = tk.StringVar(value='Ready')
        tk.Label(frame, textvariable=self._status, font=normal,
                 fg='#444', pady=4).pack(fill=tk.X)
        self._state = tk.StringVar(value='state: —')
        tk.Label(frame, textvariable=self._state, font=bold,
                 fg='#1a1a1a').pack(fill=tk.X)

        self._poll_state()

    def _set_status(self, ok: bool, msg: str) -> None:
        self._root.after(
            0, lambda: self._status.set(f'{"OK" if ok else "FAIL"}: {msg}'))

    def _call(self, name: str) -> None:
        self._status.set(f'{name}…')
        self._node.call(name, self._set_status)

    def _on_apply(self) -> None:
        try:
            step = int(self._step_var.get())
        except ValueError:
            self._status.set('FAIL: step must be an integer')
            return
        self._node.goto(step)
        self._status.set(f'jumped to {step} (paused — press Resume)')

    def _poll_state(self) -> None:
        self._state.set(f'state: {self._node.state_text}')
        self._root.after(150, self._poll_state)

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
