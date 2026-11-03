"""Bounded, latest-frame ZMQ replay source. Never opens a chassis connection."""
from collections import deque
import threading
import time
import math

from .anchor_map import decode_rgbd_pose_message, decode_explore_map


class ReplayFinished(EOFError):
    pass


class ReplayStopped(RuntimeError):
    pass


class ReplayRobot:
    def __init__(self, *, map_id, map_revision, endpoint, map_endpoint,
                 idle_timeout_s=30., max_map_age_s=3.):
        if endpoint == map_endpoint:
            raise ValueError('Replay camera and map endpoints must differ')
        if any(not math.isfinite(v) or v <= 0 for v in (idle_timeout_s, max_map_age_s)):
            raise ValueError('Replay timeouts must be positive and finite')
        self.map_id, self.map_revision = map_id, map_revision
        self.endpoint, self.map_endpoint = endpoint, map_endpoint
        self.idle_timeout_s, self.max_map_age_s = idle_timeout_s, max_map_age_s
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.ready = None
        self.finished = False
        self.error = None
        self.stats = dict(received_frames=0, received_maps=0, overwritten_frames=0,
                          unmatched_frames=0, processed_frames=0)

    def __enter__(self):
        self.thread = threading.Thread(target=self._receive, name='replay-receiver')
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()

    def capture_pair(self, *, timeout_s=5.):
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while True:
                if self.error:
                    raise ReplayStopped(str(self.error)) from self.error
                if self.ready is not None:
                    pair, self.ready = self.ready, None
                    self.stats['processed_frames'] += 1
                    return pair
                if self.finished:
                    raise ReplayFinished('回放完成')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('等待回放发布端或可配对的 RGB-D / 地图')
                self.condition.wait(remaining)

    def diagnostics(self):
        with self.condition:
            return dict(self.stats)

    def _receive(self):
        import zmq
        import msgpack
        context = zmq.Context()
        sockets = []
        maps = deque(maxlen=128)
        pending = None
        last_received = None
        ended = set()
        epoch = None
        last_frame_time = last_map_time = -1
        try:
            poller = zmq.Poller()
            for endpoint, topic in ((self.endpoint, b'rgbd.pose'), (self.map_endpoint, b'hermes.map')):
                socket = context.socket(zmq.SUB)
                sockets.append(socket)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVHWM, 8)
                socket.setsockopt(zmq.MAXMSGSIZE, 128*1024*1024)
                socket.setsockopt(zmq.SUBSCRIBE, topic)
                socket.setsockopt(zmq.SUBSCRIBE, b'replay.control')
                socket.connect(endpoint)
                poller.register(socket, zmq.POLLIN)
            while not self.stop.is_set():
                events = dict(poller.poll(100))
                for socket in sockets:
                    if socket not in events:
                        continue
                    parts = socket.recv_multipart()
                    last_received = time.monotonic()
                    if parts[0] == b'replay.control':
                        control = msgpack.unpackb(parts[1], raw=False)
                        if control.get('event') == 'end':
                            ended.add(socket)
                            with self.condition:
                                self.stats['publisher_counts'] = control['published']
                        continue
                    if socket in ended:
                        raise ValueError('New data after replay end; restart main for each recording')
                    if socket is sockets[0]:
                        frame = decode_rgbd_pose_message(parts, map_id=self.map_id, map_revision=self.map_revision)
                        if frame.timestamp_ns <= last_frame_time:
                            raise ValueError('Replay frame timestamps must increase')
                        last_frame_time = frame.timestamp_ns
                        with self.condition:
                            self.stats['received_frames'] += 1
                            if pending is not None:
                                self.stats['unmatched_frames'] += 1
                        pending = frame
                    else:
                        if len(parts) != 3 or parts[0] != b'hermes.map':
                            raise ValueError('Invalid replay map message')
                        meta = msgpack.unpackb(parts[1], raw=False)
                        if epoch is not None and meta['epoch'] != epoch:
                            raise ValueError('Map epoch changed; use --after-map-reset and restart main')
                        epoch = meta['epoch']
                        stamp = (meta['request_ns'] + meta['response_ns']) // 2
                        if stamp < last_map_time:
                            raise ValueError('Replay map timestamps regressed')
                        last_map_time = stamp
                        grid = decode_explore_map(parts[2], map_id=self.map_id,
                                                  map_revision=self.map_revision, timestamp_ns=stamp)
                        maps.append(grid)
                        with self.condition:
                            self.stats['received_maps'] += 1
                    if pending is not None:
                        grid = next((g for g in reversed(maps) if g.timestamp_ns <= pending.timestamp_ns), None)
                        if grid is not None and (pending.timestamp_ns-grid.timestamp_ns)/1e9 <= self.max_map_age_s:
                            with self.condition:
                                if self.ready is not None:
                                    self.stats['overwritten_frames'] += 1
                                self.ready = (pending, grid)
                                self.condition.notify_all()
                            pending = None
                if len(ended) == 2:
                    with self.condition:
                        if pending is not None:
                            self.stats['unmatched_frames'] += 1
                        self.finished = True
                        self.condition.notify_all()
                    return
                if last_received is not None and time.monotonic()-last_received > self.idle_timeout_s:
                    raise TimeoutError('回放数据流中断或缺少结束通知；保留最后成功地图，请重启发布端和 main')
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()
        finally:
            for socket in sockets:
                socket.close()
            context.term()
