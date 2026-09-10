from collections import defaultdict, deque


class SequenceBuffer:
    def __init__(self, maxlen=15):
        self.maxlen = maxlen
        self.buffers = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, track_id, frame_idx, frame, bbox):
        self.buffers[track_id].append({
            "frame_idx": frame_idx,
            "frame": frame.copy(),
            "bbox": bbox
        })

    def is_ready(self, track_id):
        return len(self.buffers[track_id]) == self.maxlen

    def get_sequence(self, track_id):
        if not self.is_ready(track_id):
            return None

        return list(self.buffers[track_id])

    def remove(self, track_id):
        if track_id in self.buffers:
            del self.buffers[track_id]