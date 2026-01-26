import threading
class Telemetry:
    def __init__(self):
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.is_ready = False
        self.lock = threading.Lock()