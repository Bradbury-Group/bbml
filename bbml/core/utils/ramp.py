

class Ramp:
    def __init__(self, start:int, duration:int):
        """
        0->1 over duration after start (for gates)
        """
        self.start = start
        self.duration = duration

    def __call__(self, current_step):
        if current_step < self.start: return 0.0
        if self.duration <= 0: return 1.0
        return min(1.0, (current_step - self.start) / self.duration)