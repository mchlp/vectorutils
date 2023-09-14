import time

class TimerLogger:
    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter_ns()
        self.interval = self.end - self.start
        print(f"Time elapsed: {self.interval / 1E9:.4f}")