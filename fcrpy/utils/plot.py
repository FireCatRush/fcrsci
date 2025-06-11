import time
from collections import deque

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None


class DataBuffer:
    def __init__(self, maxlen=None):
        self.maxlen = maxlen
        self._buf = {}

    def add_point(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self._buf:
                self._buf[key] = deque(maxlen=self.maxlen)
            self._buf[key].append(value)

    def get(self, key):
        return list(self._buf.get(key, []))

    def keys(self):
        return list(self._buf.keys())


class PlotEngine:
    def __init__(self, core="matplotlib", maxlen=100, x_key="x", y_key="y"):
        self.core = core.lower()
        self.buffer = DataBuffer(maxlen=maxlen)
        self.x_key = x_key
        self.y_key = y_key
        self._init_plot()

    def _init_plot(self):
        if self.core == "matplotlib":
            if plt is None:
                raise ImportError("matplotlib is not installed")
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.line, = self.ax.plot([], [], '-o', label=f"{self.y_key} vs {self.x_key}")
            self.ax.set_xlabel(self.x_key)
            self.ax.set_ylabel(self.y_key)
            self.ax.legend()
        elif self.core == "plotly":
            if go is None:
                raise ImportError("plotly is not installed")
            self.fig = make_subplots(rows=1, cols=1)
            self.trace = go.Scatter(x=[], y=[], mode='lines+markers', name=f"{self.y_key} vs {self.x_key}")
            self.fig.add_trace(self.trace)
            self.fig.update_layout(xaxis_title=self.x_key, yaxis_title=self.y_key)
            self.fig.show()
        else:
            raise ValueError(f"Unknown core: {self.core}")

    def set_keys(self, x_key=None, y_key=None):
        """
        Update data field names and reinitialize the plot labels.
        """
        if x_key:
            self.x_key = x_key
        if y_key:
            self.y_key = y_key
        # update plot components
        if self.core == "matplotlib":
            self.line.set_label(f"{self.y_key} vs {self.x_key}")
            self.ax.set_xlabel(self.x_key)
            self.ax.set_ylabel(self.y_key)
            self.ax.legend()
        else:
            self.fig.data[0].name = f"{self.y_key} vs {self.x_key}"
            self.fig.update_layout(xaxis_title=self.x_key, yaxis_title=self.y_key)

    def add_point(self, **kwargs):
        """Add a new data point; keys must include x_key and y_key."""
        self.buffer.add_point(**kwargs)
        if self.core == "plotly":
            xs = self.buffer.get(self.x_key)
            ys = self.buffer.get(self.y_key)
            with self.fig.batch_update():
                self.fig.data[0].x = xs
                self.fig.data[0].y = ys

    def refresh(self):
        """
        Manually refresh the plot (only for matplotlib).
        Call this after add_point when using matplotlib.
        """
        if self.core != "matplotlib":
            return
        xs = self.buffer.get(self.x_key)
        ys = self.buffer.get(self.y_key)
        if not xs or not ys:
            return

        self.line.set_data(xs, ys)
        self.ax.set_xlim(min(xs), max(xs))
        self.ax.set_ylim(min(ys), max(ys))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# Example usage:
engine = PlotEngine(core="matplotlib", maxlen=50)
for i in range(50):
    engine.add_point(time=i, value=i**2)
    engine.refresh()
engine.set_keys(x_key="time", y_key="value")
for i in range(50, 100):
    engine.add_point(time=i, value=i**2)
    engine.refresh()

