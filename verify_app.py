"""Headless smoke test for the Streamlit interface.

Installs a minimal stub implementing the Streamlit API surface used by app.py, then
executes the application with default widget values and the action buttons enabled.
Verifies that every service call succeeds and that the expected components render.

This exercises data flow and service integration, not user interface behaviour.
"""
from __future__ import annotations

import runpy
import sys
import types


class _SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


class _Shim:
    """Stub standing in for the streamlit module and its container objects."""

    def __init__(self):
        self.session_state = _SessionState()
        self.counts = {"metric": 0, "line_chart": 0, "pyplot": 0,
                       "dataframe": 0, "error": 0, "button": 0}
        self.force_buttons = True
        self.text_overrides = {}

    # Layout and containers.
    def __enter__(self): return self
    def __exit__(self, *a): return False
    sidebar = property(lambda self: self)
    def columns(self, n, **k): return [self] * (n if isinstance(n, int) else len(n))
    def tabs(self, labels, **k): return [self] * len(labels)
    def spinner(self, *a, **k): return self
    def expander(self, *a, **k): return self
    def container(self, *a, **k): return self
    def set_page_config(self, *a, **k): pass

    # Text output.
    def _noop(self, *a, **k): pass
    title = markdown = header = subheader = caption = write = _noop
    divider = success = info = warning = stop = rerun = _noop
    def error(self, *a, **k): self.counts["error"] += 1

    # Widgets return their default so downstream logic executes.
    def selectbox(self, label, options, index=0, **k): return list(options)[index]
    def multiselect(self, label, options, default=None, **k):
        return list(default) if default is not None else list(options)
    def text_input(self, label, value="", **k): return self.text_overrides.get(label, value)
    def date_input(self, label, value=None, **k): return value
    def slider(self, label, mn, mx, value=None, **k): return value if value is not None else mn
    def number_input(self, label, value=0, **k): return value
    def checkbox(self, label, value=False, **k): return value
    def button(self, *a, **k):
        self.counts["button"] += 1
        return self.force_buttons

    # Rendered components.
    def metric(self, *a, **k): self.counts["metric"] += 1
    def line_chart(self, *a, **k): self.counts["line_chart"] += 1
    def bar_chart(self, *a, **k): self.counts["line_chart"] += 1
    def pyplot(self, *a, **k): self.counts["pyplot"] += 1
    def dataframe(self, *a, **k): self.counts["dataframe"] += 1
    def table(self, *a, **k): self.counts["dataframe"] += 1

    # Cache decorators pass through.
    def cache_data(self, *a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f
    cache_resource = cache_data


def run() -> None:
    import shutil
    import tempfile
    from datetime import date
    from build_store import build
    tmp = tempfile.mkdtemp()
    build('fixture', tmp, [], date(2015, 1, 1), date(2023, 12, 31), backend='auto', seed=0)
    shim = _Shim()
    shim.text_overrides = {'Store path': tmp}
    mod = types.ModuleType("streamlit")
    for name in dir(shim):
        if not name.startswith("__"):
            setattr(mod, name, getattr(shim, name))
    # session_state and the cache decorators must be the live objects.
    mod.session_state = shim.session_state
    mod.cache_data = shim.cache_data
    mod.cache_resource = shim.cache_resource
    sys.modules["streamlit"] = mod

    runpy.run_path("app.py", run_name="__main__")

    c = shim.counts
    print(f"app executed. rendered: metrics={c['metric']} line_charts={c['line_chart']} "
          f"pyplots={c['pyplot']} dataframes={c['dataframe']} buttons={c['button']} "
          f"errors={c['error']}")
    assert c["error"] == 0, "app hit an st.error path (store missing?)"
    assert c["metric"] >= 8, "expected metrics across tabs"
    assert c["line_chart"] >= 1 and c["pyplot"] >= 1, "expected charts to render"
    assert c["dataframe"] >= 1, "expected at least one table"
    shutil.rmtree(tmp, ignore_errors=True)
    print("RESULT: PASS")


if __name__ == "__main__":
    run()
