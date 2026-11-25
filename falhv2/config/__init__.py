"""Project defaults and per-run configuration overrides."""
import runpy
from pathlib import Path
from types import SimpleNamespace

from . import defaults as settings


def load_config(path=None):
    """Select a Python config in the entry point; omitted fields inherit defaults."""
    if path is None:
        return settings
    path = Path(path).expanduser().resolve()
    if path.suffix != '.py':
        raise ValueError('--config must name a Python configuration file')
    selected = runpy.run_path(str(path))
    values = {key: value for key, value in vars(settings).items() if key.isupper()}
    for key, value in selected.items():
        if not key.isupper():
            continue
        if isinstance(values.get(key), Path) and isinstance(value, (str, Path)):
            value = Path(value).expanduser()
            value = value if value.is_absolute() else path.parent / value
        if isinstance(value, dict) and isinstance(values.get(key), dict):
            value = {**values[key], **value}
        values[key] = value
    return SimpleNamespace(**values)
