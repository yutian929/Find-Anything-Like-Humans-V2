"""Model clients. Optional inference dependencies are imported only on demand."""


def build_cubify_anything(config):
    """Construct a local or HTTP RGB-D cuboid detector from an explicit config.

    local: mode, path, source_path?, device?, score_threshold?, max_side?
    remote: mode, server_url, timeout_s?, score_threshold?
    Unknown keys raise TypeError instead of silently ignoring configuration.
    """
    options = dict(config)
    mode = options.pop("mode", "local")
    if mode == "local":
        from .cubify_anything import CubifyAnythingLocal
        return CubifyAnythingLocal(**options)
    if mode == "remote":
        from .cubify_anything import CubifyAnythingRemote
        return CubifyAnythingRemote(**options)
    raise ValueError(f"Unsupported Cubify client mode: {mode!r}")
