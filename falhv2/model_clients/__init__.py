"""Model clients. Optional inference dependencies are imported only on demand."""

_LOCAL_KEYS = {"path", "source_path", "device", "fp16", "max_side", "score_threshold"}
_REMOTE_KEYS = {"url", "timeout", "score_threshold"}


def get_cubify_config(config):
    """Validate explicitly supplied model parameters; never read project config."""
    options = {}
    if config is not None:
        overrides = dict(config)
        for old, new in (("server_url", "url"), ("timeout_s", "timeout")):
            if old in overrides:
                if new in overrides:
                    raise TypeError(f"Specify only {new}, not both {old} and {new}")
                overrides[new] = overrides.pop(old)
        options.update(overrides)
    options.setdefault("mode", "local")
    unknown = options.keys() - (_LOCAL_KEYS | _REMOTE_KEYS | {"mode"})
    if unknown:
        raise TypeError(f"Unknown Cubify configuration keys: {sorted(unknown)}")
    if options["mode"] not in ("local", "remote"):
        raise ValueError(f"Unsupported Cubify client mode: {options['mode']!r}")
    return options


def build_cubify_anything(config):
    """Construct a client from parameters supplied by the application entry point."""
    options = get_cubify_config(config)
    if options["mode"] == "local":
        from .cubify_anything import CubifyAnythingLocal
        return CubifyAnythingLocal(**{k:v for k,v in options.items() if k in _LOCAL_KEYS})
    from .cubify_anything import CubifyAnythingRemote
    return CubifyAnythingRemote(options["url"], timeout_s=options.get("timeout", 60.),
                                score_threshold=options.get("score_threshold", .25))
