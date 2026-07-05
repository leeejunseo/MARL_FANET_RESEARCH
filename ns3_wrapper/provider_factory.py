import os

from ns3_wrapper.link_provider import CsvLinkTraceProvider


def _env_bool(name, default=None):
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def build_link_provider(config, num_drones):
    bridge_cfg = config.get("ns3_bridge", {})
    env_enabled = _env_bool("NS3_BRIDGE_ENABLED", default=None)
    enabled = bridge_cfg.get("enabled", False) if env_enabled is None else env_enabled
    if not enabled:
        return None

    provider_type = os.getenv("NS3_BRIDGE_PROVIDER", bridge_cfg.get("provider", "csv_trace"))
    if provider_type != "csv_trace":
        raise ValueError(f"Unsupported ns3 bridge provider: {provider_type}")

    csv_path = os.getenv("NS3_BRIDGE_PATH", bridge_cfg.get("csv_trace_path", ""))
    if not csv_path:
        raise ValueError("ns3_bridge.csv_trace_path must be set when bridge is enabled")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ns3 bridge trace not found: {csv_path}")

    env_strict = _env_bool("NS3_BRIDGE_STRICT", default=None)
    strict = bridge_cfg.get("strict", False) if env_strict is None else env_strict
    return CsvLinkTraceProvider(csv_path=csv_path, num_drones=num_drones, strict=strict)
