"""Read RealSense calibration directly with the pyrealsense2 Python SDK.

Parameters go to outputs/. Fill the null fields, then copy the JSON into
data/camera.d435i.json for offline use. Reads calibration without starting video.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from . import config as settings
except ImportError:
    import config as settings

DEFAULT_CONFIG = settings.CAMERA_PATH


class CameraReadError(ValueError):
    pass


def load_sdk():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise CameraReadError("缺少 Python SDK，请在 conda test 环境运行：python -m pip install pyrealsense2") from exc
    return rs


def enum_name(value) -> str:
    return str(value).rsplit(".", 1)[-1]


def device_info(device, rs) -> dict:
    fields = {"Name": "name", "Serial Number": "serial_number", "Firmware Version": "firmware_version",
              "Product Line": "product_line", "Usb Type Descriptor": "usb_type_descriptor"}
    return {label: device.get_info(getattr(rs.camera_info, field)) for label, field in fields.items()
            if device.supports(getattr(rs.camera_info, field))}


def select_device(devices: list, rs, serial: str | None):
    matches = [device for device in devices
               if serial is None or device_info(device, rs).get("Serial Number") == serial]
    if not matches:
        raise CameraReadError("没有找到对应 RealSense 相机，请检查 USB 连接、序列号与设备权限")
    if len(matches) != 1:
        serials = ", ".join(device_info(device, rs).get("Serial Number", "unknown") for device in matches)
        raise CameraReadError(f"检测到多台相机，请用 --serial 选择：{serials}")
    return matches[0]


def video_profiles(device, rs, stream: str) -> list:
    profiles = []
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.is_video_stream_profile() and profile.stream_type() == getattr(rs.stream, stream):
                profiles.append(profile.as_video_stream_profile())
    return profiles


def profile_info(profile) -> dict:
    return {"stream": enum_name(profile.stream_type()).title(), "width": profile.width(),
            "height": profile.height(), "fps": profile.fps(),
            "format": enum_name(profile.format()).upper(), "stream_index": profile.stream_index()}


def select_profile(profiles: list, width: int, height: int, fps: int, image_format: str):
    matches = [profile for profile in profiles if profile.width() == width and profile.height() == height
               and profile.fps() == fps and enum_name(profile.format()).upper() == image_format.upper()]
    if len(matches) != 1:
        raise CameraReadError(f"未找到唯一的 {width}x{height} {image_format} @{fps} 标定模式，请先运行 --list-modes")
    return matches[0]


def make_camera_config(profile) -> dict:
    intr = profile.get_intrinsics()
    mode = profile_info(profile)
    fx, fy, cx, cy = (float(value) for value in (intr.fx, intr.fy, intr.ppx, intr.ppy))
    coefficients = [float(value) for value in intr.coeffs]
    if (intr.width != mode["width"] or intr.height != mode["height"]
            or intr.width <= 0 or intr.height <= 0 or fx <= 0 or fy <= 0
            or len(coefficients) != 5
            or not all(math.isfinite(v) for v in [fx, fy, cx, cy, *coefficients])):
        raise CameraReadError("SDK 返回了不完整或无效的相机内参")
    return {
        "camera": {
            "width": intr.width, "height": intr.height,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "distortion_model": enum_name(intr.model),
            "distortion_coefficients": coefficients,
        },
        "mount": {
            "position_in_base_m": [None, None, None],
            "roll_deg": None, "pitch_deg": None, "yaw_deg": None,
        },
        "observation_quality": {
            "min_distance_m": None, "max_distance_m": None,
        },
    }


def save_config(config: dict, output: Path) -> None:
    content = json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stream", choices=("color", "depth"), default="color")
    parser.add_argument("--format", help="默认 color=BGR8，depth=Z16")
    parser.add_argument("--serial", help="多台相机时明确选择设备")
    parser.add_argument("--list-modes", action="store_true", help="列出分辨率、格式与帧率，不写文件")
    parser.add_argument("--output", type=Path, default=settings.CAMERA_EXPORT_PATH)
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.fps) <= 0:
        parser.error("宽高和 fps 必须为正整数")
    try:
        rs = load_sdk()
        # Keep context alive while accessing device, sensor and profile handles.
        context = rs.context()
        devices = list(context.query_devices())
        if args.list_modes:
            selected = [select_device(devices, rs, args.serial)] if args.serial else devices
            if not selected:
                raise CameraReadError("未检测到 RealSense 相机，请检查 USB 连接和设备权限")
            for device in selected:
                info = device_info(device, rs)
                print(info.get("Name"), info.get("Serial Number"))
                modes = {tuple(profile_info(profile).values()) for profile in video_profiles(device, rs, args.stream)}
                for stream, width, height, fps, fmt, index in sorted(modes):
                    print(f"  {stream} {width}x{height} {fmt} @{fps} Hz (index={index})")
            return 0
        device = select_device(devices, rs, args.serial)
        profile = select_profile(video_profiles(device, rs, args.stream), args.width, args.height,
                                 args.fps, args.format or ("BGR8" if args.stream == "color" else "Z16"))
        config = make_camera_config(profile)
        save_config(config, args.output)
    except (CameraReadError, RuntimeError, OSError) as exc:
        parser.exit(2, f"相机参数读取失败：{exc}\n")
    intr = config["camera"]
    print(f"已通过 Python SDK 读取 {device_info(device, rs).get('Name')} / {args.stream} "
          f"{args.width}x{args.height} @{args.fps} Hz")
    print(f"fx={intr['fx']:.6f}, fy={intr['fy']:.6f}, cx={intr['cx']:.6f}, cy={intr['cy']:.6f}")
    print(f"畸变：{intr['distortion_model']} {intr['distortion_coefficients']}")
    print(f"读取结果：{args.output}")
    print("请人工填写 mount 与 observation_quality 中的 null；本次未验证图像采集。")
    if args.stream == "color":
        print(f"补全后请自行将文件放到：{DEFAULT_CONFIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
