#!/usr/bin/env python3
#======================================================================#
# Module:      NGV_SensorPractice_13_sensorsim.py
# Description: Sensor Practice + Spectator Follow + Manual Control
#
# Authors: Seokhwan Jeong (shjeong00@hanyang.ac.kr)
#
# Revision History
#      Feb  11, 2026: Seokhwan Jeong - Created.
#      Feb  11, 2026: Seokhwan Jeong - LiDAR background fix + colormap option
#======================================================================#

import os
import time
import math
import argparse
import random
import configparser
import colorsys

import numpy as np

try:
    import pygame
    from pygame.locals import K_ESCAPE
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame is installed")

import carla

import json
import threading
from pathlib import Path
from PIL import Image as PILImage


# ==============================
# INI
# ==============================
INI_PATH_DEFAULT = "NGV_sensorpractice.ini"


# ==============================
# Utility
# ==============================
def _get_bool(cfg, section, key, fallback=False):
    try:
        return cfg.getboolean(section, key, fallback=fallback)
    except Exception:
        return fallback

def _get_float(cfg, section, key, fallback=0.0):
    try:
        return cfg.getfloat(section, key, fallback=fallback)
    except Exception:
        return fallback

def _get_int(cfg, section, key, fallback=0):
    try:
        return cfg.getint(section, key, fallback=fallback)
    except Exception:
        return fallback

def _get_str(cfg, section, key, fallback=""):
    try:
        return cfg.get(section, key, fallback=fallback)
    except Exception:
        return fallback

def is_alive(actor):
    try:
        return actor is not None and actor.is_alive
    except Exception:
        return False

def compute_map_center_from_spawnpoints(spawn_points):
    if not spawn_points:
        return carla.Location(0.0, 0.0, 0.0)
    sx = sum(sp.location.x for sp in spawn_points)
    sy = sum(sp.location.y for sp in spawn_points)
    sz = sum(sp.location.z for sp in spawn_points)
    n = float(len(spawn_points))
    return carla.Location(x=sx/n, y=sy/n, z=sz/n)

def parse_display_pos(s: str):
    try:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 2:
            return None
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return None

def parse_transform(cfg, section):
    x = _get_float(cfg, section, "x", 0.0)
    y = _get_float(cfg, section, "y", 0.0)
    z = _get_float(cfg, section, "z", 0.0)
    roll  = _get_float(cfg, section, "roll", 0.0)
    pitch = _get_float(cfg, section, "pitch", 0.0)
    yaw   = _get_float(cfg, section, "yaw", 0.0)
    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
    )

def get_blueprint_attributes_from_section(cfg, section, reserved_keys_lower):
    out = {}
    if not cfg.has_section(section):
        return out
    for k, v in cfg.items(section):
        if k.lower() in reserved_keys_lower:
            continue
        out[k] = v
    return out

def safe_set_attributes(bp, attrs: dict, prefix=""):
    for k, v in attrs.items():
        try:
            if bp.has_attribute(k):
                bp.set_attribute(k, str(v))
            else:
                pass
        except Exception:
            pass


# ==============================
# Color palettes
# ==============================
def build_palette_hsv(n=256, hue_end=0.83):
    """
    HSV hue sweep:
      hue=0.0 red -> hue_end (0.66 blue, 0.83 purple-ish)
    NOTE:
      index 0 is forced to BLACK for background.
      actual coloring uses indices 1..255
    """
    pal = np.zeros((n, 3), dtype=np.uint8)
    pal[0] = (0, 0, 0)  # background fixed BLACK
    for i in range(1, n):
        h = (i / (n - 1)) * hue_end
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        pal[i, 0] = int(r * 255)
        pal[i, 1] = int(g * 255)
        pal[i, 2] = int(b * 255)
    return pal


# ==============================
# Display / Visual Sensor Managers
# ==============================
class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.grid_size = grid_size  # [rows, cols]
        self.window_size = window_size
        self.sensor_list = []
        self.font = pygame.font.SysFont("Arial", 18)

    def set_grid_size(self, grid_size):
        self.grid_size = grid_size

    def get_display_size(self):
        return [int(self.window_size[0] / self.grid_size[1]), int(self.window_size[1] / self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def render(self, overlay_lines=None):
        if self.display is None:
            return

        for s in self.sensor_list:
            s.render()

        if overlay_lines:
            y = 8
            for line in overlay_lines:
                surf = self.font.render(line, True, (240, 240, 240))
                self.display.blit(surf, (10, y))
                y += 22

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            try:
                s.destroy()
            except Exception:
                pass
        self.sensor_list.clear()


class VisualSensorManager:
    def __init__(self, world, display_man, sensor_kind, transform, attached, sensor_options, display_pos,
                 converter_name=None, viz_cfg=None):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor_kind = sensor_kind
        self.sensor_options = sensor_options
        self.converter_name = (converter_name or "").strip()
        self.viz_cfg = viz_cfg or {}

        cm = str(self.viz_cfg.get("lidar_colormap", "RAINBOW")).upper()
        if cm == "RED_BLUE":
            self._lidar_palette = build_palette_hsv(256, hue_end=0.66)
        else:
            self._lidar_palette = build_palette_hsv(256, hue_end=0.83)

        self.sensor = self._init_sensor(sensor_kind, transform, attached, sensor_options)
        self.display_man.add_sensor(self)

    def get_sensor(self):
        return self.sensor

    def _apply_converter(self, image):
        if self.sensor_kind == "DepthCamera":
            name = (self.converter_name or "LogarithmicDepth").lower()
            if name == "depth":
                image.convert(carla.ColorConverter.Depth)
            elif name == "raw":
                image.convert(carla.ColorConverter.Raw)
            else:
                image.convert(carla.ColorConverter.LogarithmicDepth)
            return

        if self.sensor_kind == "SemanticSegCamera":
            name = (self.converter_name or "CityScapesPalette").lower()
            if name == "raw":
                image.convert(carla.ColorConverter.Raw)
            else:
                image.convert(carla.ColorConverter.CityScapesPalette)
            return

        image.convert(carla.ColorConverter.Raw)

    def _init_sensor(self, sensor_kind, transform, attached, sensor_options):
        bp_lib = self.world.get_blueprint_library()
        disp_size = self.display_man.get_display_size()  # [w,h]

        if sensor_kind == "RGBCamera":
            bp = bp_lib.find("sensor.camera.rgb")
            if "image_size_x" not in sensor_options:
                bp.set_attribute("image_size_x", str(disp_size[0]))
            if "image_size_y" not in sensor_options:
                bp.set_attribute("image_size_y", str(disp_size[1]))
            safe_set_attributes(bp, sensor_options, prefix="RGBCamera")
            cam = self.world.spawn_actor(bp, transform, attach_to=attached, attachment_type=carla.AttachmentType.Rigid)
            cam.listen(self._cb_camera_image)
            return cam

        if sensor_kind == "DepthCamera":
            bp = bp_lib.find("sensor.camera.depth")
            if "image_size_x" not in sensor_options:
                bp.set_attribute("image_size_x", str(disp_size[0]))
            if "image_size_y" not in sensor_options:
                bp.set_attribute("image_size_y", str(disp_size[1]))
            safe_set_attributes(bp, sensor_options, prefix="DepthCamera")
            cam = self.world.spawn_actor(bp, transform, attach_to=attached, attachment_type=carla.AttachmentType.Rigid)
            cam.listen(self._cb_camera_image)
            return cam

        if sensor_kind == "SemanticSegCamera":
            bp = bp_lib.find("sensor.camera.semantic_segmentation")
            if "image_size_x" not in sensor_options:
                bp.set_attribute("image_size_x", str(disp_size[0]))
            if "image_size_y" not in sensor_options:
                bp.set_attribute("image_size_y", str(disp_size[1]))
            safe_set_attributes(bp, sensor_options, prefix="SemanticSegCamera")
            cam = self.world.spawn_actor(bp, transform, attach_to=attached, attachment_type=carla.AttachmentType.Rigid)
            cam.listen(self._cb_camera_image)
            return cam

        if sensor_kind == "LiDAR":
            bp = bp_lib.find("sensor.lidar.ray_cast")
            safe_set_attributes(bp, sensor_options, prefix="LiDAR")
            lidar = self.world.spawn_actor(bp, transform, attach_to=attached)
            lidar.listen(self._cb_lidar)
            return lidar

        if sensor_kind == "Radar":
            bp = bp_lib.find("sensor.other.radar")
            safe_set_attributes(bp, sensor_options, prefix="Radar")
            radar = self.world.spawn_actor(bp, transform, attach_to=attached)
            radar.listen(self._cb_radar)
            return radar

        return None

    def _cb_camera_image(self, image):
        self._apply_converter(image)

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]  # BGR->RGB

        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def _cb_lidar(self, image):
        disp_size = self.display_man.get_display_size()  # [w,h]
        w, h = disp_size[0], disp_size[1]

        lidar_range = 2.0 * float(self.sensor_options.get("range", 50.0))

        pts = np.frombuffer(image.raw_data, dtype=np.float32)
        pts = np.reshape(pts, (int(pts.shape[0] / 4), 4))
        xy = pts[:, :2]  # x,y
        intensity = pts[:, 3]  # 0..1
        dist = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)

        xy = np.array(xy)
        zoom = float(self.viz_cfg.get("lidar_zoom", 1.0))
        xy *= (min(disp_size) / max(lidar_range, 1e-6)) * zoom
        xy += (0.5 * w, 0.5 * h)
        xy = np.fabs(xy).astype(np.int32)

        xs = np.clip(xy[:, 0], 0, w - 1)
        ys = np.clip(xy[:, 1], 0, h - 1)

        mode = str(self.viz_cfg.get("lidar_point_mode", "NORMAL")).upper()

        if mode == "NORMAL":
            lidar_img = np.zeros((w, h, 3), dtype=np.uint8)
            lidar_img[xs, ys] = (255, 255, 255)

        else:
            if mode == "DISTANCE":
                dmin = float(self.viz_cfg.get("lidar_dist_min", 0.0))
                dmax = float(self.viz_cfg.get("lidar_dist_max", max(1.0, float(np.max(dist)) if dist.size else 1.0)))
                t = np.clip((dist - dmin) / max(dmax - dmin, 1e-6), 0.0, 1.0)
            else:
                imin = float(self.viz_cfg.get("lidar_intensity_min", 0.0))
                imax = float(self.viz_cfg.get("lidar_intensity_max", 1.0))
                gamma = float(self.viz_cfg.get("lidar_gamma", 1.0))
                t = np.clip((intensity - imin) / max(imax - imin, 1e-6), 0.0, 1.0)
                t = np.power(t, gamma)

            acc = np.zeros((w, h), dtype=np.float32)
            r = int(self.viz_cfg.get("lidar_point_radius", 0))
            if r <= 0:
                np.maximum.at(acc, (xs, ys), t.astype(np.float32))
            else:
                tt = t.astype(np.float32)
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        xs2 = np.clip(xs + dx, 0, w - 1)
                        ys2 = np.clip(ys + dy, 0, h - 1)
                        np.maximum.at(acc, (xs2, ys2), tt)

            pal = self._lidar_palette
            n = pal.shape[0]

            idx = np.zeros((w, h), dtype=np.uint8)  # background -> 0
            mask = acc > 0.0
            idx[mask] = (1 + (acc[mask] * float(n - 2))).astype(np.uint8)

            lidar_img = pal[idx]

        self.surface = pygame.surfarray.make_surface(lidar_img)

    def _cb_radar(self, radar_data):
        disp_size = self.display_man.get_display_size()
        w, h = disp_size[0], disp_size[1]
        radar_range = float(self.sensor_options.get("range", 80.0))

        radar_img = np.zeros((w, h, 3), dtype=np.uint8)

        ORIGIN_Y_RATIO = 0.88
        ORIGIN_X_RATIO = 0.50
        ZOOM = 1.0
        cx = int(w * ORIGIN_X_RATIO)
        cy = int(h * ORIGIN_Y_RATIO)
        scale = (h * 0.95) / max(radar_range, 1e-6)
        scale *= ZOOM

        mode = str(self.viz_cfg.get("radar_point_mode", "NORMAL")).upper()
        vmin_d = float(self.viz_cfg.get("radar_depth_min", 0.0))
        vmax_d = float(self.viz_cfg.get("radar_depth_max", radar_range))
        vmin_v = float(self.viz_cfg.get("radar_vel_min", -20.0))
        vmax_v = float(self.viz_cfg.get("radar_vel_max",  20.0))

        for detect in radar_data:
            az = detect.azimuth
            depth = detect.depth
            vel = detect.velocity

            x = depth * np.cos(az)
            y = depth * np.sin(az)

            px = int(cx + y * scale)
            py = int(cy - x * scale)

            if not (0 <= px < w and 0 <= py < h):
                continue

            if mode == "NORMAL":
                col = (255, 255, 255)

            elif mode == "DEPTH":
                t = np.clip((depth - vmin_d) / max(vmax_d - vmin_d, 1e-6), 0.0, 1.0)
                g = int(t * 255)
                col = (g, g, g)

            else:
                max_abs = max(abs(vmin_v), abs(vmax_v), 1e-6)
                s = float(np.clip(vel / max_abs, -1.0, 1.0))  # -1..1
                a = int(abs(s) * 255)
                base = 255 - a

                if abs(s) < 0.05:
                    col = (255, 255, 255)
                elif s < 0:
                    col = (255, base, base)
                else:
                    col = (base, base, 255)

            r = int(self.viz_cfg.get("radar_point_radius", 2))
            if r <= 0:
                radar_img[px, py] = col
            else:
                x0 = max(px - r, 0)
                x1 = min(px + r + 1, w)
                y0 = max(py - r, 0)
                y1 = min(py + r + 1, h)
                radar_img[x0:x1, y0:y1] = col


        self.surface = pygame.surfarray.make_surface(radar_img)

    def render(self):
        if self.surface is None or self.display_pos is None:
            return
        offset = self.display_man.get_display_offset(self.display_pos)
        self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        try:
            if self.sensor is not None:
                self.sensor.stop()
        except Exception:
            pass
        try:
            if self.sensor is not None:
                self.sensor.destroy()
        except Exception:
            pass

def make_throttled_printer(name, interval_sec, state_dict):
    def allow_print():
        now = time.time()
        last = state_dict.get(name, 0.0)
        if now - last >= interval_sec:
            state_dict[name] = now
            return True
        return False
    return allow_print

def attach_terminal_sensor(world, bp_id, transform, ego, attrs, callback):
    bp = world.get_blueprint_library().find(bp_id)
    safe_set_attributes(bp, attrs, prefix=bp_id)
    sensor = world.spawn_actor(bp, transform, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
    sensor.listen(callback)
    return sensor

SPECTATOR_MODES = ["CHASE", "TOP", "FIRST_PERSON"]

def spectator_follow_tf(ego_tf, mode, spec_cfg):
    if mode == "CHASE":
        back = float(spec_cfg["chase_back"])
        up = float(spec_cfg["chase_up"])
        pitch = float(spec_cfg["chase_pitch"])
        offset = carla.Location(x=-back, y=0.0, z=up)
        cam_loc = ego_tf.transform(offset)
        cam_rot = carla.Rotation(pitch=pitch, yaw=ego_tf.rotation.yaw, roll=0.0)
        return carla.Transform(cam_loc, cam_rot)

    if mode == "TOP":
        top_z = float(spec_cfg["top_z"])
        top_pitch = float(spec_cfg["top_pitch"])
        top_yaw = float(spec_cfg["top_yaw"])
        loc = carla.Location(x=ego_tf.location.x, y=ego_tf.location.y, z=ego_tf.location.z + top_z)
        rot = carla.Rotation(pitch=top_pitch, yaw=top_yaw, roll=0.0)
        return carla.Transform(loc, rot)

    fp_front = float(spec_cfg["fp_front"])
    fp_up = float(spec_cfg["fp_up"])
    fp_pitch = float(spec_cfg["fp_pitch"])
    offset = carla.Location(x=fp_front, y=0.0, z=fp_up)
    cam_loc = ego_tf.transform(offset)
    cam_rot = carla.Rotation(pitch=fp_pitch, yaw=ego_tf.rotation.yaw, roll=0.0)
    return carla.Transform(cam_loc, cam_rot)

def map_view_tf_from_cfg(map_center, spec_cfg):
    z = float(spec_cfg["map_view_z"])
    pitch = float(spec_cfg["map_view_pitch"])
    yaw = float(spec_cfg["map_view_yaw"])
    return carla.Transform(
        carla.Location(x=map_center.x, y=map_center.y, z=map_center.z + z),
        carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)
    )

def build_vehicle_control(keys, steer_cache, reverse_toggle, manual_cfg):
    steer_step = float(manual_cfg["steer_step"])
    steer_return = float(manual_cfg["steer_return"])
    throttle_val = float(manual_cfg["throttle"])
    brake_val = float(manual_cfg["brake"])

    if keys[pygame.K_a]:
        steer_cache -= steer_step
    elif keys[pygame.K_d]:
        steer_cache += steer_step
    else:
        steer_cache *= steer_return

    steer_cache = max(-1.0, min(1.0, steer_cache))

    throttle = 0.0
    brake = 0.0
    if keys[pygame.K_w] and not keys[pygame.K_s]:
        throttle = throttle_val
    elif keys[pygame.K_s] and not keys[pygame.K_w]:
        brake = brake_val

    hand_brake = bool(keys[pygame.K_SPACE])

    ctrl = carla.VehicleControl(
        throttle=float(throttle),
        steer=float(steer_cache),
        brake=float(brake),
        hand_brake=hand_brake,
        reverse=bool(reverse_toggle)
    )
    return ctrl, steer_cache

VISUAL_SECTIONS_ORDER = ["RGBCamera", "DepthCamera", "SemanticSegCamera", "LiDAR", "Radar"]

def load_ini(path: str):
    cfg = configparser.ConfigParser()
    ok = cfg.read(path, encoding="utf-8")
    if not ok:
        return None
    return cfg

def extract_visualization_cfg(cfg):
    return {
        "lidar_point_mode": _get_str(cfg, "visualization", "lidar_point_mode", "NORMAL").upper(),
        "lidar_colormap": _get_str(cfg, "visualization", "lidar_colormap", "RAINBOW").upper(),
        "lidar_dist_min": _get_float(cfg, "visualization", "lidar_dist_min", 0.0),
        "lidar_dist_max": _get_float(cfg, "visualization", "lidar_dist_max", 100.0),
        "lidar_intensity_min": _get_float(cfg, "visualization", "lidar_intensity_min", 0.0),
        "lidar_intensity_max": _get_float(cfg, "visualization", "lidar_intensity_max", 1.0),
        "lidar_gamma": _get_float(cfg, "visualization", "lidar_gamma", 1.0),
        "lidar_zoom": _get_float(cfg, "visualization", "lidar_zoom", 1.0),
        "lidar_point_radius": _get_float(cfg, "visualization", "lidar_point_radius", 0),

        "radar_point_mode": _get_str(cfg, "visualization", "radar_point_mode", "NORMAL").upper(),
        "radar_depth_min": _get_float(cfg, "visualization", "radar_depth_min", 0.0),
        "radar_depth_max": _get_float(cfg, "visualization", "radar_depth_max", 80.0),
        "radar_vel_min": _get_float(cfg, "visualization", "radar_vel_min", -20.0),
        "radar_vel_max": _get_float(cfg, "visualization", "radar_vel_max", 20.0),
        "radar_point_radius": _get_float(cfg, "visualization", "radar_point_radius", 2)
    }

def extract_runtime_cfg(cfg):
    reload_interval = max(0.2, _get_float(cfg, "sensor_system", "reload_interval_sec", 1.0))
    term_interval = max(0.1, _get_float(cfg, "sensor_system", "terminal_print_interval_sec", 0.5))

    rows = max(1, _get_int(cfg, "display", "rows", 2))
    cols = max(1, _get_int(cfg, "display", "cols", 3))

    spec_cfg = {
        "chase_back": _get_float(cfg, "spectator", "chase_back", 8.0),
        "chase_up": _get_float(cfg, "spectator", "chase_up", 3.0),
        "chase_pitch": _get_float(cfg, "spectator", "chase_pitch", -10.0),

        "top_z": _get_float(cfg, "spectator", "top_z", 60.0),
        "top_pitch": _get_float(cfg, "spectator", "top_pitch", -89.0),
        "top_yaw": _get_float(cfg, "spectator", "top_yaw", 0.0),

        "fp_front": _get_float(cfg, "spectator", "fp_front", 1.8),
        "fp_up": _get_float(cfg, "spectator", "fp_up", 1.35),
        "fp_pitch": _get_float(cfg, "spectator", "fp_pitch", -2.0),

        "map_view_z": _get_float(cfg, "spectator", "map_view_z", 220.0),
        "map_view_pitch": _get_float(cfg, "spectator", "map_view_pitch", -89.0),
        "map_view_yaw": _get_float(cfg, "spectator", "map_view_yaw", -90.0),
    }

    manual_cfg = {
        "throttle": _get_float(cfg, "manual_vehicle", "throttle", 0.7),
        "brake": _get_float(cfg, "manual_vehicle", "brake", 1.0),
        "steer_step": _get_float(cfg, "manual_vehicle", "steer_step", 0.03),
        "steer_return": _get_float(cfg, "manual_vehicle", "steer_return", 0.80),
    }

    viz_cfg = extract_visualization_cfg(cfg)

    return {
        "reload_interval": reload_interval,
        "term_interval": term_interval,
        "grid": (rows, cols),
        "spec_cfg": spec_cfg,
        "manual_cfg": manual_cfg,
        "viz_cfg": viz_cfg,
    }

def spawn_sensors_from_ini(world, display_manager, ego_vehicle, cfg, term_print_interval, viz_cfg):
    rows, cols = display_manager.grid_size
    auto_positions = [(r, c) for r in range(rows) for c in range(cols)]
    used_positions = set()

    def alloc_pos(preferred):
        if preferred and preferred in auto_positions and preferred not in used_positions:
            used_positions.add(preferred)
            return preferred
        for p in auto_positions:
            if p not in used_positions:
                used_positions.add(p)
                return p
        return None

    for sec in VISUAL_SECTIONS_ORDER:
        if not cfg.has_section(sec):
            continue
        if not _get_bool(cfg, sec, "enabled", False):
            continue

        tf = parse_transform(cfg, sec)
        reserved = {"enabled", "display_pos", "x", "y", "z", "roll", "pitch", "yaw", "color_converter"}
        attrs = get_blueprint_attributes_from_section(cfg, sec, reserved_keys_lower=reserved)

        dp_s = _get_str(cfg, sec, "display_pos", "").strip()
        disp_pos = parse_display_pos(dp_s) if dp_s else None
        disp_pos = alloc_pos(disp_pos)

        converter = _get_str(cfg, sec, "color_converter", "").strip()

        mgr = VisualSensorManager(
            world=world,
            display_man=display_manager,
            sensor_kind=sec,
            transform=tf,
            attached=ego_vehicle,
            sensor_options=attrs,
            display_pos=disp_pos,
            converter_name=converter,
            viz_cfg=viz_cfg
        )
        if mgr.get_sensor() is not None:
            print(f"[SENSOR MANAGER] Visual ON: {sec} at {disp_pos}")
        else:
            print(f"[SENSOR MANAGER] Failed to spawn visual sensor: {sec}")

    terminal_sensors = []
    print_state = {}

    if cfg.has_section("GNSS") and _get_bool(cfg, "GNSS", "enabled", False):
        tf = parse_transform(cfg, "GNSS")
        reserved = {"enabled", "x", "y", "z", "roll", "pitch", "yaw"}
        attrs = get_blueprint_attributes_from_section(cfg, "GNSS", reserved_keys_lower=reserved)
        allow = make_throttled_printer("GNSS", term_print_interval, print_state)

        def cb_gnss(data):
            if allow():
                print(f"[GNSS] lat={data.latitude:.6f}, lon={data.longitude:.6f}, alt={data.altitude:.2f}")

        s = attach_terminal_sensor(world, "sensor.other.gnss", tf, ego_vehicle, attrs, cb_gnss)
        terminal_sensors.append(s)
        print("[SENSOR MANAGER] Terminal ON: GNSS")

    if cfg.has_section("IMU") and _get_bool(cfg, "IMU", "enabled", False):
        tf = parse_transform(cfg, "IMU")
        reserved = {"enabled", "x", "y", "z", "roll", "pitch", "yaw"}
        attrs = get_blueprint_attributes_from_section(cfg, "IMU", reserved_keys_lower=reserved)
        allow = make_throttled_printer("IMU", term_print_interval, print_state)

        def cb_imu(data):
            if allow():
                ax, ay, az = data.accelerometer.x, data.accelerometer.y, data.accelerometer.z
                gx, gy, gz = data.gyroscope.x, data.gyroscope.y, data.gyroscope.z
                print(f"[IMU] acc=({ax:+.3f},{ay:+.3f},{az:+.3f}) | gyro=({gx:+.3f},{gy:+.3f},{gz:+.3f}) | compass={data.compass:+.3f}")

        s = attach_terminal_sensor(world, "sensor.other.imu", tf, ego_vehicle, attrs, cb_imu)
        terminal_sensors.append(s)
        print("[SENSOR MANAGER] Terminal ON: IMU")

    if cfg.has_section("Collision") and _get_bool(cfg, "Collision", "enabled", False):
        tf = parse_transform(cfg, "Collision")
        reserved = {"enabled", "x", "y", "z", "roll", "pitch", "yaw"}
        attrs = get_blueprint_attributes_from_section(cfg, "Collision", reserved_keys_lower=reserved)
        allow = make_throttled_printer("Collision", term_print_interval, print_state)

        def cb_col(ev):
            if allow():
                other = ev.other_actor.type_id if ev.other_actor else "None"
                intensity = math.sqrt(ev.normal_impulse.x**2 + ev.normal_impulse.y**2 + ev.normal_impulse.z**2)
                print(f"[COLLISION] other={other} | intensity={intensity:.1f}")

        s = attach_terminal_sensor(world, "sensor.other.collision", tf, ego_vehicle, attrs, cb_col)
        terminal_sensors.append(s)
        print("[SENSOR MANAGER] Terminal ON: Collision")

    if cfg.has_section("LaneInvasion") and _get_bool(cfg, "LaneInvasion", "enabled", False):
        tf = parse_transform(cfg, "LaneInvasion")
        reserved = {"enabled", "x", "y", "z", "roll", "pitch", "yaw"}
        attrs = get_blueprint_attributes_from_section(cfg, "LaneInvasion", reserved_keys_lower=reserved)
        allow = make_throttled_printer("LaneInvasion", term_print_interval, print_state)

        def cb_lane(ev):
            if allow():
                marks = [str(x.type) for x in ev.crossed_lane_markings] if ev.crossed_lane_markings else []
                print(f"[LANE] crossed={marks}")

        s = attach_terminal_sensor(world, "sensor.other.lane_invasion", tf, ego_vehicle, attrs, cb_lane)
        terminal_sensors.append(s)
        print("[SENSOR MANAGER] Terminal ON: LaneInvasion")

    if cfg.has_section("Obstacle") and _get_bool(cfg, "Obstacle", "enabled", False):
        tf = parse_transform(cfg, "Obstacle")
        reserved = {"enabled", "x", "y", "z", "roll", "pitch", "yaw"}
        attrs = get_blueprint_attributes_from_section(cfg, "Obstacle", reserved_keys_lower=reserved)
        allow = make_throttled_printer("Obstacle", term_print_interval, print_state)

        def cb_obs(ev):
            if allow():
                other = ev.other_actor.type_id if ev.other_actor else "None"
                print(f"[OBSTACLE] other={other} | distance={ev.distance:.2f} m")

        s = attach_terminal_sensor(world, "sensor.other.obstacle", tf, ego_vehicle, attrs, cb_obs)
        terminal_sensors.append(s)
        print("[SENSOR MANAGER] Terminal ON: Obstacle")

    return terminal_sensors

def destroy_terminal_sensors(terminal_sensors):
    for s in terminal_sensors:
        try:
            if s is not None:
                s.stop()
        except Exception:
            pass
        try:
            if s is not None:
                s.destroy()
        except Exception:
            pass
    terminal_sensors.clear()


# ==============================
# Data Recording
# ==============================

def _weather_to_str(weather) -> str:
    parts = []
    if weather.cloudiness > 60:
        parts.append("Cloudy")
    elif weather.cloudiness > 20:
        parts.append("PartlyCloudy")
    else:
        parts.append("Clear")
    if weather.precipitation > 40:
        parts.append("HeavyRain")
    elif weather.precipitation > 5:
        parts.append("Rain")
    if weather.fog_density > 30:
        parts.append("Foggy")
    return "_".join(parts) if parts else "Clear"


class _FrameBuffer:
    """Thread-safe single-slot frame buffer for the recording camera."""
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def put(self, arr):
        with self._lock:
            self._frame = arr

    def get(self):
        with self._lock:
            return self._frame


class DataRecorder:
    """Records per-step RGB frames and vehicle controls to disk.

    Output layout::

        <output_dir>/episodes/
            ep_0000/
                frames/0000.jpg, 0001.jpg, ...
                episode.json    # list of per-step dicts
                metadata.json   # map, weather, instruction, spawn
    """

    def __init__(self, output_dir: str, steps_per_episode: int):
        self._root = Path(output_dir) / "episodes"
        self._spe = steps_per_episode
        self._ep_dir = None
        self._frames_dir = None
        self._steps = []
        self._step_idx = 0

    @property
    def steps_per_episode(self):
        return self._spe

    def start_episode(self, ep_idx: int, map_name: str, weather_str: str, spawn_tf):
        self._ep_dir = self._root / f"ep_{ep_idx:04d}"
        self._frames_dir = self._ep_dir / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._steps = []
        self._step_idx = 0
        meta = {
            "episode": ep_idx,
            "map": map_name,
            "weather": weather_str,
            "spawn_x": float(spawn_tf.location.x),
            "spawn_y": float(spawn_tf.location.y),
            "spawn_z": float(spawn_tf.location.z),
            "spawn_yaw": float(spawn_tf.rotation.yaw),
            "instruction": f"Drive on {map_name}. Weather: {weather_str}. Follow traffic rules.",
        }
        with open(self._ep_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[RECORDER] Episode {ep_idx:04d} started → {self._ep_dir}")

    def record_step(self, rgb_array, control, velocity, timestamp: float):
        if rgb_array is None or self._ep_dir is None:
            return
        frame_path = self._frames_dir / f"{self._step_idx:04d}.jpg"
        PILImage.fromarray(rgb_array).save(str(frame_path), quality=95)
        self._steps.append({
            "step": self._step_idx,
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
            "hand_brake": float(control.hand_brake),
            "reverse": float(control.reverse),
            "vel_x": float(velocity.x),
            "vel_y": float(velocity.y),
            "timestamp": timestamp,
        })
        self._step_idx += 1

    def end_episode(self):
        if self._ep_dir is None:
            return
        with open(self._ep_dir / "episode.json", "w") as f:
            json.dump(self._steps, f)
        print(f"[RECORDER] Episode done: {self._step_idx} steps → {self._ep_dir}")
        self._ep_dir = None
        self._frames_dir = None
        self._steps = []
        self._step_idx = 0


def _default_runtime_cfg():
    """Minimal runtime config used when no INI file is available (recording mode)."""
    return {
        "reload_interval": 99999.0,
        "term_interval": 0.5,
        "grid": (1, 1),
        "spec_cfg": {
            "chase_back": 8.0, "chase_up": 3.0, "chase_pitch": -10.0,
            "top_z": 60.0, "top_pitch": -89.0, "top_yaw": 0.0,
            "fp_front": 1.8, "fp_up": 1.35, "fp_pitch": -2.0,
            "map_view_z": 220.0, "map_view_pitch": -89.0, "map_view_yaw": -90.0,
        },
        "manual_cfg": {
            "throttle": 0.7, "brake": 1.0, "steer_step": 0.03, "steer_return": 0.80,
        },
        "viz_cfg": {
            "lidar_point_mode": "NORMAL", "lidar_colormap": "RAINBOW",
            "lidar_dist_min": 0.0, "lidar_dist_max": 100.0,
            "lidar_intensity_min": 0.0, "lidar_intensity_max": 1.0,
            "lidar_gamma": 1.0, "lidar_zoom": 1.0, "lidar_point_radius": 0,
            "radar_point_mode": "NORMAL", "radar_depth_min": 0.0,
            "radar_depth_max": 80.0, "radar_vel_min": -20.0,
            "radar_vel_max": 20.0, "radar_point_radius": 2,
        },
    }


def run_simulation(args, client, ini_path):
    recording = getattr(args, 'record', False)

    cfg = load_ini(ini_path)
    if cfg is None:
        if recording:
            print(f"[INFO] INI not found ({ini_path}). Using defaults for recording mode.")
            rt = _default_runtime_cfg()
            cfg = None  # sensors skipped below when cfg is None
        else:
            raise RuntimeError(f"Cannot read ini: {ini_path}")
    else:
        rt = extract_runtime_cfg(cfg)
    reload_interval = rt["reload_interval"]
    term_interval = rt["term_interval"]
    grid_rows, grid_cols = rt["grid"]
    spec_cfg = rt["spec_cfg"]
    manual_cfg = rt["manual_cfg"]
    viz_cfg = rt["viz_cfg"]

    # Recording mode
    recording = getattr(args, 'record', False)
    rec_output_dir = getattr(args, 'output_dir', 'carla_episodes')
    max_episodes = getattr(args, 'max_episodes', 50)
    steps_per_episode = getattr(args, 'steps_per_episode', 200)

    if recording and not args.sync:
        print("[RECORDER] WARNING: recording requires --sync, forcing sync mode.")
        args.sync = True

    display_manager = None
    world = None
    original_settings = None

    ego_vehicle = None
    actor_id_list = []
    terminal_sensors = []
    spectator = None
    map_view_tf = None

    frame_buf = None
    rec_cam = None
    recorder = None
    ep_idx = 0
    ep_step = 0

    follow_enabled = True
    cam_mode_idx = 0
    manual_enabled = False
    reverse_toggle = False
    steer_cache = 0.0

    last_mtime = None
    next_ini_check = 0.0

    traffic_manager = None
    tm_port = None

    try:
        world = client.get_world()
        original_settings = world.get_settings()

        traffic_manager = client.get_trafficmanager(8000)
        if args.sync:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
        else:
            traffic_manager.set_synchronous_mode(False)
        tm_port = traffic_manager.get_port()

        spawn_points = world.get_map().get_spawn_points()
        ego_transform = random.choice(spawn_points)

        bp = world.get_blueprint_library().filter("charger_2020")[0]
        ego_vehicle = world.spawn_actor(bp, ego_transform)
        actor_id_list.append(ego_vehicle.id)

        ego_vehicle.set_autopilot(True, tm_port)

        # Dedicated 256×256 front camera for data recording
        if recording:
            frame_buf = _FrameBuffer()
            rec_bp = world.get_blueprint_library().find("sensor.camera.rgb")
            rec_bp.set_attribute("image_size_x", "256")
            rec_bp.set_attribute("image_size_y", "256")
            rec_bp.set_attribute("fov", "90")
            rec_cam_tf = carla.Transform(carla.Location(x=1.6, z=1.7))
            rec_cam = world.spawn_actor(rec_bp, rec_cam_tf, attach_to=ego_vehicle,
                                        attachment_type=carla.AttachmentType.Rigid)
            actor_id_list.append(rec_cam.id)

            def _on_rec_frame(image):
                # CARLA gives BGRA; convert to RGB
                arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                    (image.height, image.width, 4))
                frame_buf.put(arr[:, :, 2::-1].copy())

            rec_cam.listen(_on_rec_frame)

            map_name = world.get_map().name.split("/")[-1]
            weather_str = _weather_to_str(world.get_weather())
            recorder = DataRecorder(rec_output_dir, steps_per_episode)
            recorder.start_episode(ep_idx, map_name, weather_str, ego_transform)
            print(f"[RECORDER] Recording mode ON — {max_episodes} episodes × {steps_per_episode} steps")

        num_npc = 30
        blueprints = world.get_blueprint_library().filter("vehicle.*")
        blueprints = [
            bp for bp in blueprints
            if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels")) == 4
        ]

        def far_enough(sp):
            return sp.location.distance(ego_transform.location) > 8.0

        candidate_points = [sp for sp in spawn_points if far_enough(sp)]
        random.shuffle(candidate_points)
        num_npc = min(num_npc, len(candidate_points))

        batch = []
        for i in range(num_npc):
            npc_bp = random.choice(blueprints)
            if npc_bp.has_attribute("color"):
                npc_bp.set_attribute("color", random.choice(npc_bp.get_attribute("color").recommended_values))
            if npc_bp.has_attribute("driver_id"):
                npc_bp.set_attribute("driver_id", random.choice(npc_bp.get_attribute("driver_id").recommended_values))
            npc_bp.set_attribute("role_name", "autopilot")

            transform = candidate_points[i]
            batch.append(
                carla.command.SpawnActor(npc_bp, transform).then(
                    carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port)
                )
            )

        responses = client.apply_batch_sync(batch, args.sync)
        for r in responses:
            if not r.error:
                actor_id_list.append(r.actor_id)

        spectator = world.get_spectator()
        map_center = compute_map_center_from_spawnpoints(spawn_points)
        map_view_tf = map_view_tf_from_cfg(map_center, spec_cfg)

        display_manager = DisplayManager(grid_size=[grid_rows, grid_cols], window_size=[args.width, args.height])
        pygame.display.set_caption(
            "SensorPractice | C:CamMode | V:MapTop | Q:Auto<->Manual(WASD) | R:ReverseToggle | ESC/Ctrl+Q Quit"
        )

        if cfg is not None:
            terminal_sensors = spawn_sensors_from_ini(
                world, display_manager, ego_vehicle, cfg, term_interval, viz_cfg
            )
        else:
            terminal_sensors = []

        try:
            last_mtime = os.path.getmtime(ini_path)
        except Exception:
            last_mtime = None
        next_ini_check = time.time() + reload_interval

        call_exit = False
        while True:
            if args.sync:
                world.tick()
                dt = 0.05
            else:
                world.wait_for_tick()
                snap = world.get_snapshot()
                dt = float(snap.timestamp.delta_seconds) if snap else 0.05

            now = time.time()

            if now >= next_ini_check:
                try:
                    mtime = os.path.getmtime(ini_path)
                except Exception:
                    mtime = None

                if mtime is not None and mtime != last_mtime:
                    new_cfg = load_ini(ini_path)
                    if new_cfg is not None:
                        new_rt = extract_runtime_cfg(new_cfg)

                        reload_interval = new_rt["reload_interval"]
                        term_interval = new_rt["term_interval"]
                        new_rows, new_cols = new_rt["grid"]
                        spec_cfg = new_rt["spec_cfg"]
                        manual_cfg = new_rt["manual_cfg"]
                        viz_cfg = new_rt["viz_cfg"]

                        if (new_rows, new_cols) != (grid_rows, grid_cols):
                            grid_rows, grid_cols = new_rows, new_cols
                            display_manager.set_grid_size([grid_rows, grid_cols])

                        display_manager.destroy()
                        display_manager.display.fill((0, 0, 0))
                        pygame.display.flip()
                        destroy_terminal_sensors(terminal_sensors)
                        terminal_sensors = spawn_sensors_from_ini(
                            world, display_manager, ego_vehicle, new_cfg, term_interval, viz_cfg
                        )
                        cfg = new_cfg
                        last_mtime = mtime

                        print("[INI UPDATER] Reloaded new INI file.")
                    else:
                        print("[WARN] INI changed but parse failed. Keep current sensors.")

                next_ini_check = now + reload_interval

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True

                elif event.type == pygame.KEYDOWN:
                    mods = pygame.key.get_mods()

                    if event.key == K_ESCAPE or (event.key == pygame.K_q and (mods & pygame.KMOD_CTRL)):
                        call_exit = True

                    elif event.key == pygame.K_c:
                        cam_mode_idx = (cam_mode_idx + 1) % len(SPECTATOR_MODES)

                    elif event.key == pygame.K_v:
                        follow_enabled = not follow_enabled
                        if not follow_enabled and spectator is not None and map_view_tf is not None:
                            spectator.set_transform(map_view_tf)

                    elif event.key == pygame.K_q:
                        manual_enabled = not manual_enabled
                        reverse_toggle = False
                        steer_cache = 0.0

                        if manual_enabled:
                            ego_vehicle.set_autopilot(False, tm_port)
                            print("[AUTOPILOT] OFF, MANUAL CONTROL ENABLED")
                        else:
                            ego_vehicle.set_autopilot(True, tm_port)
                            print("[AUTOPILOT] ON, MANUAL CONTROL DISABLED")

                    elif event.key == pygame.K_r:
                        if manual_enabled:
                            reverse_toggle = not reverse_toggle

            if call_exit:
                break

            # Per-step recording
            if recording and recorder is not None and is_alive(ego_vehicle):
                snap = world.get_snapshot()
                ts = float(snap.timestamp.elapsed_seconds) if snap else time.time()
                ctrl = ego_vehicle.get_control()
                vel = ego_vehicle.get_velocity()
                recorder.record_step(frame_buf.get(), ctrl, vel, ts)
                ep_step += 1

                if ep_step >= steps_per_episode:
                    recorder.end_episode()
                    ep_idx += 1
                    ep_step = 0

                    if ep_idx >= max_episodes:
                        print(f"[RECORDER] Collected {max_episodes} episodes. Stopping.")
                        call_exit = True
                    else:
                        # Teleport to a new random spawn point and continue
                        ego_transform = random.choice(spawn_points)
                        ego_vehicle.set_transform(ego_transform)
                        ego_vehicle.apply_control(carla.VehicleControl())
                        if args.sync:
                            world.tick()
                        ego_vehicle.set_autopilot(True, tm_port)
                        map_name = world.get_map().name.split("/")[-1]
                        weather_str = _weather_to_str(world.get_weather())
                        recorder.start_episode(ep_idx, map_name, weather_str, ego_transform)

            if manual_enabled and is_alive(ego_vehicle):
                keys = pygame.key.get_pressed()
                ctrl, steer_cache = build_vehicle_control(keys, steer_cache, reverse_toggle, manual_cfg)
                ego_vehicle.apply_control(ctrl)

            if follow_enabled and spectator is not None and is_alive(ego_vehicle):
                mode = SPECTATOR_MODES[cam_mode_idx]
                spectator.set_transform(spectator_follow_tf(ego_vehicle.get_transform(), mode, spec_cfg))

            hud = [
                f"C:{SPECTATOR_MODES[cam_mode_idx]} | V:{'FOLLOW' if follow_enabled else 'MAP_TOP'} | Q:{'MANUAL' if manual_enabled else 'AUTO'} | R(reverse):{'ON' if reverse_toggle else 'OFF'}",
                f"LiDAR:{viz_cfg.get('lidar_point_mode','NORMAL')} ({viz_cfg.get('lidar_colormap','RAINBOW')}) | Radar:{viz_cfg.get('radar_point_mode','NORMAL')} | INI hot-reload:{reload_interval:.1f}s",
            ]
            if recording:
                hud.append(f"[REC] ep={ep_idx}/{max_episodes}  step={ep_step}/{steps_per_episode}")
            display_manager.render(overlay_lines=hud)

    finally:
        if recording and recorder is not None:
            recorder.end_episode()

        try:
            if spectator is not None and map_view_tf is not None:
                spectator.set_transform(map_view_tf)
                if args.sync and world is not None:
                    world.tick()
                elif world is not None:
                    world.wait_for_tick()
        except Exception:
            pass

        try:
            if display_manager is not None:
                display_manager.destroy()
        except Exception:
            pass

        try:
            destroy_terminal_sensors(terminal_sensors)
        except Exception:
            pass

        try:
            if actor_id_list:
                client.apply_batch([carla.command.DestroyActor(x) for x in actor_id_list])
        except Exception:
            pass

        try:
            if world is not None and original_settings is not None:
                world.apply_settings(original_settings)
        except Exception:
            pass


def main():
    argparser = argparse.ArgumentParser(description="CARLA sensor practice + data collector (INI-based)")
    argparser.add_argument("--host", default="127.0.0.1")
    argparser.add_argument("-p", "--port", default=2000, type=int)
    argparser.add_argument("--sync", action="store_true", help="Synchronous mode")
    argparser.add_argument("--async", dest="sync", action="store_false", help="Asynchronous mode")
    argparser.set_defaults(sync=True)
    argparser.add_argument("--res", default="1920x1080", help="window resolution WIDTHxHEIGHT")
    argparser.add_argument("--ini", default=INI_PATH_DEFAULT, help="ini file path")
    # Data recording args
    argparser.add_argument("--record", action="store_true",
                           help="Enable trajectory recording for Stage 3 data collection")
    argparser.add_argument("--output-dir", default="carla_episodes",
                           help="Root directory for recorded episodes (default: carla_episodes)")
    argparser.add_argument("--max-episodes", type=int, default=50,
                           help="Stop after this many episodes (default: 50)")
    argparser.add_argument("--steps-per-episode", type=int, default=200,
                           help="Frames per episode (default: 200)")
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split("x")]

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)

    run_simulation(args, client, args.ini)


if __name__ == "__main__":
    main()
