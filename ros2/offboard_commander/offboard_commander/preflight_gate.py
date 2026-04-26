#!/usr/bin/env python3
"""
Pre-flight viability gate for optical-flow hover.

Blocks arming if flow sensor or lidar is unhealthy, or if EKF is configured
not to fuse flow.  Designed to catch sensor-level faults before they show up
as EKF divergence in flight.

Checks (all must pass):
  1. EKF2_OF_CTRL == 1 via MAVLink param query (flow fusion enabled).
  2. /fmu/out/sensor_optical_flow stream rate >= MIN_FLOW_RATE Hz.  The
     plugin still emits messages on the ground even with quality=0 and
     integration_timespan_us=0 (nothing to integrate when stationary), so
     rate is our liveness signal.  Deeper in-flight correctness checks
     (quality >= 100, xy_valid goes True) are performed by hover_test.py
     after takeoff.
  3. /fmu/out/distance_sensor stream rate >= MIN_LIDAR_RATE Hz, no NaN, and
     every sample is one of:
       (a) in-band: min_distance <= current_distance <= max_distance, OR
       (b) below-range on-ground: current_distance == 0 AND signal_quality == 0
     (b) is expected in this sim because the iris_opt_flow drone body
     occludes the lidar at spawn height — the lidar plugin returns 0 with
     signal_quality=0 when the ray hits within min_distance.

Exit code 0 on PASS, 1 on BLOCK; block reasons printed to stderr.

Notes on what this gate does NOT cover:
  - Flow quality threshold (>= 100) cannot be checked pre-arm with camera-flow
    because correlation returns quality=0 without body motion.  That check
    is performed in-flight after the attitude-takeoff phase of hover_test.py.
  - Illumination floor: a lux-sensor proxy is the hardware-module plan; the
    sim version relies on the post-takeoff quality check.
"""

import argparse
import struct
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from px4_msgs.msg import SensorOpticalFlow, DistanceSensor


def query_ekf2_of_ctrl(udp_port: int, timeout_s: float = 3.0):
    """Return EKF2_OF_CTRL int value, or None if query failed."""
    from pymavlink import mavutil
    try:
        mav = mavutil.mavlink_connection(
            f"udpin:0.0.0.0:{udp_port}", source_system=250)
        mav.wait_heartbeat(timeout=timeout_s)
    except Exception as e:
        print(f'  EKF2_OF_CTRL: MAVLink connect failed ({e})', file=sys.stderr)
        return None

    mav.mav.param_request_read_send(
        mav.target_system, mav.target_component,
        b"EKF2_OF_CTRL", -1)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if m and m.param_id == "EKF2_OF_CTRL":
            return struct.unpack('<i', struct.pack('<f', m.param_value))[0]
    return None


class GateSampler(Node):
    def __init__(self, sample_seconds: float):
        super().__init__('preflight_gate')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=30,
        )
        self.flow_samples = []
        self.flow_first_ts = None
        self.flow_last_ts = None
        self.lidar_samples = []
        self.lidar_first_ts = None
        self.lidar_last_ts = None
        self.sample_seconds = sample_seconds
        self.start_wall = time.time()

        self.create_subscription(
            SensorOpticalFlow, '/fmu/out/sensor_optical_flow',
            self._flow_cb, qos)
        self.create_subscription(
            DistanceSensor, '/fmu/out/distance_sensor',
            self._lidar_cb, qos)

    def _flow_cb(self, msg):
        now = time.time()
        if self.flow_first_ts is None:
            self.flow_first_ts = now
        self.flow_last_ts = now
        self.flow_samples.append({
            'quality': int(msg.quality),
            'integ_us': int(msg.integration_timespan_us),
        })

    def _lidar_cb(self, msg):
        now = time.time()
        if self.lidar_first_ts is None:
            self.lidar_first_ts = now
        self.lidar_last_ts = now
        self.lidar_samples.append({
            'distance': float(msg.current_distance),
            'min_m': float(msg.min_distance),
            'max_m': float(msg.max_distance),
            'signal_quality': int(msg.signal_quality),
        })

    def done(self) -> bool:
        return time.time() - self.start_wall >= self.sample_seconds


def main():
    ap = argparse.ArgumentParser(description='Optical-flow pre-flight gate')
    ap.add_argument('--sample-seconds', type=float, default=3.0)
    ap.add_argument('--min-flow-rate', type=float, default=15.0,
                    help='Plugin outputRate=20 Hz; DDS sampling can show ~18-20.')
    ap.add_argument('--min-lidar-rate', type=float, default=10.0,
                    help='Lidar update_rate=20 Hz; use 10 to absorb jitter.')
    ap.add_argument('--mavlink-port', type=int, default=14540)
    ap.add_argument('--skip-mavlink', action='store_true',
                    help='Skip EKF2_OF_CTRL param query')
    args = ap.parse_args()

    print('=== Pre-flight viability gate ===', file=sys.stderr)
    failures = []

    if not args.skip_mavlink:
        of_ctrl = query_ekf2_of_ctrl(args.mavlink_port)
        if of_ctrl is None:
            failures.append(
                'EKF2_OF_CTRL: MAVLink query timed out '
                f'(udpin:{args.mavlink_port})')
        elif of_ctrl != 1:
            failures.append(
                f'EKF2_OF_CTRL={of_ctrl} (expected 1; flow fusion disabled)')
        else:
            print('  EKF2_OF_CTRL=1 OK', file=sys.stderr)

    rclpy.init()
    node = GateSampler(sample_seconds=args.sample_seconds)
    try:
        while rclpy.ok() and not node.done():
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    # Flow stream check
    if node.flow_first_ts is None or len(node.flow_samples) < 2:
        failures.append(
            f'sensor_optical_flow: no samples in {args.sample_seconds}s '
            '(plugin dead or DDS bridge down)')
    else:
        duration = max(node.flow_last_ts - node.flow_first_ts, 1e-3)
        rate = (len(node.flow_samples) - 1) / duration
        if rate < args.min_flow_rate:
            failures.append(
                f'sensor_optical_flow: {rate:.1f} Hz < min {args.min_flow_rate} Hz')
        else:
            integ_ok = sum(1 for s in node.flow_samples if s['integ_us'] > 0)
            print(
                f'  flow: rate={rate:.1f} Hz, {len(node.flow_samples)} samples '
                f'({integ_ok} with integ>0) OK',
                file=sys.stderr)

    # Lidar check
    if node.lidar_first_ts is None or len(node.lidar_samples) < 2:
        failures.append(
            f'distance_sensor: no samples in {args.sample_seconds}s')
    else:
        duration = max(node.lidar_last_ts - node.lidar_first_ts, 1e-3)
        rate = (len(node.lidar_samples) - 1) / duration
        import math
        nan_count = sum(
            1 for s in node.lidar_samples if math.isnan(s['distance']))
        # "healthy" = in-band OR (0.0 AND signal_quality==0) i.e. occluded on-ground.
        healthy = 0
        in_band = 0
        occluded = 0
        for s in node.lidar_samples:
            if math.isnan(s['distance']):
                continue
            lo, hi = s['min_m'], s['max_m']
            if lo <= s['distance'] <= hi:
                healthy += 1
                in_band += 1
            elif s['distance'] == 0.0 and s['signal_quality'] == 0:
                healthy += 1
                occluded += 1
        if rate < args.min_lidar_rate:
            failures.append(
                f'distance_sensor: {rate:.1f} Hz < min {args.min_lidar_rate} Hz')
        elif nan_count > 0:
            failures.append(
                f'distance_sensor: {nan_count}/{len(node.lidar_samples)} NaN samples')
        elif healthy < len(node.lidar_samples):
            bad = len(node.lidar_samples) - healthy
            first_bad = next(
                s for s in node.lidar_samples
                if not math.isnan(s['distance'])
                and not (s['min_m'] <= s['distance'] <= s['max_m'])
                and not (s['distance'] == 0.0 and s['signal_quality'] == 0))
            failures.append(
                f'distance_sensor: {bad}/{len(node.lidar_samples)} invalid '
                f'(first bad: dist={first_bad["distance"]:.3f} m, '
                f'band=[{first_bad["min_m"]:.2f},{first_bad["max_m"]:.2f}], '
                f'sq={first_bad["signal_quality"]})')
        else:
            tag = 'in-band' if in_band else 'occluded-on-ground (dist=0 sq=0)'
            print(
                f'  lidar: rate={rate:.1f} Hz, {tag} '
                f'({in_band} in-band, {occluded} occluded, '
                f'{len(node.lidar_samples)} total) OK',
                file=sys.stderr)

    if failures:
        print('=== PRE-FLIGHT GATE BLOCKED ===', file=sys.stderr)
        for f in failures:
            print(f'  BLOCK: {f}', file=sys.stderr)
        sys.exit(1)
    print('=== PRE-FLIGHT GATE PASS ===', file=sys.stderr)
    sys.exit(0)


if __name__ == '__main__':
    main()
