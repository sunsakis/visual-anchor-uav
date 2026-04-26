# Drone motion sensor output — single schema, two timescales via `reference`.
# PREVIOUS_FRAME: 100 Hz short-window displacement from optical flow.
# ANCHOR: 10 Hz displacement against a stored reference frame; drift-free while LOCKED.

# reference values
uint8 REFERENCE_PREVIOUS_FRAME = 0
uint8 REFERENCE_ANCHOR = 1

# lock_status values (ANCHOR only; PREVIOUS_FRAME stream sets lock_status=0)
uint8 LOCK_IDLE = 0
uint8 LOCK_SEARCHING = 1
uint8 LOCK_LOCKED = 2
uint8 LOCK_DEGRADED = 3
uint8 LOCK_LOST = 4

std_msgs/Header header
uint64 timestamp_end              # measurement end (microseconds, epoch)
uint64 timestamp_start            # earlier frame compared against (microseconds)
uint8 reference                   # PREVIOUS_FRAME or ANCHOR
uint32 reference_id               # anchor id if reference=ANCHOR, 0 otherwise
float32 dx                        # meters, ground-plane, body-frame-flat aligned to anchor yaw
float32 dy                        # meters, ground-plane, body-frame-flat aligned to anchor yaw
float32 dyaw                      # radians
float32[3] cov                    # diagonal variances: [dx, dy, dyaw]
uint8 quality                     # 0-255
uint16 matched_features           # diagnostics
uint8 lock_status                 # LOCK_* (ANCHOR only)
