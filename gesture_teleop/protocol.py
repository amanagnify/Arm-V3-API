from __future__ import annotations

"""Single source of truth for the host <-> firmware control packet (V3).

Every joint is transmitted as an ABSOLUTE servo angle in [0, 180] degrees.
Quantization is UNIFORM for all six channels, which removes the per-joint
decode-range coupling that the old V2 packet had (host quantized lower over
[0,45] while firmware decoded it over [90,180]).

V3 packet layout (19 bytes, big-endian):

    byte 0     : MAGIC   (0xA6)
    byte 1     : VERSION (0x03)
    byte 2..3  : sequence (uint16)
    byte 4     : mode     'A' active | 'H' hold | 'M' home
    byte 5     : mask     (0x3F -> all six joints present)
    byte 6..7  : base    angle q16
    byte 8..9  : p1      angle q16   (lower pitch)
    byte 10..11: p2      angle q16   (middle pitch, host p2 direction is configurable)
    byte 12..13: p3      angle q16   (upper pitch)
    byte 14..15: roll    angle q16
    byte 16..17: gripper angle q16   (absolute servo angle: open=166.5, closed=36)
    byte 18    : checksum = XOR of bytes 0..17

Quant / dequant (firmware must mirror this):
    q     = round(clamp(angle, 0, 180) / 180 * 65535)
    angle = q / 65535 * 180

The host converts the gripper open fraction to a servo angle BEFORE building
the packet, so all six channels decode identically. Firmware is responsible
for clamping each decoded angle to that joint's own safe mechanical limit
(base 0-180, p1 0-100, p2 45-180, p3 0-180, roll 0-180, gripper 36-166.5).
The p2 direction is handled host-side by `config.mapping.p2_invert`.
"""

MAGIC = 0xA6
VERSION = 0x03
MASK_ALL = 0x3F          # 6 joints: base|p1|p2|p3|roll|gripper
PACKET_LEN = 19
ANGLE_MAX_DEG = 180.0

JOINT_ORDER = ("base", "p1", "p2", "p3", "roll", "gripper")

VALID_MODES = ("A", "H", "M")


def quantize_angle(angle_deg: float) -> int:
    v = 0.0 if angle_deg < 0.0 else ANGLE_MAX_DEG if angle_deg > ANGLE_MAX_DEG else angle_deg
    return int(round(v / ANGLE_MAX_DEG * 65535.0))


def dequantize_angle(q: int) -> float:
    return (q & 0xFFFF) / 65535.0 * ANGLE_MAX_DEG


def build_packet(
    sequence: int,
    mode: str,
    base_deg: float,
    p1_deg: float,
    p2_deg: float,
    p3_deg: float,
    roll_deg: float,
    gripper_deg: float,
) -> bytes:
    """Encode one control frame. All six values are absolute servo angles in
    degrees (the host has already converted the gripper fraction to its servo
    angle, open=166.5 / closed=36)."""
    seq = sequence & 0xFFFF
    mode_char = mode[:1] if mode else "H"
    if mode_char not in VALID_MODES:
        raise ValueError(f"Unsupported V3 mode: {mode_char!r}")
    mode_byte = ord(mode_char)

    values = (
        base_deg,
        p1_deg,
        p2_deg,
        p3_deg,
        roll_deg,
        gripper_deg,
    )

    pkt = bytearray(PACKET_LEN)
    pkt[0] = MAGIC
    pkt[1] = VERSION
    pkt[2] = (seq >> 8) & 0xFF
    pkt[3] = seq & 0xFF
    pkt[4] = mode_byte & 0xFF
    pkt[5] = MASK_ALL

    idx = 6
    for value in values:
        q = quantize_angle(value)
        pkt[idx] = (q >> 8) & 0xFF
        pkt[idx + 1] = q & 0xFF
        idx += 2

    checksum = 0
    for i in range(PACKET_LEN - 1):
        checksum ^= pkt[i]
    pkt[PACKET_LEN - 1] = checksum & 0xFF

    return bytes(pkt)


def parse_packet(pkt: bytes) -> dict | None:
    """Reference decoder mirroring the firmware spec. Used by tests and by the
    diagnostic readers; NOT used on the hot transmit path."""
    if len(pkt) != PACKET_LEN:
        return None
    if pkt[0] != MAGIC or pkt[1] != VERSION or pkt[5] != MASK_ALL:
        return None
    mode = chr(pkt[4])
    if mode not in VALID_MODES:
        return None

    checksum = 0
    for i in range(PACKET_LEN - 1):
        checksum ^= pkt[i]
    if (checksum & 0xFF) != pkt[PACKET_LEN - 1]:
        return None

    def rd(offset: int) -> float:
        return dequantize_angle((pkt[offset] << 8) | pkt[offset + 1])

    return {
        "seq": (pkt[2] << 8) | pkt[3],
        "mode": mode,
        "base": rd(6),
        "p1": rd(8),
        "p2": rd(10),
        "p3": rd(12),
        "roll": rd(14),
        "gripper": rd(16),
    }
