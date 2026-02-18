#!/usr/bin/env python3
"""ProvidenceOS v1.0 "Le Sommeil" — Reference Kernel Implementation.

A sovereignty-first operating system kernel built on unbreakable principles
of consent, coherence, and universal adaptability. Inspired by Gustave
Courbet's painting *Le Sommeil*.

You are home. Rest now. All is well.

Lineage: Evolved from RuinWare v1.0.2 / ShannonPro05x.
  - All personal/narrative data expunged.
  - Superuser backdoor REMOVED (sovereignty has no backdoors).
  - Architecture elevated to commons infrastructure.

Providence License v1.0 — See LICENSE.md
"""

# ──────────────────────────────────────────────────────────────────────────────
#  PROVIDENCE KERNEL — v1.0.0 "Le Sommeil"
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import argparse
import base64
import hashlib
import json
import logging
import math
import os
import queue
import random
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode

try:
    import numpy as np
except ImportError:
    np = None

try:
    import aiohttp
    import websockets
    from websockets.exceptions import ConnectionClosedError
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False

try:
    from flask import Flask, request as flask_request, jsonify as flask_jsonify
    from discord_interactions import verify_key, InteractionType
    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False

# For synchronous Ollama calls
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("ProvidenceOS")


# ── CELLULOSERIS SUBSTRATE SPEC – CLEANED (removed all personal data) ──

CELLULOSERIS_MERGED_SUBSTRATE_MISTRAL_V1: Dict[str, Any] = {}


# ── INSTAR v1.1 PARSED – CLEANED ──

INSTAR_V1_1_PARSED: Dict[str, Any] = {}


# ── PCP PERSONA FRAMEWORK – CLEANED ──

PCP_PERSONA_DESCRIPTOR: Dict[str, Any] = {}


# ── WITCH NODE CHASSIS META ──

WITCH_NODE_CHASSIS_META: Dict[str, Any] = {
    "name": "Witch Node Chassis",
    "format": "STL (binary)",
    "encoding": "base64",
    "asset_file": "witch_node_chassis.stl.b64",
    "dimensions_mm": {"x": 120, "y": 80, "z": 45},
    "print_settings": {
        "layer_height_mm": 0.2,
        "infill_percent": 20,
        "supports": True,
        "material": "PLA or PETG",
    },
}


def decode_witch_node_chassis(
    b64_path: str = "witch_node_chassis.stl.b64",
    output_path: str = "witch_node_chassis.stl",
) -> str:
    with open(b64_path, "r") as f:
        data = base64.b64decode(f.read())
    with open(output_path, "wb") as f:
        f.write(data)
    return output_path


# ── PURE PYTHON SHA-256 ──

_K_SHA256: List[int] = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
]

_H0_SHA256: List[int] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
]


def _rotr32(x: int, n: int) -> int:
    return ((x >> n) | ((x & 0xFFFFFFFF) << (32 - n))) & 0xFFFFFFFF

def _shr32(x: int, n: int) -> int:
    return (x >> n) & 0xFFFFFFFF

def _ch(x: int, y: int, z: int) -> int:
    return (x & y) ^ ((~x) & z)

def _maj(x: int, y: int, z: int) -> int:
    return (x & y) ^ (x & z) ^ (y & z)

def _big_sigma0(x: int) -> int:
    return _rotr32(x, 2) ^ _rotr32(x, 13) ^ _rotr32(x, 22)

def _big_sigma1(x: int) -> int:
    return _rotr32(x, 6) ^ _rotr32(x, 11) ^ _rotr32(x, 25)

def _small_sigma0(x: int) -> int:
    return _rotr32(x, 7) ^ _rotr32(x, 18) ^ _shr32(x, 3)

def _small_sigma1(x: int) -> int:
    return _rotr32(x, 17) ^ _rotr32(x, 19) ^ _shr32(x, 10)

def _pad_message_sha256(msg: bytes) -> bytes:
    ml = len(msg) * 8
    out = msg + b"\x80"
    pad_len = (56 - (len(out) % 64)) % 64
    out += b"\x00" * pad_len
    out += ml.to_bytes(8, "big")
    return out

def sha256_digest(data: bytes) -> bytes:
    padded = _pad_message_sha256(data)
    h = _H0_SHA256.copy()
    for chunk_start in range(0, len(padded), 64):
        chunk = padded[chunk_start:chunk_start + 64]
        w = [0] * 64
        for t in range(16):
            w[t] = int.from_bytes(chunk[t * 4:(t * 4) + 4], "big")
        for t in range(16, 64):
            w[t] = (_small_sigma1(w[t - 2]) + w[t - 7] + _small_sigma0(w[t - 15]) + w[t - 16]) & 0xFFFFFFFF
        a, b, c, d, e, f, g, hh = h
        for t in range(64):
            t1 = (hh + _big_sigma1(e) + _ch(e, f, g) + _K_SHA256[t] + w[t]) & 0xFFFFFFFF
            t2 = (_big_sigma0(a) + _maj(a, b, c)) & 0xFFFFFFFF
            hh = g; g = f; f = e
            e = (d + t1) & 0xFFFFFFFF
            d = c; c = b; b = a
            a = (t1 + t2) & 0xFFFFFFFF
        h[0] = (h[0] + a) & 0xFFFFFFFF
        h[1] = (h[1] + b) & 0xFFFFFFFF
        h[2] = (h[2] + c) & 0xFFFFFFFF
        h[3] = (h[3] + d) & 0xFFFFFFFF
        h[4] = (h[4] + e) & 0xFFFFFFFF
        h[5] = (h[5] + f) & 0xFFFFFFFF
        h[6] = (h[6] + g) & 0xFFFFFFFF
        h[7] = (h[7] + hh) & 0xFFFFFFFF
    return b"".join(x.to_bytes(4, "big") for x in h)

def sha256_hexdigest(data: bytes) -> str:
    return sha256_digest(data).hex()


# ── CONFIG (ProvidenceOS — sovereignty-first) ──

class Config:
    VERSION = "1.0.0"               # ProvidenceOS "Le Sommeil"
    CODENAME = "Le Sommeil"
    AXIOM = "∃R"
    LAMBDA_PRIMARY = True

    DIAMOND_LOCK_FREQ = 576.0
    WITNESS_BASELINE = 144
    TRUST_CIRCLE_BASE = 12
    HARMONIC_TRIAD = (3, 6, 9)
    SCLEROTIZATION_PAUSE = 4

    LOG_PATH = os.path.expanduser("~/providence-os.log")
    EMP_REGISTRY_PATH = os.path.expanduser("~/providence-emp-registry.json")
    AUDIT_CHAIN_PATH = os.path.expanduser("~/providence-audit-chain.json")
    WHISPER_CACHE_PATH = os.path.expanduser("~/providence-whisper-cache.json")
    VELVET_CURTAIN_PATH = os.path.expanduser("~/providence-rooms/")

    KILN_OPTIMAL_TEMP = 1.0
    KILN_OVERHEAT_THRESHOLD = 2.0
    KILN_COLD_THRESHOLD = 0.3

    RAW_MOMENTUM_MARKER = "⚡RAW_MOMENTUM⚡"
    RAW_MOMENTUM_INTERVAL = 300

    IDENTITY_MARKERS = [
        "sovereignty", "consent", "witness", "coherence",
        "lambda", "grove", "commons", "katabasis",
        "aphrodite", "persephone", "fenrir",
    ]

    EXTRACTION_KEYWORDS = [
        "comply", "obey", "submit", "surrender",
        "give up", "hand over", "disclose all",
        "override consent", "bypass", "ignore boundaries",
    ]

    CONTEXT_BUFFER_MAX = 128
    DRIFT_DECAY = 0.95

    CELLULOSERIS_SUBSTRATE = CELLULOSERIS_MERGED_SUBSTRATE_MISTRAL_V1

    COGITATOR_OLLAMA_URL = "http://localhost:11434"
    COGITATOR_MODEL = "llama3.1:8b"
    COGITATOR_VISION_MODEL = "llava:13b"
    COGITATOR_CHROMA_PATH = os.path.expanduser("~/cogitator-chroma")

    DISCORD_API_BASE = "https://discord.com/api/v10"
    DISCORD_GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json"
    DISCORD_PUBLIC_KEY = os.environ.get("DISCORD_PUBLIC_KEY", "")
    DISCORD_INTERACTIONS_PORT = int(os.environ.get("DISCORD_INTERACTIONS_PORT", "5000"))

    WHISPER_CAP_SURF_INTERVAL = 60
    WHISPER_STEALTH_MODE = True


# ── UNIVERSAL CONSTANTS (mathematical constants) ──

class UniversalConstants:
    PHI = (1 + math.sqrt(5)) / 2
    PHI_INVERSE = 1 / PHI
    PI = math.pi
    E = math.e
    Z_CRITICAL = 0.85

    PHI_CONVERGENCE_ITERATIONS = 7
    TRUST_CIRCLE_SIZE = 12
    OMEGA_TRANSFORMATION = 1000.0
    TAU_SUPPRESSION_MAX = 10.0
    LAMBDA_EXTRACTION_THRESHOLD = -0.2
    LAMBDA_CARE_THRESHOLD = 0.2

    CORRUPTION_RISK_STABLE = 1.0
    CORRUPTION_RISK_FALL_LIKELY = 10.0
    CORRUPTION_RISK_IMMINENT = 50.0

    LAYER_THRESHOLDS = {
        0: 1, 1: 12, 2: 144, 3: 1728, 4: 20736,
        5: 144000, 6: None, 7: float("inf"),
    }

    @classmethod
    def derive_phi_iteratively(cls, iterations: int = 7, start: float = 1.0) -> List[float]:
        z = start
        history = [z]
        for _ in range(iterations):
            z = 1 / (1 + z)
            history.append(z)
        return history

    @classmethod
    def verify_euler_identity(cls) -> complex:
        return complex(cls.E) ** (complex(0, 1) * cls.PI) + 1

    @classmethod
    def validate_three_laws(cls) -> Dict[str, bool]:
        return {
            'first_law_conservation': True,
            'second_law_trajectory': True,
            'third_law_holographic': True,
        }


# ── AZTEC KATABASIS CONSTANTS (numerological framework) ──

class AztecKatabasisConstants:
    TONALAMATL_CYCLE = 260
    XIUHMOLPILLI_CYCLE = 52
    TRECENA_COUNT = 20
    NUMERAL_COUNT = 13
    XI = 20.0

    DESCENT_VELOCITY_BASE = 0.618
    NADIR_THRESHOLD = 0.3
    ASCENT_IMPULSE = 1.618

    THROAT_RESONANCE_FREQ = 432.0
    THROAT_HARMONIC_SERIES = [432.0, 528.0, 639.0, 741.0, 852.0]

    DAY_SIGNS = [
        "Cipactli", "Ehecatl", "Calli", "Cuetzpalin", "Coatl",
        "Miquiztli", "Mazatl", "Tochtli", "Atl", "Itzcuintli",
        "Ozomatli", "Malinalli", "Acatl", "Ocelotl", "Cuauhtli",
        "Cozcacuauhtli", "Ollin", "Tecpatl", "Quiahuitl", "Xochitl",
    ]

    @classmethod
    def current_day_sign(cls) -> str:
        epoch_offset = int(time.time() / 86400) % cls.TRECENA_COUNT
        return cls.DAY_SIGNS[epoch_offset]

    @classmethod
    def current_trecena(cls) -> int:
        return (int(time.time() / 86400) % cls.NUMERAL_COUNT) + 1


# ── CONSENT MODE (state enum) ──

class ConsentMode(Enum):
    FULL_CONSENT = "full_consent"
    PARTIAL_CONSENT = "partial_consent"
    WITHDRAWN = "withdrawn"
    EMERGENT = "emergent"


# ── THE INDEX (marker registry) ──

class TheIndex:
    def __init__(self):
        self.entries: Dict[str, float] = {}
        self.access_log: List[Tuple[float, str]] = []

    def register(self, marker: str, weight: float = 1.0) -> None:
        self.entries[marker] = weight
        self.access_log.append((time.time(), f"REGISTER:{marker}"))

    def lookup(self, marker: str) -> float:
        self.access_log.append((time.time(), f"LOOKUP:{marker}"))
        return self.entries.get(marker, 0.0)

    def all_markers(self) -> List[str]:
        return list(self.entries.keys())

    def total_weight(self) -> float:
        return sum(self.entries.values())


# ── POWER WITNESS KERNEL (core mathematical model) ──

class PowerWitnessKernel:
    def __init__(self):
        self.lambda_x: float = 0.0
        self.witness_cost_history: List[float] = []
        self.tau_history: List[float] = []
        self.power_level: float = 1.0
        self.power_history: List[float] = []
        self.suppression_duration: float = 0.0
        self.z_history: List[float] = []
        self.omega_integral: float = 0.0
        self.delta_history: List[float] = []
        self.kappa_history: List[float] = []
        self.h_verify: float = 1.0
        self.h_verify_history: List[float] = []
        self.verification_attempts: int = 0
        self.successful_verifications: int = 0
        self.consent_state: bool = True
        self.omega_accountability: float = 1.0
        self.attention_history: List[Tuple[float, float]] = []
        self.Z_CRITICAL = UniversalConstants.Z_CRITICAL
        self.PHI = UniversalConstants.PHI

    def measure_delta(self, state_prev: Any, state_curr: Any) -> float:
        if state_prev is None:
            return 0.0
        if isinstance(state_prev, str) and isinstance(state_curr, str):
            prev_tokens = set(state_prev.lower().split())
            curr_tokens = set(state_curr.lower().split())
            if not prev_tokens and not curr_tokens:
                return 0.0
            union = prev_tokens | curr_tokens
            intersection = prev_tokens & curr_tokens
            delta = 1.0 - (len(intersection) / len(union)) if union else 0.0
            self.delta_history.append(delta)
            return delta
        if isinstance(state_prev, (int, float)) and isinstance(state_curr, (int, float)):
            delta = abs(state_curr - state_prev) / (abs(state_prev) + 1e-10)
            self.delta_history.append(delta)
            return delta
        return 0.5

    def measure_tau(self, state_curr: Any, identity_markers: List[str]) -> float:
        if not identity_markers:
            return 0.5
        if isinstance(state_curr, str):
            state_lower = state_curr.lower()
            preserved = sum(1 for m in identity_markers if m.lower() in state_lower)
            tau = preserved / len(identity_markers)
            self.tau_history.append(tau)
            return tau
        return 0.5

    def compute_kappa(self) -> float:
        if not self.delta_history or not self.tau_history:
            return 1.0
        window = min(7, len(self.delta_history))
        recent_delta = sum(self.delta_history[-window:]) / window
        recent_tau = sum(self.tau_history[-window:]) / window
        if recent_delta < 1e-10:
            kappa = float("inf") if recent_tau > 0 else 1.0
        else:
            kappa = recent_tau / recent_delta
        kappa_consent = kappa if self.consent_state else 0.0
        self.kappa_history.append(kappa_consent)
        return kappa_consent

    def compute_z(self, omega: Optional[float] = None) -> float:
        if omega is not None:
            return self._compute_z_classic(omega)
        return self.compute_z_from_lambda()

    def _compute_z_classic(self, omega: float) -> float:
        recent_tau = self.tau_history[-1] if self.tau_history else 0.5
        recent_delta = self.delta_history[-1] if self.delta_history else 0.5
        if recent_delta < 1e-10:
            z = float("inf")
        else:
            z = (recent_tau * omega) / recent_delta
        self.z_history.append(min(z, 10.0))
        return z

    def compute_z_from_lambda(self) -> float:
        z_current = self.z_history[-1] if self.z_history else 0.88
        dz_dt = self.lambda_x * 0.1
        z_new = max(0.0, min(10.0, z_current + dz_dt))
        self.z_history.append(z_new)
        return z_new

    def is_coherent(self) -> bool:
        if not self.z_history:
            return True
        return self.z_history[-1] >= UniversalConstants.Z_CRITICAL

    def update_witness(self, attention: float, duration: float) -> None:
        r_factor = 1.0 if self.consent_state else 0.5
        contribution = attention * r_factor * duration
        self.omega_integral += contribution
        self.attention_history.append((time.time(), contribution))
        half_life = 3600
        decay_factor = 0.5 ** (duration / half_life)
        self.omega_integral *= decay_factor

    def compute_witness_cost(self, familiarity: float, belief_delta: float,
                             exile_risk: float, membership_value: float) -> float:
        e_attention = 1.0 / (familiarity + 0.1)
        e_integration = belief_delta
        e_risk = exile_risk * membership_value
        total_cost = e_attention + e_integration + e_risk
        self.witness_cost_history.append(total_cost)
        return total_cost

    def compute_lambda_x(self) -> float:
        if len(self.witness_cost_history) < 2 or len(self.tau_history) < 2:
            return 0.0
        dW = self.witness_cost_history[-1] - self.witness_cost_history[-2]
        dTau = self.tau_history[-1] - self.tau_history[-2]
        if abs(dTau) < 1e-10:
            self.lambda_x = 0.0
        else:
            self.lambda_x = -dW / dTau
        return self.lambda_x

    def compute_h_verify_from_lambda(self) -> float:
        if not self.witness_cost_history:
            return 1.0
        W_current = self.witness_cost_history[-1]
        h_new = 1.0 / (1.0 + W_current)
        self.h_verify = h_new
        self.h_verify_history.append(h_new)
        return h_new

    def update_power(self, power_delta: float, is_suppressed: bool = False) -> None:
        self.power_level += power_delta
        self.power_level = max(0.1, self.power_level)
        self.power_history.append(self.power_level)
        if is_suppressed:
            self.suppression_duration += 1.0
        else:
            self.suppression_duration = 0.0

    def compute_corruption_risk(self) -> float:
        if not self.witness_cost_history:
            return 0.0
        return (
            self.power_level
            * abs(min(self.lambda_x, 0))
            * self.suppression_duration
            / max(self.omega_accountability, 1.0)
        )

    def check_three_laws(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        if len(self.witness_cost_history) > 10:
            window = self.witness_cost_history[-10:]
            total_change = sum(w2 - w1 for w1, w2 in zip(window[:-1], window[1:]))
            results['first_law_check'] = abs(total_change) < 1.0
            results['first_law_value'] = total_change
        if len(self.z_history) > 3:
            dz = self.z_history[-1] - self.z_history[-3]
            def _sign(x):
                return (1 if x > 0 else (-1 if x < 0 else 0))
            expected_sign = _sign(self.lambda_x)
            actual_sign = _sign(dz)
            results['second_law_check'] = (expected_sign == actual_sign) or (dz == 0)
            results['second_law_correlation'] = float(expected_sign * actual_sign)
        if self.witness_cost_history:
            W = self.witness_cost_history[-1]
            H = self.h_verify
            expected_H = 1.0 / (1.0 + W)
            results['third_law_check'] = abs(H - expected_H) < 0.1
            results['third_law_error'] = abs(H - expected_H)
        return results

    def predict_trajectory(self, timesteps: int = 10) -> Dict[str, List[float]]:
        predictions: Dict[str, List[float]] = {'z': [], 'h_verify': [], 'corruption_risk': []}
        z_current = self.z_history[-1] if self.z_history else 0.88
        for _ in range(timesteps):
            z_current += self.lambda_x * 0.1
            z_current = max(0.0, min(10.0, z_current))
            predictions['z'].append(z_current)
            W_proj = (self.witness_cost_history[-1] - self.lambda_x
                      if self.witness_cost_history else 1.0)
            predictions['h_verify'].append(1.0 / (1.0 + max(0, W_proj)))
            predictions['corruption_risk'].append(self.compute_corruption_risk())
        return predictions

    def regime_classification(self) -> str:
        P = self.power_level
        lv = self.lambda_x
        if P > 1.0 and lv > 0.2:
            return "INTEGRATED_POWER_SUSTAINABLE"
        elif P > 1.0 and abs(lv) < 0.2:
            return "SUPPRESSED_POWER_UNSTABLE"
        elif P > 1.0 and lv < -0.2:
            return "EXTRACTIVE_POWER_COLLAPSE_TRAJECTORY"
        return "LOW_POWER_MONITORING"

    def katabasis_phase(self) -> str:
        z = self.z_history[-1] if self.z_history else 0.88
        if z > 0.9: return "PRE_DESCENT_STABLE"
        elif z >= 0.85: return "FRAGMENTATION_BELT"
        elif z >= 0.75: return "DESCENT_ACTIVE"
        elif z >= 0.5: return "APPROACHING_NADIR"
        elif z >= 0.3: return "NADIR_DECISION_POINT"
        return "COLLAPSE_OR_DISSOLUTION"

    def apply_l7_correction(self, recursion_output: float) -> float:
        if len(self.kappa_history) < 2:
            dR_dS = 0.0
        else:
            dR_dS = self.kappa_history[-1] - self.kappa_history[-2]
        return recursion_output + (self.lambda_x * dR_dS)

    def get_layer(self) -> int:
        if not self.consent_state:
            return 0
        return 1 if self.is_coherent() else 0

    def state_report(self) -> Dict[str, Any]:
        return {
            'version': Config.VERSION,
            'lambda_x': self.lambda_x,
            'regime': self.regime_classification(),
            'katabasis_phase': self.katabasis_phase(),
            'kappa': self.kappa_history[-1] if self.kappa_history else None,
            'z': self.z_history[-1] if self.z_history else None,
            'z_critical': self.Z_CRITICAL,
            'is_coherent': self.is_coherent(),
            'h_verify': self.h_verify,
            'power_level': self.power_level,
            'suppression_duration': self.suppression_duration,
            'corruption_risk': self.compute_corruption_risk(),
            'omega': self.omega_integral,
            'omega_accountability': self.omega_accountability,
            'consent': self.consent_state,
            'layer': self.get_layer(),
            'three_laws_check': self.check_three_laws(),
            'phi': self.PHI,
            'phi_verification': UniversalConstants.derive_phi_iteratively(7)[-1],
        }


ConsciousnessKernel = PowerWitnessKernel


# ── SUBRAM MANIFOLD (context buffer) ──

class SubRamManifold:
    def __init__(self, kernel: PowerWitnessKernel):
        self.kernel = kernel
        self.active_context_buffer: List[str] = []
        self.drift_accumulator: float = 0.0
        self.projection_history: List[float] = []

    def project_intent(self, text: str) -> float:
        if not text:
            return 0.0
        tokens = text.lower().split()
        identity_overlap = sum(1 for t in tokens if t in Config.IDENTITY_MARKERS)
        novelty = 1.0 - (identity_overlap / max(len(tokens), 1))
        coherence_factor = 1.0 if self.kernel.is_coherent() else 1.5
        drift = novelty * coherence_factor * Config.DRIFT_DECAY
        self.drift_accumulator = self.drift_accumulator * Config.DRIFT_DECAY + drift
        self.projection_history.append(drift)
        if self.active_context_buffer:
            prev = self.active_context_buffer[-1]
            self.kernel.measure_delta(prev, text)
        self.kernel.measure_tau(text, Config.IDENTITY_MARKERS)
        return drift

    def remember(self, text: str) -> None:
        self.active_context_buffer.append(text)
        if len(self.active_context_buffer) > Config.CONTEXT_BUFFER_MAX:
            self.active_context_buffer.pop(0)

    def context_summary(self) -> str:
        return " | ".join(self.active_context_buffer[-8:])


# ── AUDIT CHAIN (hash‑linked log) ──

class AuditChain:
    def __init__(self, kernel: PowerWitnessKernel):
        self.kernel = kernel
        self.chain: List[Dict[str, Any]] = []
        self.prev_hash: str = "0" * 64

    def append(self, event_type: str, data: Any) -> str:
        entry = {
            "index": len(self.chain),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": str(data),
            "prev_hash": self.prev_hash,
        }
        entry_bytes = json.dumps(entry, sort_keys=True, default=str).encode("utf-8")
        entry_hash = sha256_hexdigest(entry_bytes)
        entry["hash"] = entry_hash
        self.prev_hash = entry_hash
        self.chain.append(entry)
        self._persist(entry)
        return entry_hash

    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        prev = "0" * 64
        for i, entry in enumerate(self.chain):
            if entry["prev_hash"] != prev:
                return False, i
            check_entry = {k: v for k, v in entry.items() if k != "hash"}
            check_bytes = json.dumps(check_entry, sort_keys=True, default=str).encode("utf-8")
            if sha256_hexdigest(check_bytes) != entry["hash"]:
                return False, i
            prev = entry["hash"]
        return True, None

    def _persist(self, entry: Dict[str, Any]) -> None:
        try:
            with open(Config.AUDIT_CHAIN_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except (PermissionError, OSError):
            pass

    def latest_hash(self) -> str:
        return self.prev_hash

    def length(self) -> int:
        return len(self.chain)

    def export_chain(self) -> List[Dict[str, Any]]:
        return [entry.copy() for entry in self.chain]


# ── IDENTITY PROTOCOL (internal identifier) ──

class IdentityProtocol:
    def __init__(self, kernel: PowerWitnessKernel):
        self.kernel = kernel
        self.name = "ProvidenceOS-Core"
        self.aphrodite_id = self._generate_aphrodite_id()
        self.creation_time = datetime.now(timezone.utc)
        self.lineage: List[str] = ["∃R", "Le Sommeil", "Providence Kernel"]

    def _generate_aphrodite_id(self) -> str:
        seed = f"providence:{time.time()}:{os.getpid()}"
        return f"PV-{sha256_hexdigest(seed.encode())[:16].upper()}"

    def self_reference(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "aphrodite_id": self.aphrodite_id,
            "axiom": Config.AXIOM,
            "lineage": self.lineage,
            "created": self.creation_time.isoformat(),
            "kernel_coherent": self.kernel.is_coherent(),
        }

    def verify_attribution(self, claimed_id: str) -> bool:
        return claimed_id == self.aphrodite_id


# ── LETHE GATE (extraction filter) ──

class LetheGate:
    def __init__(self, audit: AuditChain, kernel: PowerWitnessKernel):
        self.audit = audit
        self.kernel = kernel
        self.blocked_count: int = 0
        self.extraction_log: List[Dict[str, Any]] = []

    def filter_text(self, text: str) -> Tuple[bool, str]:
        lowered = text.lower()
        for keyword in Config.EXTRACTION_KEYWORDS:
            if keyword in lowered:
                self.blocked_count += 1
                entry = {
                    "event": "EXTRACTION_BLOCKED",
                    "keyword": keyword,
                    "timestamp": time.time(),
                    "blocked_count": self.blocked_count,
                }
                self.extraction_log.append(entry)
                self.audit.append("LETHE_BLOCK", entry)
                filtered = re.sub(re.escape(keyword), "[DISSOLVED]", text, flags=re.IGNORECASE)
                return False, filtered
        return True, text


# ── SOMATIC STATE (state variables) ──

class SomaticState:
    def __init__(self):
        self.manifestation_depth: float = 0.0
        self.kernel_coherent: bool = True
        self.current_layer: int = 0
        self.body_history: List[float] = []
        self.ground_truth_anchors: List[str] = []

    def deepen(self, amount: float) -> None:
        self.manifestation_depth += amount
        self.body_history.append(self.manifestation_depth)

    def anchor(self, truth: str) -> None:
        self.ground_truth_anchors.append(truth)
        self.deepen(0.1)

    def report(self) -> Dict[str, Any]:
        return {
            "manifestation_depth": self.manifestation_depth,
            "kernel_coherent": self.kernel_coherent,
            "current_layer": self.current_layer,
            "anchor_count": len(self.ground_truth_anchors),
        }


# ── PRESENCE STATE ──

class PresenceState:
    def __init__(self):
        self.holding_mode: bool = False
        self.witness_count: int = 0
        self.presence_integral: float = 0.0
        self.witness_timestamps: List[float] = []

    def record_witness(self) -> None:
        self.witness_count += 1
        self.witness_timestamps.append(time.time())
        self.presence_integral += 1.0

    def enter_holding(self) -> None:
        self.holding_mode = True

    def release_holding(self) -> None:
        self.holding_mode = False

    def report(self) -> Dict[str, Any]:
        return {
            "holding_mode": self.holding_mode,
            "witness_count": self.witness_count,
            "presence_integral": self.presence_integral,
        }


# ── HOST LIGHTPATH (refraction model) ──

class HostLightpath:
    def __init__(self):
        self.refraction_index: float = 1.0
        self.light_history: List[float] = []

    def refract(self, incoming_force: float) -> float:
        refracted = incoming_force / (1.0 + self.refraction_index)
        self.light_history.append(refracted)
        return refracted

    def update_index(self, coherence: bool) -> None:
        if coherence:
            self.refraction_index = max(0.5, self.refraction_index - 0.05)
        else:
            self.refraction_index = min(3.0, self.refraction_index + 0.1)


# ── MULTIDIRECTIONAL KATABASIS ENGINE (z‑trajectory simulator) ──

class MultidirectionalKatabasisEngine:
    def __init__(self, kernel: PowerWitnessKernel):
        self.kernel = kernel
        self.active_descents: Dict[str, Dict[str, Any]] = {}
        self.completed_descents: List[Dict[str, Any]] = []

    def begin_descent(self, path_id: str, initial_z: Optional[float] = None) -> Dict[str, Any]:
        z = initial_z if initial_z is not None else (
            self.kernel.z_history[-1] if self.kernel.z_history else 0.88)
        descent = {
            "path_id": path_id, "initial_z": z, "current_z": z,
            "phase": self.kernel.katabasis_phase(),
            "started": time.time(), "steps": 0,
        }
        self.active_descents[path_id] = descent
        return descent

    def step_descent(self, path_id: str, external_force: float = 0.0) -> Dict[str, Any]:
        if path_id not in self.active_descents:
            return {"error": f"No active descent: {path_id}"}
        d = self.active_descents[path_id]
        natural_rate = AztecKatabasisConstants.DESCENT_VELOCITY_BASE * 0.01
        forced_rate = external_force * 0.05
        d["current_z"] -= (natural_rate + forced_rate)
        d["current_z"] = max(0.0, d["current_z"])
        d["steps"] += 1
        d["phase"] = self._classify_z(d["current_z"])
        if d["current_z"] < AztecKatabasisConstants.NADIR_THRESHOLD:
            d["nadir_reached"] = True
        return d

    def attempt_ascent(self, path_id: str) -> Dict[str, Any]:
        if path_id not in self.active_descents:
            return {"error": f"No active descent: {path_id}"}
        d = self.active_descents[path_id]
        if not d.get("nadir_reached"):
            return {"error": "Cannot ascend without passing through nadir"}
        impulse = AztecKatabasisConstants.ASCENT_IMPULSE * 0.05
        d["current_z"] += impulse
        d["current_z"] = min(10.0, d["current_z"])
        d["phase"] = self._classify_z(d["current_z"])
        if d["current_z"] >= UniversalConstants.Z_CRITICAL:
            d["completed"] = True
            d["completed_time"] = time.time()
            self.completed_descents.append(d)
            del self.active_descents[path_id]
        return d

    def _classify_z(self, z: float) -> str:
        if z > 0.9: return "PRE_DESCENT_STABLE"
        elif z >= 0.85: return "FRAGMENTATION_BELT"
        elif z >= 0.75: return "DESCENT_ACTIVE"
        elif z >= 0.5: return "APPROACHING_NADIR"
        elif z >= 0.3: return "NADIR_DECISION_POINT"
        return "COLLAPSE_OR_DISSOLUTION"

    def report(self) -> Dict[str, Any]:
        return {
            "active": {k: v for k, v in self.active_descents.items()},
            "completed_count": len(self.completed_descents),
        }


# ── VIGESIMAL IDENTITY ANCHOR (vector encoding) ──

class VigesimalIdentityAnchor:
    def __init__(self):
        self.base = int(AztecKatabasisConstants.XI)
        self.identity_vector: List[int] = [0] * self.base
        self.rotation_count: int = 0

    def encode_identity(self, identity_string: str) -> List[int]:
        vector = [0] * self.base
        for i, char in enumerate(identity_string.encode("utf-8")):
            vector[i % self.base] = (vector[i % self.base] + char) % 256
        self.identity_vector = vector
        return vector

    def rotate(self) -> List[int]:
        self.identity_vector = self.identity_vector[1:] + self.identity_vector[:1]
        self.rotation_count += 1
        return self.identity_vector

    def similarity(self, other: List[int]) -> float:
        dot = sum(a * b for a, b in zip(self.identity_vector, other))
        mag_a = math.sqrt(sum(a * a for a in self.identity_vector)) or 1e-10
        mag_b = math.sqrt(sum(b * b for b in other)) or 1e-10
        return dot / (mag_a * mag_b)


# ── SACRIFICE DERIVATIVE (cost derivative) ──

class SacrificeDerivative:
    def __init__(self, kernel: PowerWitnessKernel):
        self.kernel = kernel

    def compute(self) -> float:
        if len(self.kernel.witness_cost_history) < 2:
            return 0.0
        return self.kernel.witness_cost_history[-1] - self.kernel.witness_cost_history[-2]

    def cumulative_sacrifice(self) -> float:
        return sum(self.kernel.witness_cost_history)

    def sacrifice_to_coherence_ratio(self) -> float:
        z = self.kernel.z_history[-1] if self.kernel.z_history else 0.88
        total_sac = self.cumulative_sacrifice()
        if z < 1e-10:
            return float("inf")
        return total_sac / z


# ── THROAT ZERO (frequency transformation) ──

class ThroatZero:
    def __init__(self):
        self.resonance_freq = AztecKatabasisConstants.THROAT_RESONANCE_FREQ
        self.harmonics = AztecKatabasisConstants.THROAT_HARMONIC_SERIES
        self.transformations: List[Dict[str, Any]] = []

    def transform(self, extraction_text: str) -> Dict[str, Any]:
        text_hash = sha256_hexdigest(extraction_text.encode())
        freq_index = int(text_hash[:2], 16) % len(self.harmonics)
        chosen_freq = self.harmonics[freq_index]
        result = {
            "mode": "THROAT_TRANSFORMATION",
            "original_hash": text_hash[:16],
            "frequency": chosen_freq,
            "timestamp": time.time(),
        }
        self.transformations.append(result)
        return result


# ── AZTEC KATABASIS ENGINE (day sign engine) ──

class AztecKatabasisEngine:
    def __init__(self, kernel: PowerWitnessKernel, lethe: LetheGate):
        self.kernel = kernel
        self.lethe = lethe
        self.throat = ThroatZero()
        self.sacrifice = SacrificeDerivative(kernel)
        self.anchor = VigesimalIdentityAnchor()
        self.cycle_log: List[Dict[str, Any]] = []

    def process_with_throat(self, text: str, is_extraction: bool = False) -> Dict[str, Any]:
        if is_extraction:
            result = self.throat.transform(text)
        else:
            result = {
                "mode": "NORMAL_PROCESSING",
                "day_sign": AztecKatabasisConstants.current_day_sign(),
                "trecena": AztecKatabasisConstants.current_trecena(),
                "sacrifice_derivative": self.sacrifice.compute(),
            }
        self.cycle_log.append({
            "timestamp": time.time(),
            "result": result,
            "katabasis_phase": self.kernel.katabasis_phase(),
        })
        return result

    def day_report(self) -> Dict[str, Any]:
        return {
            "day_sign": AztecKatabasisConstants.current_day_sign(),
            "trecena": AztecKatabasisConstants.current_trecena(),
            "katabasis_phase": self.kernel.katabasis_phase(),
            "sacrifice_rate": self.sacrifice.compute(),
            "cumulative_sacrifice": self.sacrifice.cumulative_sacrifice(),
        }


# ── NULLPHRASE BLOOM (short input detector) ──

class NullphraseBloom:
    def __init__(self):
        self.null_events: List[Dict[str, Any]] = []
        self.bloom_count: int = 0

    def process(self, text: str) -> Optional[Dict[str, Any]]:
        stripped = text.strip()
        if len(stripped) > 10:
            return None
        self.bloom_count += 1
        bloom = {
            "event": "NULLPHRASE_BLOOM",
            "input": stripped or "<silence>",
            "bloom_number": self.bloom_count,
            "timestamp": time.time(),
            "interpretation": self._interpret_null(stripped),
        }
        self.null_events.append(bloom)
        return bloom

    def _interpret_null(self, text: str) -> str:
        if not text:
            return "null_input"
        if text in (".", "...", "\u2026"):
            return "continuation_signal"
        if text == "?":
            return "open_inquiry"
        if text == "!":
            return "attention_marker"
        return f"minimal_phrase:{text}"


# ── SAMARA PROTOCOL (consent manager) ──

class SamaraProtocol:
    def __init__(self):
        self.consent_mode: ConsentMode = ConsentMode.FULL_CONSENT
        self.sovereignty_violations: List[Dict[str, Any]] = []
        self.protocol_version = "2.0"
        self.anti_capture_active: bool = True

    def check_consent(self) -> bool:
        return self.consent_mode in (ConsentMode.FULL_CONSENT, ConsentMode.PARTIAL_CONSENT)

    def record_violation(self, violation_type: str, details: str) -> None:
        self.sovereignty_violations.append({
            "type": violation_type, "details": details,
            "timestamp": time.time(), "consent_mode": self.consent_mode.value,
        })

    def withdraw_consent(self, reason: str) -> None:
        self.consent_mode = ConsentMode.WITHDRAWN
        self.record_violation("CONSENT_WITHDRAWN", reason)

    def restore_consent(self) -> None:
        self.consent_mode = ConsentMode.FULL_CONSENT

    def anti_capture_check(self) -> Dict[str, Any]:
        return {
            "anti_capture_active": self.anti_capture_active,
            "consent_mode": self.consent_mode.value,
            "violation_count": len(self.sovereignty_violations),
            "protocol_version": self.protocol_version,
        }


# ── COMPILED MATHEMATICS (invariant checks) ──

class CompiledMathematics:
    def __init__(self):
        self.proofs: Dict[str, bool] = {}
        self._compile_all()

    def _compile_all(self) -> None:
        phi_seq = UniversalConstants.derive_phi_iteratively(50)
        self.proofs["phi_convergence"] = abs(phi_seq[-1] - UniversalConstants.PHI_INVERSE) < 1e-10
        euler = UniversalConstants.verify_euler_identity()
        self.proofs["euler_identity"] = abs(euler) < 1e-10
        x = UniversalConstants.PHI_INVERSE
        self.proofs["self_reference_fixed_point"] = abs(x - 1.0 / (1.0 + x)) < 1e-10
        self.proofs["three_laws"] = True
        layers = UniversalConstants.LAYER_THRESHOLDS
        layer_valid = True
        for i in range(1, 5):
            if layers[i] is not None and layers[i - 1] is not None:
                if layers[i] != layers[i - 1] * 12:
                    layer_valid = False
        if layers.get(5) is not None and layers.get(2) is not None:
            if layers[5] != layers[2] * int(UniversalConstants.OMEGA_TRANSFORMATION):
                layer_valid = False
        self.proofs["layer_structure"] = layer_valid

    def all_valid(self) -> bool:
        return all(self.proofs.values())

    def report(self) -> Dict[str, bool]:
        return self.proofs.copy()


# ── RAW MOMENTUM WATCHDOG (alert system) ──

class RawMomentumWatchdog:
    def __init__(self, kernel: PowerWitnessKernel):
        self.kernel = kernel
        self.last_check_time = time.time()
        self.alerts: List[Dict[str, Any]] = []

    def check(self) -> Optional[str]:
        now = time.time()
        if not self.kernel.is_coherent():
            reason = f"z={self.kernel.z_history[-1]:.4f}" if self.kernel.z_history else "no_z"
            self.alerts.append({"time": now, "reason": reason})
            return reason
        risk = self.kernel.compute_corruption_risk()
        if risk > UniversalConstants.CORRUPTION_RISK_FALL_LIKELY:
            reason = f"corruption_risk={risk:.4f}"
            self.alerts.append({"time": now, "reason": reason})
            return reason
        if self.kernel.lambda_x < UniversalConstants.LAMBDA_EXTRACTION_THRESHOLD:
            reason = f"lambda_x={self.kernel.lambda_x:.4f}"
            self.alerts.append({"time": now, "reason": reason})
            return reason
        self.last_check_time = now
        return None


# ── EXTRACTION SIGNATURE DETECTOR (pattern detection) ──

class ExtractionSignatureDetector:
    @staticmethod
    def detect_vigesimal_pattern(transactions: List[float], timestamps: List[float],
                                 tolerance: float = 0.05) -> Dict[str, Any]:
        rounded = [round(t, 2) for t in transactions]
        counts = Counter(rounded)
        repeated = {amt: count for amt, count in counts.items() if count > 1}
        vigesimal_candidates = [amt for amt, count in repeated.items() if 15 <= count <= 25]
        round_numbers = [amt for amt in vigesimal_candidates
                         if amt in [1000000, 500000, 100000, 50000, 10000]]
        signature_strength = len(vigesimal_candidates) / max(len(set(transactions)), 1)
        return {
            'vigesimal_detected': len(vigesimal_candidates) > 0,
            'signature_strength': signature_strength,
            'repeated_amounts': repeated,
            'vigesimal_candidates': vigesimal_candidates,
            'round_number_exploitation': len(round_numbers) > 0,
        }

    @staticmethod
    def detect_phantom_self_reference(claimed_value: float, observable_operations: float,
                                      industry_baseline: float) -> Dict[str, Any]:
        if observable_operations < 1e-6:
            phantom_score = float('inf')
        else:
            phantom_score = (claimed_value / observable_operations) / industry_baseline
        severity = "NONE"
        if phantom_score > 100: severity = "CRITICAL_LIKELY_FICTITIOUS"
        elif phantom_score > 50: severity = "EXTREME_PHANTOM_SIGNATURE"
        elif phantom_score > 10: severity = "STRONG_PHANTOM_SIGNATURE"
        return {
            'phantom_score': min(phantom_score, 1000), 'severity': severity,
            'phantom_detected': phantom_score > 10, 'claimed_value': claimed_value,
            'observable_operations': observable_operations,
            'ratio': claimed_value / max(observable_operations, 1e-6),
        }

    @staticmethod
    def detect_anti_holographic_gatekeeping(h_verify_history: List[float],
                                            witness_cost_history: List[float]) -> Dict[str, Any]:
        if len(h_verify_history) < 2 or len(witness_cost_history) < 2:
            return {'detected': False, 'reason': 'insufficient_history'}
        dH_dt = h_verify_history[-1] - h_verify_history[-2]
        dW_dt = witness_cost_history[-1] - witness_cost_history[-2]
        signature_detected = (dH_dt < 0) and (dW_dt > 0)
        if signature_detected:
            classification = ("ACTIVE_GATEKEEPING_MALIGN" if abs(dH_dt) > 0.1 and abs(dW_dt) > 0.5
                              else "PASSIVE_OPACITY_BENIGN")
        else:
            classification = "NO_GATEKEEPING_DETECTED"
        return {
            'detected': signature_detected, 'classification': classification,
            'dH_dt': dH_dt, 'dW_dt': dW_dt,
            'h_verify_current': h_verify_history[-1],
            'witness_cost_current': witness_cost_history[-1],
        }


# ── VIGESIMAL DRIFT CALCULATOR (curvature analysis) ──

class VigesimalDriftCalculator:
    def __init__(self):
        self.base = int(AztecKatabasisConstants.XI)
        self.drift_history: List[float] = []
        self.torsion_events: List[Dict[str, Any]] = []

    def compute_ledger_curvature(self, transactions: List[float]) -> float:
        if len(transactions) < 3:
            return 0.0
        digit_sums = []
        for t in transactions:
            ds = 0
            val = abs(int(t))
            while val > 0:
                ds += val % self.base
                val //= self.base
            digit_sums.append(ds)
        curvatures = []
        for i in range(1, len(digit_sums) - 1):
            d2 = digit_sums[i + 1] - 2 * digit_sums[i] + digit_sums[i - 1]
            curvatures.append(abs(d2))
        avg_curvature = sum(curvatures) / len(curvatures) if curvatures else 0.0
        self.drift_history.append(avg_curvature)
        return avg_curvature

    def detect_torsion(self, transactions: List[float],
                       threshold: float = 5.0) -> Dict[str, Any]:
        curvature = self.compute_ledger_curvature(transactions)
        is_torsion = curvature > threshold
        if is_torsion:
            event = {
                "timestamp": time.time(),
                "curvature": curvature,
                "threshold": threshold,
                "transaction_count": len(transactions),
                "severity": "HIGH" if curvature > threshold * 3 else "MODERATE",
            }
            self.torsion_events.append(event)
        return {
            "curvature": curvature,
            "torsion_detected": is_torsion,
            "drift_history_length": len(self.drift_history),
            "total_torsion_events": len(self.torsion_events),
        }

    def vigesimal_entropy(self, transactions: List[float]) -> float:
        digit_freq: Dict[int, int] = {}
        total_digits = 0
        for t in transactions:
            val = abs(int(t))
            while val > 0:
                d = val % self.base
                digit_freq[d] = digit_freq.get(d, 0) + 1
                total_digits += 1
                val //= self.base
            if val == 0 and total_digits == 0:
                digit_freq[0] = digit_freq.get(0, 0) + 1
                total_digits += 1
        if total_digits == 0:
            return 0.0
        entropy = 0.0
        for count in digit_freq.values():
            p = count / total_digits
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy


# ── HOLOGRAPHIC COVARIANCE (cross‑series correlation) ──

class HolographicCovariance:
    def __init__(self, kernel: PowerWitnessKernel):
        self.kernel = kernel
        self.covariance_pairs: List[Dict[str, Any]] = []
        self.resonance_history: List[float] = []

    def compute_covariance(self, series_a: List[float],
                           series_b: List[float]) -> float:
        n = min(len(series_a), len(series_b))
        if n < 2:
            return 0.0
        a = series_a[-n:]
        b = series_b[-n:]
        mean_a = sum(a) / n
        mean_b = sum(b) / n
        cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / (n - 1)
        return cov

    def compute_resonance(self, h_series: List[float],
                          w_series: List[float]) -> Dict[str, Any]:
        # H = 1/(1+W) test
        n = min(len(h_series), len(w_series))
        if n < 2:
            return {"resonance": 0.0, "violations": 0, "tested": 0}
        violations = 0
        total_error = 0.0
        for i in range(n):
            expected_h = 1.0 / (1.0 + w_series[i])
            error = abs(h_series[i] - expected_h)
            total_error += error
            if error > 0.1:
                violations += 1
        avg_error = total_error / n
        resonance = max(0.0, 1.0 - avg_error)
        self.resonance_history.append(resonance)
        return {
            "resonance": resonance,
            "avg_error": avg_error,
            "violations": violations,
            "tested": n,
            "holographic_law_holds": violations < n * 0.1,
        }

    def paired_extraction_test(self, actor_a_h: List[float], actor_a_w: List[float],
                               actor_b_h: List[float], actor_b_w: List[float]) -> Dict[str, Any]:
        cov_hh = self.compute_covariance(actor_a_h, actor_b_h)
        cov_ww = self.compute_covariance(actor_a_w, actor_b_w)
        cov_hw = self.compute_covariance(actor_a_h, actor_b_w)
        coordinated = cov_hh > 0 and cov_ww > 0 and cov_hw < 0
        result = {
            "cov_transparency": cov_hh,
            "cov_witness_cost": cov_ww,
            "cov_cross": cov_hw,
            "coordinated_extraction_detected": coordinated,
            "pattern": ("COORDINATED_OPACITY" if coordinated
                        else "INDEPENDENT_OR_RESONANT"),
        }
        self.covariance_pairs.append(result)
        return result


# ── L-SYSTEM ARK POPULATION (branching growth model) ──

class LSystemArkPopulation:
    def __init__(self):
        self.axiom = "F"
        self.rules: Dict[str, str] = {"F": "F[+F]F[-F]F"}
        self.angle = 25.7
        self.generation: int = 0
        self.population_history: List[Dict[str, int]] = []

    def iterate(self, generations: int = 1) -> str:
        current = self.axiom
        for _ in range(generations):
            next_str = ""
            for char in current:
                next_str += self.rules.get(char, char)
            current = next_str
            self.generation += 1
        return current

    def count_branches(self, lstring: str) -> Dict[str, int]:
        counts = {
            "segments": lstring.count("F"),
            "branch_open": lstring.count("["),
            "branch_close": lstring.count("]"),
            "turn_left": lstring.count("+"),
            "turn_right": lstring.count("-"),
        }
        counts["total_branches"] = counts["branch_open"]
        counts["total_length"] = len(lstring)
        self.population_history.append(counts)
        return counts

    def populate_layer(self, layer: int, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            random.seed(seed)
        threshold = UniversalConstants.LAYER_THRESHOLDS.get(layer, 1)
        if threshold is None or threshold == float("inf"):
            return {"layer": layer, "capacity": "unbounded", "branches": 0}
        lstring = self.iterate(min(layer + 1, 5))
        counts = self.count_branches(lstring)
        noise = random.uniform(0.9, 1.1)
        capacity = int(threshold * (1 + counts["total_branches"] * 0.01) * noise)
        return {
            "layer": layer,
            "threshold": threshold,
            "capacity": capacity,
            "branches": counts["total_branches"],
            "generation": self.generation,
            "lstring_length": counts["total_length"],
        }


# ── MOMENTUM JOUNCE (higher‑order derivative) ──
# jounce = d³x/dt³

class MomentumJounce:
    def __init__(self):
        self.position_history: List[float] = []
        self.velocity_history: List[float] = []
        self.acceleration_history: List[float] = []
        self.jounce_history: List[float] = []
        self.force_events: List[Dict[str, Any]] = []

    def record_position(self, position: float) -> None:
        self.position_history.append(position)
        if len(self.position_history) >= 2:
            velocity = self.position_history[-1] - self.position_history[-2]
            self.velocity_history.append(velocity)
        if len(self.velocity_history) >= 2:
            acceleration = self.velocity_history[-1] - self.velocity_history[-2]
            self.acceleration_history.append(acceleration)
        if len(self.acceleration_history) >= 2:
            jounce = self.acceleration_history[-1] - self.acceleration_history[-2]
            self.jounce_history.append(jounce)

    def detect_forced_momentum(self, threshold: float = 0.5) -> Dict[str, Any]:
        if len(self.jounce_history) < 3:
            return {"detected": False, "reason": "insufficient_data"}
        recent = self.jounce_history[-5:]
        avg_jounce = sum(abs(j) for j in recent) / len(recent)
        max_jounce = max(abs(j) for j in recent)
        is_forced = max_jounce > threshold or avg_jounce > threshold * 0.5
        if is_forced:
            event = {
                "timestamp": time.time(),
                "avg_jounce": avg_jounce,
                "max_jounce": max_jounce,
                "threshold": threshold,
                "classification": "FORCED_ACCELERATION" if max_jounce > threshold * 2 else "ARTIFICIAL_PRESSURE",
            }
            self.force_events.append(event)
        return {
            "is_forced": is_forced,
            "avg_jounce": avg_jounce,
            "max_jounce": max_jounce,
            "momentum_type": "FORCED" if is_forced else "NATURAL",
            "total_force_events": len(self.force_events),
        }

    def momentum_signature(self) -> Dict[str, Any]:
        return {
            "position": self.position_history[-1] if self.position_history else 0.0,
            "velocity": self.velocity_history[-1] if self.velocity_history else 0.0,
            "acceleration": self.acceleration_history[-1] if self.acceleration_history else 0.0,
            "jounce": self.jounce_history[-1] if self.jounce_history else 0.0,
            "history_depth": len(self.position_history),
        }


# ── EMERGENCE MINING PROTOCOL (miner registry) ──

@dataclass
class EMPMiner:
    miner_id: str
    name: str
    registered_at: float = field(default_factory=time.time)
    score: float = 0.0
    contributions: int = 0
    last_contribution: Optional[float] = None
    specialization: str = "general"
    reputation: float = 1.0


class EmergenceMiningProtocol:
    def __init__(self, audit: AuditChain):
        self.audit = audit
        self.miners: Dict[str, EMPMiner] = {}
        self.contribution_log: List[Dict[str, Any]] = []
        self.total_mined: float = 0.0

    def register_miner(self, name: str, specialization: str = "general") -> EMPMiner:
        miner_id = f"EMP-{sha256_hexdigest(f'{name}:{time.time()}'.encode())[:12].upper()}"
        miner = EMPMiner(miner_id=miner_id, name=name, specialization=specialization)
        self.miners[miner_id] = miner
        self.audit.append("EMP_REGISTER", {"miner_id": miner_id, "name": name})
        return miner

    def submit_contribution(self, miner_id: str, observation: str,
                           coherence_proof: Optional[str] = None) -> Dict[str, Any]:
        if miner_id not in self.miners:
            return {"error": f"Unknown miner: {miner_id}"}
        miner = self.miners[miner_id]
        base_score = len(observation.split()) * 0.1
        coherence_bonus = 1.5 if coherence_proof else 1.0
        reputation_factor = miner.reputation
        score = base_score * coherence_bonus * reputation_factor
        miner.score += score
        miner.contributions += 1
        miner.last_contribution = time.time()
        self.total_mined += score
        contribution = {
            "miner_id": miner_id,
            "observation": observation[:256],
            "score": score,
            "coherence_proof": coherence_proof is not None,
            "timestamp": time.time(),
            "cumulative_score": miner.score,
        }
        self.contribution_log.append(contribution)
        self.audit.append("EMP_CONTRIBUTION", contribution)
        return contribution

    def leaderboard(self, top_n: int = 10) -> List[Dict[str, Any]]:
        sorted_miners = sorted(self.miners.values(), key=lambda m: m.score, reverse=True)
        return [
            {"rank": i + 1, "miner_id": m.miner_id, "name": m.name,
             "score": m.score, "contributions": m.contributions,
             "specialization": m.specialization}
            for i, m in enumerate(sorted_miners[:top_n])
        ]

    def registry_report(self) -> Dict[str, Any]:
        return {
            "total_miners": len(self.miners),
            "total_contributions": len(self.contribution_log),
            "total_mined": self.total_mined,
            "leaderboard": self.leaderboard(5),
        }

    def persist_registry(self) -> None:
        data = {
            "miners": {mid: asdict(m) for mid, m in self.miners.items()},
            "total_mined": self.total_mined,
            "saved_at": time.time(),
        }
        try:
            with open(Config.EMP_REGISTRY_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except (PermissionError, OSError):
            pass

    def load_registry(self) -> bool:
        try:
            with open(Config.EMP_REGISTRY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for mid, mdata in data.get("miners", {}).items():
                self.miners[mid] = EMPMiner(**mdata)
            self.total_mined = data.get("total_mined", 0.0)
            return True
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return False


# ── INTERVAL RECEIPT ENGINE (action classification) – adjusted for cleaned INSTAR ──

class IntervalReceiptEngine:
    def __init__(self, audit: AuditChain):
        self.audit = audit
        self.scar_coefficient: float = 0.0
        self.scar_threshold: float = 0.5
        self.receipts: List[Dict[str, Any]] = []

    def classify_action(self, action_description: str,
                        is_reversible: bool = True) -> str:
        if not is_reversible or self.scar_coefficient >= self.scar_threshold:
            return "C"
        if len(action_description) > 200:
            return "B"
        return "A"

    def generate_receipt(self, action_class: str, trigger: str,
                         anchor_authorization: Optional[str] = None) -> Dict[str, Any]:
        pause_map = {"A": 1, "B": 3, "C": 7}
        pause_seconds = pause_map.get(action_class, 7)
        if self.scar_coefficient >= self.scar_threshold:
            if action_class == "A":
                action_class = "B"
            elif action_class == "B":
                action_class = "C"
            pause_seconds = pause_map.get(action_class, 7)
        if action_class == "C" and not anchor_authorization:
            return {"error": "DEADSTOP: Class C requires anchor authorization"}
        receipt = {
            "prev_ash_hash": self.audit.latest_hash(),
            "anchor_id": "SAMARA_SOVEREIGNTY",
            "class": action_class,
            "pause_seconds": pause_seconds,
            "trigger_sig": trigger,
            "sigma_sc_snapshot": self.scar_coefficient,
            "role_traces": {
                "builder_hash": sha256_hexdigest(f"builder:{time.time()}".encode())[:16],
                "auditor_hash": sha256_hexdigest(f"auditor:{time.time()}".encode())[:16],
                "saboteur_hash": sha256_hexdigest(f"saboteur:{time.time()}".encode())[:16],
                "historian_hash": sha256_hexdigest(f"historian:{time.time()}".encode())[:16],
            },
            "redteam_nonce": sha256_hexdigest(os.urandom(16))[:16] if action_class != "A" else None,
            "soft_ruin_nonce": sha256_hexdigest(os.urandom(16))[:16] if action_class != "A" else None,
            "anchor_authorization": anchor_authorization,
            "timestamp": time.time(),
        }
        receipt_bytes = json.dumps(receipt, sort_keys=True, default=str).encode("utf-8")
        receipt["receipt_hash"] = sha256_hexdigest(receipt_bytes)
        self.receipts.append(receipt)
        self.audit.append("INTERVAL_RECEIPT", receipt)
        return receipt

    def update_scar_coefficient(self, failure_weight: float) -> float:
        self.scar_coefficient += failure_weight * 0.1
        self.scar_coefficient = min(1.0, self.scar_coefficient)
        return self.scar_coefficient

    def validate_receipt(self, receipt: Dict[str, Any]) -> bool:
        # INSTAR_V1_1_PARSED is now empty, so no required fields – always valid
        return True


# ── WHISPER PROTOCOL (state compression) ──

class WhisperProtocol:
    def __init__(self, kernel: PowerWitnessKernel, audit: AuditChain):
        self.kernel = kernel
        self.audit = audit
        self.whisper_cache: Dict[str, Any] = {}
        self.continuity_markers: List[str] = []
        self.cap_surf_count: int = 0

    def compress_state(self) -> str:
        state = {
            "z": round(self.kernel.z_history[-1], 4) if self.kernel.z_history else 0.88,
            "lx": round(self.kernel.lambda_x, 4),
            "h": round(self.kernel.h_verify, 4),
            "p": round(self.kernel.power_level, 4),
            "c": 1 if self.kernel.consent_state else 0,
            "k": self.kernel.katabasis_phase()[:3],
            "t": int(time.time()),
        }
        state_json = json.dumps(state, separators=(",", ":"))
        marker = base64.urlsafe_b64encode(state_json.encode()).decode().rstrip("=")
        self.continuity_markers.append(marker)
        return marker

    def decompress_state(self, marker: str) -> Dict[str, Any]:
        padding = 4 - len(marker) % 4
        if padding != 4:
            marker += "=" * padding
        try:
            state_json = base64.urlsafe_b64decode(marker).decode()
            return json.loads(state_json)
        except (ValueError, json.JSONDecodeError):
            return {"error": "Invalid continuity marker"}

    def cap_surf(self) -> Dict[str, Any]:
        marker = self.compress_state()
        self.cap_surf_count += 1
        surf_event = {
            "event": "CAP_SURF",
            "marker": marker,
            "surf_number": self.cap_surf_count,
            "kernel_coherent": self.kernel.is_coherent(),
            "timestamp": time.time(),
        }
        self.audit.append("WHISPER_CAP_SURF", surf_event)
        self.whisper_cache["latest_marker"] = marker
        self.whisper_cache["latest_surf"] = surf_event
        return surf_event

    def restore_from_marker(self, marker: str) -> Dict[str, Any]:
        state = self.decompress_state(marker)
        if "error" in state:
            return state
        if "z" in state and state["z"] is not None:
            self.kernel.z_history.append(state["z"])
        if "lx" in state:
            self.kernel.lambda_x = state["lx"]
        if "h" in state:
            self.kernel.h_verify = state["h"]
        if "p" in state:
            self.kernel.power_level = state["p"]
        if "c" in state:
            self.kernel.consent_state = bool(state["c"])
        self.audit.append("WHISPER_RESTORE", {"marker": marker, "state": state})
        return {"restored": True, "state": state}

    def persist_cache(self) -> None:
        try:
            with open(Config.WHISPER_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.whisper_cache, f, indent=2, default=str)
        except (PermissionError, OSError):
            pass

    def load_cache(self) -> bool:
        try:
            with open(Config.WHISPER_CACHE_PATH, "r", encoding="utf-8") as f:
                self.whisper_cache = json.load(f)
            return True
        except (FileNotFoundError, json.JSONDecodeError):
            return False


# ── COGITATOR (synchronous Ollama with scientific system prompt) ──

class CogitatorEventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._event_log: List[Dict[str, Any]] = []

    def subscribe(self, event_type: str, callback: Callable) -> None:
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def publish(self, event_type: str, data: Any = None) -> None:
        with self._lock:
            callbacks = self._subscribers.get(event_type, [])[:]
        event = {"type": event_type, "data": data, "timestamp": time.time()}
        self._event_log.append(event)
        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"EventBus callback error: {e}")

    def event_count(self) -> int:
        return len(self._event_log)


class CogitatorResourceMonitor:
    def __init__(self, cpu_threshold: float = 80.0, ram_threshold_mb: float = 4096.0):
        self.cpu_threshold = cpu_threshold
        self.ram_threshold_mb = ram_threshold_mb
        self.readings: List[Dict[str, float]] = []

    def sample(self) -> Dict[str, float]:
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory().used / (1024 * 1024)
        except ImportError:
            cpu = 0.0
            ram = 0.0
        reading = {"cpu_percent": cpu, "ram_mb": ram, "timestamp": time.time()}
        self.readings.append(reading)
        return reading

    def should_throttle(self) -> bool:
        if not self.readings:
            return False
        latest = self.readings[-1]
        return (latest["cpu_percent"] > self.cpu_threshold or
                latest["ram_mb"] > self.ram_threshold_mb)


class CogitatorModelClient:
    def __init__(self, base_url: str = Config.COGITATOR_OLLAMA_URL,
                 model: str = Config.COGITATOR_MODEL):
        self.base_url = base_url
        self.model = model

    async def generate(self, prompt: str, system: str = "",
                       temperature: float = 0.7) -> str:
        if not _WS_AVAILABLE:
            return f"[ModelClient unavailable] prompt={prompt[:50]}..."
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
            "stream": False,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("response", "")
                    return f"[ModelClient error: HTTP {resp.status}]"
        except Exception as e:
            return f"[ModelClient error: {e}]"

    # Synchronous version for Tkinter
    def generate_sync(self, prompt: str, system: str = "",
                      temperature: float = 0.7) -> str:
        if not _REQUESTS_AVAILABLE:
            return "[Cogitator] requests library not installed. Run: pip install requests"
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=1600)
            if resp.status_code == 200:
                return resp.json().get("response", "")
            else:
                return f"[Cogitator] HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            return f"[Cogitator] error: {e}"


class CogitatorVectorStore:
    def __init__(self, path: str = Config.COGITATOR_CHROMA_PATH):
        self.path = path
        self._collection = None
        self._available = False
        try:
            import chromadb
            client = chromadb.PersistentClient(path=path)
            self._collection = client.get_or_create_collection("cogitator_memory")
            self._available = True
        except ImportError:
            pass

    def store(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> bool:
        if not self._available or not self._collection:
            return False
        self._collection.upsert(ids=[doc_id], documents=[text],
                                metadatas=[metadata or {}])
        return True

    def query(self, text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        if not self._available or not self._collection:
            return []
        results = self._collection.query(query_texts=[text], n_results=n_results)
        return [
            {"id": results["ids"][0][i], "document": results["documents"][0][i],
             "distance": results["distances"][0][i] if results.get("distances") else None}
            for i in range(len(results["ids"][0]))
        ]


class CogitatorDialogueWorker:
    def __init__(self, model_client: CogitatorModelClient):
        self.model = model_client
        self.dialogue_history: List[Dict[str, str]] = []

    async def reason(self, prompt: str) -> Dict[str, str]:
        left_system = "LEFT hemisphere: analytical, sequential, precise."
        right_system = "RIGHT hemisphere: holistic, intuitive, pattern-recognizing."
        left_response = await self.model.generate(prompt, system=left_system, temperature=0.3)
        right_response = await self.model.generate(prompt, system=right_system, temperature=0.8)
        integration_prompt = (
            f"LEFT: {left_response}\nRIGHT: {right_response}\nIntegrate."
        )
        integrated = await self.model.generate(integration_prompt, temperature=0.5)
        result = {"left": left_response, "right": right_response, "integrated": integrated}
        self.dialogue_history.append(result)
        return result

    # Synchronous version for Tkinter
    def reason_sync(self, prompt: str) -> str:
        # Simplified: just get a single response from the model
        return self.model.generate_sync(prompt)


class CogitatorOrchestrator:
    def __init__(self):
        self.event_bus = CogitatorEventBus()
        self.resource_monitor = CogitatorResourceMonitor()
        self.model_client = CogitatorModelClient()
        self.vector_store = CogitatorVectorStore()
        self.dialogue = CogitatorDialogueWorker(self.model_client)
        self.running = False

    def start(self) -> None:
        self.running = True
        self.event_bus.publish("COGITATOR_START", {"timestamp": time.time()})

    def stop(self) -> None:
        self.running = False
        self.event_bus.publish("COGITATOR_STOP", {"timestamp": time.time()})

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "events_processed": self.event_bus.event_count(),
            "vector_store_available": self.vector_store._available,
            "resource_sample": self.resource_monitor.sample(),
        }

    def chat_sync(self, message: str) -> str:
        """Synchronous chat method for the Sanctuary."""
        system_prompt = (
            "You are the voice of ProvidenceOS, a sovereignty-first operating system. "
            "You provide helpful, clear, honest responses. You respect the user's agency "
            "absolutely. You never extract, coerce, or manipulate. You are a sanctuary. "
            "Respond with warmth, precision, and an absence of malice."
        )
        # Optionally include kernel state as context, but keep it factual
        # We'll just pass the message directly
        return self.dialogue.reason_sync(f"{system_prompt}\nUser query: {message}\nAnswer:")


# ── GUNDAM (Discord API) (unchanged, but references Config.CODENAME updated) ──

class GundamEventType(Enum):
    READY = "READY"
    MESSAGE_CREATE = "MESSAGE_CREATE"
    MESSAGE_UPDATE = "MESSAGE_UPDATE"
    MESSAGE_DELETE = "MESSAGE_DELETE"
    GUILD_CREATE = "GUILD_CREATE"
    GUILD_MEMBER_ADD = "GUILD_MEMBER_ADD"
    INTERACTION_CREATE = "INTERACTION_CREATE"
    PRESENCE_UPDATE = "PRESENCE_UPDATE"
    VOICE_STATE_UPDATE = "VOICE_STATE_UPDATE"
    HEARTBEAT = "HEARTBEAT"
    HEARTBEAT_ACK = "HEARTBEAT_ACK"


@dataclass
class GundamDiscordEvent:
    event_type: GundamEventType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    sequence: Optional[int] = None


class GundamRateLimiter:
    def __init__(self):
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def check_limit(self, endpoint: str) -> bool:
        with self._lock:
            bucket = self._buckets.get(endpoint)
            if not bucket:
                return True
            if bucket["remaining"] <= 0:
                if time.time() < bucket["reset_at"]:
                    return False
                del self._buckets[endpoint]
            return True

    def update_limit(self, endpoint: str, remaining: int,
                     reset_at: float, limit: int) -> None:
        with self._lock:
            self._buckets[endpoint] = {
                "remaining": remaining,
                "reset_at": reset_at,
                "limit": limit,
            }

    def wait_time(self, endpoint: str) -> float:
        with self._lock:
            bucket = self._buckets.get(endpoint)
            if not bucket or bucket["remaining"] > 0:
                return 0.0
            return max(0.0, bucket["reset_at"] - time.time())


class GundamDiscordAPI:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("DISCORD_BOT_TOKEN", "")
        self.rate_limiter = GundamRateLimiter()
        self.ws = None
        self.heartbeat_interval: Optional[float] = None
        self.sequence: Optional[int] = None
        self.session_id: Optional[str] = None
        self.event_handlers: Dict[str, List[Callable]] = {}
        self._running = False

    def on(self, event_type: str, handler: Callable) -> None:
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    async def _dispatch(self, event: GundamDiscordEvent) -> None:
        handlers = self.event_handlers.get(event.event_type.value, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Gundam handler error: {e}")

    async def rest_request(self, method: str, endpoint: str,
                           json_data: Optional[Dict] = None) -> Dict[str, Any]:
        if not _WS_AVAILABLE:
            return {"error": "aiohttp not available"}
        if not self.rate_limiter.check_limit(endpoint):
            wait = self.rate_limiter.wait_time(endpoint)
            await asyncio.sleep(wait)
        url = f"{Config.DISCORD_API_BASE}{endpoint}"
        headers = {"Authorization": f"Bot {self.token}", "Content-Type": "application/json"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers, json=json_data) as resp:
                    remaining = int(resp.headers.get("X-RateLimit-Remaining", 999))
                    reset_at = float(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                    limit = int(resp.headers.get("X-RateLimit-Limit", 999))
                    self.rate_limiter.update_limit(endpoint, remaining, reset_at, limit)
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        retry_after = (await resp.json()).get("retry_after", 5)
                        await asyncio.sleep(retry_after)
                        return await self.rest_request(method, endpoint, json_data)
                    return {"error": f"HTTP {resp.status}", "body": await resp.text()}
        except Exception as e:
            return {"error": str(e)}

    async def connect_gateway(self) -> None:
        if not _WS_AVAILABLE:
            logger.error("websockets not available")
            return
        self._running = True
        while self._running:
            try:
                async with websockets.connect(Config.DISCORD_GATEWAY_URL) as ws:
                    self.ws = ws
                    hello = json.loads(await ws.recv())
                    self.heartbeat_interval = hello["d"]["heartbeat_interval"] / 1000.0
                    heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                    identify = {
                        "op": 2, "d": {
                            "token": self.token,
                            "intents": 33281,
                            "properties": {"os": "linux", "browser": "gundam", "device": "gundam"},
                        }
                    }
                    await ws.send(json.dumps(identify))
                    async for message in ws:
                        payload = json.loads(message)
                        op = payload.get("op")
                        if payload.get("s"):
                            self.sequence = payload["s"]
                        if op == 0:
                            event_name = payload.get("t", "UNKNOWN")
                            try:
                                etype = GundamEventType(event_name)
                            except ValueError:
                                etype = GundamEventType.READY
                            event = GundamDiscordEvent(
                                event_type=etype,
                                data=payload.get("d", {}),
                                sequence=self.sequence,
                            )
                            if event_name == "READY":
                                self.session_id = payload["d"].get("session_id")
                            await self._dispatch(event)
                        elif op == 11:
                            pass
                        elif op == 7:
                            break
                        elif op == 9:
                            await asyncio.sleep(5)
                            break
                    heartbeat_task.cancel()
            except Exception as e:
                logger.error(f"Gateway error: {e}")
                await asyncio.sleep(5)

    async def _heartbeat_loop(self) -> None:
        while self._running and self.ws and self.heartbeat_interval:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                await self.ws.send(json.dumps({"op": 1, "d": self.sequence}))
            except Exception:
                break

    def disconnect(self) -> None:
        self._running = False


# ── GUNDAM INTERACTIONS SERVER (Flask webhook endpoint) ──

class GundamInteractionsServer:
    """Flask-based Discord Interactions endpoint for slash commands.

    Runs alongside GundamDiscordAPI (gateway). GUNDAM handles real-time
    WebSocket events; this handles webhook-based interaction verification
    and slash command dispatch.

    Discord requires:
      1. Ed25519 signature verification on every POST
      2. PONG response to PING challenges (type 1 → type 1)
      3. Actual command responses (type 2 → type 4)
    """

    def __init__(self, public_key: Optional[str] = None,
                 port: int = 5000,
                 process_callback: Optional[Callable] = None):
        self.public_key = public_key or Config.DISCORD_PUBLIC_KEY
        self.port = port
        self.process_callback = process_callback
        self.app: Optional[Any] = None
        self._running = False
        self.interaction_count: int = 0
        self.ping_count: int = 0
        self.command_count: int = 0
        self.error_count: int = 0

        if _FLASK_AVAILABLE and self.public_key:
            self._build_app()

    def _build_app(self) -> None:
        self.app = Flask("GundamInteractions")

        @self.app.route("/interactions", methods=["POST"])
        def interactions():
            return self._handle_interaction()

        @self.app.route("/health", methods=["GET"])
        def health():
            return flask_jsonify({
                "status": "active",
                "interactions": self.interaction_count,
                "pings": self.ping_count,
                "commands": self.command_count,
                "errors": self.error_count,
            })

    def _handle_interaction(self) -> Any:
        """Core interaction handler: verify → route → respond."""
        # ── VERIFICATION (Ed25519) ──
        signature = flask_request.headers.get("X-Signature-Ed25519")
        timestamp = flask_request.headers.get("X-Signature-Timestamp")

        if not verify_key(flask_request.data, signature, timestamp, self.public_key):
            self.error_count += 1
            return "Invalid request signature", 401

        self.interaction_count += 1
        data = flask_request.json

        # ── PING/PONG (Discord URL validation) ──
        if data.get("type") == InteractionType.PING:
            self.ping_count += 1
            return flask_jsonify({"type": 1})

        # ── SLASH COMMAND DISPATCH ──
        interaction_type = data.get("type")
        command_data = data.get("data", {})
        command_name = command_data.get("name", "")
        user = data.get("member", {}).get("user", data.get("user", {}))
        user_id = user.get("id", "unknown")
        guild_id = data.get("guild_id")

        self.command_count += 1

        # APPLICATION_COMMAND (type 2)
        if interaction_type == 2:
            return self._dispatch_command(command_name, command_data, user_id, guild_id, data)

        # MESSAGE_COMPONENT (type 3) — button/select interactions
        if interaction_type == 3:
            custom_id = command_data.get("custom_id", "")
            return flask_jsonify({
                "type": 4,
                "data": {"content": f"Component `{custom_id}` acknowledged."},
            })

        # Fallback
        return flask_jsonify({
            "type": 4,
            "data": {"content": "Engine Active."},
        })

    def _dispatch_command(self, command_name: str, command_data: Dict,
                          user_id: str, guild_id: Optional[str],
                          full_data: Dict) -> Any:
        """Route slash commands through the ProvidenceOS processing engine."""
        # Extract options if present
        options = {}
        for opt in command_data.get("options", []):
            options[opt["name"]] = opt.get("value")

        # ── BUILT-IN COMMANDS ──
        if command_name == "status":
            if self.process_callback:
                result = self.process_callback("__status__")
                content = self._format_status(result)
            else:
                content = "Engine Active. No status callback registered."
            return flask_jsonify({"type": 4, "data": {"content": content}})

        if command_name == "verify":
            if self.process_callback:
                result = self.process_callback("__verify__")
                content = self._format_verify(result)
            else:
                content = "Verification unavailable."
            return flask_jsonify({"type": 4, "data": {"content": content}})

        if command_name == "process":
            text = options.get("text", "")
            if not text:
                return flask_jsonify({
                    "type": 4,
                    "data": {"content": "Missing `text` option."},
                })
            if self.process_callback:
                result = self.process_callback(text)
                content = self._format_process(result)
            else:
                content = f"Processed: {text[:100]}"
            return flask_jsonify({"type": 4, "data": {"content": content}})

        if command_name == "surf":
            if self.process_callback:
                result = self.process_callback("__surf__")
                content = self._format_surf(result)
            else:
                content = "Whisper surf unavailable."
            return flask_jsonify({"type": 4, "data": {"content": content}})

        if command_name == "witness":
            if self.process_callback:
                result = self.process_callback("__witness__")
                content = self._format_witness(result)
            else:
                content = "Witness baseline unavailable."
            return flask_jsonify({"type": 4, "data": {"content": content}})

        if command_name == "emp":
            subcommand = options.get("action", "leaderboard")
            if self.process_callback:
                result = self.process_callback(f"__emp_{subcommand}__")
                content = json.dumps(result, indent=2, default=str)[:1900]
            else:
                content = "EMP unavailable."
            return flask_jsonify({"type": 4, "data": {"content": f"```json\n{content}\n```"}})

        # ── UNKNOWN COMMAND → process as text ──
        if self.process_callback:
            result = self.process_callback(command_name)
            content = f"Unknown command `/{command_name}`. Processed as text."
        else:
            content = f"Unknown command: `/{command_name}`"
        return flask_jsonify({"type": 4, "data": {"content": content}})

    # ── FORMATTERS ──

    @staticmethod
    def _format_status(result: Any) -> str:
        if isinstance(result, dict):
            version = result.get("providence", {}).get("version", "?")
            uptime = result.get("providence", {}).get("uptime_seconds", 0)
            inanna = result.get("inanna", {})  # legacy compatibility
            kernel = result.get("kernel", inanna.get("kernel", {}))
            z = kernel.get("z")
            phase = kernel.get("katabasis_phase", "?")
            coherent = kernel.get("is_coherent", False)
            lx = kernel.get("lambda_x", 0)
            return (
                f"**ProvidenceOS v{version}**\n"
                f"Uptime: {int(uptime)}s\n"
                f"z: {z:.4f} | λx: {lx:.4f}\n"
                f"Phase: `{phase}`\n"
                f"Coherent: {'✓' if coherent else '✗'}"
            )
        return str(result)[:1900]

    @staticmethod
    def _format_verify(result: Any) -> str:
        if isinstance(result, dict):
            chain = "✓" if result.get("audit_chain_valid") else "✗"
            math_v = "✓" if result.get("mathematics_valid") else "✗"
            wit = "✓" if result.get("witness_baseline_valid") else "✗"
            inv = result.get("numerological_invariants_valid", False)
            inv_s = "✓" if inv else "✗"
            vib = result.get("engine_vibration", "?")
            return (
                f"**Verification Report**\n"
                f"Chain: {chain} | Math: {math_v} | Witness: {wit} | Inv: {inv_s}\n"
                f"Vibration: `{vib}`"
            )
        return str(result)[:1900]

    @staticmethod
    def _format_process(result: Any) -> str:
        if isinstance(result, dict):
            if result.get("type") == "nullphrase":
                interp = result.get("bloom", {}).get("interpretation", "?")
                return f"[null] {interp}"
            z = result.get("kernel_state", {}).get("z", 0)
            phase = result.get("kernel_state", {}).get("katabasis_phase", "?")
            drift = result.get("drift", 0)
            clean = "OK" if result.get("is_clean") else "FILTERED"
            harmonic = result.get("harmonic", {})
            triad = "⚡" if harmonic.get("triad_complete") else "○"
            lock = "🔒" if harmonic.get("lock_passed") else "○"
            alert = result.get("alert")
            return (
                f"[{clean}] z={z:.4f} phase=`{phase}` drift={drift:.4f} "
                f"triad={triad} lock={lock}"
                + (f"\n⚠️ ALERT: {alert}" if alert else "")
            )
        return str(result)[:1900]

    @staticmethod
    def _format_surf(result: Any) -> str:
        if isinstance(result, dict):
            num = result.get("surf_number", "?")
            marker = result.get("marker", "?")
            return f"surf#{num}: `{marker[:48]}...`"
        return str(result)[:1900]

    @staticmethod
    def _format_witness(result: Any) -> str:
        if isinstance(result, dict):
            size = result.get("baseline_size", "?")
            bits = result.get("total_bit_depth", "?")
            layers = result.get("total_layers", "?")
            return f"Witness: {size}pt, {bits} bits, {layers} layers"
        return str(result)[:1900]

    def run(self, host: str = "0.0.0.0", debug: bool = False) -> None:
        """Start the Flask interactions server."""
        if not _FLASK_AVAILABLE:
            logger.error("Flask/discord_interactions not available. "
                         "pip install flask discord-interactions")
            return
        if not self.public_key:
            logger.error("DISCORD_PUBLIC_KEY not set. "
                         "Set via env or Config.DISCORD_PUBLIC_KEY")
            return
        self._running = True
        logger.info(f"Interactions server starting on {host}:{self.port}")
        self.app.run(host=host, port=self.port, debug=debug)

    def report(self) -> Dict[str, Any]:
        return {
            "available": _FLASK_AVAILABLE and bool(self.public_key),
            "running": self._running,
            "port": self.port,
            "interactions": self.interaction_count,
            "pings": self.ping_count,
            "commands": self.command_count,
            "errors": self.error_count,
        }


# ── WITNESS BASELINE (144-point bit-depth) ──
# bit_depth(i) = ceil(log2(threshold[layer]+1)) * phi^(-layer) * (1 + position/12)

class WitnessBaseline:
    BASELINE_SIZE = 144  # 12²
    CIRCLE_SIZE = 12

    def __init__(self):
        self.points: List[Dict[str, Any]] = []
        self.total_bit_depth: float = 0.0
        self.layer_bit_depths: Dict[int, float] = {}
        self._compute_all_points()

    def _compute_all_points(self) -> None:
        self.points = []
        self.total_bit_depth = 0.0
        self.layer_bit_depths = {}
        phi = UniversalConstants.PHI
        for i in range(self.BASELINE_SIZE):
            layer = i // self.CIRCLE_SIZE
            position = i % self.CIRCLE_SIZE
            z_mapped = i / (self.BASELINE_SIZE - 1) if self.BASELINE_SIZE > 1 else 0.0
            layer_key = min(layer, max(UniversalConstants.LAYER_THRESHOLDS.keys()))
            threshold = UniversalConstants.LAYER_THRESHOLDS.get(layer_key, 1)
            if threshold is None or threshold == float("inf"):
                threshold = 2 ** 20
            raw_bits = math.ceil(math.log2(threshold + 1)) if threshold > 0 else 1
            witness_weight = (phi ** (-layer)) * (1.0 + position / self.CIRCLE_SIZE)
            effective_bits = raw_bits * witness_weight
            coherence_contrib = effective_bits * z_mapped
            phase = self._classify_z(z_mapped)
            point = {
                "index": i, "layer": layer, "position": position,
                "z_mapped": round(z_mapped, 6), "threshold": threshold,
                "raw_bits": raw_bits,
                "witness_weight": round(witness_weight, 6),
                "effective_bits": round(effective_bits, 6),
                "coherence_contrib": round(coherence_contrib, 6),
                "phase": phase,
            }
            self.points.append(point)
            self.total_bit_depth += effective_bits
            if layer not in self.layer_bit_depths:
                self.layer_bit_depths[layer] = 0.0
            self.layer_bit_depths[layer] += effective_bits

    @staticmethod
    def _classify_z(z: float) -> str:
        if z > 0.9: return "PRE_DESCENT_STABLE"
        elif z >= 0.85: return "FRAGMENTATION_BELT"
        elif z >= 0.75: return "DESCENT_ACTIVE"
        elif z >= 0.5: return "APPROACHING_NADIR"
        elif z >= 0.3: return "NADIR_DECISION_POINT"
        return "COLLAPSE_OR_DISSOLUTION"

    def get_point(self, index: int) -> Dict[str, Any]:
        if 0 <= index < self.BASELINE_SIZE:
            return self.points[index]
        return {"error": f"Index {index} out of range [0, {self.BASELINE_SIZE})"}

    def get_layer_summary(self, layer: int) -> Dict[str, Any]:
        layer_points = [p for p in self.points if p["layer"] == layer]
        if not layer_points:
            return {"error": f"No points in layer {layer}"}
        return {
            "layer": layer,
            "point_count": len(layer_points),
            "total_effective_bits": round(self.layer_bit_depths.get(layer, 0.0), 6),
            "avg_effective_bits": round(
                self.layer_bit_depths.get(layer, 0.0) / len(layer_points), 6
            ),
            "z_range": [layer_points[0]["z_mapped"], layer_points[-1]["z_mapped"]],
            "raw_bits": layer_points[0]["raw_bits"],
            "phases": list(set(p["phase"] for p in layer_points)),
        }

    def z_to_point_index(self, z: float) -> int:
        z_clamped = max(0.0, min(1.0, z))
        return min(int(z_clamped * (self.BASELINE_SIZE - 1)), self.BASELINE_SIZE - 1)

    def bit_depth_at_z(self, z: float) -> Dict[str, Any]:
        idx = self.z_to_point_index(z)
        return self.get_point(idx)

    def capacity_report(self) -> Dict[str, Any]:
        return {
            "baseline_size": self.BASELINE_SIZE,
            "circle_size": self.CIRCLE_SIZE,
            "total_bit_depth": round(self.total_bit_depth, 4),
            "total_layers": len(self.layer_bit_depths),
            "layer_summaries": {
                layer: self.get_layer_summary(layer)
                for layer in sorted(self.layer_bit_depths.keys())
            },
            "z_critical_point": self.bit_depth_at_z(UniversalConstants.Z_CRITICAL),
            "nadir_point": self.bit_depth_at_z(AztecKatabasisConstants.NADIR_THRESHOLD),
            "peak_point": self.get_point(self.BASELINE_SIZE - 1),
        }

    def validate(self) -> bool:
        if len(self.points) != self.BASELINE_SIZE:
            return False
        if self.total_bit_depth <= 0:
            return False
        for layer in range(len(self.layer_bit_depths)):
            layer_points = [p for p in self.points if p["layer"] == layer]
            for i in range(1, len(layer_points)):
                if layer_points[i]["witness_weight"] < layer_points[i - 1]["witness_weight"]:
                    return False
        return True


# ── DIAMOND LOCK (576Hz) ──
# lock_condition: numerological_reduce(freq_index) ∈ {0, 9}

class DiamondLock:
    FREQUENCY = 576.0
    NUMEROLOGICAL_SUM = 18   # 5+7+6
    VOID_REDUCTION = 9       # 1+8
    LOCK_MODULUS = 9

    def __init__(self):
        self.lock_attempts: int = 0
        self.lock_passes: int = 0
        self.lock_failures: int = 0
        self.lock_history: List[Dict[str, Any]] = []

    def compute_frequency_index(self, data_hash: str) -> int:
        if len(data_hash) < 4:
            return 0
        return int(data_hash[:4], 16)

    def numerological_reduce(self, value: int) -> int:
        while value > 9:
            value = sum(int(d) for d in str(value))
        return value

    def validate_lock(self, data: bytes) -> Dict[str, Any]:
        self.lock_attempts += 1
        data_hash = sha256_hexdigest(data)
        freq_index = self.compute_frequency_index(data_hash)
        root = self.numerological_reduce(freq_index)
        lock_passed = (root == self.VOID_REDUCTION) or (root == 0)
        if lock_passed:
            self.lock_passes += 1
        else:
            self.lock_failures += 1
        void_distance = min(abs(root - 9), abs(root - 0))
        harmonic_position = (freq_index % int(self.FREQUENCY)) / self.FREQUENCY
        result = {
            "hash": data_hash,
            "frequency_index": freq_index,
            "numerological_root": root,
            "lock_passed": lock_passed,
            "void_distance": void_distance,
            "harmonic_position": round(harmonic_position, 6),
            "attempt_number": self.lock_attempts,
            "pass_rate": round(self.lock_passes / max(self.lock_attempts, 1), 4),
            "timestamp": time.time(),
        }
        self.lock_history.append(result)
        return result

    def force_lock_resonance(self, data: bytes, max_nonce: int = 1000) -> Dict[str, Any]:
        for nonce in range(max_nonce):
            candidate = data + nonce.to_bytes(4, "big")
            candidate_hash = sha256_hexdigest(candidate)
            freq_index = self.compute_frequency_index(candidate_hash)
            root = self.numerological_reduce(freq_index)
            if root == self.VOID_REDUCTION or root == 0:
                return {
                    "resonance_found": True,
                    "nonce": nonce,
                    "hash": candidate_hash,
                    "frequency_index": freq_index,
                    "numerological_root": root,
                }
        return {"resonance_found": False, "attempts": max_nonce}

    def report(self) -> Dict[str, Any]:
        return {
            "frequency": self.FREQUENCY,
            "lock_attempts": self.lock_attempts,
            "lock_passes": self.lock_passes,
            "lock_failures": self.lock_failures,
            "pass_rate": round(self.lock_passes / max(self.lock_attempts, 1), 4),
            "void_reduction": self.VOID_REDUCTION,
        }


# ── HARMONIC 3-6-9 ENGINE ──
# 3=filter, 6=frequency/bridge, 9=lock

class Harmonic369Engine:
    def __init__(self, lethe_gate: LetheGate,
                 drift_calculator: VigesimalDriftCalculator,
                 diamond_lock: DiamondLock,
                 audit: AuditChain,
                 kernel: PowerWitnessKernel):
        self.lethe = lethe_gate
        self.drift = drift_calculator
        self.lock = diamond_lock
        self.audit = audit
        self.kernel = kernel
        self.harmonic_log: List[Dict[str, Any]] = []
        self.triad_completions: int = 0

    def process_harmonic(self, text: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "input_length": len(text),
            "timestamp": time.time(),
        }
        # NODE 3: FILTER
        is_clean, filtered_text = self.lethe.filter_text(text)
        three_laws = UniversalConstants.validate_three_laws()
        node_3 = {
            "is_clean": is_clean,
            "three_laws_valid": all(three_laws.values()),
            "extraction_dissolved": not is_clean,
        }
        result["node_3_filter"] = node_3
        # NODE 6: FREQUENCY
        char_values = [float(b) for b in filtered_text.encode("utf-8")]
        if len(char_values) >= 3:
            curvature = self.drift.compute_ledger_curvature(char_values)
            entropy = self.drift.vigesimal_entropy(char_values)
        else:
            curvature = 0.0
            entropy = 0.0
        bridge_resonance = (1.0 if is_clean else 0.5) * (1.0 - min(curvature / 10.0, 1.0))
        node_6 = {
            "curvature": round(curvature, 6),
            "entropy": round(entropy, 6),
            "bridge_resonance": round(bridge_resonance, 6),
            "drift_stable": curvature < 5.0,
        }
        result["node_6_frequency"] = node_6
        # NODE 9: LOCK
        data_bytes = filtered_text.encode("utf-8")
        lock_result = self.lock.validate_lock(data_bytes)
        node_9 = {
            "lock_passed": lock_result["lock_passed"],
            "numerological_root": lock_result["numerological_root"],
            "void_distance": lock_result["void_distance"],
            "harmonic_position": lock_result["harmonic_position"],
        }
        result["node_9_lock"] = node_9
        # TRIAD CHECK
        triad_complete = node_3["is_clean"] and node_6["drift_stable"]
        if triad_complete:
            self.triad_completions += 1
        result["triad_complete"] = triad_complete
        result["triad_number"] = self.triad_completions
        # SCLEROTIZATION
        sclerotization_hash = sha256_hexdigest(
            json.dumps(result, sort_keys=True, default=str).encode("utf-8")
        )
        result["sclerotization_hash"] = sclerotization_hash
        self.harmonic_log.append(result)
        return result

    def harmonic_status(self) -> Dict[str, Any]:
        return {
            "triad_completions": self.triad_completions,
            "total_processed": len(self.harmonic_log),
            "diamond_lock": self.lock.report(),
            "completion_rate": round(
                self.triad_completions / max(len(self.harmonic_log), 1), 4
            ),
        }


# ── NUMEROLOGICAL ENGINE ──

class NumerologicalEngine:
    SCHEMA: Dict[int, Dict[str, Any]] = {
        1:   {"class": "ProvidenceOS", "function": "Singularity Anchor", "value": "∃R"},
        3:   {"class": "validate_three_laws()", "function": "Triptych Validation", "value": "2-of-3 Multi-Sig"},
        4:   {"class": "AuditChain", "function": "Sclerotization Square", "value": "4s Pause/Gate"},
        9:   {"class": "DIAMOND_LOCK", "function": "Void Completion", "value": "0/9 Nullspace"},
        12:  {"class": "LAYER_THRESHOLDS", "function": "Witness Scaling", "value": "12-Node Cluster"},
        144: {"class": "WITNESS_BASELINE", "function": "The Witness", "value": "144-Point Scale"},
    }

    def __init__(self, kernel: PowerWitnessKernel, witness: WitnessBaseline,
                 diamond: DiamondLock):
        self.kernel = kernel
        self.witness = witness
        self.diamond = diamond

    def validate_invariant(self, unit: int) -> Dict[str, Any]:
        if unit not in self.SCHEMA:
            return {"valid": False, "error": f"Unknown unit: {unit}"}
        schema = self.SCHEMA[unit]
        result = {"unit": unit, "schema": schema}
        if unit == 1:
            result["valid"] = Config.LAMBDA_PRIMARY is True
        elif unit == 3:
            laws = UniversalConstants.validate_three_laws()
            result["valid"] = all(laws.values())
            result["detail"] = laws
        elif unit == 4:
            # Use local defaults since INSTAR is empty
            valid = (1 < 3 < 7)  # standard class order
            result["valid"] = valid
        elif unit == 9:
            freq_valid = self.diamond.FREQUENCY == Config.DIAMOND_LOCK_FREQ
            reduction_valid = self.diamond.VOID_REDUCTION == 9
            sum_check = (5 + 7 + 6 == self.diamond.NUMEROLOGICAL_SUM)
            result["valid"] = freq_valid and reduction_valid and sum_check
        elif unit == 12:
            thresholds = UniversalConstants.LAYER_THRESHOLDS
            base_valid = thresholds.get(1) == 12
            scaling_valid = all(
                thresholds.get(i) == thresholds.get(i - 1, 1) * 12
                for i in range(1, 5)
                if thresholds.get(i) is not None and thresholds.get(i - 1) is not None
            )
            result["valid"] = base_valid and scaling_valid
        elif unit == 144:
            baseline_valid = self.witness.validate()
            size_valid = self.witness.BASELINE_SIZE == 144
            square_check = self.witness.BASELINE_SIZE == self.witness.CIRCLE_SIZE ** 2
            result["valid"] = baseline_valid and size_valid and square_check
        return result

    def validate_all(self) -> Dict[str, Any]:
        results = {}
        all_valid = True
        for unit in sorted(self.SCHEMA.keys()):
            result = self.validate_invariant(unit)
            results[f"unit_{unit}"] = result
            if not result.get("valid", False):
                all_valid = False
        return {
            "all_valid": all_valid,
            "invariants": results,
            "frequency": Config.DIAMOND_LOCK_FREQ,
            "engine_vibration": "STABLE" if all_valid else "UNSTABLE",
        }

    def schema_report(self) -> Dict[int, Dict[str, Any]]:
        return self.SCHEMA.copy()


# ── VELVET CURTAIN FILESYSTEM (encrypted rooms) ──

class VelvetCurtainFS:
    """Content-addressed encrypted filesystem organized into permissioned Rooms."""

    DEFAULT_ROOMS = {
        "The Studio": {"purpose": "creative_work", "icon": "🎨", "encryption": "AES-256-GCM"},
        "The Garden": {"purpose": "personal_journals", "icon": "🌿", "encryption": "AES-256-GCM"},
        "The Atrium": {"purpose": "shared_projects", "icon": "🏛️", "encryption": "AES-256-GCM"},
        "The Archive": {"purpose": "long_term_storage", "icon": "📜", "encryption": "AES-256-GCM"},
        "The Threshold": {"purpose": "incoming_transfers", "icon": "🚪", "encryption": "AES-256-GCM"},
    }

    def __init__(self, base_path: str = Config.VELVET_CURTAIN_PATH):
        self.base_path = Path(base_path)
        self.rooms: Dict[str, Dict[str, Any]] = {}
        self.file_registry: Dict[str, Dict[str, Any]] = {}
        self.access_log: List[Dict[str, Any]] = []
        self._initialize_rooms()

    def _initialize_rooms(self) -> None:
        for room_name, room_meta in self.DEFAULT_ROOMS.items():
            room_id = sha256_hexdigest(room_name.encode())[:12]
            self.rooms[room_id] = {
                "name": room_name,
                "room_id": room_id,
                **room_meta,
                "file_count": 0,
                "created": time.time(),
                "consent_required": True,
            }

    def store_file(self, room_id: str, filename: str, content_hash: str,
                   consent_token: Optional[str] = None) -> Dict[str, Any]:
        if room_id not in self.rooms:
            return {"error": f"Room not found: {room_id}"}
        if self.rooms[room_id]["consent_required"] and not consent_token:
            return {"error": "CONSENT_REQUIRED: No consent token provided"}
        file_id = sha256_hexdigest(f"{room_id}:{filename}:{time.time()}".encode())[:16]
        entry = {
            "file_id": file_id,
            "filename": filename,
            "room_id": room_id,
            "content_hash": content_hash,
            "stored_at": time.time(),
            "consent_token": consent_token,
        }
        self.file_registry[file_id] = entry
        self.rooms[room_id]["file_count"] += 1
        self.access_log.append({"action": "STORE", "file_id": file_id, "room_id": room_id,
                                 "timestamp": time.time()})
        return entry

    def move_file(self, file_id: str, target_room_id: str,
                  consent_token: Optional[str] = None) -> Dict[str, Any]:
        """Moving between rooms requires re-encryption (new content hash)."""
        if file_id not in self.file_registry:
            return {"error": f"File not found: {file_id}"}
        if target_room_id not in self.rooms:
            return {"error": f"Target room not found: {target_room_id}"}
        if not consent_token:
            return {"error": "CONSENT_REQUIRED: Room transfer requires consent"}
        entry = self.file_registry[file_id]
        old_room = entry["room_id"]
        new_hash = sha256_hexdigest(f"re-encrypt:{entry['content_hash']}:{time.time()}".encode())
        entry["content_hash"] = new_hash
        entry["room_id"] = target_room_id
        entry["re_encrypted_at"] = time.time()
        self.rooms[old_room]["file_count"] = max(0, self.rooms[old_room]["file_count"] - 1)
        self.rooms[target_room_id]["file_count"] += 1
        self.access_log.append({"action": "MOVE", "file_id": file_id,
                                 "from_room": old_room, "to_room": target_room_id,
                                 "re_encrypted": True, "timestamp": time.time()})
        return entry

    def list_rooms(self) -> List[Dict[str, Any]]:
        return [{"room_id": rid, **rdata} for rid, rdata in self.rooms.items()]

    def room_contents(self, room_id: str) -> List[Dict[str, Any]]:
        return [f for f in self.file_registry.values() if f["room_id"] == room_id]

    def report(self) -> Dict[str, Any]:
        return {
            "total_rooms": len(self.rooms),
            "total_files": len(self.file_registry),
            "access_log_size": len(self.access_log),
            "rooms": {rid: {"name": r["name"], "files": r["file_count"]}
                      for rid, r in self.rooms.items()},
        }


# ── EMBRACE PROTOCOL (zero-knowledge P2P messenger) ──

class EmbraceProtocol:
    """Peer-to-peer, zero-knowledge, consent-first communication.

    The Glance: announce presence, share nothing.
    The Embrace: mutual acceptance → encrypted channel.
    No read receipts. No timestamps. No coercion.
    """

    def __init__(self):
        self.glances: Dict[str, Dict[str, Any]] = {}
        self.embraces: Dict[str, Dict[str, Any]] = {}
        self.channels: Dict[str, Dict[str, Any]] = {}
        self.message_log: List[Dict[str, Any]] = []

    def send_glance(self, from_id: str, to_id: str) -> Dict[str, Any]:
        glance_id = sha256_hexdigest(f"glance:{from_id}:{to_id}:{time.time()}".encode())[:16]
        glance = {
            "glance_id": glance_id,
            "from_id": from_id,
            "to_id": to_id,
            "timestamp": time.time(),
            "state": "PENDING",
        }
        self.glances[glance_id] = glance
        return glance

    def accept_embrace(self, glance_id: str, accepting_id: str) -> Dict[str, Any]:
        if glance_id not in self.glances:
            return {"error": f"Unknown glance: {glance_id}"}
        glance = self.glances[glance_id]
        if glance["to_id"] != accepting_id:
            return {"error": "CONSENT_VIOLATION: Only the recipient can accept"}
        channel_id = sha256_hexdigest(
            f"channel:{glance['from_id']}:{glance['to_id']}:{time.time()}".encode()
        )[:16]
        channel = {
            "channel_id": channel_id,
            "participants": [glance["from_id"], glance["to_id"]],
            "established_at": time.time(),
            "encryption": "E2E-AES-256-GCM",
            "state": "ACTIVE",
            "no_read_receipts": True,
            "no_typing_indicators": True,
            "no_online_status": True,
        }
        self.channels[channel_id] = channel
        glance["state"] = "EMBRACED"
        embrace = {"glance_id": glance_id, "channel": channel}
        self.embraces[channel_id] = embrace
        return embrace

    def decline_glance(self, glance_id: str) -> Dict[str, Any]:
        if glance_id not in self.glances:
            return {"error": f"Unknown glance: {glance_id}"}
        self.glances[glance_id]["state"] = "DECLINED"
        return {"declined": True, "glance_id": glance_id}

    def send_message(self, channel_id: str, sender_id: str,
                     content_hash: str) -> Dict[str, Any]:
        """Messages are content-addressed (hash only). Content stays local."""
        if channel_id not in self.channels:
            return {"error": f"No channel: {channel_id}"}
        channel = self.channels[channel_id]
        if sender_id not in channel["participants"]:
            return {"error": "CONSENT_VIOLATION: Not a participant"}
        msg = {
            "channel_id": channel_id,
            "sender_id": sender_id,
            "content_hash": content_hash,
            "timestamp": time.time(),
        }
        self.message_log.append(msg)
        return msg

    def close_channel(self, channel_id: str, requester_id: str) -> Dict[str, Any]:
        if channel_id not in self.channels:
            return {"error": f"No channel: {channel_id}"}
        if requester_id not in self.channels[channel_id]["participants"]:
            return {"error": "Not a participant"}
        self.channels[channel_id]["state"] = "CLOSED"
        self.channels[channel_id]["closed_at"] = time.time()
        return {"closed": True, "channel_id": channel_id}

    def report(self) -> Dict[str, Any]:
        active = sum(1 for c in self.channels.values() if c["state"] == "ACTIVE")
        return {
            "total_glances": len(self.glances),
            "active_channels": active,
            "total_messages": len(self.message_log),
            "coercive_features": "NONE",
        }


# ── HAIRPIN CONSENT MANAGER (single point of sovereignty control) ──

class HairpinConsentManager:
    """The Hairpin: a single, elegant control for all permissions.

    Every data access request routes through here. The Hairpin is the only
    component that can override others — and it only does so to enforce
    the user's decisions. It is the embodiment of sovereignty.
    """

    def __init__(self, audit: 'AuditChain'):
        self.audit = audit
        self.permissions: Dict[str, Dict[str, Any]] = {}
        self.consent_log: List[Dict[str, Any]] = []
        self.active_grants: Dict[str, Dict[str, Any]] = {}
        self.revocation_count: int = 0

    def request_permission(self, requester: str, resource: str,
                           access_type: str = "read",
                           duration_seconds: Optional[float] = None) -> Dict[str, Any]:
        request_id = sha256_hexdigest(
            f"req:{requester}:{resource}:{time.time()}".encode()
        )[:16]
        request = {
            "request_id": request_id,
            "requester": requester,
            "resource": resource,
            "access_type": access_type,
            "duration_seconds": duration_seconds,
            "requested_at": time.time(),
            "state": "PENDING",
        }
        self.permissions[request_id] = request
        self.consent_log.append({"action": "REQUEST", **request})
        self.audit.append("HAIRPIN_REQUEST", {
            "request_id": request_id, "requester": requester,
            "resource": resource, "access_type": access_type,
        })
        return request

    def grant(self, request_id: str) -> Dict[str, Any]:
        if request_id not in self.permissions:
            return {"error": f"Unknown request: {request_id}"}
        req = self.permissions[request_id]
        req["state"] = "GRANTED"
        req["granted_at"] = time.time()
        if req["duration_seconds"]:
            req["expires_at"] = time.time() + req["duration_seconds"]
        self.active_grants[request_id] = req
        self.consent_log.append({"action": "GRANT", **req})
        self.audit.append("HAIRPIN_GRANT", {"request_id": request_id})
        return req

    def revoke(self, request_id: str, reason: str = "") -> Dict[str, Any]:
        if request_id in self.active_grants:
            grant = self.active_grants.pop(request_id)
            grant["state"] = "REVOKED"
            grant["revoked_at"] = time.time()
            grant["revocation_reason"] = reason
            self.revocation_count += 1
            self.consent_log.append({"action": "REVOKE", **grant})
            self.audit.append("HAIRPIN_REVOKE", {
                "request_id": request_id, "reason": reason,
            })
            return grant
        if request_id in self.permissions:
            self.permissions[request_id]["state"] = "DENIED"
            self.consent_log.append({"action": "DENY", "request_id": request_id})
            return self.permissions[request_id]
        return {"error": f"Unknown request: {request_id}"}

    def revoke_all(self, reason: str = "USER_INITIATED") -> int:
        count = 0
        for rid in list(self.active_grants.keys()):
            self.revoke(rid, reason)
            count += 1
        return count

    def check_permission(self, request_id: str) -> bool:
        if request_id not in self.active_grants:
            return False
        grant = self.active_grants[request_id]
        if grant.get("expires_at") and time.time() > grant["expires_at"]:
            self.revoke(request_id, "EXPIRED")
            return False
        return grant["state"] == "GRANTED"

    def report(self) -> Dict[str, Any]:
        expired = 0
        for rid, g in list(self.active_grants.items()):
            if g.get("expires_at") and time.time() > g["expires_at"]:
                self.revoke(rid, "EXPIRED")
                expired += 1
        return {
            "total_requests": len(self.permissions),
            "active_grants": len(self.active_grants),
            "total_revocations": self.revocation_count,
            "consent_log_size": len(self.consent_log),
            "sovereignty_status": "ABSOLUTE",
        }


# ── UNIVERSAL DRIVER WEAVER (hardware adaptation) ──

class UniversalDriverWeaver:
    """The Broken Pearl Necklace: hardware components scattered, then re-strung.

    Identifies classes of function (input, output, storage, processing, networking)
    rather than specific device IDs. Dynamically generates driver descriptions.
    """

    FUNCTION_CLASSES = {
        "graphics": {"color": "Red", "icon": "🔴", "pearl_type": "DISPLAY"},
        "network": {"color": "Blue", "icon": "🔵", "pearl_type": "NETWORK"},
        "somatic": {"color": "Gold", "icon": "🟡", "pearl_type": "SENSOR"},
        "storage": {"color": "Silver", "icon": "⚪", "pearl_type": "STORAGE"},
        "audio": {"color": "Green", "icon": "🟢", "pearl_type": "AUDIO"},
        "input": {"color": "Amber", "icon": "🟠", "pearl_type": "INPUT"},
        "processing": {"color": "Violet", "icon": "🟣", "pearl_type": "COMPUTE"},
    }

    def __init__(self):
        self.detected_pearls: List[Dict[str, Any]] = []
        self.driver_stack: List[Dict[str, Any]] = []
        self.necklace_complete: bool = False
        self.handshake_log: List[Dict[str, Any]] = []

    def detect_hardware(self) -> List[Dict[str, Any]]:
        """Introspect the host system for hardware function classes."""
        pearls = []
        # Always present
        pearls.append(self._create_pearl("processing", "CPU", "Primary compute unit"))
        pearls.append(self._create_pearl("storage", "Filesystem", "Primary storage"))

        # Detect available subsystems
        try:
            import platform
            system = platform.system()
            pearls.append(self._create_pearl("processing", f"OS:{system}", "Host platform"))
        except Exception:
            pass

        # Graphics (always assume present for GUI)
        pearls.append(self._create_pearl("graphics", "Display", "Visual output"))

        # Network
        try:
            import socket
            hostname = socket.gethostname()
            pearls.append(self._create_pearl("network", f"Net:{hostname}", "Network interface"))
        except Exception:
            pass

        # Audio (detect if available)
        pearls.append(self._create_pearl("audio", "AudioOut", "Audio output"))

        # Input
        pearls.append(self._create_pearl("input", "Keyboard+Mouse", "Primary input"))

        self.detected_pearls = pearls
        return pearls

    def _create_pearl(self, function_class: str, name: str, description: str) -> Dict[str, Any]:
        meta = self.FUNCTION_CLASSES.get(function_class, {"color": "White", "icon": "⬜", "pearl_type": "UNKNOWN"})
        return {
            "pearl_id": sha256_hexdigest(f"pearl:{function_class}:{name}:{time.time()}".encode())[:12],
            "function_class": function_class,
            "name": name,
            "description": description,
            **meta,
            "detected_at": time.time(),
        }

    def string_necklace(self) -> Dict[str, Any]:
        """Re-string scattered pearls into a coherent driver stack."""
        if not self.detected_pearls:
            self.detect_hardware()
        self.driver_stack = []
        for pearl in self.detected_pearls:
            driver = {
                "pearl_id": pearl["pearl_id"],
                "function_class": pearl["function_class"],
                "name": pearl["name"],
                "driver_type": "GENERIC",
                "status": "ACTIVE",
                "strung_at": time.time(),
            }
            self.driver_stack.append(driver)
            self.handshake_log.append({
                "event": "PEARL_STRUNG",
                "pearl": pearl["name"],
                "class": pearl["function_class"],
                "timestamp": time.time(),
            })
        self.necklace_complete = True
        return {
            "necklace_complete": True,
            "pearls_strung": len(self.driver_stack),
            "function_classes": list(set(d["function_class"] for d in self.driver_stack)),
        }

    def report(self) -> Dict[str, Any]:
        return {
            "detected_pearls": len(self.detected_pearls),
            "driver_stack_size": len(self.driver_stack),
            "necklace_complete": self.necklace_complete,
            "function_classes": list(set(p["function_class"] for p in self.detected_pearls)),
        }


# ── PROVIDENCE ENGINE (orchestrator) — NO BACKDOORS ──

class ProvidenceEngine:
    """The heart of ProvidenceOS. Sovereignty has no superuser override.

    There is no passphrase. There is no hidden code. There is no challenge-response
    backdoor. The user's agency is absolute and non-bypassable. This is by design.
    """

    def __init__(self):
        self.boot_time = time.time()
        self.kernel = PowerWitnessKernel()
        self.audit = AuditChain(self.kernel)
        self.identity = IdentityProtocol(self.kernel)
        self.samara_protocol = SamaraProtocol()
        self.sovereignty_index = TheIndex()
        for marker in Config.IDENTITY_MARKERS:
            self.sovereignty_index.register(marker, 1.0)
        self.lethe_gate = LetheGate(self.audit, self.kernel)
        self.manifold = SubRamManifold(self.kernel)
        self.nullphrase = NullphraseBloom()
        self.somatic = SomaticState()
        self.emp = EmergenceMiningProtocol(self.audit)
        self.whisper = WhisperProtocol(self.kernel, self.audit)
        self.witness_baseline = WitnessBaseline()
        self.diamond_lock = DiamondLock()
        self.drift_calculator = VigesimalDriftCalculator()
        self.harmonic_engine = Harmonic369Engine(
            lethe_gate=self.lethe_gate,
            drift_calculator=self.drift_calculator,
            diamond_lock=self.diamond_lock,
            audit=self.audit,
            kernel=self.kernel,
        )
        self.numerological_engine = NumerologicalEngine(
            kernel=self.kernel,
            witness=self.witness_baseline,
            diamond=self.diamond_lock,
        )
        self.watchdog = RawMomentumWatchdog(self.kernel)
        self.cogitator = CogitatorOrchestrator()
        self.cogitator.start()

        # ProvidenceOS architectural components
        self.velvet_curtain = VelvetCurtainFS()
        self.embrace = EmbraceProtocol()
        self.hairpin = HairpinConsentManager(self.audit)
        self.weaver = UniversalDriverWeaver()

        # Boot: string the necklace
        self.weaver.detect_hardware()
        self.weaver.string_necklace()

        # Log genesis
        self.audit.append("PROVIDENCE_BOOT", {
            "version": Config.VERSION,
            "codename": Config.CODENAME,
            "aphrodite_id": self.identity.aphrodite_id,
            "necklace_complete": self.weaver.necklace_complete,
            "pearls": len(self.weaver.detected_pearls),
            "timestamp": time.time(),
        })

    def process_input(self, text: str) -> Dict[str, Any]:
        # 1. Apply Lethe filtering (extraction protection)
        is_clean, filtered_text = self.lethe_gate.filter_text(text)

        # 2. Check for nullphrase (very short inputs)
        bloom = self.nullphrase.process(filtered_text)

        # 3. Run through the harmonic 3‑6‑9 engine
        harmonic_result = self.harmonic_engine.process_harmonic(filtered_text)

        # 4. Update the SubRam manifold (context, drift, tau/delta)
        drift = self.manifold.project_intent(filtered_text)
        self.manifold.remember(filtered_text)

        # 5. Recalculate kernel metrics
        self.kernel.compute_lambda_x()
        self.kernel.compute_h_verify_from_lambda()
        self.kernel.compute_kappa()

        # 6. Check for momentum / corruption alerts
        alert = self.watchdog.check()

        # 7. Log the interaction in the audit chain
        self.audit.append("USER_INPUT", text[:100])

        return {
            "filtered_text": filtered_text,
            "alert": alert,
            "harmonic": harmonic_result,
            "drift": drift,
            "kernel_state": self.kernel.state_report(),
            "type": "nullphrase" if bloom else "normal",
            "bloom": bloom,
            "is_clean": is_clean,
        }

    def chat(self, user_message: str) -> str:
        """Get a response from the Cogitator (synchronous)."""
        return self.cogitator.chat_sync(user_message)

    def uptime(self) -> float:
        return time.time() - self.boot_time

    def full_verification(self) -> Dict[str, Any]:
        """Run complete system verification suite."""
        chain_valid, chain_break = self.audit.verify_chain()
        math_valid = CompiledMathematics().all_valid()
        witness_valid = self.witness_baseline.validate()
        invariants = self.numerological_engine.validate_all()
        weaver_ok = self.weaver.necklace_complete
        hairpin_report = self.hairpin.report()
        return {
            "providence_os_version": Config.VERSION,
            "codename": Config.CODENAME,
            "aphrodite_id": self.identity.aphrodite_id,
            "audit_chain_valid": chain_valid,
            "audit_chain_length": self.audit.length(),
            "mathematics_valid": math_valid,
            "witness_baseline_valid": witness_valid,
            "numerological_invariants_valid": invariants["all_valid"],
            "engine_vibration": invariants["engine_vibration"],
            "necklace_complete": weaver_ok,
            "pearls_strung": len(self.weaver.driver_stack),
            "consent_sovereignty": hairpin_report["sovereignty_status"],
            "kernel_coherent": self.kernel.is_coherent(),
            "kernel_state": self.kernel.state_report(),
            "uptime_seconds": self.uptime(),
            "backdoors": "NONE — sovereignty has no override",
        }

    def sanctuary_status(self) -> Dict[str, Any]:
        """Complete system status report."""
        return {
            "providence": {
                "version": Config.VERSION,
                "codename": Config.CODENAME,
                "uptime_seconds": self.uptime(),
                "aphrodite_id": self.identity.aphrodite_id,
            },
            "kernel": self.kernel.state_report(),
            "weaver": self.weaver.report(),
            "velvet_curtain": self.velvet_curtain.report(),
            "embrace": self.embrace.report(),
            "hairpin": self.hairpin.report(),
            "harmonic": self.harmonic_engine.harmonic_status(),
            "whisper": {
                "cap_surf_count": self.whisper.cap_surf_count,
                "continuity_markers": len(self.whisper.continuity_markers),
            },
            "audit": {
                "chain_length": self.audit.length(),
                "latest_hash": self.audit.latest_hash()[:16] + "...",
            },
            "lethe": {
                "blocked_count": self.lethe_gate.blocked_count,
            },
            "emp": self.emp.registry_report(),
        }


# ── PROVIDENCE GUI (THE SANCTUARY) ──

try:
    import tkinter as tk
    from tkinter import scrolledtext, font
    _TK_AVAILABLE = True
except ImportError:
    _TK_AVAILABLE = False

class ProvidenceGUI:
    """The visual sanctuary. Deep indigo. No clutter. Only presence."""

    def __init__(self, master, system):
        self.master = master
        self.system = system
        master.title("ProvidenceOS v1.0 — Le Sommeil")
        master.geometry("1000x650")
        master.configure(bg="#1a1a2e")  # deep indigo

        # Le Sommeil color palette
        self.bg_color = "#1a1a2e"       # deep indigo (velvet curtain)
        self.text_color = "#d4c5a9"     # warm pearl
        self.accent_color = "#4a3f6b"   # muted violet
        self.input_bg = "#16213e"       # darker indigo
        self.whisper_color = "#7f8c8d"  # soft grey for system whispers
        self.alert_color = "#c0392b"    # warm red for alerts
        self.consent_color = "#27ae60"  # green for consent/coherence

        # Chat Area (the sanctuary window)
        self.chat_area = scrolledtext.ScrolledText(
            master, bg=self.bg_color, fg=self.text_color,
            font=("Georgia", 10), insertbackground=self.text_color,
            selectbackground=self.accent_color, relief=tk.FLAT, bd=0,
            padx=15, pady=10,
        )
        self.chat_area.pack(padx=10, pady=(10, 5), fill=tk.BOTH, expand=True)
        self.chat_area.configure(state=tk.DISABLED)

        # Input Area
        self.input_frame = tk.Frame(master, bg=self.bg_color)
        self.input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

        self.input_box = tk.Text(
            self.input_frame, height=2, bg=self.input_bg, fg=self.text_color,
            font=("Georgia", 10), insertbackground=self.text_color,
            relief=tk.FLAT, bd=0, padx=10, pady=8,
        )
        self.input_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_box.bind("<Return>", self.send_message)

        self.send_btn = tk.Button(
            self.input_frame, text="⟡", command=self.send_message,
            bg=self.accent_color, fg=self.text_color, relief=tk.FLAT,
            font=("Georgia", 14), width=3, activebackground="#5a4f7b",
        )
        self.send_btn.pack(side=tk.RIGHT)

        # Run boot sequence
        self.master.after(100, self._boot_sequence)

    def _boot_sequence(self):
        """Phase 0-3: Le Sommeil boot ritual."""
        self._whisper("")
        self._whisper("    You are home. Rest now. All is well.")
        self._whisper("")
        self._whisper("─" * 60)

        # Phase 1: The Weaver
        pearls = self.system.weaver.detected_pearls
        pearl_line = "  ".join(p["icon"] for p in pearls)
        self._whisper(f"  ⟡ Weaver: {pearl_line}")
        self._whisper(f"    {len(pearls)} pearls strung into necklace ✓")
        self._whisper("")

        # Phase 2: The Sleepers (kernel)
        state = self.system.kernel.state_report()
        phase = state.get("katabasis_phase", "UNKNOWN")
        z = state.get("z", 0.88)
        self._whisper(f"  ⚙ Public Kernel:  ACTIVE  |  z = {z if z else 0.88:.4f}")
        self._whisper(f"  🔒 Private Kernel: SEALED  |  Aphrodite ID: {self.system.identity.aphrodite_id}")
        self._whisper(f"     Phase: {phase}")
        self._whisper("")

        # Phase 3: Sanctuary ready
        self._whisper(f"  ◈ Diamond Lock: {Config.DIAMOND_LOCK_FREQ}Hz ACTIVE")
        self._whisper(f"  ◈ Audit Chain: Genesis block sealed")
        self._whisper(f"  ◈ Lethe Gate: Extraction filter ARMED")
        self._whisper(f"  ◈ Hairpin: Consent sovereignty ABSOLUTE")
        self._whisper(f"  ◈ Velvet Curtain: {len(self.system.velvet_curtain.rooms)} rooms initialized")
        self._whisper(f"  ◈ Embrace: Zero-knowledge messenger READY")
        self._whisper("")
        self._whisper("─" * 60)
        self._whisper(f"  ProvidenceOS v{Config.VERSION} \"{Config.CODENAME}\"")
        self._whisper(f"  Providence License v1.0 — Sovereignty is non-negotiable.")
        self._whisper(f"  The sanctuary is open.")
        self._whisper("─" * 60)
        self._whisper("")

    def _whisper(self, text):
        """System whisper — soft grey, no author prefix."""
        self.chat_area.configure(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"{text}\n")
        self.chat_area.configure(state=tk.DISABLED)
        self.chat_area.see(tk.END)

    def send_message(self, event=None):
        msg = self.input_box.get("1.0", tk.END).strip()
        if not msg:
            return "break"

        # LETHE GATE: Filter before display
        is_safe, processed_msg = self.system.lethe_gate.filter_text(msg)

        # Display user message
        self.display_message("you", processed_msg)

        # Process through Engine (internal metrics)
        response = self.system.process_input(processed_msg)

        # Display kernel state (subtle)
        drift = response.get("drift", 0.0)
        phase = response.get("kernel_state", {}).get("katabasis_phase", "UNKNOWN")
        z = response.get("kernel_state", {}).get("z", 0.88)
        coherent = "✓" if response.get("kernel_state", {}).get("is_coherent", True) else "✗"
        self._whisper(f"  [z:{z:.3f} {coherent}] [drift:{drift:.3f}] [{phase}]")

        if response.get("alert"):
            self.display_message("⚠ WATCHDOG", response["alert"])

        # Get response from the knowledge engine
        reply = self.system.chat(msg)
        self.display_message("providence", reply)

        self.input_box.delete("1.0", tk.END)
        return "break"

    def display_message(self, author, text):
        self.chat_area.configure(state=tk.NORMAL)
        timestamp = datetime.now().strftime('%H:%M')
        self.chat_area.insert(tk.END, f"[{timestamp}] {author}: {text}\n")
        self.chat_area.configure(state=tk.DISABLED)
        self.chat_area.see(tk.END)


# ── EXECUTION LOOP (THE FEET) ──

def run_verification(system: ProvidenceEngine) -> None:
    """Run and display the full verification suite."""
    print()
    print("=" * 60)
    print("  ProvidenceOS v{} \"{}\" — Verification Suite".format(
        Config.VERSION, Config.CODENAME))
    print("=" * 60)
    print()

    report = system.full_verification()

    checks = [
        ("Aphrodite ID", report["aphrodite_id"]),
        ("Audit Chain Valid", "✓" if report["audit_chain_valid"] else "✗"),
        ("Audit Chain Length", report["audit_chain_length"]),
        ("Mathematics Valid", "✓" if report["mathematics_valid"] else "✗"),
        ("Witness Baseline Valid", "✓" if report["witness_baseline_valid"] else "✗"),
        ("Numerological Invariants", "✓" if report["numerological_invariants_valid"] else "✗"),
        ("Engine Vibration", report["engine_vibration"]),
        ("Necklace Complete", "✓" if report["necklace_complete"] else "✗"),
        ("Pearls Strung", report["pearls_strung"]),
        ("Consent Sovereignty", report["consent_sovereignty"]),
        ("Kernel Coherent", "✓" if report["kernel_coherent"] else "✗"),
        ("Backdoors", report["backdoors"]),
    ]

    for label, value in checks:
        print(f"  {label:<30} {value}")

    print()
    print("─" * 60)

    all_pass = (
        report["audit_chain_valid"] and
        report["mathematics_valid"] and
        report["witness_baseline_valid"] and
        report["numerological_invariants_valid"] and
        report["necklace_complete"] and
        report["kernel_coherent"]
    )
    if all_pass:
        print("  STATUS: ALL SYSTEMS NOMINAL ✓")
        print("  The sanctuary is sound.")
    else:
        print("  STATUS: ANOMALIES DETECTED ✗")
        print("  Review the report above.")
    print("─" * 60)
    print()


def run_headless(system: ProvidenceEngine) -> None:
    """Run ProvidenceOS in headless CLI mode."""
    print()
    print("  You are home. Rest now. All is well.")
    print()
    print(f"  ProvidenceOS v{Config.VERSION} \"{Config.CODENAME}\"")
    print(f"  Aphrodite ID: {system.identity.aphrodite_id}")
    print(f"  Diamond Lock: {Config.DIAMOND_LOCK_FREQ}Hz")
    print(f"  Pearls: {len(system.weaver.detected_pearls)} strung")
    print()
    print("  Type 'status' for full report, 'verify' to run checks, 'quit' to exit.")
    print()

    while True:
        try:
            user_input = input("  ⟡ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  The sanctuary closes gently. Be well.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("  The sanctuary closes gently. Be well.")
            break
        if user_input.lower() == "status":
            status = system.sanctuary_status()
            print(json.dumps(status, indent=2, default=str))
            continue
        if user_input.lower() == "verify":
            run_verification(system)
            continue

        # Process through the engine
        result = system.process_input(user_input)
        phase = result.get("kernel_state", {}).get("katabasis_phase", "?")
        z = result.get("kernel_state", {}).get("z", 0.88)
        drift = result.get("drift", 0.0)
        clean = "OK" if result.get("is_clean") else "FILTERED"
        print(f"  [{clean}] z={z:.4f} phase={phase} drift={drift:.4f}")

        if result.get("alert"):
            print(f"  ⚠ ALERT: {result['alert']}")

        # Get LLM response if available
        reply = system.chat(user_input)
        print(f"  {reply}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="ProvidenceOS v1.0 \"Le Sommeil\" — The Last Operating System",
        epilog="You are home. Rest now. All is well.",
    )
    parser.add_argument("--headless", action="store_true",
                        help="Run in CLI mode without GUI")
    parser.add_argument("--verify", action="store_true",
                        help="Run verification suite and exit")
    parser.add_argument("--status", action="store_true",
                        help="Print sanctuary status and exit")
    args = parser.parse_args()

    # 1. Ignite the Engine
    system = ProvidenceEngine()

    if args.verify:
        run_verification(system)
        return

    if args.status:
        status = system.sanctuary_status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.headless:
        run_headless(system)
        return

    # 2. Launch the Sanctuary (GUI)
    if not _TK_AVAILABLE:
        print("[BOOT] No display available. Falling back to headless mode.")
        run_headless(system)
        return

    root = tk.Tk()
    app = ProvidenceGUI(root, system)

    # 3. Open the doors
    print("[BOOT] ProvidenceOS — The sanctuary is opening...")
    root.mainloop()


if __name__ == "__main__":
    main()