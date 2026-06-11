# SPDX-FileCopyrightText: 2026 Roni Tervo
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import moderngl
import moderngl_window as mglw
import numpy as np
from moderngl_window import geometry

try:
    import pyaudio
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

CREATE_NO_WINDOW = 0x08000000
CPU_SENSOR_POWERSHELL = """
$ErrorActionPreference = 'Stop'
$namespaces = @('root\\LibreHardwareMonitor', 'root\\OpenHardwareMonitor')
$preferredPattern = 'CPU Package|Tctl/Tdie|Tdie|Core Max|CPU CCD|CPU Die|Package'
foreach ($ns in $namespaces) {
    try {
        $sensors = Get-CimInstance -Namespace $ns -ClassName Sensor |
            Where-Object { $_.SensorType -eq 'Temperature' } |
            ForEach-Object { [PSCustomObject]@{ Name = [string]$_.Name; Value = [double]$_.Value } }
        if ($sensors) {
            $preferred = $sensors | Where-Object { $_.Name -match $preferredPattern } | Sort-Object Value -Descending | Select-Object -First 1
            if (-not $preferred) { $preferred = $sensors | Sort-Object Value -Descending | Select-Object -First 1 }
            if ($preferred) {
                $value = $preferred.Value.ToString('0.0', [System.Globalization.CultureInfo]::InvariantCulture)
                Write-Output ('{0}|{1}' -f $preferred.Name, $value)
                exit 0
            }
        }
    } catch {}
}
exit 1
""".strip()

FONT_3X5 = {
    " ": ("000", "000", "000", "000", "000"), "-": ("000", "000", "111", "000", "000"),
    ".": ("000", "000", "000", "000", "010"), ":": ("000", "010", "000", "010", "000"),
    "?": ("111", "001", "011", "000", "010"), "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"), "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"), "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"), "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"), "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"), "A": ("111", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"), "C": ("111", "100", "100", "100", "111"),
    "D": ("110", "101", "101", "101", "110"), "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"), "G": ("111", "100", "101", "101", "111"),
    "H": ("101", "101", "111", "101", "101"), "I": ("111", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "101", "111"), "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"), "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"), "O": ("111", "101", "101", "101", "111"),
    "P": ("111", "101", "111", "100", "100"), "Q": ("111", "101", "101", "111", "011"),
    "R": ("110", "101", "110", "101", "101"), "S": ("111", "100", "111", "001", "111"),
    "T": ("111", "010", "010", "010", "010"), "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"), "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"), "Y": ("101", "101", "010", "010", "010"),
    "Z": ("111", "001", "010", "100", "111"),
}

CHUNKS_PER_SEC = 60
SAMPLE_RATE = 44100
CHUNK_SIZE = SAMPLE_RATE // CHUNKS_PER_SEC
FFT_BINS = 128

class GenerativeAudioEngine:
    def __init__(self):
        self.active = True
        self.time_val = 0.0
        self.fft_data = np.zeros(FFT_BINS, dtype=np.float32)
        self.wave_data = np.zeros(FFT_BINS, dtype=np.float32)
        self.energy = 0.0
        self.bass = 0.0
        self.treble = 0.0

        self.chords = [
            [32.70, 65.41, 98.00, 155.56, 196.00], 
            [27.50, 55.00, 82.41, 138.59, 164.81], 
            [21.83, 43.65, 65.41, 103.83, 130.81], 
            [36.71, 73.42, 110.00, 174.61, 220.00] 
        ]
        self.current_chord_idx = 0
        self.phase = np.zeros(5, dtype=np.float32)
        self.lock = threading.Lock()

        self._pa = None
        self._stream = None
        if HAS_AUDIO:
            self._start_audio_stream()
        else:
            self._sim_thread = threading.Thread(target=self._simulate_audio, daemon=True)
            self._sim_thread.start()

    def _generate_chunk(self) -> np.ndarray:
        dt = CHUNK_SIZE / SAMPLE_RATE
        t_seq = np.linspace(self.time_val, self.time_val + dt, CHUNK_SIZE, endpoint=False)
        self.time_val += dt

        macro_time = self.time_val * 0.15
        next_chord_idx = (self.current_chord_idx + 1) % len(self.chords)
        blend = smoothstep(0.8, 1.0, macro_time % 1.0)

        if (macro_time % 1.0) < 0.05 and macro_time > self.current_chord_idx + 1:
            self.current_chord_idx = next_chord_idx

        chord_a = np.array(self.chords[self.current_chord_idx])
        chord_b = np.array(self.chords[next_chord_idx])
        freqs = chord_a * (1 - blend) + chord_b * blend

        out = np.zeros(CHUNK_SIZE, dtype=np.float32)
        for i, f in enumerate(freqs):
            fm = np.sin(2.0 * np.pi * (f * 0.5) * t_seq) * (1.2 + 0.8 * np.sin(self.time_val * 0.2 + i))
            phase_inc = 2.0 * np.pi * f * t_seq + fm
            amp = 0.12 + 0.08 * np.sin(self.time_val * 0.4 + i * 1.6)
            out += np.sin(phase_inc + self.phase[i]) * amp
            self.phase[i] = (self.phase[i] + 2.0 * np.pi * f * dt) % (2.0 * np.pi)

        # Pulse kick
        kick_env = max(0, np.sin(self.time_val * np.pi * 2.0 * 1.5)) ** 24.0
        kick = np.sin(2.0 * np.pi * 45.0 * t_seq - kick_env * 15.0) * kick_env * 0.6
        out += kick

        # Noise sweeps
        noise_env = max(0, np.sin(self.time_val * np.pi * 0.25)) ** 4.0
        out += np.random.normal(0, 0.01 + 0.03 * noise_env, CHUNK_SIZE)

        out = np.clip(out, -1.0, 1.0).astype(np.float32)

        with self.lock:
            window = np.hanning(CHUNK_SIZE)
            fft_complex = np.fft.rfft(out * window)
            fft_mag = np.abs(fft_complex[:FFT_BINS]) / (CHUNK_SIZE / 2)
            self.fft_data = self.fft_data * 0.6 + fft_mag * 0.4
            
            wave_downsampled = out[::CHUNK_SIZE//FFT_BINS][:FFT_BINS]
            self.wave_data = self.wave_data * 0.5 + wave_downsampled * 0.5

            self.energy = float(np.mean(self.fft_data))
            self.bass = float(np.mean(self.fft_data[:8]))
            self.treble = float(np.mean(self.fft_data[64:]))

        return out

    def _audio_callback(self, in_data, frame_count, time_info, status):
        out = self._generate_chunk()
        return (out.tobytes(), pyaudio.paContinue)

    def _start_audio_stream(self):
        try:
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE,
                output=True, frames_per_buffer=CHUNK_SIZE, stream_callback=self._audio_callback
            )
            self._stream.start_stream()
        except Exception:
            global HAS_AUDIO
            HAS_AUDIO = False
            self._sim_thread = threading.Thread(target=self._simulate_audio, daemon=True)
            self._sim_thread.start()

    def _simulate_audio(self):
        while self.active:
            self._generate_chunk()
            time.sleep(1.0 / CHUNKS_PER_SEC)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        with self.lock:
            return self.fft_data.copy(), self.wave_data.copy(), self.energy, self.bass, self.treble

    def destroy(self):
        self.active = False
        if HAS_AUDIO and self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._pa.terminate()

def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


VERT_SHADER = """
#version 450
in vec2 in_position;
out vec2 uv;
void main() {
    uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

SIM_FRAG_SHADER = """
#version 450
in vec2 uv;
out vec4 fragColor;
uniform sampler2D stateTex;
uniform sampler1D audioFft;

uniform vec2 resolution;
uniform float time;
uniform float feed;
uniform float kill;
uniform float diffU;
uniform float diffV;
uniform float dt;
uniform float laplaceScale;
uniform float noiseStrength;
uniform float parameterDrift;
uniform float audioEnergy;
uniform float audioBass;
uniform float audioTreble;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

void main() {
    vec2 texel = 1.0 / resolution;
    vec4 c = texture(stateTex, uv);
    vec4 r = texture(stateTex, uv + vec2(texel.x, 0.0));
    vec4 l = texture(stateTex, uv - vec2(texel.x, 0.0));
    vec4 t = texture(stateTex, uv + vec2(0.0, texel.y));
    vec4 b = texture(stateTex, uv - vec2(0.0, texel.y));
    vec4 tr = texture(stateTex, uv + texel);
    vec4 tl = texture(stateTex, uv + vec2(-texel.x, texel.y));
    vec4 br = texture(stateTex, uv + vec2(texel.x, -texel.y));
    vec4 bl = texture(stateTex, uv - texel);

    float lScale = laplaceScale * (1.0 + audioBass * 0.8);
    float lapU = ((r.r + l.r + t.r + b.r) * 0.2 + (tr.r + tl.r + br.r + bl.r) * 0.05 - c.r) * lScale;
    float lapV = ((r.g + l.g + t.g + b.g) * 0.2 + (tr.g + tl.g + br.g + bl.g) * 0.05 - c.g) * lScale;

    vec2 gradH = vec2(r.b - l.b, t.b - b.b);
    float flowStrength = 0.8 + audioEnergy * 2.0;
    float advectU = dot(gradH, vec2(r.r - l.r, t.r - b.r)) * flowStrength;
    float advectV = dot(gradH, vec2(r.g - l.g, t.g - b.g)) * flowStrength;

    float wetNoise = (hash(uv * resolution + floor(time * 0.35)) - 0.5) * noiseStrength * (1.0 + audioTreble * 15.0);
    float localFeed = feed + c.a * 0.015 * sin(time * 0.1 + uv.x * 20.0) + wetNoise * 0.18 + audioBass * 0.03;
    float localKill = kill + (1.0 - c.a) * 0.01 * cos(time * 0.15 + uv.y * 15.0) + parameterDrift * 0.6 - audioEnergy * 0.01;

    float reaction = c.r * c.g * c.g * (1.0 + audioTreble * 2.0);
    
    float du = (diffU * lapU) - reaction + localFeed * (1.0 - c.r) - advectU;
    float dv = (diffV * lapV) + reaction - (localFeed + localKill) * c.g - advectV;

    float fftSample = texture(audioFft, uv.x).r;
    float dh = (c.g * 0.01 - 0.002 + wetNoise * 0.015 + fftSample * 0.008) * dt * c.a;

    fragColor = vec4(
        clamp(c.r + du * dt, 0.0, 1.0),
        clamp(c.g + dv * dt, 0.0, 1.0),
        clamp(c.b + dh * dt, 0.0, 1.0),
        c.a
    );
}
"""

DISPLAY_FRAG_SHADER = """
#version 450
in vec2 uv;
out vec4 fragColor;
uniform sampler2D stateTex;
uniform sampler1D audioFft;
uniform sampler1D audioWave;

uniform vec2 resolution;
uniform float time;
uniform float exposure;
uniform float glow;
uniform float gamma;
uniform float contourContrast;
uniform float colorShift;
uniform float cameraSpeed;
uniform float fxIntensity;
uniform int raySteps;
uniform float audioEnergy;
uniform float audioBass;

float hash(vec2 p) { return fract(sin(dot(p, vec2(41.7, 289.1))) * 45758.5453); }

float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}

mat3 setCamera(in vec3 ro, in vec3 ta, float cr) {
    vec3 cw = normalize(ta - ro);
    vec3 cp = vec3(sin(cr), cos(cr), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = normalize(cross(cu, cw));
    return mat3(cu, cv, cw);
}

float map(in vec3 p, out float matID, out vec4 stateOut) {
    vec2 mapUV = p.xz * 0.04;
    vec4 state = textureLod(stateTex, mapUV, 0.0);

    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float waveDisturb = texture(audioWave, fract(p.x * 0.1 + time * 0.2)).r * audioBass * 0.4;
    float baseHeight = state.b * mix(2.7, 3.9, safeFx) + waveDisturb;
    float biomassExtrusion = smoothstep(0.16, 0.86, state.g) * mix(0.85, 2.15, safeFx) * (1.0 + audioBass);
    float h = baseHeight + biomassExtrusion;

    float detail = sin(p.x * 8.0) * cos(p.z * 8.3) * 0.035 + sin(p.x * 15.0) * cos(p.z * 14.1) * 0.014;
    detail += (noise(p.xz * 2.4 + time * 0.02) - 0.5) * 0.045;
    h += detail * smoothstep(0.2, 0.8, state.b) * safeFx;

    float dTerrain = p.y - h;
    float dWater = p.y - 1.55 + sin(time * 2.0 + p.x * 4.0) * audioBass * 0.15;

    if (dTerrain < dWater) {
        matID = 1.0; stateOut = state; return dTerrain * 0.72;
    } else {
        matID = 0.0; stateOut = state; return dWater * 0.9;
    }
}

vec3 calcNormal(in vec3 p) {
    vec2 e = vec2(0.04, 0); float dummyMat; vec4 dummyState;
    return normalize(vec3(
        map(p + e.xyy, dummyMat, dummyState) - map(p - e.xyy, dummyMat, dummyState),
        map(p + e.yxy, dummyMat, dummyState) - map(p - e.yxy, dummyMat, dummyState),
        map(p + e.yyx, dummyMat, dummyState) - map(p - e.yyx, dummyMat, dummyState)
    ));
}

float calcShadow(in vec3 ro, in vec3 rd) {
    float res = 1.0; float t = 0.1; float dMat; vec4 dSt;
    for (int i = 0; i < 30; i++) {
        float h = map(ro + rd * t, dMat, dSt);
        res = min(res, 8.0 * h / t); t += clamp(h, 0.05, 0.75);
        if (h < 0.001 || t > 10.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    int safeRaySteps = clamp(raySteps, 32, 160);

    float camShake = noise(vec2(time * 10.0, 0.0)) * audioEnergy * 0.05;
    float camTime = time * 0.15 * max(cameraSpeed, 0.05);
    vec3 ro = vec3(camTime * 5.0, 7.0 + sin(camTime * 0.5) * 1.2 + camShake, camTime * 4.0);
    vec3 ta = vec3(ro.x + 4.8, 2.1, ro.z + 4.8 + sin(camTime));

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.3) * 0.1 + camShake);
    vec3 rd = ca * normalize(vec3(p.xy, 2.0 - audioBass * 0.2));

    vec3 lightDir = normalize(vec3(0.8, 0.62, -0.4));
    float sun = pow(max(0.0, dot(rd, lightDir)), 220.0);
    float skyRise = smoothstep(-0.2, 0.9, rd.y);
    vec3 skyColor = mix(vec3(0.008, 0.014, 0.032), vec3(0.09, 0.18, 0.31), skyRise);
    skyColor += vec3(1.00, 0.72, 0.36) * sun * (0.8 + safeFx * 0.8);
    
    float auroraFft = texture(audioFft, abs(p.x)).r;
    float aurora = smoothstep(0.74, 0.98, rd.y + 0.12 * sin(p.x * 3.0 + time * 0.22));
    aurora *= 0.5 + 0.5 * sin(p.x * 9.0 + time * 0.65 + colorShift) + auroraFft;
    skyColor += aurora * mix(vec3(0.08, 0.55, 0.46), vec3(0.8, 0.2, 0.5), audioBass) * safeFx;

    float tMax = 50.0; float t = 0.0; float matID = -1.0;
    vec4 state = vec4(0.0); vec3 volumeGlow = vec3(0.0);

    for (int i = 0; i < 160; i++) {
        if (i >= safeRaySteps) break;
        vec3 pos = ro + rd * t;
        float currMat; vec4 currState;
        float h = map(pos, currMat, currState);

        if (currMat == 1.0) {
            float bio = smoothstep(0.2, 0.8, currState.g);
            vec3 emColor = mix(vec3(0.0, 1.0, 0.8), vec3(1.0, 0.2, 0.5), currState.r + audioEnergy);
            volumeGlow += emColor * bio * (0.012 + glow * 0.003) * safeFx / (1.0 + abs(h) * 10.0) * (1.0 + audioBass * 3.0);
        }
        if (h < max(0.003, 0.0015 * t) || t > tMax) {
            matID = currMat; state = currState; break;
        }
        t += clamp(h, 0.035, 0.85);
    }

    vec3 color = skyColor;
    if (t < tMax) {
        vec3 pos = ro + rd * t;
        vec3 nor = calcNormal(pos);

        float occ = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
        float sha = calcShadow(pos + nor * 0.01, lightDir);
        float dif = clamp(dot(nor, lightDir), 0.0, 1.0) * sha;
        float fre = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 2.0);

        vec3 matColor;
        if (matID == 1.0) {
            vec3 rock = vec3(0.1, 0.12, 0.15); vec3 sand = vec3(0.3, 0.25, 0.18);
            vec3 bio = mix(vec3(0.05, 0.2, 0.15), vec3(0.8, 1.0, 0.9), state.g);
            matColor = mix(rock, sand, smoothstep(0.4, 0.6, nor.y));
            matColor = mix(matColor, bio, smoothstep(0.1, 0.5, state.g));

            vec3 bioGlow = mix(vec3(0.0, 0.9, 0.7), vec3(1.0, 0.13, 0.42), state.r);
            float pulse = 1.0 + 0.5 * sin(time * 3.0 - pos.x + colorShift * 3.0) + audioBass * 2.0;
            matColor += bioGlow * pow(state.g, 3.0) * (1.6 + glow * 0.65) * pulse * safeFx;
            matColor += vec3(1.0, 0.72, 0.22) * pow(state.a, 2.0) * (1.0 + glow * 0.45) * safeFx;
            float contour = 1.0 - smoothstep(0.018, 0.045 + contourContrast * 0.03, abs(fract(state.b * 18.0) - 0.5));
            matColor += contour * vec3(0.08, 0.12, 0.11) * contourContrast;
        } else {
            float depth = clamp((1.55 - state.b * 3.7) * 0.5, 0.0, 1.0);
            matColor = mix(vec3(0.0, 0.4, 0.5), vec3(0.0, 0.05, 0.15), depth);
            matColor += vec3(0.05, 0.24, 0.22) * (0.5 + 0.5 * sin(pos.x * 3.0 + pos.z * 2.0 + time * 1.3)) * safeFx;
            vec3 ref = reflect(rd, nor);
            float spe = pow(clamp(dot(ref, lightDir), 0.0, 1.0), 32.0) * sha;
            matColor += vec3(1.0) * spe * 0.5 + volumeGlow * 0.18;
        }

        vec3 lin = 2.15 * dif * vec3(1.0, 0.88, 0.76) + 0.55 * clamp(0.5 + 0.5 * nor.y, 0.0, 1.0) * vec3(0.20, 0.32, 0.45) * occ + 0.2 * fre;
        color = mix(matColor * lin, skyColor, 1.0 - exp(-0.0016 * t * t));
    }

    color += volumeGlow * (0.8 + safeFx * 0.45);
    
    // Chromatic aberration from audio
    float caShift = audioBass * 0.015;
    color.r += texture(stateTex, uv + vec2(caShift, 0)).r * caShift;
    color.b += texture(stateTex, uv - vec2(caShift, 0)).b * caShift;

    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(color, vec3(1.0 / max(gamma, 0.2)));
    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.5 + 0.5 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.25);

    fragColor = vec4(color, 1.0);
}
"""

MSG_FRAG_SHADER = """
#version 450
in vec2 uv;
out vec4 fragColor;
uniform sampler2D displayTex;
void main() { fragColor = texture(displayTex, uv); }
"""

@dataclass(frozen=True)
class ThermalHoldState:
    lines: Sequence[str]
    log_path: Path

def _sanitize_text(line: str) -> str:
    return "".join(ch if ch in FONT_3X5 else "?" for ch in line.upper())

def _draw_glyph(canvas: np.ndarray, glyph: Sequence[str], x: int, y: int, scale: int, color: Sequence[int]) -> None:
    h, w, _ = canvas.shape
    for ri, row in enumerate(glyph):
        for ci, cell in enumerate(row):
            if cell != "1": continue
            y0, x0 = y + ri * scale, x + ci * scale
            y1, x1 = min(y0 + scale, h), min(x0 + scale, w)
            if x1 > 0 and y1 > 0 and x0 < w and y0 < h:
                canvas[max(y0, 0):y1, max(x0, 0):x1] = color

def _draw_text_line(canvas: np.ndarray, line: str, y: int, scale: int, color: Sequence[int], shadow: bool = True, x: Optional[int] = None, align: str = "center") -> None:
    sanitized = _sanitize_text(line)
    g_w, spacing = 3 * scale, scale
    l_w = max(0, len(sanitized) * (g_w + spacing) - spacing)
    if x is None: x = canvas.shape[1] - l_w - scale * 3 if align == "right" else scale * 3 if align == "left" else (canvas.shape[1] - l_w) // 2
    elif align == "right": x -= l_w
    elif align == "center": x -= l_w // 2
    x = max(int(x), scale * 2)
    
    if shadow:
        s_off = max(1, scale // 3)
        s_col = np.clip(np.array(color, dtype=np.int16) // 4, 0, 255).astype(np.uint8)
        c_x = x + s_off
        for char in sanitized:
            _draw_glyph(canvas, FONT_3X5.get(char, FONT_3X5["?"]), c_x, y + s_off, scale, s_col)
            c_x += g_w + spacing
    
    c_x = x
    for char in sanitized:
        _draw_glyph(canvas, FONT_3X5.get(char, FONT_3X5["?"]), c_x, y, scale, color)
        c_x += g_w + spacing

def _text_width(line: str, scale: int) -> int:
    return len(_sanitize_text(line)) * (4 * scale) - scale if _sanitize_text(line) else 0

def _fill_rect(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: Sequence[int]) -> None:
    h, w, _ = canvas.shape
    l, r = max(0, min(w, int(x0))), max(0, min(w, int(x1)))
    t, b = max(0, min(h, int(y0))), max(0, min(h, int(y1)))
    if r > l and b > t: canvas[t:b, l:r] = color

def build_hold_frame(lines: Sequence[str], size: Sequence[int]) -> np.ndarray:
    w, h = max(int(size[0]), 320), max(int(size[1]), 180)
    xg = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    yg = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    stripe = 0.5 + 0.5 * np.sin(xg * 18.0 + yg * 11.0)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[..., 0] = np.clip(18.0 + 40.0 * (1.0 - yg) + stripe * 12.0, 0, 255).astype(np.uint8)
    frame[..., 1] = np.clip(11.0 + 20.0 * (1.0 - yg) + stripe * 8.0, 0, 255).astype(np.uint8)
    frame[..., 2] = np.clip(10.0 + 16.0 * (1.0 - yg) + stripe * 7.0, 0, 255).astype(np.uint8)
    
    margin = max(24, min(w, h) // 12)
    max_chars = max((len(_sanitize_text(line)) for line in lines), default=1)
    scale = max(2, min(max(w - 2 * margin, 120) // max(1, max_chars * 4 - 1), max(h - 2 * margin, 80) // max(1, len(lines) * 7 - 2), 32))
    l_h, l_g = 5 * scale, 2 * scale
    start_y = max((h - (len(lines) * l_h + max(0, len(lines) - 1) * l_g)) // 2, margin)
    
    for i, line in enumerate(lines):
        color = (255, 132, 78) if i == 0 else (204, 214, 220) if i >= len(lines) - 2 else (248, 238, 228)
        _draw_text_line(frame, line, start_y + i * (l_h + l_g), scale, color)
    return frame

def build_hud_frame(left: Sequence[str], right: Sequence[str], size: Sequence[int], hud_scale: float) -> np.ndarray:
    w, h = max(int(size[0]), 320), max(int(size[1]), 180)
    frame = np.zeros((h, w, 4), dtype=np.uint8)
    margin = max(14, min(w, h) // 44)
    scale = max(2, min(int(round((min(w, h) / 360.0) * max(0.5, hud_scale))), 6))
    t_scale, l_gap, pad_x, pad_y = max(scale + 1, 3), max(3, scale), scale * 4, scale * 3
    
    l_tit, l_bod = left[:1], left[1:]
    l_w = min(w - margin * 2, max([_text_width(l, t_scale) for l in l_tit] + [_text_width(l, scale) for l in l_bod] + [scale * 32]) + pad_x * 2)
    r_w = min(w - margin * 2, max([_text_width(l, scale) for l in right] + [scale * 34]) + pad_x * 2)
    l_h = max(pad_y * 2 + (5 * t_scale if l_tit else 0) + (l_gap * 2 if l_tit and l_bod else 0) + len(l_bod) * (5 * scale + l_gap), scale * 20)
    r_h = max(pad_y * 2 + len(right) * (5 * scale + l_gap), scale * 20)
    
    lx, ly, rx, ry = margin, margin, w - margin - r_w, margin
    if rx < lx + l_w + max(8, margin // 2): rx, ry = margin, margin + l_h + max(8, margin // 2)
    
    for x, y, pw, ph, lines in [(lx, ly, l_w, l_h, left), (rx, ry, r_w, r_h, right)]:
        _fill_rect(frame, x, y, x + pw, y + ph, (6, 10, 14, 172))
        _fill_rect(frame, x, y, x + scale, y + ph, (36, 214, 192, 210))
        cy = y + pad_y
        for i, line in enumerate(lines):
            s = t_scale if x == lx and i == 0 else scale
            col = (238, 252, 247, 245) if x == lx and i == 0 else (255, 186, 87, 238) if "OFF" in line else (190, 222, 224, 230)
            _draw_text_line(frame, line, cy, s, col, x=x + pad_x, align="left")
            cy += 5 * s + (l_gap * 2 if x == lx and i == 0 else l_gap)
            
    return frame

class GarageHeatShow(mglw.WindowConfig):
    title = "Garage Life Lab - Audio-Reactive 3D Bio-World"
    gl_version = (4, 5)
    resource_dir = Path(__file__).parent
    window_size = (1920, 1080)
    aspect_ratio = window_size[0] / window_size[1]
    samples = 0
    fullscreen = False
    vsync = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args = getattr(type(self), "argv", None)
        if self.args is None: raise RuntimeError("Requires command-line arguments")
        
        self.audio = GenerativeAudioEngine()
        self.stop_event = threading.Event()
        self.thermal_thread_stop = threading.Event()
        self.telemetry_lock = threading.Lock()
        
        self.latest_temperatures: Dict[str, float] = {}
        self.cpu_sensor_name: Optional[str] = None
        self.cpu_sensor_retry_after, self.gpu_sensor_failures = 0.0, 0
        self.next_title_refresh_at, self.next_hud_refresh_at = 0.0, 0.0
        self.thermal_hold: Optional[ThermalHoldState] = None
        self.hold_texture, self.hud_texture = None, None
        self.started_at, self.fps_estimate = time.monotonic(), 0.0
        self.offscreen_texture, self.offscreen_framebuffer = None, None
        self.cpu_threads: List[threading.Thread] = []

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.quad = geometry.quad_fs()

        self.update_program = self.ctx.program(vertex_shader=VERT_SHADER, fragment_shader=SIM_FRAG_SHADER)
        self.display_program = self.ctx.program(vertex_shader=VERT_SHADER, fragment_shader=DISPLAY_FRAG_SHADER)
        self.message_program = self.ctx.program(vertex_shader=VERT_SHADER, fragment_shader=MSG_FRAG_SHADER)

        self.update_program["stateTex"].value = 0
        self.update_program["audioFft"].value = 1
        self.display_program["stateTex"].value = 0
        self.display_program["audioFft"].value = 1
        self.display_program["audioWave"].value = 2
        self.message_program["displayTex"].value = 0

        self.audio_fft_tex = self.ctx.texture((FFT_BINS, 1), 1, dtype='f4')
        self.audio_fft_tex.filter = (moderngl.LINEAR, moderngl.NEAREST)
        self.audio_wave_tex = self.ctx.texture((FFT_BINS, 1), 1, dtype='f4')
        self.audio_wave_tex.filter = (moderngl.LINEAR, moderngl.NEAREST)

        if (self.args.width, self.args.height) != self.window_size:
            self.wnd.resize(self.args.width, self.args.height)

        self._init_simulation_resources()
        self._sync_static_uniforms()

        if self.args.cpu_workers > 0: self._spin_cpu_workers()
        if not self.args.no_thermal_hold: self._spin_thermal_watchdog()

    def _run_cmd(self, cmd: Sequence[str], timeout: float) -> Optional[subprocess.CompletedProcess[str]]:
        try: return subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, creationflags=CREATE_NO_WINDOW, check=False)
        except Exception: return None

    def _read_gpu_temp(self) -> Optional[float]:
        res = self._run_cmd(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], 3.0)
        try: return float([L.strip() for L in res.stdout.splitlines() if L.strip()][0]) if res and res.returncode == 0 else None
        except Exception: return None

    def _read_cpu_temp(self) -> Optional[float]:
        now = time.monotonic()
        if now < self.cpu_sensor_retry_after: return None
        res = self._run_cmd(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", CPU_SENSOR_POWERSHELL], 4.0)
        if not res or res.returncode != 0 or "|" not in res.stdout:
            self.cpu_sensor_retry_after = now + 60.0
            return None
        name, val = res.stdout.strip().splitlines()[-1].strip().split("|", 1)
        try:
            self.cpu_sensor_name = name.strip() or "CPU"
            self.cpu_sensor_retry_after = now + max(1.0, float(self.args.thermal_poll_seconds))
            return float(val)
        except ValueError:
            self.cpu_sensor_retry_after = now + 60.0
            return None

    def _spin_thermal_watchdog(self) -> None:
        threading.Thread(target=self._thermal_watchdog, name="thermal-watchdog", daemon=True).start()

    def _thermal_watchdog(self) -> None:
        poll_interval = max(1.0, float(self.args.thermal_poll_seconds))
        while not self.thermal_thread_stop.is_set() and self.thermal_hold is None:
            gpu_t = self._read_gpu_temp() if self.args.max_gpu_temp > 0 else None
            cpu_t = self._read_cpu_temp() if self.args.max_cpu_temp > 0 else None

            with self.telemetry_lock:
                if gpu_t is None: self.latest_temperatures.pop("GPU", None)
                else: self.latest_temperatures["GPU"] = gpu_t
                if cpu_t is None: self.latest_temperatures.pop("CPU", None)
                else: self.latest_temperatures["CPU"] = cpu_t

            reasons, notes = [], []
            if self.args.max_gpu_temp > 0:
                if gpu_t is None:
                    self.gpu_sensor_failures += 1
                    if self.gpu_sensor_failures >= 3: reasons.append("GPU SENSOR OFFLINE")
                else:
                    self.gpu_sensor_failures = 0
                    if gpu_t > self.args.max_gpu_temp: reasons.append(f"GPU {gpu_t:.1f}C OVER LIMIT {self.args.max_gpu_temp:.1f}C")

            if self.args.max_cpu_temp > 0:
                if cpu_t is None: notes.append("CPU SENSOR OFFLINE")
                elif cpu_t > self.args.max_cpu_temp: reasons.append(f"CPU {cpu_t:.1f}C OVER LIMIT {self.args.max_cpu_temp:.1f}C")

            if reasons:
                self._trigger_thermal_hold(reasons, notes)
                return
            if self.thermal_thread_stop.wait(poll_interval): return

    def _trigger_thermal_hold(self, reasons: Sequence[str], notes: Sequence[str]) -> None:
        if self.thermal_hold: return
        log_dir = self.resource_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "thermal_events.log"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"[{ts}] THERMAL HOLD", *reasons, *notes]
        if "GPU" in self.latest_temperatures: lines.append(f"LAST GPU TEMP {self.latest_temperatures['GPU']:.1f}C")
        if "CPU" in self.latest_temperatures: lines.append(f"LAST {self.cpu_sensor_name or 'CPU'} TEMP {self.latest_temperatures['CPU']:.1f}C")
        lines.extend(["LOADS STOPPED TO COOL SYSTEM", ""])
        with log_path.open("a", encoding="utf-8") as f: f.write("\n".join(lines))
        
        h_lines = ["THERMAL HOLD", *reasons, *notes, "LOADS STOPPED TO COOL SYSTEM", ts, "SEE LOGS THERMAL EVENTS LOG", "PRESS ESC TO EXIT"]
        self.thermal_hold = ThermalHoldState(lines=h_lines, log_path=log_path)
        self.stop_event.set()
        self.thermal_thread_stop.set()
        self.wnd.title = f"{self.base_title} | THERMAL HOLD"
        
        f_img = build_hold_frame(h_lines, self.wnd.buffer_size)
        if self.hold_texture: self.hold_texture.release()
        self.hold_texture = self.ctx.texture(self.wnd.buffer_size, 3, data=np.flipud(f_img).tobytes(), alignment=1)
        self.hold_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def _display_target(self):
        if self.ctx.screen: return self.ctx.screen
        if not self.offscreen_texture or self.offscreen_texture.size != self.wnd.buffer_size:
            if self.offscreen_framebuffer: self.offscreen_framebuffer.release()
            if self.offscreen_texture: self.offscreen_texture.release()
            self.offscreen_texture = self.ctx.texture(self.wnd.buffer_size, 4)
            self.offscreen_framebuffer = self.ctx.framebuffer(color_attachments=[self.offscreen_texture])
        return self.offscreen_framebuffer

    def _refresh_window_title(self) -> None:
        if time.monotonic() < self.next_title_refresh_at or self.thermal_hold: return
        self.next_title_refresh_at = time.monotonic() + 1.0
        parts = []
        with self.telemetry_lock:
            g, c = self.latest_temperatures.get("GPU"), self.latest_temperatures.get("CPU")
        if g is not None: parts.append(f"GPU {g:.0f}C")
        if c is not None: parts.append(f"CPU {c:.0f}C")
        elif self.args.max_cpu_temp > 0 and not self.args.no_thermal_hold: parts.append("CPU SENSOR OFFLINE")
        self.wnd.title = f"{self.base_title} | " + " | ".join(parts) if parts else self.base_title

    def _hud_lines(self) -> Sequence[Sequence[str]]:
        w, h = self.wnd.buffer_size
        ts = max(2, int(self.args.tile_size))
        with self.telemetry_lock:
            g, c = self.latest_temperatures.get("GPU"), self.latest_temperatures.get("CPU")
        audio_stat = "AUDIO ONLINE" if HAS_AUDIO else "AUDIO SIMULATED"
        secs = max(0, int(time.monotonic() - self.started_at))
        return (
            ["GARAGE LIFE LAB", "3D VOLUMETRIC STRESS", f"{w}X{h} MAP {max(1, int(np.ceil(w/ts)))}X{max(1, int(np.ceil(h/ts)))}", f"SIM STEP {self.args.substeps} RAYMARCH {self.args.ray_steps}", f"FX {self.args.fx_intensity:.1f} CAM {self.args.camera_speed:.1f}", f"CPU WORKERS {self.args.cpu_workers}"],
            [(f"GPU {g:.0f}C LIMIT {self.args.max_gpu_temp:.0f}C" if g is not None else "GPU OFFLINE") if self.args.max_gpu_temp>0 else "GPU HOLD OFF",
             (f"CPU {c:.0f}C LIMIT {self.args.max_cpu_temp:.0f}C" if c is not None else "CPU OFFLINE") if self.args.max_cpu_temp>0 else "CPU HOLD OFF",
             f"FPS {self.fps_estimate:.0f}" if self.fps_estimate > 0 else "FPS --", audio_stat, f"UP {secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}", "THERMAL HOLD OFF" if self.args.no_thermal_hold else "THERMAL HOLD ARMED"]
        )

    def _init_simulation_resources(self) -> None:
        if hasattr(self, "state_textures"):
            for fbo in getattr(self, "framebuffers", []): fbo.release()
            for tex in self.state_textures: tex.release()
        self.state_textures = [self.ctx.texture(self.wnd.buffer_size, 4, dtype="f4") for _ in range(2)]
        for tex in self.state_textures:
            tex.filter, tex.repeat_x, tex.repeat_y = (moderngl.LINEAR, moderngl.LINEAR), True, True
        self.framebuffers = [self.ctx.framebuffer(color_attachments=[tex]) for tex in self.state_textures]
        self.active_state = 0
        
        w, h = self.state_textures[0].size
        ts = max(2, int(self.args.tile_size))
        tx, ty = max(1, int(np.ceil(w / ts))), max(1, int(np.ceil(h / ts)))
        rng = np.random.default_rng(2026)
        ty_m, tx_m = np.meshgrid(np.arange(ty, dtype=np.float32), np.arange(tx, dtype=np.float32), indexing="ij")
        xn, yn = tx_m / max(tx - 1, 1), ty_m / max(ty - 1, 1)
        
        ht = 0.45 + 0.15 * np.sin(xn * 7.3 + 0.9) + 0.11 * np.cos(yn * 5.7 - 1.2) + rng.standard_normal((ty, tx), dtype=np.float32) * 0.025
        for _ in range(max(5, (tx * ty) // 4000)):
            cx, cy, rx, ry = rng.uniform(0.0, tx), rng.uniform(0.0, ty), rng.uniform(max(5.0, tx * 0.04), max(12.0, tx * 0.18)), rng.uniform(max(5.0, ty * 0.04), max(12.0, ty * 0.18))
            ht += np.clip(1.0 - (((tx_m - cx) / rx) ** 2 + ((ty_m - cy) / ry) ** 2), 0.0, 1.0) * rng.uniform(0.10, 0.24)
        for _ in range(max(4, max(5, (tx * ty) // 4000) // 2)):
            cx, cy, rx, ry = rng.uniform(0.0, tx), rng.uniform(0.0, ty), rng.uniform(max(6.0, tx * 0.05), max(14.0, tx * 0.16)), rng.uniform(max(6.0, ty * 0.05), max(14.0, ty * 0.16))
            ht -= np.clip(1.0 - (((tx_m - cx) / rx) ** 2 + ((ty_m - cy) / ry) ** 2), 0.0, 1.0) * rng.uniform(0.08, 0.18)
            
        ht = np.clip(ht + np.clip(np.sin(xn * 21.0 + np.cos(yn * 9.0) * 2.3) - 0.45, 0.0, 1.0) * 0.08, 0.0, 1.0)
        ocean = (ht < 0.46).astype(np.float32)
        moist = np.clip(0.16 + ocean * 0.52 + np.clip(1.0 - np.abs(ht - 0.46) / 0.07, 0.0, 1.0) * 0.22 + (1.0 - np.abs(yn * 2.0 - 1.0)) * 0.12 + rng.standard_normal((ty, tx), dtype=np.float32) * 0.03, 0.0, 1.0)
        bio = np.clip((1.0 - ocean) * (0.06 + moist * 0.62 + (1.0 - np.abs(yn * 2.0 - 1.0)) * 0.16 - np.clip(ht - 0.72, 0.0, 1.0) * 0.50), 0.0, 1.0)
        
        setl = np.zeros((ty, tx), dtype=np.float32)
        cands = np.argwhere((ocean < 0.5) & (np.clip(1.0 - np.abs(ht - 0.46) / 0.07, 0.0, 1.0) > 0.35) & (bio > 0.28) & (ht < 0.74))
        if len(cands) > 0:
            for idx in rng.choice(len(cands), size=min(max(8, (tx * ty) // 1800), len(cands)), replace=False):
                cy, cx = cands[idx]
                rad = int(rng.integers(1, 3))
                y0, y1, x0, x1 = max(cy - rad, 0), min(cy + rad + 1, ty), max(cx - rad, 0), min(cx + rad + 1, tx)
                py, px = np.meshgrid(np.arange(y0, y1, dtype=np.float32), np.arange(x0, x1, dtype=np.float32), indexing="ij")
                setl[y0:y1, x0:x1] = np.maximum(setl[y0:y1, x0:x1], np.clip(1.0 - np.sqrt((px - cx) ** 2 + (py - cy) ** 2) / max(rad + 0.5, 1.0), 0.0, 1.0) * rng.uniform(0.35, 0.78))

        fld = np.repeat(np.repeat(np.stack([np.clip(1.0 - bio * 0.36 + moist * 0.08, 0.0, 1.0).astype(np.float32), np.clip(bio * 0.82 + moist * 0.12, 0.0, 1.0).astype(np.float32), ht.astype(np.float32), setl.astype(np.float32)], axis=-1), ts, axis=0), ts, axis=1)[:h, :w]
        for tex in self.state_textures: tex.write(fld.tobytes())
        
        res = np.array([w, h], dtype="f4")
        self.update_program["resolution"].write(res)
        self.display_program["resolution"].write(res)

    def _sync_static_uniforms(self) -> None:
        for k, v in [("diffU", self.args.diff_u), ("diffV", self.args.diff_v), ("dt", self.args.time_step), 
                     ("laplaceScale", self.args.laplace_scale), ("noiseStrength", self.args.noise_strength), 
                     ("parameterDrift", self.args.param_drift)]: self.update_program[k].value = v
        for k, v in [("exposure", self.args.exposure), ("glow", self.args.glow), ("gamma", self.args.gamma), 
                     ("contourContrast", self.args.contour_contrast), ("cameraSpeed", self.args.camera_speed), 
                     ("fxIntensity", self.args.fx_intensity), ("raySteps", int(max(32, min(160, self.args.ray_steps))))]: self.display_program[k].value = v

    def _spin_cpu_workers(self) -> None:
        for w_id in range(self.args.cpu_workers):
            th = threading.Thread(target=self._cpu_burner, args=(w_id,), name=f"cpu-burner-{w_id}", daemon=True)
            th.start()
            self.cpu_threads.append(th)

    def _cpu_burner(self, worker_id: int) -> None:
        rng = np.random.default_rng(worker_id + 42)
        n, a, b = self.args.cpu_matrix, rng.standard_normal((self.args.cpu_matrix, self.args.cpu_matrix), dtype=np.float32), rng.standard_normal((self.args.cpu_matrix, self.args.cpu_matrix), dtype=np.float32)
        while not self.stop_event.is_set():
            np.matmul(a, b, out=a)
            if (norm := np.linalg.norm(a)) > 0: a /= norm
            _ = np.fft.fft((rng.standard_normal(n * 8, dtype=np.float32) + 1j * rng.standard_normal(n * 8, dtype=np.float32)).astype(np.complex64))
            a, b = b, a

    def render(self, time_value: float, frame_time: float) -> None:
        if frame_time > 0: self.fps_estimate = 1.0 / frame_time if self.fps_estimate <= 0 else self.fps_estimate * 0.92 + (1.0 / frame_time) * 0.08
        if self.thermal_hold:
            if not self.hold_texture or self.hold_texture.size != self.wnd.buffer_size:
                f_img = build_hold_frame(self.thermal_hold.lines, self.wnd.buffer_size)
                if self.hold_texture: self.hold_texture.release()
                self.hold_texture = self.ctx.texture(self.wnd.buffer_size, 3, data=np.flipud(f_img).tobytes(), alignment=1)
                self.hold_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._display_target().use()
            self.ctx.viewport = (0, 0, *self.wnd.buffer_size)
            self.hold_texture.use(location=0)
            self.quad.render(self.message_program)
            return

        if self.state_textures[0].size != self.wnd.buffer_size:
            self._init_simulation_resources()
            self._sync_static_uniforms()

        self._refresh_window_title()
        
        fft_data, wave_data, energy, bass, treble = self.audio.get_data()
        self.audio_fft_tex.write(fft_data.tobytes())
        self.audio_wave_tex.write(wave_data.tobytes())

        a_time = time_value * self.args.anim_speed
        for _ in range(max(1, self.args.substeps)):
            curr, nxt = self.state_textures[self.active_state], 1 - self.active_state
            self.framebuffers[nxt].use()
            self.ctx.viewport = (0, 0, *curr.size)
            curr.use(location=0)
            self.audio_fft_tex.use(location=1)
            self.update_program["time"].value = a_time
            self.update_program["feed"].value = self.args.feed
            self.update_program["kill"].value = self.args.kill
            self.update_program["audioEnergy"].value = energy
            self.update_program["audioBass"].value = bass
            self.update_program["audioTreble"].value = treble
            self.quad.render(self.update_program)
            self.active_state = nxt

        self._display_target().use()
        self.ctx.viewport = (0, 0, *self.wnd.buffer_size)
        self.state_textures[self.active_state].use(location=0)
        self.audio_fft_tex.use(location=1)
        self.audio_wave_tex.use(location=2)
        self.display_program["time"].value = a_time
        self.display_program["colorShift"].value = (a_time * self.args.color_shift_speed) % 10.0
        self.display_program["audioEnergy"].value = energy
        self.display_program["audioBass"].value = bass
        self.quad.render(self.display_program)

        if not self.args.no_hud:
            if not self.hud_texture or self.hud_texture.size != self.wnd.buffer_size or time.monotonic() >= self.next_hud_refresh_at:
                h_img = build_hud_frame(*self._hud_lines(), self.wnd.buffer_size, self.args.hud_scale)
                if not self.hud_texture or self.hud_texture.size != self.wnd.buffer_size:
                    if self.hud_texture: self.hud_texture.release()
                    self.hud_texture = self.ctx.texture(self.wnd.buffer_size, 4, data=np.flipud(h_img).tobytes(), alignment=1)
                    self.hud_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
                else: self.hud_texture.write(np.flipud(h_img).tobytes())
                self.next_hud_refresh_at = time.monotonic() + 0.5
            self._display_target().use()
            self.ctx.viewport = (0, 0, *self.wnd.buffer_size)
            self.hud_texture.use(location=0)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.quad.render(self.message_program)
            self.ctx.disable(moderngl.BLEND)

    def resize(self, width: int, height: int):  # type: ignore[override]
        if self.thermal_hold: return
        self._init_simulation_resources()
        self._sync_static_uniforms()

    def destroy(self) -> None:
        self.stop_event.set()
        self.thermal_thread_stop.set()
        for t in self.cpu_threads: t.join(timeout=1.0)
        self.cpu_threads.clear()
        self.audio.destroy()
        if self.hold_texture: self.hold_texture.release()
        if self.hud_texture: self.hud_texture.release()
        if self.offscreen_framebuffer: self.offscreen_framebuffer.release()
        if self.offscreen_texture: self.offscreen_texture.release()
        super().destroy()

    @classmethod
    def add_arguments(cls, parser) -> None:  # type: ignore[override]
        parser.add_argument("--width", type=int, default=cls.window_size[0], help="Render width")
        parser.add_argument("--height", type=int, default=cls.window_size[1], help="Render height")
        parser.add_argument("--feed", type=float, default=0.035, help="Gray-Scott base feed rate")
        parser.add_argument("--kill", type=float, default=0.060, help="Gray-Scott base kill rate")
        parser.add_argument("--diff-u", type=float, default=0.16, help="Diffusion rate for U")
        parser.add_argument("--diff-v", type=float, default=0.08, help="Diffusion rate for V")
        parser.add_argument("--time-step", dest="time_step", type=float, default=1.0, help="Simulation time step")
        parser.add_argument("--substeps", type=int, default=8, help="Simulation steps per frame")
        parser.add_argument("--laplace-scale", type=float, default=1.0, help="Global laplacian multiplier")
        parser.add_argument("--noise-strength", type=float, default=0.015, help="Stochastic noise injected each step")
        parser.add_argument("--param-drift", type=float, default=0.004, help="Sinusoidal feed/kill drift amplitude")
        parser.add_argument("--anim-speed", type=float, default=1.0, help="Global animation multiplier")
        parser.add_argument("--color-shift-speed", type=float, default=0.05, help="Palette cycle speed")
        parser.add_argument("--exposure", type=float, default=1.4, help="Display exposure")
        parser.add_argument("--glow", type=float, default=1.1, help="Display glow factor")
        parser.add_argument("--gamma", type=float, default=1.2, help="Display gamma correction")
        parser.add_argument("--contour-contrast", type=float, default=0.75, help="Contour emphasis strength")
        parser.add_argument("--ray-steps", type=int, default=96, help="Maximum raymarch steps per pixel")
        parser.add_argument("--fx-intensity", type=float, default=1.0, help="Cinematic glow, aurora, terrain, and material intensity")
        parser.add_argument("--camera-speed", type=float, default=1.0, help="Cinematic camera speed multiplier")
        parser.add_argument("--cpu-workers", type=int, default=0, help="CPU burner thread count")
        parser.add_argument("--cpu-matrix", type=int, default=896, help="CPU burner matrix size")
        parser.add_argument("--tile-size", type=int, default=12, help="Base resolution downscale factor for fluid sim")
        parser.add_argument("--max-cpu-temp", type=float, default=75.0, help="Hold the show if the CPU exceeds this temperature in Celsius")
        parser.add_argument("--max-gpu-temp", type=float, default=70.0, help="Hold the show if the GPU exceeds this temperature in Celsius")
        parser.add_argument("--thermal-poll-seconds", type=float, default=5.0, help="Sensor poll interval in seconds")
        parser.add_argument("--no-thermal-hold", action="store_true", help="Disable the thermal watchdog and hold screen")
        parser.add_argument("--no-hud", action="store_true", help="Hide the in-frame show status overlay")
        parser.add_argument("--hud-scale", type=float, default=1.0, help="Scale the in-frame show status overlay")

def main() -> None:
    mglw.run_window_config(GarageHeatShow)

if __name__ == "__main__":
    main()
