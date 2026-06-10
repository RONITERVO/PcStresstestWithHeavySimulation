"""Garage Life Lab: 1080p bio-simulation space heater (3D Raymarched Volumetric Edition)."""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import moderngl
import moderngl_window as mglw
import numpy as np
from moderngl_window import geometry

CREATE_NO_WINDOW = 0x08000000
CPU_SENSOR_POWERSHELL = """
$ErrorActionPreference = 'Stop'
$namespaces = @('root\\LibreHardwareMonitor', 'root\\OpenHardwareMonitor')
$preferredPattern = 'CPU Package|Tctl/Tdie|Tdie|Core Max|CPU CCD|CPU Die|Package'
foreach ($ns in $namespaces) {
    try {
        $sensors = Get-CimInstance -Namespace $ns -ClassName Sensor |
            Where-Object { $_.SensorType -eq 'Temperature' } |
            ForEach-Object {
                [PSCustomObject]@{
                    Name = [string]$_.Name
                    Value = [double]$_.Value
                }
            }
        if ($sensors) {
            $preferred = $sensors |
                Where-Object { $_.Name -match $preferredPattern } |
                Sort-Object Value -Descending |
                Select-Object -First 1
            if (-not $preferred) {
                $preferred = $sensors |
                    Sort-Object Value -Descending |
                    Select-Object -First 1
            }
            if ($preferred) {
                $value = $preferred.Value.ToString('0.0', [System.Globalization.CultureInfo]::InvariantCulture)
                Write-Output ('{0}|{1}' -f $preferred.Name, $value)
                exit 0
            }
        }
    } catch {
    }
}
exit 1
""".strip()

FONT_3X5 = {
    " ": ("000", "000", "000", "000", "000"),
    "-": ("000", "000", "111", "000", "000"),
    ".": ("000", "000", "000", "000", "010"),
    ":": ("000", "010", "000", "010", "000"),
    "?": ("111", "001", "011", "000", "010"),
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "A": ("111", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("111", "100", "100", "100", "111"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"),
    "G": ("111", "100", "101", "101", "111"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "101", "111"),
    "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("111", "101", "101", "101", "111"),
    "P": ("111", "101", "111", "100", "100"),
    "Q": ("111", "101", "101", "111", "011"),
    "R": ("110", "101", "110", "101", "101"),
    "S": ("111", "100", "111", "001", "111"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
    "Z": ("111", "001", "010", "100", "111"),
}

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
uniform vec2 resolution;
uniform float time;
uniform float feed;
uniform float kill;
uniform float diffU;
uniform float diffV;
uniform float dt;
uniform float laplaceScale;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 getEventTarget(float id, float seed) {
    return vec2(hash12(vec2(id, seed)), hash12(vec2(id, seed + 1.0)));
}

void main() {
    vec2 texel = 1.0 / resolution;

    // Channels:
    // R (x): Substrate / Empty Grid (U)
    // G (y): Active Neural Nodes (V)
    // B (z): Crystal / Bismuth Matrix Elevation
    // A (w): Computational Heat / Energy Load

    vec4 c  = texture(stateTex, uv);
    vec4 r  = texture(stateTex, uv + vec2(texel.x, 0.0));
    vec4 l  = texture(stateTex, uv - vec2(texel.x, 0.0));
    vec4 t  = texture(stateTex, uv + vec2(0.0, texel.y));
    vec4 b_ = texture(stateTex, uv - vec2(0.0, texel.y));
    vec4 tr = texture(stateTex, uv + vec2(texel.x, texel.y));
    vec4 tl = texture(stateTex, uv + vec2(-texel.x, texel.y));
    vec4 br = texture(stateTex, uv + vec2(texel.x, -texel.y));
    vec4 bl = texture(stateTex, uv - texel);

    float lapU = (r.x + l.x + t.x + b_.x) * 0.2 + (tr.x + tl.x + br.x + bl.x) * 0.05 - c.x;
    float lapV = (r.y + l.y + t.y + b_.y) * 0.2 + (tr.y + tl.y + br.y + bl.y) * 0.05 - c.y;
    float lapH = (r.z + l.z + t.z + b_.z) * 0.2 + (tr.z + tl.z + br.z + bl.z) * 0.05 - c.z;
    float lapA = (r.w + l.w + t.w + b_.w) * 0.2 + (tr.w + tl.w + br.w + bl.w) * 0.05 - c.w;

    float reaction = c.x * c.y * c.y;

    // Matrix heat slows down structural feed, pushing network to branch
    float matrixFeed = feed - c.w * 0.012;
    float matrixKill = kill + c.w * 0.012;

    float du = (diffU * lapU * laplaceScale) - reaction + matrixFeed * (1.0 - c.x);
    float dv = (diffV * lapV * laplaceScale) + reaction - (matrixFeed + matrixKill) * c.y;

    // Matrix Height organically targets active node locations
    float targetH = smoothstep(0.25, 0.75, c.y); // Requires higher concentration to raise
    float dh = (targetH - c.z) * 0.05 + lapH * 0.2;

    // Computational Surges (Quantum tunnel events)
    float surgeInterval = 6.0;
    float surgeId = floor(time / surgeInterval);
    float surgeLocalTime = fract(time / surgeInterval) * surgeInterval;
    vec2 sTarget = getEventTarget(surgeId, 99.0);
    float sDist = length(uv - sTarget);

    float surgePulse = exp(-pow((surgeLocalTime - 1.0) * 15.0, 2.0));
    float surgeCore = 1.0 - smoothstep(0.0, 0.03, sDist);
    
    // Inject extreme heat and force nodes to activate
    float heatGain = surgePulse * surgeCore * 15.0;
    dv += surgePulse * surgeCore * 0.5;

    // Heat diffuses rapidly and is consumed by growth
    float dw = lapA * 0.85 + heatGain + reaction * 2.5 - c.w * 0.15;

    fragColor = vec4(
        clamp(c.x + du * dt, 0.0, 1.0),
        clamp(c.y + dv * dt, 0.0, 1.0),
        clamp(c.z + dh * dt, 0.0, 1.0),
        clamp(c.w + dw * dt, 0.0, 1.0)
    );
}
"""

DISPLAY_FRAG_SHADER = """
#version 450
in vec2 uv;
out vec4 fragColor;

uniform sampler2D stateTex;
uniform vec2 resolution;
uniform float time;
uniform float exposure;
uniform float glow;
uniform float gamma;
uniform float contourContrast;
uniform float cameraSpeed;
uniform float fxIntensity;
uniform int raySteps;

#define MAX_STEPS 160
#define MAX_DIST 40.0
#define SURF_DIST 0.002

mat3 setCamera(in vec3 ro, in vec3 ta, float cr) {
    vec3 cw = normalize(ta - ro);
    vec3 cp = vec3(sin(cr), cos(cr), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = normalize(cross(cu, cw));
    return mat3(cu, cv, cw);
}

float hash12(vec2 p) {
    vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i + vec2(0.0, 0.0)), hash12(i + vec2(1.0, 0.0)), f.x),
               mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), f.x), f.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    mat2 rot = mat2(0.866, -0.5, 0.5, 0.866);
    for (int i = 0; i < 4; i++) {
        v += a * noise(p);
        p = rot * p * 2.0 + vec2(100.0);
        a *= 0.5;
    }
    return v;
}

vec2 getEventTarget(float id, float seed) {
    return vec2(hash12(vec2(id, seed)), hash12(vec2(id, seed + 1.0)));
}

float map(in vec3 p, out int matID, out vec4 stateOut) {
    vec2 mapUV = p.xz * 0.06;
    vec4 state = textureLod(stateTex, mapUV, 0.0);

    // Quantize height to create Bismuth-like crystalline circuits. 
    // Smooth the step slightly to prevent SDF raymarching discontinuities.
    float hRaw = state.z * 2.5; 
    float levels = mix(4.0, 10.0, smoothstep(0.2, 1.5, fxIntensity));
    float q = hRaw * levels;
    float hStep = (floor(q) + smoothstep(0.0, 0.2, fract(q))) / levels * 1.5;

    // Tech grid gaps and trenches
    float gridX = abs(fract(p.x * 1.5) - 0.5) * 2.0;
    float gridZ = abs(fract(p.z * 1.5) - 0.5) * 2.0;
    float trench = 1.0 - smoothstep(0.02, 0.12, min(gridX, gridZ));
    
    // Deform terrain with trenches and structural steps
    float hTerrain = hStep - trench * 0.5 * smoothstep(0.2, 0.8, noise(p.xz * 3.0));
    float dTerrain = p.y - hTerrain;

    // Data Swarm / Quantum Plasma Volume
    float swarmH = state.w * 0.8 + state.y * 0.5 + fbm(p.xz * 1.5 - vec2(time * 0.4, 0.0)) * 0.4 - 0.2;
    float dSwarm = p.y - swarmH;

    if (dTerrain < dSwarm) {
        matID = 1; // Neural Bismuth Matrix
        stateOut = state;
        return dTerrain * 0.5;
    } else {
        matID = 0; // Data Swarm
        stateOut = state;
        return dSwarm * 0.7;
    }
}

vec3 calcNormal(in vec3 p) {
    const vec2 h = vec2(0.01, 0.0);
    int dMat; vec4 dState;
    return normalize(vec3(
        map(p + h.xyy, dMat, dState) - map(p - h.xyy, dMat, dState),
        map(p + h.yxy, dMat, dState) - map(p - h.yxy, dMat, dState),
        map(p + h.yyx, dMat, dState) - map(p - h.yyx, dMat, dState)
    ));
}

float calcShadow(in vec3 ro, in vec3 rd) {
    float res = 1.0;
    float t = 0.05;
    int dMat; vec4 dState;
    for (int i = 0; i < 32; i++) {
        float h = map(ro + rd * t, dMat, dState);
        res = min(res, 12.0 * h / t);
        t += clamp(h, 0.02, 0.5);
        if (h < 0.001 || t > 15.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float contourPower = clamp(contourContrast, 0.0, 1.6);
    int maxRaySteps = clamp(raySteps, 32, MAX_STEPS);

    float camTime = time * 0.08 * max(cameraSpeed, 0.01);
    // Lift camera baseline and look slightly down so it never dips below max terrain bounds
    vec3 ro = vec3(camTime * 4.0, 5.5 + sin(camTime * 0.3) * 1.2, camTime * 3.0);
    vec3 ta = ro + vec3(cos(camTime * 0.25), -0.6, sin(camTime * 0.25));

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.15) * 0.08);
    vec3 rd = ca * normalize(vec3(p.xy, 2.0));

    vec3 lightDir = normalize(vec3(0.5, 0.8, -0.6));

    // Deep void cyber-sky
    vec3 skyColor = vec3(0.01, 0.02, 0.04);
    float sun = pow(max(0.0, dot(rd, lightDir)), 120.0);
    skyColor += vec3(0.0, 0.4, 0.8) * sun * safeFx * 0.4;

    // Quantum Surge Flash
    float surgeInterval = 6.0;
    float surgeId = floor(time / surgeInterval);
    float surgeLocalTime = fract(time / surgeInterval) * surgeInterval;

    if (surgeLocalTime > 0.5 && surgeLocalTime < 2.0) {
        vec2 sTargetUV = getEventTarget(surgeId, 99.0);
        vec3 sTargetWorld = vec3(sTargetUV.x / 0.06, 0.0, sTargetUV.y / 0.06);
        vec3 toTarget = normalize(sTargetWorld - ro);
        float sProj = max(0.0, dot(rd, toTarget));
        
        float flash = exp(-pow(surgeLocalTime - 1.0, 2.0) * 80.0);
        skyColor += vec3(0.0, 1.0, 0.8) * pow(sProj, 250.0) * 6.0 * flash * safeFx;
        skyColor += vec3(1.0, 0.0, 0.6) * flash * 1.5 * safeFx;
    }

    float t = 0.0;
    int matID = -1;
    vec4 state = vec4(0.0);
    vec3 volumeGlow = vec3(0.0);

    for (int i = 0; i < MAX_STEPS; i++) {
        if (i >= maxRaySteps) break;
        vec3 pos = ro + rd * t;
        int currMat; vec4 currState;
        float h = map(pos, currMat, currState);

        if (currMat == 0) {
            float vDense = smoothstep(0.1, 0.8, currState.y);
            float wHot = smoothstep(0.1, 1.0, currState.w);
            vec3 swarmColor = mix(vec3(0.0, 0.7, 1.0), vec3(1.0, 0.0, 0.8), wHot);
            volumeGlow += swarmColor * (vDense + wHot * 0.5) * (0.04 * glow) * safeFx / (1.0 + abs(h) * 6.0);
        }

        if (h < SURF_DIST * (1.0 + t * 0.1) || t > MAX_DIST) {
            matID = currMat;
            state = currState;
            break;
        }
        t += clamp(h, 0.02, 0.8);
    }

    vec3 color = skyColor;

    if (t < MAX_DIST) {
        vec3 pos = ro + rd * t;
        vec3 nor = calcNormal(pos);

        float occ = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
        float sha = calcShadow(pos + nor * 0.02, lightDir);
        float dif = clamp(dot(nor, lightDir), 0.0, 1.0) * sha;
        float sky = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
        float fre = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 5.0);

        vec3 matColor;
        vec3 emission = vec3(0.0);

        if (matID == 1) {
            // Obsidian / Bismuth Matrix Base
            matColor = vec3(0.03, 0.04, 0.06);

            // Bismuth Iridescence
            vec3 iridescence = 0.5 + 0.5 * cos(6.28318 * (state.z * 1.5 + vec3(0.0, 0.33, 0.67)));
            matColor += iridescence * 0.15 * sha * safeFx;

            // Compute grid location for glowing circuit traces
            float gridX = abs(fract(pos.x * 1.5) - 0.5) * 2.0;
            float gridZ = abs(fract(pos.z * 1.5) - 0.5) * 2.0;
            float isTrench = 1.0 - smoothstep(0.02, 0.06, min(gridX, gridZ));
            float edgeGlow = smoothstep(0.08, 0.0, min(gridX, gridZ));
            float contourLine = 1.0 - smoothstep(
                0.0,
                0.055,
                abs(fract((pos.x * 0.35 + pos.z * 0.45 + pos.y * 1.8) * 5.0) - 0.5)
            );
            matColor += vec3(0.0, 0.45, 0.55) * contourLine * contourPower * 0.18;

            // Data Path Emissions
            vec3 pathGlow = mix(vec3(0.0, 0.8, 1.0), vec3(1.0, 0.2, 0.6), state.w);
            emission += pathGlow * isTrench * state.y * 3.0 * safeFx;
            emission += pathGlow * edgeGlow * state.w * 4.0 * safeFx;

            // Shiny matrix reflections
            float spec = pow(clamp(dot(reflect(rd, nor), lightDir), 0.0, 1.0), 32.0) * sha;
            emission += vec3(0.5, 0.8, 1.0) * spec * 0.5;

        } else {
            // Surface of the dense Quantum Swarm
            matColor = mix(vec3(0.0, 0.15, 0.25), vec3(0.6, 0.1, 0.4), state.y);
            emission += matColor * state.w * 1.5 * safeFx;
            
            float spec = pow(clamp(dot(reflect(rd, nor), lightDir), 0.0, 1.0), 8.0) * sha;
            emission += vec3(0.0, 0.6, 0.8) * spec * 0.4;
            emission += volumeGlow * 0.3;
        }

        vec3 lin = vec3(0.0);
        lin += 1.2 * dif * vec3(0.8, 0.9, 1.0);
        lin += 0.6 * sky * vec3(0.1, 0.2, 0.3) * occ;
        lin += 0.8 * fre * vec3(0.4, 0.8, 1.0) * occ;

        color = matColor * lin + emission;

        float fog = 1.0 - exp(-0.0015 * t * t);
        color = mix(color, skyColor, fog);
    }

    color += volumeGlow * (1.0 + safeFx * 0.5);

    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(color, vec3(1.0 / max(gamma, 0.1)));

    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.5 + 0.5 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.2);

    fragColor = vec4(color, 1.0);
}
"""

MSG_FRAG_SHADER = """
#version 450
in vec2 uv;
out vec4 fragColor;
uniform sampler2D displayTex;
void main() {
    fragColor = texture(displayTex, uv);
}
"""

@dataclass(frozen=True)
class ThermalHoldState:
    lines: Sequence[str]
    log_path: Path

def _sanitize_text(line: str) -> str:
    return "".join(ch if ch in FONT_3X5 else "?" for ch in line.upper())

def _draw_glyph(
    canvas: np.ndarray,
    glyph: Sequence[str],
    x: int,
    y: int,
    scale: int,
    color: Sequence[int],
) -> None:
    height, width, _ = canvas.shape
    for row_index, row in enumerate(glyph):
        for col_index, cell in enumerate(row):
            if cell != "1":
                continue
            y0 = y + row_index * scale
            x0 = x + col_index * scale
            y1 = min(y0 + scale, height)
            x1 = min(x0 + scale, width)
            if x1 > 0 and y1 > 0 and x0 < width and y0 < height:
                canvas[max(y0, 0):y1, max(x0, 0):x1] = color

def _draw_text_line(
    canvas: np.ndarray,
    line: str,
    y: int,
    scale: int,
    color: Sequence[int],
    shadow: bool = True,
    x: Optional[int] = None,
    align: str = "center",
) -> None:
    sanitized = _sanitize_text(line)
    glyph_width = 3 * scale
    spacing = scale
    line_width = max(0, len(sanitized) * (glyph_width + spacing) - spacing)
    if x is None:
        if align == "right":
            x = canvas.shape[1] - line_width - scale * 3
        elif align == "left":
            x = scale * 3
        else:
            x = (canvas.shape[1] - line_width) // 2
    elif align == "right":
        x -= line_width
    elif align == "center":
        x -= line_width // 2
    x = max(int(x), scale * 2)
    if shadow:
        shadow_offset = max(1, scale // 3)
        shadow_color = np.clip(np.array(color, dtype=np.int16) // 4, 0, 255).astype(np.uint8)
        cursor = x + shadow_offset
        for char in sanitized:
            _draw_glyph(canvas, FONT_3X5.get(char, FONT_3X5["?"]), cursor, y + shadow_offset, scale, shadow_color)
            cursor += glyph_width + spacing
    cursor = x
    for char in sanitized:
        _draw_glyph(canvas, FONT_3X5.get(char, FONT_3X5["?"]), cursor, y, scale, color)
        cursor += glyph_width + spacing

def _text_width(line: str, scale: int) -> int:
    sanitized = _sanitize_text(line)
    if not sanitized:
        return 0
    return len(sanitized) * (4 * scale) - scale

def _fill_rect(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Sequence[int],
) -> None:
    height, width, _ = canvas.shape
    left = max(0, min(width, int(x0)))
    right = max(0, min(width, int(x1)))
    top = max(0, min(height, int(y0)))
    bottom = max(0, min(height, int(y1)))
    if right > left and bottom > top:
        canvas[top:bottom, left:right] = color

def build_hold_frame(lines: Sequence[str], size: Sequence[int]) -> np.ndarray:
    width = max(int(size[0]), 320)
    height = max(int(size[1]), 180)
    x_gradient = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y_gradient = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    stripe = 0.5 + 0.5 * np.sin(x_gradient * 18.0 + y_gradient * 11.0)

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = np.clip(10.0 + 10.0 * (1.0 - y_gradient) + stripe * 5.0, 0, 255).astype(np.uint8)
    frame[..., 1] = np.clip(5.0 + 10.0 * (1.0 - y_gradient) + stripe * 5.0, 0, 255).astype(np.uint8)
    frame[..., 2] = np.clip(30.0 + 20.0 * (1.0 - y_gradient) + stripe * 10.0, 0, 255).astype(np.uint8)

    margin = max(24, min(width, height) // 12)
    max_chars = max((len(_sanitize_text(line)) for line in lines), default=1)
    usable_width = max(width - 2 * margin, 120)
    usable_height = max(height - 2 * margin, 80)
    scale_by_width = usable_width // max(1, max_chars * 4 - 1)
    scale_by_height = usable_height // max(1, len(lines) * 7 - 2)
    scale = max(2, min(scale_by_width, scale_by_height, 32))
    line_height = 5 * scale
    line_gap = 2 * scale
    total_height = len(lines) * line_height + max(0, len(lines) - 1) * line_gap
    start_y = max((height - total_height) // 2, margin)

    for index, line in enumerate(lines):
        if index == 0:
            color = (255, 50, 100)
        elif index >= len(lines) - 2:
            color = (100, 200, 255)
        else:
            color = (200, 255, 255)
        line_y = start_y + index * (line_height + line_gap)
        _draw_text_line(frame, line, line_y, scale, color)

    return frame

def build_hud_frame(
    left_lines: Sequence[str],
    right_lines: Sequence[str],
    size: Sequence[int],
    hud_scale: float,
) -> np.ndarray:
    width = max(int(size[0]), 320)
    height = max(int(size[1]), 180)
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    margin = max(14, min(width, height) // 44)
    gap = max(8, margin // 2)
    base_scale = int(round((min(width, height) / 360.0) * max(0.5, hud_scale)))
    scale = max(2, min(base_scale, 6))
    title_scale = max(scale + 1, 3)
    line_gap = max(3, scale)
    pad_x = scale * 4
    pad_y = scale * 3

    left_title = left_lines[:1]
    left_body = left_lines[1:]
    left_width = max(
        [_text_width(line, title_scale) for line in left_title]
        + [_text_width(line, scale) for line in left_body]
        + [scale * 32]
    )
    right_width = max([_text_width(line, scale) for line in right_lines] + [scale * 34])
    left_panel_width = min(width - margin * 2, left_width + pad_x * 2)
    right_panel_width = min(width - margin * 2, right_width + pad_x * 2)
    left_panel_height = (
        pad_y * 2
        + (5 * title_scale if left_title else 0)
        + (line_gap * 2 if left_title and left_body else 0)
        + len(left_body) * (5 * scale + line_gap)
    )
    left_panel_height = max(left_panel_height, scale * 20)
    right_panel_height = pad_y * 2 + len(right_lines) * (5 * scale + line_gap)
    right_panel_height = max(right_panel_height, scale * 20)

    left_x = margin
    left_y = margin
    right_x = width - margin - right_panel_width
    right_y = margin
    if right_x < left_x + left_panel_width + gap:
        right_x = margin
        right_y = margin + left_panel_height + gap

    panel_color = (5, 10, 20, 200)
    line_color = (0, 255, 180, 210)
    title_color = (255, 255, 255, 245)
    body_color = (150, 220, 255, 230)
    warn_color = (255, 50, 100, 238)

    _fill_rect(frame, left_x, left_y, left_x + left_panel_width, left_y + left_panel_height, panel_color)
    _fill_rect(frame, left_x, left_y, left_x + scale, left_y + left_panel_height, line_color)
    cursor_y = left_y + pad_y
    if left_title:
        _draw_text_line(
            frame,
            left_title[0],
            cursor_y,
            title_scale,
            title_color,
            x=left_x + pad_x,
            align="left",
        )
        cursor_y += 5 * title_scale + line_gap * 2
    for line in left_body:
        _draw_text_line(
            frame,
            line,
            cursor_y,
            scale,
            body_color,
            x=left_x + pad_x,
            align="left",
        )
        cursor_y += 5 * scale + line_gap

    _fill_rect(frame, right_x, right_y, right_x + right_panel_width, right_y + right_panel_height, panel_color)
    _fill_rect(frame, right_x, right_y, right_x + scale, right_y + right_panel_height, line_color)
    cursor_y = right_y + pad_y
    for line in right_lines:
        color = warn_color if "OFFLINE" in line or "OFF" in line or "LOST" in line else body_color
        _draw_text_line(
            frame,
            line,
            cursor_y,
            scale,
            color,
            x=right_x + pad_x,
            align="left",
        )
        cursor_y += 5 * scale + line_gap

    return frame

class GarageHeatShow(mglw.WindowConfig):
    title = "Garage Life Lab - NEURAL MATRIX CONTAINMENT"
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
        if self.args is None:
            raise RuntimeError("GarageHeatShow requires command-line arguments")

        self.base_title = type(self).title
        self.stop_event = threading.Event()
        self.thermal_thread_stop = threading.Event()
        self.telemetry_lock = threading.Lock()
        self.latest_temperatures: Dict[str, float] = {}
        self.cpu_sensor_name: Optional[str] = None
        self.cpu_sensor_retry_after = 0.0
        self.gpu_sensor_failures = 0
        self.next_title_refresh_at = 0.0
        self.thermal_hold: Optional[ThermalHoldState] = None
        self.hold_texture = None
        self.hud_texture = None
        self.next_hud_refresh_at = 0.0
        self.started_at = time.monotonic()
        self.fps_estimate = 0.0
        self.offscreen_texture = None
        self.offscreen_framebuffer = None
        self.cpu_threads: List[threading.Thread] = []

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.quad = geometry.quad_fs()

        self.update_program = self.ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=SIM_FRAG_SHADER,
        )
        self.display_program = self.ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=DISPLAY_FRAG_SHADER,
        )
        self.message_program = self.ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=MSG_FRAG_SHADER,
        )

        self.update_program["stateTex"].value = 0
        self.display_program["stateTex"].value = 0
        self.message_program["displayTex"].value = 0

        if (self.args.width, self.args.height) != self.window_size:
            self.wnd.resize(self.args.width, self.args.height)

        self._init_simulation_resources()
        self._sync_static_uniforms()

        if self.args.cpu_workers > 0:
            self._spin_cpu_workers()
        if not self.args.no_thermal_hold:
            self._spin_thermal_watchdog()

    def _run_command(self, command: Sequence[str], timeout: float) -> Optional[subprocess.CompletedProcess[str]]:
        try:
            return subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=CREATE_NO_WINDOW,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return None

    def _read_gpu_temp(self) -> Optional[float]:
        result = self._run_command(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            timeout=3.0,
        )
        if result is None or result.returncode != 0:
            return None
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return None
        try:
            return float(lines[0])
        except ValueError:
            return None

    def _read_cpu_temp(self) -> Optional[float]:
        now = time.monotonic()
        if now < self.cpu_sensor_retry_after:
            return None
        result = self._run_command(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", CPU_SENSOR_POWERSHELL],
            timeout=4.0,
        )
        if result is None or result.returncode != 0:
            self.cpu_sensor_name = None
            self.cpu_sensor_retry_after = now + 60.0
            return None
        line = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ""
        if "|" not in line:
            self.cpu_sensor_name = None
            self.cpu_sensor_retry_after = now + 60.0
            return None
        name, value = line.split("|", 1)
        try:
            temperature = float(value)
        except ValueError:
            self.cpu_sensor_name = None
            self.cpu_sensor_retry_after = now + 60.0
            return None
        self.cpu_sensor_name = name.strip() or "CPU"
        self.cpu_sensor_retry_after = now + max(1.0, float(self.args.thermal_poll_seconds))
        return temperature

    def _spin_thermal_watchdog(self) -> None:
        thread = threading.Thread(
            target=self._thermal_watchdog,
            name="thermal-watchdog",
            daemon=True,
        )
        thread.start()

    def _thermal_watchdog(self) -> None:
        poll_interval = max(1.0, float(self.args.thermal_poll_seconds))
        while not self.thermal_thread_stop.is_set() and self.thermal_hold is None:
            gpu_temp = self._read_gpu_temp() if self.args.max_gpu_temp > 0 else None
            cpu_temp = self._read_cpu_temp() if self.args.max_cpu_temp > 0 else None

            with self.telemetry_lock:
                if gpu_temp is None:
                    self.latest_temperatures.pop("GPU", None)
                else:
                    self.latest_temperatures["GPU"] = gpu_temp
                if cpu_temp is None:
                    self.latest_temperatures.pop("CPU", None)
                else:
                    self.latest_temperatures["CPU"] = cpu_temp

            reasons: List[str] = []
            notes: List[str] = []

            if self.args.max_gpu_temp > 0:
                if gpu_temp is None:
                    self.gpu_sensor_failures += 1
                    if self.gpu_sensor_failures >= 3:
                        reasons.append("GPU TELEMETRY LOST")
                else:
                    self.gpu_sensor_failures = 0
                    if gpu_temp > self.args.max_gpu_temp:
                        reasons.append(
                            f"GPU {gpu_temp:.1f}C EXCEEDS LIMIT {self.args.max_gpu_temp:.1f}C"
                        )

            if self.args.max_cpu_temp > 0:
                if cpu_temp is None:
                    notes.append("CPU TELEMETRY LOST")
                elif cpu_temp > self.args.max_cpu_temp:
                    reasons.append(
                        f"CPU {cpu_temp:.1f}C EXCEEDS LIMIT {self.args.max_cpu_temp:.1f}C"
                    )

            if reasons:
                self._trigger_thermal_hold(reasons, notes)
                return

            if self.thermal_thread_stop.wait(poll_interval):
                return

    def _write_thermal_logs(self, reasons: Sequence[str], notes: Sequence[str]) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_dir = self.resource_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "thermal_events.log"
        last_event_path = log_dir / "last_thermal_hold.txt"

        log_lines = [f"[{timestamp}] CONTAINMENT BREACH AVERTED: THERMAL HOLD", *reasons]
        if notes:
            log_lines.extend(notes)
        gpu_temp = self.latest_temperatures.get("GPU")
        cpu_temp = self.latest_temperatures.get("CPU")
        if gpu_temp is not None:
            log_lines.append(f"LAST GPU TEMP {gpu_temp:.1f}C")
        if cpu_temp is not None:
            sensor_name = self.cpu_sensor_name or "CPU"
            log_lines.append(f"LAST {sensor_name.upper()} TEMP {cpu_temp:.1f}C")
        log_lines.append("QUANTUM LOADS HALTED TO COOL SYSTEM")
        log_lines.append("")

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(log_lines))
        last_event_path.write_text("\n".join(log_lines[:-1]), encoding="utf-8")
        return log_path

    def _build_hold_texture(self, lines: Sequence[str]) -> None:
        frame = build_hold_frame(lines, self.wnd.buffer_size)
        if self.hold_texture is not None:
            self.hold_texture.release()
        self.hold_texture = self.ctx.texture(
            self.wnd.buffer_size,
            3,
            data=np.flipud(frame).tobytes(),
            alignment=1,
        )
        self.hold_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def _display_target(self):
        if self.ctx.screen is not None:
            return self.ctx.screen
        if (
            self.offscreen_texture is None
            or self.offscreen_framebuffer is None
            or self.offscreen_texture.size != self.wnd.buffer_size
        ):
            if self.offscreen_framebuffer is not None:
                self.offscreen_framebuffer.release()
            if self.offscreen_texture is not None:
                self.offscreen_texture.release()
            self.offscreen_texture = self.ctx.texture(self.wnd.buffer_size, 4)
            self.offscreen_framebuffer = self.ctx.framebuffer(
                color_attachments=[self.offscreen_texture]
            )
        return self.offscreen_framebuffer

    def _trigger_thermal_hold(self, reasons: Sequence[str], notes: Sequence[str]) -> None:
        if self.thermal_hold is not None:
            return
        log_path = self._write_thermal_logs(reasons, notes)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hold_lines = [
            "CONTAINMENT BREACH AVERTED",
            *reasons,
            *notes,
            "QUANTUM LOADS HALTED TO COOL SYSTEM",
            timestamp,
            "SEE LOGS THERMAL EVENTS LOG",
            "PRESS ESC TO EXIT",
        ]
        self.thermal_hold = ThermalHoldState(lines=hold_lines, log_path=log_path)
        self.stop_event.set()
        self.thermal_thread_stop.set()
        self.wnd.title = f"{self.base_title} | THERMAL HOLD"
        self._build_hold_texture(hold_lines)

    def _refresh_window_title(self) -> None:
        now = time.monotonic()
        if now < self.next_title_refresh_at or self.thermal_hold is not None:
            return
        self.next_title_refresh_at = now + 1.0
        parts: List[str] = []
        with self.telemetry_lock:
            gpu_temp = self.latest_temperatures.get("GPU")
            cpu_temp = self.latest_temperatures.get("CPU")
        if gpu_temp is not None:
            parts.append(f"GPU {gpu_temp:.0f}C")
        if cpu_temp is not None:
            parts.append(f"CPU {cpu_temp:.0f}C")
        elif self.args.max_cpu_temp > 0 and not self.args.no_thermal_hold:
            parts.append("CPU TELEMETRY LOST")
        if parts:
            self.wnd.title = f"{self.base_title} | " + " | ".join(parts)
        else:
            self.wnd.title = self.base_title

    def _format_uptime(self) -> str:
        total_seconds = max(0, int(time.monotonic() - self.started_at))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"UP {hours:02d}:{minutes:02d}:{seconds:02d}"

    def _temperature_line(self, label: str, value: Optional[float], limit: float) -> str:
        if limit <= 0:
            return f"{label} HOLD OFF"
        if value is None:
            return f"{label} TELEMETRY LOST"
        return f"{label} {value:.0f}C LIMIT {limit:.0f}C"

    def _hud_lines(self) -> Sequence[Sequence[str]]:
        width, height = self.wnd.buffer_size
        tile_size = max(2, int(self.args.tile_size))
        tiles_x = max(1, int(np.ceil(width / tile_size)))
        tiles_y = max(1, int(np.ceil(height / tile_size)))
        with self.telemetry_lock:
            gpu_temp = self.latest_temperatures.get("GPU")
            cpu_temp = self.latest_temperatures.get("CPU")
        left_lines = [
            "GARAGE LIFE LAB",
            "NEURAL MATRIX CONTAINMENT",
            f"QUANTUM GRID {width}X{height} NODES {tiles_x}X{tiles_y}",
            f"SIM CYCLE {self.args.substeps} TENSOR {self.args.ray_steps}",
            f"OVERRIDE FX {self.args.fx_intensity:.1f} CAM {self.args.camera_speed:.1f}",
            f"ANNEALING WORKERS {self.args.cpu_workers}",
        ]
        right_lines = [
            self._temperature_line("GPU", gpu_temp, self.args.max_gpu_temp),
            self._temperature_line("CPU", cpu_temp, self.args.max_cpu_temp),
            f"FPS {self.fps_estimate:.0f}" if self.fps_estimate > 0 else "FPS --",
            self._format_uptime(),
            "THERMAL HOLD OFF" if self.args.no_thermal_hold else "THERMAL HOLD ARMED",
        ]
        return left_lines, right_lines

    def _build_hud_texture(self) -> None:
        if self.args.no_hud:
            return
        now = time.monotonic()
        if (
            self.hud_texture is not None
            and self.hud_texture.size == self.wnd.buffer_size
            and now < self.next_hud_refresh_at
        ):
            return
        left_lines, right_lines = self._hud_lines()
        frame = build_hud_frame(left_lines, right_lines, self.wnd.buffer_size, self.args.hud_scale)
        data = np.flipud(frame).tobytes()
        if self.hud_texture is None or self.hud_texture.size != self.wnd.buffer_size:
            if self.hud_texture is not None:
                self.hud_texture.release()
            self.hud_texture = self.ctx.texture(
                self.wnd.buffer_size,
                4,
                data=data,
                alignment=1,
            )
            self.hud_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        else:
            self.hud_texture.write(data)
        self.next_hud_refresh_at = now + 0.5

    def _init_simulation_resources(self) -> None:
        if hasattr(self, "state_textures"):
            for fbo in getattr(self, "framebuffers", []):
                fbo.release()
            for tex in self.state_textures:
                tex.release()
        buffer_size = self.wnd.buffer_size
        self.state_textures = [
            self.ctx.texture(buffer_size, 4, dtype="f4")
            for _ in range(2)
        ]
        for tex in self.state_textures:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = True
            tex.repeat_y = True
        self.framebuffers = [
            self.ctx.framebuffer(color_attachments=[tex])
            for tex in self.state_textures
        ]
        self.active_state = 0
        self._seed_field()
        self._update_resolution_uniforms()

    def _update_resolution_uniforms(self) -> None:
        buffer_size = self.wnd.buffer_size
        resolution = np.array([buffer_size[0], buffer_size[1]], dtype="f4")
        self.update_program["resolution"].write(resolution)
        self.display_program["resolution"].write(resolution)

    def _seed_field(self) -> None:
        width_px, height_px = self.state_textures[0].size
        tile_size = max(2, int(self.args.tile_size))
        tiles_x = max(1, int(np.ceil(width_px / tile_size)))
        tiles_y = max(1, int(np.ceil(height_px / tile_size)))

        rng = np.random.default_rng(4242)

        reaction_u = np.ones((tiles_y, tiles_x), dtype=np.float32)
        reaction_v = np.zeros((tiles_y, tiles_x), dtype=np.float32)

        seed_count = max(5, (tiles_x * tiles_y) // 300)
        for _ in range(seed_count):
            cx = rng.integers(0, tiles_x)
            cy = rng.integers(0, tiles_y)
            radius = rng.integers(2, 8)
            y, x = np.ogrid[-cy:tiles_y-cy, -cx:tiles_x-cx]
            mask = x*x + y*y <= radius*radius
            reaction_v[mask] = 1.0
            reaction_u[mask] = 0.5

        height = rng.uniform(0.0, 0.2, size=(tiles_y, tiles_x)).astype(np.float32)
        heat = np.zeros((tiles_y, tiles_x), dtype=np.float32)

        tile_field = np.stack([reaction_u, reaction_v, height, heat], axis=-1)
        field = np.repeat(np.repeat(tile_field, tile_size, axis=0), tile_size, axis=1)
        field = field[:height_px, :width_px].copy()
        
        for tex in self.state_textures:
            tex.write(field.tobytes())

    def _sync_static_uniforms(self) -> None:
        self.update_program["diffU"].value = self.args.diff_u
        self.update_program["diffV"].value = self.args.diff_v
        self.update_program["dt"].value = self.args.time_step
        self.update_program["laplaceScale"].value = self.args.laplace_scale

        self.display_program["exposure"].value = self.args.exposure
        self.display_program["glow"].value = self.args.glow
        self.display_program["gamma"].value = self.args.gamma
        self.display_program["contourContrast"].value = self.args.contour_contrast
        self.display_program["cameraSpeed"].value = self.args.camera_speed
        self.display_program["fxIntensity"].value = self.args.fx_intensity
        self.display_program["raySteps"].value = int(max(32, min(160, self.args.ray_steps)))

    def _spin_cpu_workers(self) -> None:
        for worker_id in range(self.args.cpu_workers):
            thread = threading.Thread(
                target=self._cpu_burner,
                args=(worker_id,),
                name=f"quantum-annealer-{worker_id}",
                daemon=True,
            )
            thread.start()
            self.cpu_threads.append(thread)

    def _cpu_burner(self, worker_id: int) -> None:
        matrix_n = self.args.cpu_matrix
        rng = np.random.default_rng(worker_id + 42)
        a = rng.standard_normal((matrix_n, matrix_n), dtype=np.float32)
        b = rng.standard_normal((matrix_n, matrix_n), dtype=np.float32)
        while not self.stop_event.is_set():
            np.matmul(a, b, out=a)
            norm = np.linalg.norm(a)
            if norm > 0:
                a /= norm
            real = rng.standard_normal(matrix_n * 8, dtype=np.float32)
            imag = rng.standard_normal(matrix_n * 8, dtype=np.float32)
            signal = (real + 1j * imag).astype(np.complex64)
            _ = np.fft.fft(signal)
            a, b = b, a

    def render(self, time_value: float, frame_time: float) -> None:
        if frame_time > 0:
            current_fps = 1.0 / frame_time
            if self.fps_estimate <= 0:
                self.fps_estimate = current_fps
            else:
                self.fps_estimate = self.fps_estimate * 0.92 + current_fps * 0.08

        if self.thermal_hold is not None:
            if self.hold_texture is None or self.hold_texture.size != self.wnd.buffer_size:
                self._build_hold_texture(self.thermal_hold.lines)
            self._display_target().use()
            self.ctx.viewport = (0, 0, *self.wnd.buffer_size)
            self.hold_texture.use(location=0)
            self.quad.render(self.message_program)
            return

        if self.state_textures[0].size != self.wnd.buffer_size:
            self._init_simulation_resources()
            self._sync_static_uniforms()

        self._refresh_window_title()

        animated_time = time_value * self.args.anim_speed
        substeps = max(1, self.args.substeps)
        for _ in range(substeps):
            self._step_simulation(animated_time)
        self._render_display(animated_time)

    def _step_simulation(self, animated_time: float) -> None:
        current = self.state_textures[self.active_state]
        next_index = 1 - self.active_state
        target_fbo = self.framebuffers[next_index]
        target_fbo.use()
        self.ctx.viewport = (0, 0, *current.size)
        current.use(location=0)
        self.update_program["time"].value = animated_time
        self.update_program["feed"].value = self.args.feed
        self.update_program["kill"].value = self.args.kill
        self.quad.render(self.update_program)
        self.active_state = next_index

    def _render_display(self, animated_time: float) -> None:
        self._display_target().use()
        self.ctx.viewport = (0, 0, *self.wnd.buffer_size)
        self.state_textures[self.active_state].use(location=0)
        self.display_program["time"].value = animated_time
        self.quad.render(self.display_program)
        self._render_hud()

    def _render_hud(self) -> None:
        if self.args.no_hud:
            return
        self._build_hud_texture()
        if self.hud_texture is None:
            return
        self._display_target().use()
        self.ctx.viewport = (0, 0, *self.wnd.buffer_size)
        self.hud_texture.use(location=0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.quad.render(self.message_program)
        self.ctx.disable(moderngl.BLEND)

    def resize(self, width: int, height: int):
        if self.thermal_hold is not None:
            self._build_hold_texture(self.thermal_hold.lines)
            return
        self._init_simulation_resources()
        self._sync_static_uniforms()

    def destroy(self) -> None:
        self.stop_event.set()
        self.thermal_thread_stop.set()
        for thread in self.cpu_threads:
            thread.join(timeout=1.0)
        self.cpu_threads.clear()
        if self.hold_texture is not None:
            self.hold_texture.release()
            self.hold_texture = None
        if self.hud_texture is not None:
            self.hud_texture.release()
            self.hud_texture = None
        if self.offscreen_framebuffer is not None:
            self.offscreen_framebuffer.release()
            self.offscreen_framebuffer = None
        if self.offscreen_texture is not None:
            self.offscreen_texture.release()
            self.offscreen_texture = None
        super().destroy()

    @classmethod
    def add_arguments(cls, parser) -> None:
        parser.add_argument("--width", type=int, default=cls.window_size[0], help="Render width")
        parser.add_argument("--height", type=int, default=cls.window_size[1], help="Render height")
        parser.add_argument("--feed", type=float, default=0.035, help="Neural matrix base feed rate")
        parser.add_argument("--kill", type=float, default=0.060, help="Neural matrix base kill rate")
        parser.add_argument("--diff-u", type=float, default=0.16, help="Diffusion rate for Substrate")
        parser.add_argument("--diff-v", type=float, default=0.08, help="Diffusion rate for Active Nodes")
        parser.add_argument("--time-step", dest="time_step", type=float, default=1.0, help="Simulation time step")
        parser.add_argument("--substeps", type=int, default=8, help="Simulation steps per frame")
        parser.add_argument("--laplace-scale", type=float, default=1.0, help="Global laplacian multiplier")
        parser.add_argument("--anim-speed", type=float, default=1.0, help="Global animation multiplier")
        parser.add_argument("--exposure", type=float, default=1.5, help="Display exposure")
        parser.add_argument("--glow", type=float, default=1.1, help="Display glow factor")
        parser.add_argument("--gamma", type=float, default=1.2, help="Display gamma correction")
        parser.add_argument("--contour-contrast", type=float, default=0.75, help="Contour emphasis strength")
        parser.add_argument("--ray-steps", type=int, default=120, help="Maximum raymarch steps per pixel")
        parser.add_argument("--fx-intensity", type=float, default=1.2, help="Matrix bloom and structural intensity")
        parser.add_argument("--camera-speed", type=float, default=1.0, help="Cinematic camera speed multiplier")
        parser.add_argument("--cpu-workers", type=int, default=0, help="Quantum annealing burner thread count")
        parser.add_argument("--cpu-matrix", type=int, default=896, help="Quantum annealing burner matrix size")
        parser.add_argument("--tile-size", type=int, default=10, help="Base resolution downscale factor for fluid sim")
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
