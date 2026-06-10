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

// PRNG
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
    // R (x): Nutrients/Moisture (U)
    // G (y): Biomass/Alien Life (V)
    // B (z): Topography/Height
    // A (w): Geothermal Heat & Cosmic Energy

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

    vec2 gradH = vec2(r.z - l.z, t.z - b_.z);

    // Advection: matter flows downhill
    float flow = 1.2;
    float advectU = dot(gradH, vec2(r.x - l.x, t.x - b_.x)) * flow;
    float advectV = dot(gradH, vec2(r.y - l.y, t.y - b_.y)) * flow;

    // Cosmic Events (Meteors)
    float eventInterval = 8.0;
    float eventId = floor(time / eventInterval);
    float eventLocalTime = fract(time / eventInterval) * eventInterval;
    vec2 mTarget = getEventTarget(eventId, 42.0);
    float mDist = length(uv - mTarget);

    float meteorStrike = 0.0;
    float cratering = 0.0;
    float shockwave = 0.0;

    float strikePulse = exp(-pow((eventLocalTime - 7.08) * 48.0, 2.0));
    float craterCore = 1.0 - smoothstep(0.0, 0.026, mDist);
    float heatCore = 1.0 - smoothstep(0.012, 0.070, mDist);
    float shockCore = 1.0 - smoothstep(0.035, 0.100, mDist);
    float ejectaRing = smoothstep(0.018, 0.032, mDist) * (1.0 - smoothstep(0.032, 0.052, mDist));
    cratering = strikePulse * (-0.010 * craterCore + 0.006 * ejectaRing);
    meteorStrike = strikePulse * 0.040 * heatCore;
    shockwave = -c.y * 0.020 * strikePulse * shockCore;

    // Geothermal Activity (Volcanoes)
    // Heat builds up naturally in fault lines (initialized as initial w > 0.5)
    float geothermalDrift = (hash12(uv + time * 0.01) - 0.49) * 0.01;
    float heatGain = (c.w > 0.6 ? 0.005 : -0.001) + geothermalDrift;

    float eruptionH = 0.0;
    float eruptionHeat = 0.0;
    float eruptionKill = 0.0;

    vec2 eruptionCell = floor(uv * resolution / 4.0);
    float eruptionSeed = hash12(eruptionCell + floor(time * 0.5));
    float eruptionPulse = smoothstep(0.92, 1.0, c.w) * step(0.9975, eruptionSeed);
    eruptionH = 0.012 * eruptionPulse;
    eruptionHeat = 0.045 * eruptionPulse;
    eruptionKill = -c.y * 0.018 * eruptionPulse;

    // Gray-Scott Reaction with ecological coupling
    float reaction = c.x * c.y * c.y;

    // Rainforest coupling: high moisture (c.x) and moderate heat (c.w) boosts feed, lowers kill
    float ecoFeed = feed + (c.x * 0.02) - abs(c.w - 0.3) * 0.01;
    float ecoKill = kill + c.w * 0.015;

    float du = (diffU * lapU * laplaceScale) - reaction + ecoFeed * (1.0 - c.x) - advectU;
    float dv = (diffV * lapV * laplaceScale) + reaction - (ecoFeed + ecoKill) * c.y - advectV + shockwave + eruptionKill;

    // Topography shifts slowly via life (erosion/growth) and violently via events
    float dh = lapH * 0.01 + (c.y * 0.005 - 0.001) * dt + cratering + eruptionH;

    // Heat diffusion and accumulation
    float dw = (0.2 * lapA) + heatGain + meteorStrike + eruptionHeat - (c.x * 0.002); // Water cools heat

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

float tsunamiWave(vec2 xz, float hTerrain, float safeFx, out float crest, out float trough) {
    vec2 waveDir = normalize(vec2(0.78, 0.42));
    vec2 crossDir = vec2(-waveDir.y, waveDir.x);
    float along = dot(xz, waveDir);
    float cross = dot(xz, crossDir);
    float travel = along * 0.28 - time * (0.92 + safeFx * 0.18);
    travel += fbm(vec2(cross * 0.035, along * 0.018) + time * 0.035) * 2.6;

    float cycle = sin(travel);
    crest = pow(max(cycle, 0.0), 5.0);
    trough = pow(max(-sin(travel - 0.55), 0.0), 2.2);

    float shoreAmplify = smoothstep(0.22, 1.75, hTerrain);
    float shelfDrawdown = smoothstep(0.05, 1.05, hTerrain);
    float longSwell = sin(along * 0.075 - time * 0.38 + fbm(xz * 0.025) * 4.0) * 0.075;
    float chop = (fbm(xz * 0.75 + vec2(time * 0.24, -time * 0.18)) - 0.5) * 0.055;

    float runup = crest * mix(0.55, 2.25, shoreAmplify) * safeFx;
    float drawdown = trough * mix(0.34, 0.76, shelfDrawdown) * safeFx;
    return longSwell + chop + runup - drawdown;
}

float map(in vec3 p, out int matID, out vec4 stateOut) {
    vec2 mapUV = p.xz * 0.03;
    vec4 state = textureLod(stateTex, mapUV, 0.0);

    // R: Moisture, G: Biomass, B: Height, A: Heat
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float detailFx = smoothstep(0.2, 1.6, safeFx);

    // Terrain definition
    float baseH = state.z * mix(2.4, 3.2, detailFx);

    // Rainforest Canopy (Fractal displacement based on Biomass G)
    float canopy = smoothstep(0.3, 0.9, state.y);
    float canopyDetail = fbm(p.xz * 15.0) * mix(0.22, 0.48, detailFx) * canopy;

    // Volcanic Peaks & Craters (Displacement based on Heat A)
    float rockyDetail = fbm(p.xz * 8.0) * mix(0.12, 0.30, detailFx) * smoothstep(0.4, 1.0, state.w);

    float continentLift = smoothstep(0.38, 0.72, fbm(p.xz * 0.075 + vec2(19.0, 7.0)));
    float ridgeRelief = pow(fbm(p.xz * 0.22 + vec2(5.0, 13.0)), 3.0) * mix(0.25, 1.15, detailFx);
    float microRelief = (fbm(p.xz * 2.2 + vec2(3.0)) - 0.5) * mix(0.06, 0.22, detailFx);
    float hTerrain = baseH + canopyDetail + rockyDetail + ridgeRelief + microRelief + continentLift * mix(0.45, 1.35, detailFx);
    float dTerrain = p.y - hTerrain;

    // Dynamic tsunami water: deep drawdown reveals land, crest runup climbs coastlines.
    float crest; float trough;
    float hWater = 0.26 + tsunamiWave(p.xz, hTerrain, safeFx, crest, trough);
    float dWater = p.y - hWater;

    if (dTerrain < dWater) {
        matID = 1; // Terrain/Organics
        stateOut = state;
        return dTerrain * 0.5; // Under-relax
    } else {
        matID = 0; // Water
        stateOut = state;
        return dWater * 0.8;
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

vec2 getEventTarget(float id, float seed) {
    return vec2(hash12(vec2(id, seed)), hash12(vec2(id, seed + 1.0)));
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float contourPower = clamp(contourContrast, 0.0, 1.6);
    int maxRaySteps = clamp(raySteps, 32, MAX_STEPS);

    float camTime = time * 0.12 * max(cameraSpeed, 0.01);
    vec3 ro = vec3(camTime * 6.0, 6.6 + sin(camTime * 0.4) * 0.8, camTime * 5.0);
    vec3 ta = vec3(ro.x + 5.0, 1.45, ro.z + 5.0 + sin(camTime * 0.8) * 2.0);

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.25) * 0.15);
    vec3 rd = ca * normalize(vec3(p.xy, 2.0));

    vec3 lightDir = normalize(vec3(0.7, 0.5, -0.5));

    // Meteor Sky Event
    float eventInterval = 8.0;
    float eventId = floor(time / eventInterval);
    float eventLocalTime = fract(time / eventInterval) * eventInterval;

    vec3 skyColor = mix(vec3(0.02, 0.05, 0.1), vec3(0.15, 0.08, 0.25), rd.y * 0.5 + 0.5);
    float sun = pow(max(0.0, dot(rd, lightDir)), 150.0);
    skyColor += vec3(1.0, 0.8, 0.5) * sun * safeFx;

    // Meteor entry flash
    if (eventLocalTime > 6.0 && eventLocalTime < 7.5) {
        vec2 mTargetUV = getEventTarget(eventId, 42.0);
        vec3 mTargetWorld = vec3(mTargetUV.x / 0.03, 0.0, mTargetUV.y / 0.03);

        vec3 meteorStart = mTargetWorld + vec3(20.0, 40.0, -20.0);
        vec3 meteorEnd = mTargetWorld;

        float dropPhase = smoothstep(6.0, 7.0, eventLocalTime);
        vec3 mPos = mix(meteorStart, meteorEnd, dropPhase);

        vec3 mDir = normalize(meteorEnd - meteorStart);
        float mProj = max(0.0, dot(rd, normalize(mPos - ro)));

        float flash = exp(-pow(eventLocalTime - 7.1, 2.0) * 150.0);
        skyColor += vec3(1.0, 0.5, 0.2) * pow(mProj, 800.0) * 10.0 * (1.0 - flash);
        skyColor += vec3(1.0, 0.9, 0.7) * flash * 5.0 * safeFx; // Global flash
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

        // Volumetrics
        if (currMat == 1) {
            // Rainforest Bioluminescence (G)
            float bio = smoothstep(0.4, 0.9, currState.y);
            vec3 bioGlow = mix(vec3(0.0, 1.0, 0.6), vec3(0.8, 0.2, 1.0), currState.x);
            volumeGlow += bioGlow * bio * (0.015 + glow * 0.005) * safeFx / (1.0 + abs(h) * 8.0);

            // Volcanic Lava Volumetrics (A)
            float heat = smoothstep(0.7, 1.0, currState.w);
            vec3 lavaGlow = vec3(1.0, 0.3, 0.05);
            volumeGlow += lavaGlow * heat * (0.02 + glow * 0.01) * safeFx / (1.0 + abs(h) * 5.0);
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

        float occ = clamp(0.4 + 0.6 * nor.y, 0.0, 1.0);
        float sha = calcShadow(pos + nor * 0.02, lightDir);
        float dif = clamp(dot(nor, lightDir), 0.0, 1.0) * sha;
        float sky = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
        float fre = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 3.0);

        vec3 matColor;
        vec3 emission = vec3(0.0);

        if (matID == 1) {
            // Terrain Base
            vec3 rock = vec3(0.16, 0.12, 0.17);
            vec3 soil = vec3(0.22, 0.25, 0.15);
            matColor = mix(rock, soil, smoothstep(0.3, 0.7, nor.y));

            // Rainforests (Biomass G)
            vec3 flora = mix(vec3(0.04, 0.34, 0.22), vec3(0.12, 0.74, 0.72), state.x);
            matColor = mix(matColor, flora, smoothstep(0.2, 0.8, state.y));
            matColor = mix(matColor, vec3(0.34, 0.25, 0.18), smoothstep(0.52, 0.95, state.z) * 0.35);

            // Burned Earth from Heat (A)
            matColor = mix(matColor, vec3(0.05, 0.04, 0.04), smoothstep(0.5, 0.8, state.w));

            float contourLine = (1.0 - smoothstep(0.0, 0.035, abs(fract((state.z + pos.y * 0.018) * 18.0) - 0.5))) * contourPower;
            matColor += vec3(0.10, 0.18, 0.16) * contourLine * 0.34;

            // Lava Emission (A)
            float lavaMask = smoothstep(0.85, 1.0, state.w);
            emission += vec3(2.0, 0.5, 0.1) * lavaMask * (1.0 + 0.5 * sin(time * 5.0 + pos.x)) * safeFx;

            // Bioluminescence Surface (G)
            float bioMask = smoothstep(0.6, 1.0, state.y);
            emission += mix(vec3(0.0, 0.8, 0.5), vec3(0.9, 0.2, 0.8), state.x) * bioMask * (0.5 + glow) * safeFx;

        } else {
            // Alien Ocean
            float waveCrest; float waveTrough;
            float waterLift = tsunamiWave(pos.xz, state.z * 3.2, safeFx, waveCrest, waveTrough);
            float depth = clamp((pos.y - state.z * 3.5) * 0.22 + waveTrough * 0.35 + max(waterLift, 0.0) * 0.04, 0.0, 1.0);
            vec3 shallow = vec3(0.0, 0.72, 0.62);
            vec3 deep = vec3(0.035, 0.075, 0.22);
            matColor = mix(shallow, deep, depth);
            float shoreFoam = smoothstep(0.22, 0.88, state.z) * smoothstep(0.08, 0.95, waveCrest);
            float ripFoam = smoothstep(0.54, 0.84, fbm(pos.xz * 2.1 + vec2(time * 0.35)));
            float drawdownShine = smoothstep(0.28, 0.85, waveTrough) * smoothstep(0.25, 0.75, state.z);
            matColor = mix(matColor, vec3(0.72, 0.96, 0.88), clamp(shoreFoam * (0.35 + ripFoam * 0.35), 0.0, 0.7));
            matColor += vec3(0.11, 0.22, 0.18) * drawdownShine;
            matColor += vec3(0.05, 0.16, 0.14) * contourPower * (1.0 - smoothstep(0.0, 0.08, abs(state.z - 0.46)));

            vec3 ref = reflect(rd, nor);
            float spe = pow(clamp(dot(ref, lightDir), 0.0, 1.0), 64.0) * sha;
            emission += vec3(1.0) * spe * (0.8 + waveCrest * 0.8);
            emission += vec3(0.18, 0.55, 0.42) * waveCrest * safeFx * 0.55;

            // Reflected glow
            emission += volumeGlow * 0.2;
        }

        vec3 lin = vec3(0.0);
        lin += 2.0 * dif * vec3(1.0, 0.9, 0.8);
        lin += 0.6 * sky * vec3(0.2, 0.3, 0.5) * occ;
        lin += 0.3 * fre * vec3(0.8, 0.9, 1.0) * occ;

        color = matColor * lin + emission;

        float fog = 1.0 - exp(-0.002 * t * t);
        color = mix(color, skyColor, fog);
    }

    color += volumeGlow * (0.8 + safeFx * 0.4);

    // Post-Processing: ACES Tonemapping
    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(color, vec3(1.0 / max(gamma, 0.1)));

    // Vignette
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
    frame[..., 0] = np.clip(18.0 + 40.0 * (1.0 - y_gradient) + stripe * 12.0, 0, 255).astype(np.uint8)
    frame[..., 1] = np.clip(11.0 + 20.0 * (1.0 - y_gradient) + stripe * 8.0, 0, 255).astype(np.uint8)
    frame[..., 2] = np.clip(10.0 + 16.0 * (1.0 - y_gradient) + stripe * 7.0, 0, 255).astype(np.uint8)

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
            color = (255, 132, 78)
        elif index >= len(lines) - 2:
            color = (204, 214, 220)
        else:
            color = (248, 238, 228)
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

    panel_color = (6, 10, 14, 172)
    line_color = (36, 214, 192, 210)
    title_color = (238, 252, 247, 245)
    body_color = (190, 222, 224, 230)
    warn_color = (255, 186, 87, 238)

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
        color = warn_color if "OFFLINE" in line or "OFF" in line else body_color
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
    title = "Garage Life Lab - 3D Bio-World"
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
                        reasons.append("GPU SENSOR OFFLINE")
                else:
                    self.gpu_sensor_failures = 0
                    if gpu_temp > self.args.max_gpu_temp:
                        reasons.append(
                            f"GPU {gpu_temp:.1f}C OVER LIMIT {self.args.max_gpu_temp:.1f}C"
                        )

            if self.args.max_cpu_temp > 0:
                if cpu_temp is None:
                    notes.append("CPU SENSOR OFFLINE")
                elif cpu_temp > self.args.max_cpu_temp:
                    reasons.append(
                        f"CPU {cpu_temp:.1f}C OVER LIMIT {self.args.max_cpu_temp:.1f}C"
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

        log_lines = [f"[{timestamp}] THERMAL HOLD", *reasons]
        if notes:
            log_lines.extend(notes)
        gpu_temp = self.latest_temperatures.get("GPU")
        cpu_temp = self.latest_temperatures.get("CPU")
        if gpu_temp is not None:
            log_lines.append(f"LAST GPU TEMP {gpu_temp:.1f}C")
        if cpu_temp is not None:
            sensor_name = self.cpu_sensor_name or "CPU"
            log_lines.append(f"LAST {sensor_name.upper()} TEMP {cpu_temp:.1f}C")
        log_lines.append("LOADS STOPPED TO COOL SYSTEM")
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
            "THERMAL HOLD",
            *reasons,
            *notes,
            "LOADS STOPPED TO COOL SYSTEM",
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
            parts.append("CPU SENSOR OFFLINE")
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
            return f"{label} SENSOR OFFLINE"
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
            "3D ALIEN ECOSYSTEM",
            f"{width}X{height} MAP {tiles_x}X{tiles_y}",
            f"SIM STEP {self.args.substeps} RAYMARCH {self.args.ray_steps}",
            f"FX {self.args.fx_intensity:.1f} CAM {self.args.camera_speed:.1f}",
            f"CPU WORKERS {self.args.cpu_workers}",
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
        tile_y, tile_x = np.meshgrid(
            np.arange(tiles_y, dtype=np.float32),
            np.arange(tiles_x, dtype=np.float32),
            indexing="ij",
        )
        x_norm = tile_x / max(tiles_x - 1, 1)
        y_norm = tile_y / max(tiles_y - 1, 1)

        height = (
            0.45
            + 0.15 * np.sin(x_norm * 7.3 + 0.9)
            + 0.11 * np.cos(y_norm * 5.7 - 1.2)
            + 0.08 * np.sin((x_norm + y_norm) * 11.0)
            + 0.06 * np.cos((x_norm - y_norm) * 13.0)
            + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.025
        )

        continent_count = max(5, (tiles_x * tiles_y) // 4000)
        for _ in range(continent_count):
            cx = rng.uniform(0.0, tiles_x)
            cy = rng.uniform(0.0, tiles_y)
            rx = rng.uniform(max(5.0, tiles_x * 0.04), max(12.0, tiles_x * 0.18))
            ry = rng.uniform(max(5.0, tiles_y * 0.04), max(12.0, tiles_y * 0.18))
            distance = ((tile_x - cx) / rx) ** 2 + ((tile_y - cy) / ry) ** 2
            lift = np.clip(1.0 - distance, 0.0, 1.0)
            height += lift * rng.uniform(0.10, 0.24)

        trench_count = max(4, continent_count // 2)
        for _ in range(trench_count):
            cx = rng.uniform(0.0, tiles_x)
            cy = rng.uniform(0.0, tiles_y)
            rx = rng.uniform(max(6.0, tiles_x * 0.05), max(14.0, tiles_x * 0.16))
            ry = rng.uniform(max(6.0, tiles_y * 0.05), max(14.0, tiles_y * 0.16))
            distance = ((tile_x - cx) / rx) ** 2 + ((tile_y - cy) / ry) ** 2
            carve = np.clip(1.0 - distance, 0.0, 1.0)
            height -= carve * rng.uniform(0.08, 0.18)

        ridge_bands = np.sin(x_norm * 21.0 + np.cos(y_norm * 9.0) * 2.3)
        height += np.clip(ridge_bands - 0.45, 0.0, 1.0) * 0.08
        height = np.clip(height, 0.0, 1.0)

        sea_level = 0.46
        ocean = (height < sea_level).astype(np.float32)
        coast = np.clip(1.0 - np.abs(height - sea_level) / 0.07, 0.0, 1.0)
        latitude = 1.0 - np.abs(y_norm * 2.0 - 1.0)

        moisture = np.clip(
            0.16
            + ocean * 0.52
            + coast * 0.22
            + latitude * 0.12
            + 0.08 * np.sin(x_norm * 9.0 - y_norm * 6.0)
            + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.03,
            0.0,
            1.0,
        )

        biomass = np.clip(
            (1.0 - ocean)
            * (
                0.06
                + moisture * 0.62
                + latitude * 0.16
                - np.clip(height - 0.72, 0.0, 1.0) * 0.50
            ),
            0.0,
            1.0,
        )

        geothermal_faults = np.clip(
            np.abs(np.sin(x_norm * 18.0 + np.sin(y_norm * 22.0)) * 0.5) * 2.0, 0.0, 1.0
        )
        heat = np.clip(1.0 - geothermal_faults * 3.0, 0.0, 1.0)

        reaction_u = np.clip(1.0 - biomass * 0.36 + moisture * 0.08, 0.0, 1.0)
        reaction_v = np.clip(biomass * 0.82 + moisture * 0.12, 0.0, 1.0)

        tile_field = np.stack(
            [
                reaction_u.astype(np.float32),
                reaction_v.astype(np.float32),
                height.astype(np.float32),
                heat.astype(np.float32),
            ],
            axis=-1,
        )

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
                name=f"cpu-burner-{worker_id}",
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
        parser.add_argument("--feed", type=float, default=0.035, help="Gray-Scott base feed rate")
        parser.add_argument("--kill", type=float, default=0.060, help="Gray-Scott base kill rate")
        parser.add_argument("--diff-u", type=float, default=0.16, help="Diffusion rate for U")
        parser.add_argument("--diff-v", type=float, default=0.08, help="Diffusion rate for V")
        parser.add_argument("--time-step", dest="time_step", type=float, default=1.0, help="Simulation time step")
        parser.add_argument("--substeps", type=int, default=8, help="Simulation steps per frame")
        parser.add_argument("--laplace-scale", type=float, default=1.0, help="Global laplacian multiplier")
        parser.add_argument("--anim-speed", type=float, default=1.0, help="Global animation multiplier")
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
