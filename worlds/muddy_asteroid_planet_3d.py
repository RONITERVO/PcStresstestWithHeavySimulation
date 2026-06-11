"""World definition for Muddy Asteroid Planet."""
from __future__ import annotations

import numpy as np

from .spec import WorldSpec

SIM_FRAG_SHADER = r"""
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

DISPLAY_FRAG_SHADER = r"""
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
uniform vec3 cameraOffset;
uniform vec2 cameraYawPitch;
uniform float cameraZoom;
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

mat2 cameraRotate2(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return mat2(c, -s, s, c);
}

vec3 cameraInputRay(vec2 p, float lens) {
    float zoomedLens = lens * clamp(exp(cameraZoom), 0.35, 3.0);
    vec3 ray = normalize(vec3(p.xy, zoomedLens));
    ray.yz = cameraRotate2(cameraYawPitch.y) * ray.yz;
    ray.xz = cameraRotate2(cameraYawPitch.x) * ray.xz;
    return normalize(ray);
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

    // Dynamic Ocean Waves
    float waveTime = time * 1.5;
    float wave1 = sin(p.x * 1.7 + waveTime) * cos(p.z * 1.4 + waveTime * 0.8);
    float wave2 = sin(p.x * 3.8 - waveTime * 1.2) * cos(p.z * 3.2 + waveTime);
    float hWater = 0.42 + (wave1 * 0.025 + wave2 * 0.012) * safeFx;
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
    ro += cameraOffset;
    ta += cameraOffset;

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.25) * 0.15);
    vec3 rd = ca * cameraInputRay(p, 2.0);

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
            float depth = clamp((0.42 - state.z * 3.5) * 0.7, 0.0, 1.0);
            vec3 shallow = vec3(0.0, 0.6, 0.5);
            vec3 deep = vec3(0.05, 0.1, 0.25);
            matColor = mix(shallow, deep, depth);
            matColor += vec3(0.05, 0.16, 0.14) * contourPower * (1.0 - smoothstep(0.0, 0.08, abs(state.z - 0.46)));

            vec3 ref = reflect(rd, nor);
            float spe = pow(clamp(dot(ref, lightDir), 0.0, 1.0), 64.0) * sha;
            emission += vec3(1.0) * spe * 0.8;

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


def seed_field(width_px: int, height_px: int, tile_size: int) -> np.ndarray:
    tile_size = max(2, int(tile_size))
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
    return field


SPEC = WorldSpec(
    id='muddy-asteroid-planet-3d',
    display_name='Muddy Asteroid Planet',
    window_title='Garage Life Lab - Muddy Asteroid Planet',
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={},
    preview_image='assets/world_previews/muddy-asteroid-planet-3d.png',
    stability_notes=('heavy raymarch', 'experimental'),
    hud_subtitle='MUDDY ASTEROID PLANET',
    preview_palette=('#070807', '#1e1c16', '#41382b', '#6b5a43', '#90806a', '#b7c2a6', '#d7eef2'),
)
