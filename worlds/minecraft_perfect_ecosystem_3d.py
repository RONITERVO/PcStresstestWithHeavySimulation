"""Flagship Minecraft-like living ecosystem candidate world.

State texture contract:
- R: moisture, drainage pressure, river/lake/ocean energy.
- G: biomass, canopy, crops, reeds, and post-disturbance recovery.
- B: mostly stable elevation/geology with only tiny erosion/soil deltas.
- A: settlement light, ore/cave glow, fire heat, and torch pulses.

Ecosystem rules are deliberately local and bounded: water moves downhill and
returns through rain/snowmelt, biomass grows in terrain-aware biomes, roots
reduce erosion, canopy retains moisture, and fire consumes dry biomass.
"""
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
uniform float noiseStrength;
uniform float parameterDrift;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    vec2 texel = 1.0 / resolution;

    // Channels:
    // R: water/moisture and river pressure
    // G: biome vegetation and canopy density
    // B: terrain elevation
    // A: villages, torches, caves, and ore light
    vec4 c = texture(stateTex, uv);
    vec4 r = texture(stateTex, uv + vec2(texel.x, 0.0));
    vec4 l = texture(stateTex, uv - vec2(texel.x, 0.0));
    vec4 t = texture(stateTex, uv + vec2(0.0, texel.y));
    vec4 b = texture(stateTex, uv - vec2(0.0, texel.y));
    vec4 tr = texture(stateTex, uv + texel);
    vec4 tl = texture(stateTex, uv + vec2(-texel.x, texel.y));
    vec4 br = texture(stateTex, uv + vec2(texel.x, -texel.y));
    vec4 bl = texture(stateTex, uv - texel);

    float lapR = ((r.r + l.r + t.r + b.r) * 0.2 + (tr.r + tl.r + br.r + bl.r) * 0.05 - c.r) * laplaceScale;
    float lapG = ((r.g + l.g + t.g + b.g) * 0.2 + (tr.g + tl.g + br.g + bl.g) * 0.05 - c.g) * laplaceScale;
    float lapB = ((r.b + l.b + t.b + b.b) * 0.2 + (tr.b + tl.b + br.b + bl.b) * 0.05 - c.b) * laplaceScale;
    float lapA = ((r.a + l.a + t.a + b.a) * 0.2 + (tr.a + tl.a + br.a + bl.a) * 0.05 - c.a);

    vec2 gradH = vec2(r.b - l.b, t.b - b.b);
    float higherInflow =
        max(r.b - c.b, 0.0) * r.r +
        max(l.b - c.b, 0.0) * l.r +
        max(t.b - c.b, 0.0) * t.r +
        max(b.b - c.b, 0.0) * b.r +
        0.707 * (
            max(tr.b - c.b, 0.0) * tr.r +
            max(tl.b - c.b, 0.0) * tl.r +
            max(br.b - c.b, 0.0) * br.r +
            max(bl.b - c.b, 0.0) * bl.r
        );
    float lowerDrain = c.r * (
        max(c.b - r.b, 0.0) +
        max(c.b - l.b, 0.0) +
        max(c.b - t.b, 0.0) +
        max(c.b - b.b, 0.0) +
        0.707 * (
            max(c.b - tr.b, 0.0) +
            max(c.b - tl.b, 0.0) +
            max(c.b - br.b, 0.0) +
            max(c.b - bl.b, 0.0)
        )
    );
    float catchmentFlow = clamp(higherInflow * 0.38 - lowerDrain * 0.20, -1.0, 1.0);
    float slope = clamp(length(gradH) * 5.0, 0.0, 1.0);
    float latitude = 1.0 - abs(uv.y * 2.0 - 1.0);
    float season = sin(time * 0.018 + uv.x * 6.28318) * 0.5 + 0.5;
    float daylight = sin(time * 0.048) * 0.5 + 0.5;
    float waterBelow = smoothstep(0.20, 0.72, c.r);
    float flood = smoothstep(0.76, 0.96, c.r);
    float drought = smoothstep(0.62, 0.94, 1.0 - c.r);
    float canopy = smoothstep(0.26, 0.92, c.g);
    float dryFuel = smoothstep(0.40, 0.88, c.g) * smoothstep(0.42, 0.92, 1.0 - c.r);
    float settlementLight = smoothstep(0.50, 0.92, c.a) * smoothstep(0.24, 0.76, c.g) * smoothstep(0.24, 0.80, c.r) * (1.0 - smoothstep(0.68, 0.96, c.b));
    float oreLight = smoothstep(0.50, 0.86, c.a) * smoothstep(0.58, 0.98, c.b) * (1.0 - smoothstep(0.34, 0.82, c.g) * 0.58);
    float fireBand = smoothstep(0.62, 0.96, c.a) * dryFuel * (1.0 - settlementLight * 0.95) * (1.0 - oreLight * 0.88);
    float fireHeat = clamp(fireBand, 0.0, 1.0);
    float temperature = clamp(latitude * 0.68 + season * 0.18 + fireHeat * 0.20 - c.b * 0.42, 0.0, 1.0);
    float localNoise = (hash12(uv * resolution + floor(time * 0.42)) - 0.5) * noiseStrength;
    float storm = smoothstep(0.62, 0.96, sin(time * 0.029 + hash12(floor(uv * 12.0)) * 6.28318) * 0.5 + 0.5);
    float rain = storm * (0.25 + latitude * 0.50) * (1.0 - smoothstep(0.70, 0.98, c.b));
    float snowpack = smoothstep(0.58, 0.96, c.b) * (1.0 - smoothstep(0.10, 0.38, temperature));
    float snowmelt = snowpack * smoothstep(0.36, 0.92, temperature + season * 0.22) * 0.0040;
    float valleyWetness = smoothstep(0.05, 0.32, higherInflow) * (1.0 - slope * 0.42);
    float wetlandSuitability = smoothstep(0.46, 0.84, c.r) * (1.0 - slope * 0.35) * (1.0 - smoothstep(0.70, 0.98, c.b));
    float suitability = smoothstep(0.28, 0.74, c.r) * smoothstep(0.20, 0.92, temperature) * (1.0 - slope * 0.58) * (1.0 - flood * 0.46);

    float localFeed = feed * 0.46 + 0.016 + rain * 0.010 + waterBelow * 0.006 + localNoise * 0.030;
    float localKill = kill * 0.44 + 0.021 + drought * 0.014 + flood * 0.010 + fireHeat * 0.020 + snowpack * 0.004 + parameterDrift * 0.28;
    float reaction = c.r * c.g * c.g * (0.34 + suitability * 0.50 + canopy * 0.16 + valleyWetness * 0.10);
    vec2 gradR = vec2(r.r - l.r, t.r - b.r);
    float spreadTowardMoisture = -dot(gradR, vec2(r.g - l.g, t.g - b.g)) * 0.18;
    float seedBank = smoothstep(0.24, 0.82, (r.g + l.g + t.g + b.g) * 0.25);
    float fireDamage = smoothstep(0.18, 0.88, fireHeat) * smoothstep(0.32, 0.90, c.g) * (0.034 + drought * 0.010) * (1.0 - rain * 0.42);
    float floodDamage = flood * smoothstep(0.36, 0.92, c.g) * 0.010;
    float alpineStress = smoothstep(0.68, 0.96, c.b) * smoothstep(0.42, 1.0, c.g) * (1.0 - snowpack * 0.35) * 0.005;
    float canopyRetention = canopy * rain * (1.0 - flood) * 0.0016;
    float evaporation = (0.0012 + temperature * 0.0022) * daylight * smoothstep(0.30, 0.95, c.r) * (1.0 - canopy * 0.45);

    float dr = clamp(diffU * lapR - reaction * 0.31 + localFeed * (1.0 - c.r) + catchmentFlow * 0.010 + rain * 0.0032 + snowmelt + canopyRetention - evaporation - fireHeat * 0.0024, -0.020, 0.020);
    float dg = clamp(diffV * lapG + reaction + spreadTowardMoisture + suitability * 0.0035 + wetlandSuitability * 0.0018 + seedBank * suitability * 0.0016 - (localFeed + localKill) * c.g - fireDamage - floodDamage - alpineStress, -0.026, 0.024);
    float rootArmor = smoothstep(0.32, 0.86, c.g);
    float flowEnergy = clamp((higherInflow + lowerDrain) * 0.42 * slope, 0.0, 1.0);
    float erosion = flowEnergy * 0.0037 * (1.0 - rootArmor);
    float soilReturn = (c.g - 0.42) * 0.00045 * suitability;
    float db = clamp(lapB * 0.0011 - erosion + soilReturn - fireHeat * 0.00028, -0.0016, 0.0012);

    float neighborHeat =
        r.a * smoothstep(0.40, 0.88, r.g) * smoothstep(0.42, 0.92, 1.0 - r.r) +
        l.a * smoothstep(0.40, 0.88, l.g) * smoothstep(0.42, 0.92, 1.0 - l.r) +
        t.a * smoothstep(0.40, 0.88, t.g) * smoothstep(0.42, 0.92, 1.0 - t.r) +
        b.a * smoothstep(0.40, 0.88, b.g) * smoothstep(0.42, 0.92, 1.0 - b.r) +
        tr.a * smoothstep(0.40, 0.88, tr.g) * smoothstep(0.42, 0.92, 1.0 - tr.r) +
        tl.a * smoothstep(0.40, 0.88, tl.g) * smoothstep(0.42, 0.92, 1.0 - tl.r) +
        br.a * smoothstep(0.40, 0.88, br.g) * smoothstep(0.42, 0.92, 1.0 - br.r) +
        bl.a * smoothstep(0.40, 0.88, bl.g) * smoothstep(0.42, 0.92, 1.0 - bl.r);
    float lightning = step(0.99935, hash12(uv * resolution + floor(time * 1.10))) * dryFuel * storm;
    float spreadFire = smoothstep(0.54, 1.75, neighborHeat) * dryFuel * step(0.54, hash12(uv * resolution + floor(time * 1.65)));
    vec2 eventCell = floor(uv * resolution / 8.0);
    float oreAnchor = step(0.9875, hash12(eventCell + vec2(37.0, 13.0))) * smoothstep(0.55, 0.98, c.b) * (1.0 - smoothstep(0.34, 0.78, c.g));
    float villageAnchor = step(0.9835, hash12(eventCell + vec2(107.0, 91.0))) * smoothstep(0.24, 0.70, c.g) * smoothstep(0.24, 0.76, c.r) * (1.0 - smoothstep(0.68, 0.96, c.b));
    float anchorCap = max(0.0, max(oreAnchor * (0.58 - c.a), villageAnchor * (0.64 - c.a)));
    float heatDecay = 0.0014 + fireHeat * (0.0040 + c.r * 0.0048) + (1.0 - max(oreAnchor, villageAnchor)) * 0.0007;
    float da = clamp(lapA * 0.035 - c.a * heatDecay + lightning * 0.055 + spreadFire * 0.026 + anchorCap * 0.006, -0.030, 0.030);

    fragColor = vec4(
        clamp(c.r + dr * dt, 0.0, 1.0),
        clamp(c.g + dg * dt, 0.0, 1.0),
        clamp(c.b + db * dt, 0.0, 1.0),
        clamp(c.a + da * dt, 0.0, 1.0)
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
uniform float colorShift;
uniform float cameraSpeed;
uniform float fxIntensity;
uniform vec3 cameraOffset;
uniform vec2 cameraYawPitch;
uniform float cameraZoom;
uniform int raySteps;

#define MAX_STEPS 160
#define MAX_DIST 66.0
#define SURF_DIST 0.0025
#define BLOCK_SIZE 0.72
#define MAP_SCALE 0.036

const int MAT_WATER = 0;
const int MAT_TERRAIN = 1;
const int MAT_TRUNK = 2;
const int MAT_LEAVES = 3;
const int MAT_PLANKS = 4;
const int MAT_ROOF = 5;
const int MAT_GLASS = 6;
const int MAT_TORCH = 7;
const int MAT_CAVE = 8;
const int MAT_ORE = 9;
const int MAT_REEDS = 10;
const int MAT_SHRUB = 11;
const int MAT_FIRE = 12;
const int MAT_PATH = 13;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float chance(vec2 p, float threshold) {
    return step(hash12(p), threshold);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash12(i), hash12(i + vec2(1.0, 0.0)), f.x),
        mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), f.x),
        f.y
    );
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    mat2 rot = mat2(0.866, -0.5, 0.5, 0.866);
    for (int i = 0; i < 5; i++) {
        v += a * noise(p);
        p = rot * p * 2.02 + vec2(17.31, 91.73);
        a *= 0.5;
    }
    return v;
}

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

float boxSdf(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

void takeShape(inout float bestD, inout int bestMat, float candidateD, int candidateMat) {
    if (candidateD < bestD) {
        bestD = candidateD;
        bestMat = candidateMat;
    }
}

vec4 worldState(vec2 cell) {
    return textureLod(stateTex, fract((cell + 0.5) * BLOCK_SIZE * MAP_SCALE), 0.0);
}

float terrainHeight(vec4 state, vec2 cell) {
    float macro = fbm(cell * 0.036 + vec2(3.0, 19.0));
    float cliff = pow(1.0 - abs(fbm(cell * 0.086 + vec2(41.0, 7.0)) * 2.0 - 1.0), 1.85);
    float riverCut = smoothstep(0.58, 0.94, state.r) * (1.0 - smoothstep(0.62, 0.96, state.b));
    float mountainLift = smoothstep(0.66, 0.96, state.b) * cliff * 1.45;
    float soilLift = smoothstep(0.38, 0.88, state.g) * (1.0 - smoothstep(0.70, 0.96, state.b)) * 0.38;
    float raw = state.b * 7.7 + macro * 2.35 + mountainLift + soilLift - riverCut * 1.15;
    return floor(raw) * (BLOCK_SIZE * 0.52) + 0.48;
}

float waterPresence(vec4 state, float h, float sea) {
    float lowLand = 1.0 - smoothstep(sea - 0.08, sea + 0.28, h);
    float wetValley = smoothstep(0.56, 0.88, state.r) * (1.0 - smoothstep(sea + 1.45, sea + 2.35, h));
    float basin = smoothstep(0.42, 0.72, state.r) * (1.0 - smoothstep(0.64, 0.92, state.b));
    return max(lowLand * wetValley, basin * 0.74);
}

float geologyVein(vec2 cell, vec4 state) {
    float caveLattice = abs(sin(cell.x * 0.43 + state.b * 7.0) * cos(cell.y * 0.37 - state.b * 5.0));
    float fractured = 1.0 - caveLattice;
    float seamNoise = noise(cell * 0.19 + vec2(state.b * 5.0, state.a * 2.0));
    return clamp(fractured * 0.76 + seamNoise * 0.24, 0.0, 1.0);
}

float blockGrid(vec3 p) {
    vec3 f = fract(p / BLOCK_SIZE);
    vec3 q = min(f, 1.0 - f);
    float edge = min(min(q.x, q.y), q.z);
    return 1.0 - smoothstep(0.018, 0.055, edge);
}

float daylightAmount() {
    return smoothstep(-0.22, 0.20, sin(time * 0.048));
}

float rainAmount(vec2 worldXZ) {
    vec2 region = floor(worldXZ * MAP_SCALE * 12.0);
    float localStorm = smoothstep(0.62, 0.96, sin(time * 0.029 + hash12(region) * 6.28318) * 0.5 + 0.5);
    float front = smoothstep(0.58, 0.88, fbm(worldXZ * 0.006 + vec2(time * 0.018, 23.4)));
    return clamp(localStorm * 0.68 + front * 0.32, 0.0, 1.0);
}

float map(in vec3 p, out int matID, out vec4 stateOut) {
    vec2 cell = floor(p.xz / BLOCK_SIZE);
    vec4 state = worldState(cell);
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float sea = 1.56;
    float h = terrainHeight(state, cell);

    float bestD = p.y - h;
    int bestMat = MAT_TERRAIN;
    stateOut = state;

    float waterMask = waterPresence(state, h, sea);
    if (waterMask > 0.04) {
        float ripple = (floor(noise(cell * 0.17 + vec2(time * 0.055, -time * 0.033)) * 3.0) - 1.0) * 0.010 * safeFx;
        float dWater = p.y - (sea + ripple);
        if (dWater < bestD) {
            bestD = dWater * 0.86;
            bestMat = MAT_WATER;
        }
    }

    vec2 center2 = (cell + 0.5) * BLOCK_SIZE;
    float aboveWater = step(sea + 0.20, h);
    float lowEnoughForVillage = 1.0 - step(sea + 2.55, h);
    float highMountain = smoothstep(sea + 2.65, sea + 4.35, h);
    float wetShore = (1.0 - smoothstep(0.14, 0.74, abs(h - sea))) * smoothstep(0.36, 0.88, state.r);
    float dryFuel = smoothstep(0.40, 0.86, state.g) * smoothstep(0.38, 0.86, 1.0 - state.r);
    float villageField = smoothstep(0.44, 0.82, state.a) * aboveWater * lowEnoughForVillage * smoothstep(0.16, 0.68, state.r) * (1.0 - highMountain * 0.90);
    float exposedStone = smoothstep(sea + 1.15, sea + 3.90, h) * smoothstep(0.52, 0.96, state.b) * (1.0 - wetShore * 0.62);
    float caveVein = geologyVein(cell, state);
    float caveField = smoothstep(0.70, 0.94, caveVein) * exposedStone * aboveWater * (1.0 - villageField * 0.85);
    float oreField = smoothstep(0.58, 0.96, state.a) * smoothstep(0.56, 0.88, caveVein) * exposedStone * (1.0 - smoothstep(0.40, 0.86, state.g) * 0.45);
    float pathBand = 1.0 - smoothstep(0.035, 0.150, min(abs(fract(cell.x * 0.25) - 0.5), abs(fract(cell.y * 0.25) - 0.5)));
    if (villageField * pathBand > 0.16) {
        bestMat = MAT_PATH;
    }

    float treeMask = chance(cell + vec2(11.7, 31.0), 0.255) * smoothstep(0.30, 0.82, state.g) * smoothstep(0.22, 0.78, state.r) * aboveWater * (1.0 - highMountain * 0.72) * (1.0 - villageField * 0.82);
    if (treeMask > 0.5) {
        float trunkHeight = mix(0.62, 1.08, hash12(cell + vec2(3.1, 77.7)));
        float trunk = boxSdf(
            vec3(p.x - center2.x, p.y - (h + trunkHeight * 0.52), p.z - center2.y),
            vec3(BLOCK_SIZE * 0.15, trunkHeight * 0.52, BLOCK_SIZE * 0.15)
        );
        takeShape(bestD, bestMat, trunk, MAT_TRUNK);

        float leafRise = h + trunkHeight + 0.55;
        float canopyJitter = hash12(cell + vec2(92.0, 7.0)) - 0.5;
        float leaves = boxSdf(
            vec3(p.x - center2.x + canopyJitter * 0.08, p.y - leafRise, p.z - center2.y - canopyJitter * 0.08),
            vec3(BLOCK_SIZE * 0.86, 0.76, BLOCK_SIZE * 0.86)
        );
        takeShape(bestD, bestMat, leaves, MAT_LEAVES);

        float crown = boxSdf(
            vec3(p.x - center2.x, p.y - (leafRise + 0.48), p.z - center2.y),
            vec3(BLOCK_SIZE * 0.55, 0.38, BLOCK_SIZE * 0.55)
        );
        takeShape(bestD, bestMat, crown, MAT_LEAVES);

        float sideCanopy = boxSdf(
            vec3(p.x - center2.x - canopyJitter * 0.24, p.y - (leafRise + 0.08), p.z - center2.y + canopyJitter * 0.24),
            vec3(BLOCK_SIZE * 0.68, 0.34, BLOCK_SIZE * 0.68)
        );
        takeShape(bestD, bestMat, sideCanopy, MAT_LEAVES);
    }

    float shrubMask = chance(cell + vec2(73.0, 18.0), 0.265) * smoothstep(0.20, 0.68, state.g) * aboveWater * (1.0 - highMountain) * (1.0 - villageField);
    if (shrubMask > 0.5) {
        float shrubHeight = mix(0.16, 0.36, hash12(cell + vec2(5.0, 12.0)));
        float shrub = boxSdf(
            vec3(p.x - center2.x, p.y - (h + shrubHeight), p.z - center2.y),
            vec3(BLOCK_SIZE * 0.36, shrubHeight, BLOCK_SIZE * 0.36)
        );
        takeShape(bestD, bestMat, shrub, MAT_SHRUB);
    }

    float reedMask = chance(cell + vec2(9.0, 144.0), 0.42) * wetShore * smoothstep(0.18, 0.64, state.g);
    if (reedMask > 0.16) {
        vec2 reedOffset = vec2(hash12(cell + vec2(4.1, 5.7)) - 0.5, hash12(cell + vec2(8.9, 6.3)) - 0.5) * BLOCK_SIZE * 0.24;
        float reedHeight = mix(0.30, 0.68, hash12(cell + vec2(71.0, 2.0)));
        float reeds = boxSdf(
            vec3(p.x - center2.x - reedOffset.x, p.y - (max(h, sea) + reedHeight * 0.52), p.z - center2.y - reedOffset.y),
            vec3(BLOCK_SIZE * 0.045, reedHeight * 0.52, BLOCK_SIZE * 0.045)
        );
        takeShape(bestD, bestMat, reeds, MAT_REEDS);
    }

    float activeFire = smoothstep(0.64, 0.96, state.a) * dryFuel * aboveWater * (1.0 - wetShore) * (1.0 - villageField * 0.96) * (1.0 - highMountain * 0.70);
    float fireMask = activeFire;
    if (fireMask > 0.25) {
        float flicker = 0.78 + 0.22 * hash12(cell + vec2(floor(time * 9.0)));
        float fire = boxSdf(
            vec3(p.x - center2.x, p.y - (h + 0.28 * flicker), p.z - center2.y),
            vec3(BLOCK_SIZE * 0.24, 0.28 * flicker, BLOCK_SIZE * 0.24)
        );
        takeShape(bestD, bestMat, fire, MAT_FIRE);
    }

    float villageMask = chance(cell + vec2(107.0, 13.0), 0.092) * villageField;
    if (villageMask > 0.5) {
        float yawChoice = step(0.5, hash12(cell + vec2(4.0, 91.0)));
        vec2 rel2 = p.xz - center2;
        rel2 = mix(rel2, vec2(rel2.y, rel2.x), yawChoice);
        float hut = boxSdf(
            vec3(rel2.x, p.y - (h + 0.54), rel2.y),
            vec3(BLOCK_SIZE * 0.78, 0.54, BLOCK_SIZE * 0.62)
        );
        takeShape(bestD, bestMat, hut, MAT_PLANKS);

        float roof = boxSdf(
            vec3(rel2.x, p.y - (h + 1.12), rel2.y),
            vec3(BLOCK_SIZE * 0.90, 0.24, BLOCK_SIZE * 0.74)
        );
        takeShape(bestD, bestMat, roof, MAT_ROOF);

        float glass = boxSdf(
            vec3(rel2.x - BLOCK_SIZE * 0.48, p.y - (h + 0.66), rel2.y + BLOCK_SIZE * 0.16),
            vec3(BLOCK_SIZE * 0.08, 0.12, BLOCK_SIZE * 0.16)
        );
        takeShape(bestD, bestMat, glass, MAT_GLASS);
    }

    float torchMask = chance(cell + vec2(51.0, 91.0), 0.050) * smoothstep(0.26, 0.84, state.a) * aboveWater * (1.0 - highMountain * 0.40);
    if (torchMask > 0.5) {
        float torch = boxSdf(
            vec3(p.x - center2.x, p.y - (h + 0.38), p.z - center2.y),
            vec3(BLOCK_SIZE * 0.075, 0.38, BLOCK_SIZE * 0.075)
        );
        takeShape(bestD, bestMat, torch, MAT_TORCH);
    }

    float caveMask = caveField * chance(cell + vec2(199.0, 23.0), 0.38);
    if (caveMask > 0.5) {
        vec2 rel = p.xz - center2;
        float yawChoice = step(0.5, hash12(cell + vec2(12.0, 44.0)));
        rel = mix(rel, vec2(rel.y, rel.x), yawChoice);
        float caveVoid = boxSdf(
            vec3(rel.x, p.y - (h - 0.12), rel.y),
            vec3(BLOCK_SIZE * 0.50, 0.58, BLOCK_SIZE * 0.12)
        );
        if (caveVoid < 0.10 && p.y < h + 0.44) {
            bestD = max(p.y - h, -caveVoid * 0.86);
            bestMat = MAT_CAVE;
        }

        float backWall = boxSdf(
            vec3(rel.x, p.y - (h - 0.16), rel.y + BLOCK_SIZE * 0.18),
            vec3(BLOCK_SIZE * 0.36, 0.42, BLOCK_SIZE * 0.035)
        );
        takeShape(bestD, bestMat, backWall, MAT_CAVE);
    }

    float oreMask = oreField * (0.40 + caveField * 0.75) * chance(cell + vec2(17.0, 181.0), 0.28);
    if (oreMask > 0.5) {
        float side = step(0.5, hash12(cell + vec2(61.0, 9.0))) * 2.0 - 1.0;
        float ore = boxSdf(
            vec3(p.x - center2.x - side * BLOCK_SIZE * 0.23, p.y - (h + 0.02), p.z - center2.y),
            vec3(BLOCK_SIZE * 0.18, 0.13, BLOCK_SIZE * 0.16)
        );
        takeShape(bestD, bestMat, ore, MAT_ORE);
    }

    matID = bestMat;
    return bestD;
}

vec3 calcNormal(in vec3 p) {
    const vec2 e = vec2(0.032, 0.0);
    int mat; vec4 state;
    return normalize(vec3(
        map(p + e.xyy, mat, state) - map(p - e.xyy, mat, state),
        map(p + e.yxy, mat, state) - map(p - e.yxy, mat, state),
        map(p + e.yyx, mat, state) - map(p - e.yyx, mat, state)
    ));
}

float calcSoftShadow(in vec3 ro, in vec3 rd, float mint, float tmax, float k) {
    float res = 1.0;
    float t = mint;
    int mat; vec4 state;
    for (int i = 0; i < 34; i++) {
        float h = map(ro + rd * t, mat, state);
        res = min(res, k * h / t);
        t += clamp(h, 0.025, 0.55);
        if (res < 0.004 || t > tmax) {
            break;
        }
    }
    return clamp(res, 0.0, 1.0);
}

float calcAO(vec3 pos, vec3 nor) {
    float occ = 0.0;
    float sca = 1.0;
    int mat; vec4 state;
    for (int i = 0; i < 5; i++) {
        float h = 0.02 + 0.12 * float(i);
        float d = map(pos + nor * h, mat, state);
        occ += (h - d) * sca;
        sca *= 0.70;
    }
    return clamp(1.0 - 2.15 * occ, 0.0, 1.0);
}

float squareDisc(vec2 p, float halfSize, float feather) {
    float d = max(abs(p.x), abs(p.y));
    return 1.0 - smoothstep(halfSize, halfSize + feather, d);
}

float cloudMask(vec3 rd) {
    if (rd.y <= 0.04) {
        return 0.0;
    }
    vec2 cloudPos = rd.xz / max(rd.y, 0.05) * 1.85 + vec2(time * 0.016, -time * 0.006);
    vec2 blockCloud = floor(cloudPos * 5.0);
    float body = smoothstep(0.50, 0.72, fbm(blockCloud * 0.12));
    float band = smoothstep(0.12, 0.34, rd.y) * (1.0 - smoothstep(0.68, 0.90, rd.y));
    return body * band;
}

vec3 skyColorForRay(vec3 rd, vec3 lightDir, float safeFx, float rain, float daylight) {
    float skyRise = smoothstep(-0.18, 0.86, rd.y);
    vec3 clearLow = vec3(0.46, 0.68, 0.94);
    vec3 clearHigh = vec3(0.76, 0.91, 1.00);
    vec3 stormLow = vec3(0.32, 0.40, 0.46);
    vec3 stormHigh = vec3(0.55, 0.61, 0.66);
    vec3 daySky = mix(mix(clearLow, clearHigh, skyRise), mix(stormLow, stormHigh, skyRise), rain * 0.72);
    vec3 nightLow = vec3(0.020, 0.035, 0.080);
    vec3 nightHigh = vec3(0.050, 0.075, 0.145);
    vec3 sky = mix(mix(nightLow, nightHigh, skyRise), daySky, daylight);

    vec3 sunRight = normalize(cross(lightDir, vec3(0.0, 1.0, 0.0)));
    vec3 sunUp = normalize(cross(sunRight, lightDir));
    vec2 sunUv = vec2(dot(rd, sunRight), dot(rd, sunUp));
    float disc = squareDisc(sunUv, 0.060, 0.018) * smoothstep(0.88, 0.995, dot(rd, lightDir));
    vec3 discColor = mix(vec3(0.46, 0.60, 1.00), vec3(1.0, 0.86, 0.40), daylight);
    sky += discColor * disc * (0.45 + daylight * (1.10 + safeFx)) * (1.0 - rain * 0.75);

    float clouds = cloudMask(rd);
    sky = mix(sky, vec3(0.82, 0.88, 0.94), clouds * (0.40 + rain * 0.14));
    sky = mix(sky, vec3(0.38, 0.44, 0.50), clouds * rain * 0.44);
    return sky;
}

vec3 materialColor(int matID, vec3 pos, vec3 nor, vec3 rd, vec4 state, float grid, float safeFx, float contourPower, out vec3 emission) {
    emission = vec3(0.0);
    float cellShade = hash12(floor(pos.xz / BLOCK_SIZE) + floor(pos.y / BLOCK_SIZE) * 13.7);

    if (matID == MAT_WATER) {
        vec3 shallow = vec3(0.12, 0.56, 0.84);
        vec3 deep = vec3(0.025, 0.17, 0.38);
        vec3 water = mix(shallow, deep, clamp(state.r, 0.0, 1.0));
        water += vec3(0.20, 0.45, 0.65) * grid * 0.12;
        return water;
    }
    if (matID == MAT_TRUNK) {
        return mix(vec3(0.34, 0.18, 0.07), vec3(0.57, 0.34, 0.14), step(0.5, fract(pos.y * 3.4 + cellShade)));
    }
    if (matID == MAT_LEAVES) {
        vec3 oak = vec3(0.13, 0.40, 0.11);
        vec3 lush = vec3(0.35, 0.62, 0.17);
        vec3 pale = vec3(0.48, 0.56, 0.40);
        vec3 leaves = mix(oak, lush, clamp(state.g + cellShade * 0.20, 0.0, 1.0));
        return mix(leaves, pale, smoothstep(0.10, 0.28, state.r) * smoothstep(0.12, 0.34, state.g) * 0.22);
    }
    if (matID == MAT_REEDS) {
        vec3 reed = mix(vec3(0.29, 0.42, 0.12), vec3(0.64, 0.70, 0.26), cellShade);
        emission += reed * smoothstep(0.70, 1.0, state.g) * 0.035 * safeFx;
        return reed;
    }
    if (matID == MAT_SHRUB) {
        vec3 moss = mix(vec3(0.18, 0.32, 0.10), vec3(0.46, 0.55, 0.18), clamp(state.g + cellShade * 0.18, 0.0, 1.0));
        vec3 dryMoss = vec3(0.47, 0.42, 0.20);
        return mix(moss, dryMoss, smoothstep(0.50, 0.90, 1.0 - state.r));
    }
    if (matID == MAT_PLANKS) {
        return mix(vec3(0.52, 0.32, 0.14), vec3(0.78, 0.54, 0.25), step(0.5, fract(pos.x * 2.5 + pos.y * 1.7)));
    }
    if (matID == MAT_ROOF) {
        return mix(vec3(0.35, 0.11, 0.07), vec3(0.68, 0.22, 0.12), cellShade);
    }
    if (matID == MAT_GLASS) {
        emission += vec3(0.92, 0.72, 0.34) * (0.45 + glow * 0.35) * safeFx;
        return vec3(0.95, 0.78, 0.42);
    }
    if (matID == MAT_TORCH) {
        emission += vec3(3.0, 1.30, 0.24) * (0.85 + glow) * safeFx;
        return vec3(0.44, 0.22, 0.07);
    }
    if (matID == MAT_CAVE) {
        emission += vec3(0.04, 0.10, 0.18) * glow * safeFx;
        return vec3(0.015, 0.013, 0.012);
    }
    if (matID == MAT_ORE) {
        vec3 diamond = vec3(0.15, 0.95, 0.92);
        vec3 gold = vec3(0.96, 0.70, 0.24);
        vec3 redstone = vec3(1.0, 0.14, 0.08);
        vec3 ore = mix(mix(diamond, gold, step(0.55, cellShade)), redstone, smoothstep(0.86, 0.98, cellShade));
        emission += ore * (0.38 + glow * 0.45) * safeFx;
        return ore;
    }
    if (matID == MAT_FIRE) {
        vec3 ember = mix(vec3(1.0, 0.32, 0.05), vec3(1.0, 0.86, 0.18), smoothstep(0.20, 0.86, fract(pos.y * 3.0 + time * 7.0 + cellShade)));
        emission += ember * (1.75 + glow * 1.35) * safeFx;
        return ember * 0.18;
    }
    if (matID == MAT_PATH) {
        vec3 path = mix(vec3(0.43, 0.31, 0.17), vec3(0.68, 0.52, 0.28), cellShade);
        path = mix(path, vec3(0.32, 0.28, 0.22), smoothstep(0.55, 0.95, state.a) * 0.28);
        return path * (1.0 - grid * 0.10 * contourPower);
    }

    float top = smoothstep(0.34, 0.76, nor.y);
    float altitude = smoothstep(3.10, 5.10, pos.y);
    float snow = smoothstep(4.78, 5.70, pos.y) * top * smoothstep(0.68, 0.98, state.b);
    float dry = smoothstep(0.30, 0.72, (1.0 - state.r) * (1.0 - state.g));
    float scarFuel = smoothstep(0.22, 0.76, state.g) * smoothstep(0.42, 0.92, 1.0 - state.r) * (1.0 - smoothstep(0.58, 0.96, state.b) * 0.62);
    float fireScar = smoothstep(0.54, 0.98, state.a) * scarFuel;
    float stoneFace = altitude * (1.0 - top * 0.28);
    float shore = smoothstep(0.45, 0.72, state.r) * (1.0 - altitude);

    vec3 grass = mix(vec3(0.15, 0.42, 0.10), vec3(0.46, 0.68, 0.18), clamp(state.g * 1.15 + cellShade * 0.12, 0.0, 1.0));
    vec3 dirt = mix(vec3(0.32, 0.19, 0.08), vec3(0.49, 0.31, 0.14), cellShade);
    vec3 sand = vec3(0.70, 0.60, 0.34);
    vec3 stone = mix(vec3(0.34, 0.35, 0.33), vec3(0.58, 0.59, 0.56), cellShade);
    vec3 snowColor = vec3(0.88, 0.94, 0.98);

    vec3 terrain = mix(dirt, grass, top);
    terrain = mix(terrain, sand, max(shore, dry * (1.0 - state.g)) * (1.0 - altitude));
    terrain = mix(terrain, stone, stoneFace);
    terrain = mix(terrain, snowColor, snow);
    terrain = mix(terrain, vec3(0.11, 0.10, 0.085), fireScar * (1.0 - snow) * 0.62);
    terrain *= 1.0 - grid * 0.14 * contourPower;
    return terrain;
}

vec3 waterRippleNormal(vec3 pos, float safeFx) {
    vec2 cell = floor(pos.xz / BLOCK_SIZE);
    vec2 drift = vec2(time * 0.055, -time * 0.037);
    float east = noise((cell + vec2(1.0, 0.0)) * 0.23 + drift);
    float west = noise((cell - vec2(1.0, 0.0)) * 0.23 + drift);
    float north = noise((cell + vec2(0.0, 1.0)) * 0.23 + drift.yx);
    float south = noise((cell - vec2(0.0, 1.0)) * 0.23 + drift.yx);
    float crossChop = (hash12(cell + floor(time * 0.75)) - 0.5) * 0.055;
    vec2 slope = vec2(east - west, north - south) * (0.15 + safeFx * 0.055);
    return normalize(vec3(-slope.x + crossChop, 1.0, -slope.y - crossChop));
}

float cloudTerrainShadow(vec3 pos, vec3 lightDir, float rain) {
    vec2 cloudCell = floor((pos.xz + lightDir.xz * 20.0 + vec2(time * 0.82, -time * 0.34)) * 0.10);
    float cloud = smoothstep(0.52, 0.78, fbm(cloudCell * 0.21 + vec2(11.0, 37.0)));
    float sunUp = smoothstep(0.10, 0.82, lightDir.y);
    return mix(1.0, mix(0.86, 0.66, rain), cloud * sunUp);
}

vec3 traceWaterReflection(vec3 ro, vec3 rd, vec3 lightDir, float safeFx, float daylight, float rain, out float hitWeight) {
    float t = 0.08;
    hitWeight = 0.0;
    for (int i = 0; i < 18; i++) {
        vec3 pos = ro + rd * t;
        int matID;
        vec4 state;
        float h = map(pos, matID, state);
        if (h < max(SURF_DIST * 1.6, 0.002 * t)) {
            vec3 nor = calcNormal(pos);
            float grid = blockGrid(pos);
            vec3 emission;
            vec3 albedo = materialColor(matID, pos, nor, rd, state, grid, safeFx, 0.74, emission);
            float dif = clamp(dot(nor, lightDir), 0.0, 1.0);
            float skyLight = clamp(0.44 + 0.56 * nor.y, 0.0, 1.0);
            vec3 lit = albedo * (
                0.16 +
                dif * mix(0.62, 1.34, daylight) * mix(vec3(0.52, 0.62, 1.00), vec3(1.0, 0.92, 0.76), daylight) +
                skyLight * mix(0.16, 0.34, daylight)
            ) + emission;
            float fog = clamp(1.0 - exp(-0.0020 * t * t), 0.0, 0.80);
            hitWeight = 1.0 - fog;
            return mix(lit, skyColorForRay(rd, lightDir, safeFx, rain, daylight), fog);
        }
        t += clamp(h, 0.06, 0.96);
        if (t > 18.0) {
            break;
        }
    }
    return skyColorForRay(rd, lightDir, safeFx, rain, daylight);
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float contourPower = clamp(contourContrast, 0.0, 1.8);
    int maxRaySteps = clamp(raySteps, 32, MAX_STEPS);
    float stressDetail = smoothstep(1.02, 1.55, safeFx);

    float camTime = time * 0.092 * max(cameraSpeed, 0.0);
    vec3 ro = vec3(
        camTime * 5.35 + sin(camTime * 0.27) * 2.4,
        6.35 + sin(camTime * 0.41) * 0.55,
        camTime * 4.85 + cos(camTime * 0.31) * 2.2
    );
    vec3 ta = vec3(ro.x + 5.7, 2.65 + sin(camTime * 0.30) * 0.26, ro.z + 5.1);
    ro += cameraOffset;
    ta += cameraOffset;

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.15) * 0.030);
    vec3 rd = ca * cameraInputRay(p, 2.0);

    float rain = rainAmount(ro.xz);
    float daylight = daylightAmount();
    float sunPhase = time * 0.048 + colorShift * 0.02;
    vec3 lightDir = normalize(vec3(0.54 + sin(time * 0.020 + colorShift * 0.12) * 0.10, mix(-0.32, 0.82, daylight), -0.42 + cos(sunPhase) * 0.10));
    vec3 skyColor = skyColorForRay(rd, lightDir, safeFx, rain, daylight);

    float t = 0.0;
    int matID = -1;
    vec4 state = vec4(0.0);
    vec3 volumeGlow = vec3(0.0);

    for (int i = 0; i < MAX_STEPS; i++) {
        if (i >= maxRaySteps) {
            break;
        }
        vec3 pos = ro + rd * t;
        int currMat;
        vec4 currState;
        float h = map(pos, currMat, currState);

        if (currMat == MAT_TORCH || currMat == MAT_ORE || currMat == MAT_FIRE || currState.a > 0.58) {
            vec3 glowCol = mix(vec3(1.0, 0.58, 0.15), vec3(0.22, 0.95, 0.95), step(0.72, hash12(floor(pos.xz / BLOCK_SIZE))));
            glowCol = mix(glowCol, vec3(1.0, 0.28, 0.04), float(currMat == MAT_FIRE) * 0.85);
            float pulse = 0.78 + 0.22 * sin(time * 4.0 + currState.a * 9.0);
            volumeGlow += glowCol * (currState.a + float(currMat == MAT_TORCH) * 1.35 + float(currMat == MAT_ORE) * 0.45 + float(currMat == MAT_FIRE) * 1.10) * pulse * (0.007 + glow * 0.004) * (0.30 + safeFx * 0.50) * (1.0 + (1.0 - daylight) * 1.25) / (1.0 + abs(h) * 7.5);
        }

        if (h < max(SURF_DIST, 0.00125 * t) || t > MAX_DIST) {
            matID = currMat;
            state = currState;
            break;
        }
        t += clamp(h, mix(0.030, 0.020, stressDetail), mix(0.78, 0.54, stressDetail));
    }

    vec3 color = skyColor;
    if (t < MAX_DIST) {
        vec3 pos = ro + rd * t;
        vec3 nor = calcNormal(pos);
        if (matID == MAT_WATER) {
            vec3 waterNor = waterRippleNormal(pos, safeFx);
            nor = normalize(mix(nor, waterNor, 0.62 * smoothstep(0.55, 1.45, safeFx)));
        }
        float sha = calcSoftShadow(pos + nor * 0.025, lightDir, 0.05, 14.0, 10.0);
        sha *= cloudTerrainShadow(pos, lightDir, rain);
        float ao = calcAO(pos, nor);
        if (matID == MAT_WATER) {
            sha = max(sha, 0.62);
            ao = mix(1.0, ao, 0.35);
        }
        float dif = clamp(dot(nor, lightDir), 0.0, 1.0) * sha;
        float skyLight = clamp(0.45 + 0.55 * nor.y, 0.0, 1.0);
        float fre = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 3.0);
        float grid = blockGrid(pos);

        vec3 emission;
        vec3 matColor = materialColor(matID, pos, nor, rd, state, grid, safeFx, contourPower, emission);

        if (matID == MAT_WATER) {
            vec3 ref = reflect(rd, nor);
            float spe = pow(clamp(dot(ref, lightDir), 0.0, 1.0), 42.0) * sha;
            float fresnel = pow(1.0 - max(dot(nor, -rd), 0.0), 4.0);
            emission += vec3(0.84, 0.95, 1.0) * spe * (0.50 + safeFx * 0.28);
            vec3 reflectedScene = skyColorForRay(ref, lightDir, safeFx, rain, daylight);
            float reflectedHit = 0.0;
            if (safeFx > 1.04 && maxRaySteps >= 104) {
                reflectedScene = traceWaterReflection(pos + nor * 0.045, ref, lightDir, safeFx, daylight, rain, reflectedHit);
            }
            float reflectionGain = mix(0.24, 0.48, stressDetail);
            matColor = mix(matColor, reflectedScene, clamp(fresnel * reflectionGain + reflectedHit * 0.10 * stressDetail, 0.0, 0.56));
            matColor += volumeGlow * (0.10 + stressDetail * 0.08);
        }

        vec3 lin = vec3(0.0);
        lin += mix(0.22, 1.78, daylight) * dif * mix(vec3(0.50, 0.62, 1.00), vec3(1.0, 0.92, 0.78), daylight) * ao;
        lin += mix(0.18, 0.68, daylight) * skyLight * mix(vec3(0.10, 0.16, 0.30), vec3(0.42, 0.55, 0.72), daylight) * ao;
        lin += 0.24 * fre * vec3(0.84, 0.92, 1.0);

        color = matColor * lin + emission;
        float fog = 1.0 - exp(-0.00082 * t * t);
        fog = clamp(fog + rain * smoothstep(22.0, 58.0, t) * 0.14, 0.0, 0.78);
        color = mix(color, skyColor, fog);
    }

    color += volumeGlow * (1.0 + safeFx * 0.64);

    float rainStreak = 0.0;
    if (rain > 0.05) {
        vec2 streakUv = gl_FragCoord.xy / resolution.y;
        float streaks = step(0.985, hash12(floor(vec2(streakUv.x * 220.0 + time * 18.0, streakUv.y * 76.0 - time * 34.0))));
        rainStreak = streaks * rain * smoothstep(-0.10, 0.55, rd.y);
    }
    color = mix(color, vec3(0.70, 0.80, 0.88), rainStreak * 0.18);

    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(color, vec3(1.0 / max(gamma, 0.2)));

    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.60 + 0.40 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.22);

    fragColor = vec4(color, 1.0);
}
"""


WORLD_SEED = 20260612


def _ellipse(
    tile_x: np.ndarray,
    tile_y: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
) -> np.ndarray:
    distance = ((tile_x - cx) / max(rx, 1.0)) ** 2 + ((tile_y - cy) / max(ry, 1.0)) ** 2
    return np.clip(1.0 - distance, 0.0, 1.0)


def _downhill_drainage(
    height: np.ndarray,
    rainfall: np.ndarray,
    lake_mask: np.ndarray,
) -> np.ndarray:
    """Approximate catchments by routing rainfall to lower neighboring tiles."""
    rows, cols = height.shape
    flow = (0.20 + rainfall * 0.85 + lake_mask * 1.35).astype(np.float32, copy=True)
    order = np.argsort(height, axis=None)[::-1]
    offsets = (
        (-1, 0, 1.00),
        (1, 0, 1.00),
        (0, -1, 1.00),
        (0, 1, 1.00),
        (-1, -1, 1.35),
        (-1, 1, 1.35),
        (1, -1, 1.35),
        (1, 1, 1.35),
    )

    for flat_index in order:
        y = int(flat_index // cols)
        x = int(flat_index - y * cols)
        current_h = float(height[y, x])
        best_y = y
        best_x = x
        best_drop = 0.0

        for dy, dx, cost in offsets:
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
                continue
            drop = (current_h - float(height[ny, nx])) / cost
            if drop > best_drop:
                best_drop = drop
                best_y = ny
                best_x = nx

        if best_drop > 0.0015:
            transfer = float(flow[y, x]) * min(0.92, 0.76 + best_drop * 1.8)
            flow[best_y, best_x] += transfer

    log_flow = np.log1p(np.maximum(flow, 0.0))
    low = float(np.quantile(log_flow, 0.64))
    high = float(np.quantile(log_flow, 0.985))
    drainage = (log_flow - low) / max(high - low, 1e-4)
    return np.clip(drainage, 0.0, 1.0).astype(np.float32)


def seed_field(width_px: int, height_px: int, tile_size: int) -> np.ndarray:
    tile_size = max(2, int(tile_size))
    tiles_x = max(1, int(np.ceil(width_px / tile_size)))
    tiles_y = max(1, int(np.ceil(height_px / tile_size)))

    rng = np.random.default_rng(WORLD_SEED)
    tile_y, tile_x = np.meshgrid(
        np.arange(tiles_y, dtype=np.float32),
        np.arange(tiles_x, dtype=np.float32),
        indexing="ij",
    )
    x_norm = tile_x / max(tiles_x - 1, 1)
    y_norm = tile_y / max(tiles_y - 1, 1)
    latitude = 1.0 - np.abs(y_norm * 2.0 - 1.0)

    height = (
        0.43
        + 0.18 * np.sin(x_norm * 6.6 + np.cos(y_norm * 4.8) * 1.75)
        + 0.13 * np.cos(y_norm * 8.6 - 0.55)
        + 0.09 * np.sin((x_norm + y_norm * 0.72) * 13.0)
        + 0.05 * np.cos((x_norm - y_norm) * 24.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.018
    )
    lake_mask = np.zeros((tiles_y, tiles_x), dtype=np.float32)

    for _ in range(max(8, (tiles_x * tiles_y) // 2800)):
        blob = _ellipse(
            tile_x,
            tile_y,
            rng.uniform(0.0, tiles_x),
            rng.uniform(0.0, tiles_y),
            rng.uniform(max(6.0, tiles_x * 0.05), max(14.0, tiles_x * 0.20)),
            rng.uniform(max(6.0, tiles_y * 0.05), max(14.0, tiles_y * 0.20)),
        )
        height += blob * rng.uniform(-0.10, 0.24)

    for _ in range(max(3, (tiles_x * tiles_y) // 5200)):
        basin = _ellipse(
            tile_x,
            tile_y,
            rng.uniform(0.0, tiles_x),
            rng.uniform(0.0, tiles_y),
            rng.uniform(max(3.0, tiles_x * 0.025), max(7.0, tiles_x * 0.090)),
            rng.uniform(max(3.0, tiles_y * 0.025), max(7.0, tiles_y * 0.090)),
        )
        lake_mask = np.maximum(lake_mask, basin * rng.uniform(0.45, 0.95))
        height -= basin * rng.uniform(0.07, 0.16)

    mountain_ridges = np.abs(np.sin(x_norm * 20.0 + np.cos(y_norm * 12.0) * 2.6))
    ridge_lift = np.clip(0.36 - mountain_ridges, 0.0, 1.0)
    height += ridge_lift * (0.16 + 0.05 * np.sin(y_norm * 16.0))

    rainfall = np.clip(
        0.20
        + latitude * 0.34
        + 0.10 * np.sin(x_norm * 7.0 - y_norm * 5.0)
        + 0.06 * np.cos((x_norm + y_norm) * 17.0),
        0.0,
        1.0,
    )
    drainage = _downhill_drainage(height, rainfall, lake_mask)
    meander = (
        0.76
        + 0.15 * np.sin(x_norm * 19.0 + y_norm * 5.5)
        + 0.09 * np.cos(y_norm * 15.0 - x_norm * 6.5)
    )
    river_mask = np.clip(drainage * meander, 0.0, 1.0)
    lowland_flow = river_mask * np.clip((0.82 - height) * 1.65, 0.0, 1.0)
    height -= lowland_flow * 0.19
    height = np.clip(height, 0.0, 1.0)

    grad_x = np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)
    grad_y = np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)
    slope = np.clip(np.sqrt(grad_x * grad_x + grad_y * grad_y) * 4.0, 0.0, 1.0)

    sea_level = 0.395
    ocean = np.clip((sea_level - height) / 0.16, 0.0, 1.0)
    shore = np.clip(1.0 - np.abs(height - sea_level) / 0.075, 0.0, 1.0)
    river = np.clip(lowland_flow * 0.82 + drainage * 0.26 + lake_mask * 0.76, 0.0, 1.0)

    moisture = np.clip(
        ocean * 0.84
        + river * 0.72
        + shore * 0.24
        + drainage * 0.12
        + rainfall * 0.28
        + (1.0 - slope) * 0.05
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.025,
        0.0,
        1.0,
    )

    temperature = np.clip(latitude * 0.74 + 0.14 * np.sin(x_norm * 3.2 + 0.6) - height * 0.32, 0.0, 1.0)
    wetland = np.clip((river + shore * 0.75) * (1.0 - ocean * 0.45), 0.0, 1.0)
    arid = np.clip((1.0 - moisture) * (0.55 + (1.0 - latitude) * 0.35), 0.0, 1.0)
    forest_noise = (
        0.38
        + 0.28 * np.sin(x_norm * 18.5 + y_norm * 5.5)
        + 0.16 * np.cos(y_norm * 19.0 - x_norm * 4.2)
        + 0.08 * np.sin((x_norm - y_norm) * 31.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.040
    )
    fertility = np.clip(
        (1.0 - ocean)
        * (0.22 + moisture * 0.68 + temperature * 0.22 + wetland * 0.18)
        * (1.0 - slope * 0.65)
        * (1.0 - np.clip(height - 0.78, 0.0, 1.0) * 2.2),
        0.0,
        1.0,
    )
    vegetation = np.clip(
        fertility * (0.46 + forest_noise * 0.62)
        + wetland * 0.25
        - arid * 0.22,
        0.0,
        1.0,
    )

    village = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    village_candidates = np.argwhere(
        (ocean < 0.20)
        & ((shore > 0.16) | (river > 0.28))
        & (vegetation > 0.20)
        & (slope < 0.40)
        & (height > sea_level + 0.025)
        & (height < 0.66)
    )
    if len(village_candidates) > 0:
        village_count = min(max(6, (tiles_x * tiles_y) // 2200), len(village_candidates))
        for candidate_index in rng.choice(len(village_candidates), size=village_count, replace=False):
            cy, cx = village_candidates[candidate_index]
            radius = int(rng.integers(3, 6))
            y0 = max(cy - radius, 0)
            y1 = min(cy + radius + 1, tiles_y)
            x0 = max(cx - radius, 0)
            x1 = min(cx + radius + 1, tiles_x)
            patch_y, patch_x = np.meshgrid(
                np.arange(y0, y1, dtype=np.float32),
                np.arange(x0, x1, dtype=np.float32),
                indexing="ij",
            )
            influence = np.clip(
                1.0 - np.sqrt((patch_x - cx) ** 2 + (patch_y - cy) ** 2) / max(radius + 0.5, 1.0),
                0.0,
                1.0,
            )
            village[y0:y1, x0:x1] = np.maximum(village[y0:y1, x0:x1], influence * rng.uniform(0.58, 0.94))

    vegetation = np.clip(vegetation * (1.0 - village * 0.55) + wetland * 0.06, 0.0, 1.0)

    cave_lattice = np.abs(np.sin(x_norm * 33.0 + height * 5.0) * np.cos(y_norm * 29.0 - height * 4.0))
    ore = np.clip(1.0 - cave_lattice * 3.2, 0.0, 1.0) * np.clip((height - 0.47) * 2.05, 0.0, 1.0) * (0.55 + slope * 0.55)
    mountain_light = np.clip((height - 0.74) * 2.0, 0.0, 1.0) * rng.random((tiles_y, tiles_x), dtype=np.float32) * 0.18
    dry_forest = np.argwhere((vegetation > 0.56) & (moisture < 0.50) & (height > sea_level + 0.06) & (ocean < 0.15))
    fire = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    if len(dry_forest) > 0:
        spark_count = min(max(3, (tiles_x * tiles_y) // 4200), len(dry_forest))
        for candidate_index in rng.choice(len(dry_forest), size=spark_count, replace=False):
            cy, cx = dry_forest[candidate_index]
            radius = int(rng.integers(1, 4))
            y0 = max(cy - radius, 0)
            y1 = min(cy + radius + 1, tiles_y)
            x0 = max(cx - radius, 0)
            x1 = min(cx + radius + 1, tiles_x)
            patch_y, patch_x = np.meshgrid(
                np.arange(y0, y1, dtype=np.float32),
                np.arange(x0, x1, dtype=np.float32),
                indexing="ij",
            )
            influence = np.clip(
                1.0 - np.sqrt((patch_x - cx) ** 2 + (patch_y - cy) ** 2) / max(radius + 0.5, 1.0),
                0.0,
                1.0,
            )
            fire[y0:y1, x0:x1] = np.maximum(fire[y0:y1, x0:x1], influence * rng.uniform(0.48, 0.82))
    vegetation = np.clip(vegetation * (1.0 - fire * 0.28), 0.0, 1.0)
    light = np.clip(village + ore * 0.58 + fire * 0.78 + mountain_light, 0.0, 1.0)

    field = np.stack(
        [
            moisture.astype(np.float32),
            vegetation.astype(np.float32),
            height.astype(np.float32),
            light.astype(np.float32),
        ],
        axis=-1,
    )
    field = np.nan_to_num(field, nan=0.0, posinf=1.0, neginf=0.0)
    field = np.clip(field, 0.0, 1.0).astype(np.float32)

    pixels = np.repeat(np.repeat(field, tile_size, axis=0), tile_size, axis=1)
    return pixels[:height_px, :width_px].copy()


SPEC = WorldSpec(
    id="minecraft-perfect-ecosystem-3d",
    display_name="Minecraft Perfect Ecosystem",
    window_title="Garage Life Lab - Minecraft Perfect Ecosystem",
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={
        "ray_steps": 152,
        "fx_intensity": 1.28,
        "contour_contrast": 1.14,
        "camera_speed": 0.72,
        "tile_size": 8,
        "substeps": 10,
        "glow": 1.22,
        "exposure": 1.28,
        "gamma": 1.16,
    },
    stability_notes=("candidate", "living ecosystem", "GPU stress", "blocky 3D"),
    hud_subtitle="MINECRAFT PERFECT ECOSYSTEM",
    preview_palette=("#5bb8ff", "#83c45d", "#2e7a31", "#c1a24d", "#6b6f6a", "#42d8c7", "#ff6a26"),
)
