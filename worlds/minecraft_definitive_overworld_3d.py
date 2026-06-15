"""Definitive Minecraft-inspired flagship world.

This world is intentionally built as the long-term default rather than another
one-off experiment.  The renderer uses a compact voxel DDA path, so what the
user sees is made from actual block occupancy decisions instead of a smooth
height-field with a grid texture painted over it.  The simulation remains a
simple four-channel ecological map that is easy to reason about and tune.
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
    // R: moisture / rivers / rain recharge
    // G: grass, crops, leaves, and living biome mass
    // B: long-lived stepped elevation
    // A: ore, lava, torch, and village light energy
    vec4 c  = texture(stateTex, uv);
    vec4 e  = texture(stateTex, uv + vec2(texel.x, 0.0));
    vec4 w  = texture(stateTex, uv - vec2(texel.x, 0.0));
    vec4 n  = texture(stateTex, uv + vec2(0.0, texel.y));
    vec4 s  = texture(stateTex, uv - vec2(0.0, texel.y));
    vec4 ne = texture(stateTex, uv + texel);
    vec4 nw = texture(stateTex, uv + vec2(-texel.x, texel.y));
    vec4 se = texture(stateTex, uv + vec2(texel.x, -texel.y));
    vec4 sw = texture(stateTex, uv - texel);

    float lapR = ((e.r + w.r + n.r + s.r) * 0.20 + (ne.r + nw.r + se.r + sw.r) * 0.05 - c.r) * laplaceScale;
    float lapG = ((e.g + w.g + n.g + s.g) * 0.20 + (ne.g + nw.g + se.g + sw.g) * 0.05 - c.g) * laplaceScale;
    float lapB = ((e.b + w.b + n.b + s.b) * 0.20 + (ne.b + nw.b + se.b + sw.b) * 0.05 - c.b) * laplaceScale;
    float lapA = ((e.a + w.a + n.a + s.a) * 0.20 + (ne.a + nw.a + se.a + sw.a) * 0.05 - c.a);

    vec2 gradH = vec2(e.b - w.b, n.b - s.b);
    float waterSlide = dot(gradH, vec2(e.r - w.r, n.r - s.r)) * 0.38;
    float localNoise = (hash12(uv * resolution + floor(time * 0.30)) - 0.5) * noiseStrength;
    float weather = 0.5 + 0.5 * sin(time * 0.055 + hash12(floor(uv * 18.0)) * 6.2831853);
    float rain = smoothstep(0.78, 0.98, weather) * (0.35 + 0.65 * hash12(floor(uv * 9.0) + 19.0));

    float reaction = c.r * c.g * c.g * 0.44;
    float localFeed = feed * 0.54 + 0.012 + c.r * 0.010 + localNoise * 0.030;
    float localKill = kill * 0.50 + 0.021 - c.a * 0.004 + parameterDrift * 0.36;

    float dMoisture = diffU * lapR - reaction + localFeed * (1.0 - c.r) - waterSlide + rain * 0.0024 - c.a * 0.0004;
    float dLife     = diffV * lapG + reaction - (localFeed + localKill) * c.g + c.r * 0.0047 - c.a * 0.0013;

    // Keep topography very slow so the world feels alive without melting away.
    float erosion = c.r * max(0.0, c.b - 0.42) * 0.00038;
    float growthLift = (c.g - 0.48) * 0.00022;
    float dHeight = lapB * 0.0015 + growthLift - erosion + localNoise * 0.00035;

    vec2 lightCell = floor(uv * resolution / 10.0);
    float ember = step(0.9987, hash12(lightCell + floor(time * 0.72) + 41.0));
    float villageMemory = smoothstep(0.42, 0.86, c.a) * 0.0009;
    float oreRecharge = smoothstep(0.63, 0.94, c.b) * smoothstep(0.50, 0.92, c.a) * 0.0006;
    float dLight = lapA * 0.040 + villageMemory + oreRecharge + ember * 0.020 - c.a * 0.0018;

    fragColor = vec4(
        clamp(c.r + dMoisture * dt, 0.0, 1.0),
        clamp(c.g + dLife * dt, 0.0, 1.0),
        clamp(c.b + dHeight * dt, 0.0, 1.0),
        clamp(c.a + dLight * dt, 0.0, 1.0)
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

#define MAX_RAY_STEPS 192
#define MAX_SHADOW_STEPS 34
#define FAR_DIST 92.0
#define BLOCK_SIZE 0.72
#define MAP_SCALE 0.026

const int MAT_AIR      = 0;
const int MAT_GRASS    = 1;
const int MAT_DIRT     = 2;
const int MAT_STONE    = 3;
const int MAT_WATER    = 4;
const int MAT_SAND     = 5;
const int MAT_SNOW     = 6;
const int MAT_LOG      = 7;
const int MAT_LEAVES   = 8;
const int MAT_PLANK    = 9;
const int MAT_ROOF     = 10;
const int MAT_GLASS    = 11;
const int MAT_TORCH    = 12;
const int MAT_ORE      = 13;
const int MAT_LAVA     = 14;
const int MAT_PATH     = 15;
const int MAT_CROPS    = 16;

const int SEA_LEVEL = 1;
const int SNOW_LEVEL = 11;

struct VoxelHit {
    bool hit;
    float t;
    ivec3 cell;
    vec3 normal;
    int material;
    vec4 state;
    int topY;
};

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float hash13(vec3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

float noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash12(i), hash12(i + vec2(1.0, 0.0)), f.x),
        mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), f.x),
        f.y
    );
}

float noise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float n000 = hash13(i + vec3(0.0, 0.0, 0.0));
    float n100 = hash13(i + vec3(1.0, 0.0, 0.0));
    float n010 = hash13(i + vec3(0.0, 1.0, 0.0));
    float n110 = hash13(i + vec3(1.0, 1.0, 0.0));
    float n001 = hash13(i + vec3(0.0, 0.0, 1.0));
    float n101 = hash13(i + vec3(1.0, 0.0, 1.0));
    float n011 = hash13(i + vec3(0.0, 1.0, 1.0));
    float n111 = hash13(i + vec3(1.0, 1.0, 1.0));

    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);
    float nxy0 = mix(nx00, nx10, f.y);
    float nxy1 = mix(nx01, nx11, f.y);
    return mix(nxy0, nxy1, f.z);
}

float fbm2(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    mat2 rot = mat2(0.80, -0.60, 0.60, 0.80);
    for (int i = 0; i < 4; i++) {
        v += a * noise2(p);
        p = rot * p * 2.03 + vec2(17.13, 9.71);
        a *= 0.5;
    }
    return v;
}

float fbm3(vec3 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 3; i++) {
        v += a * noise3(p);
        p = p * 2.01 + vec3(11.7, 3.1, 19.4);
        a *= 0.5;
    }
    return v;
}

vec4 sampleState(vec2 cell) {
    vec2 mapUv = fract((cell + 0.5) * MAP_SCALE);
    return clamp(textureLod(stateTex, mapUv, 0.0), 0.0, 1.0);
}

int terrainHeightBlocks(vec4 state, vec2 cell) {
    float macro = fbm2(cell * 0.030 + vec2(2.7, 8.1));
    float ridge = 1.0 - abs(fbm2(cell * 0.072 + vec2(13.0, 5.0)) * 2.0 - 1.0);
    float riverCut = smoothstep(0.55, 0.98, state.r) * (1.0 - smoothstep(0.58, 0.90, state.g));
    float raw = state.b * 13.5 + macro * 3.15 + ridge * 2.1 - riverCut * 2.25 - 4.35;
    return int(floor(clamp(raw, -5.0, 16.0)));
}

bool isCaveAir(ivec3 cell, vec4 state, int topY) {
    if (cell.y >= topY - 2 || cell.y < -9) {
        return false;
    }
    float carve = fbm3(vec3(cell) * 0.155 + vec3(23.0, 4.0, 81.0));
    float vein = fbm3(vec3(float(cell.x), float(cell.y) * 0.55, float(cell.z)) * 0.075 + 9.0);
    float threshold = mix(0.71, 0.61, clamp(state.a, 0.0, 1.0));
    return carve > threshold || (vein > 0.76 && cell.y < topY - 4);
}

bool isTreeCenter(vec2 centerCell, vec4 state, int groundY) {
    if (groundY <= SEA_LEVEL + 1 || groundY >= SNOW_LEVEL + 2) {
        return false;
    }
    float slopeProxy = abs(fbm2(centerCell * 0.18) - 0.5);
    float chance = mix(0.018, 0.125, clamp(state.g, 0.0, 1.0));
    chance *= 1.0 - smoothstep(0.52, 0.86, state.r);
    chance *= 1.0 - smoothstep(0.35, 0.50, slopeProxy);
    return hash12(centerCell + vec2(91.7, 13.3)) < chance;
}

int treeCellMaterial(ivec3 cell) {
    for (int dx = -2; dx <= 2; dx++) {
        for (int dz = -2; dz <= 2; dz++) {
            vec2 centerCell = vec2(float(cell.x - dx), float(cell.z - dz));
            vec4 state = sampleState(centerCell);
            int groundY = terrainHeightBlocks(state, centerCell);
            if (!isTreeCenter(centerCell, state, groundY)) {
                continue;
            }
            int relY = cell.y - groundY;
            int ax = abs(dx);
            int az = abs(dz);
            if (dx == 0 && dz == 0 && relY >= 1 && relY <= 4) {
                return MAT_LOG;
            }
            int radius = relY <= 4 ? 2 : 1;
            if (relY >= 3 && relY <= 6 && max(ax, az) <= radius && ax + az <= radius + 1) {
                return MAT_LEAVES;
            }
        }
    }
    return MAT_AIR;
}

bool villageChunk(vec2 chunk, vec4 baseState, int baseY) {
    if (baseY <= SEA_LEVEL || baseY > 8) {
        return false;
    }
    float roll = hash12(chunk + vec2(42.0, 77.0));
    return baseState.a > 0.46 && roll < 0.56;
}

int villageStructureMaterial(ivec3 cell) {
    vec2 c = vec2(float(cell.x), float(cell.z));
    vec2 shifted = c + vec2(4096.0);
    vec2 chunk = floor(shifted / 12.0);
    vec2 local = mod(shifted, 12.0);
    vec2 baseCell = chunk * 12.0 - vec2(4096.0) + vec2(5.5, 5.5);
    vec4 baseState = sampleState(baseCell);
    int baseY = terrainHeightBlocks(baseState, baseCell);
    if (!villageChunk(chunk, baseState, baseY)) {
        return MAT_AIR;
    }

    int lx = int(floor(local.x));
    int lz = int(floor(local.y));
    int relY = cell.y - baseY;
    bool hut = lx >= 3 && lx <= 8 && lz >= 3 && lz <= 8;
    bool wall = hut && (lx == 3 || lx == 8 || lz == 3 || lz == 8);
    bool window = relY == 2 && ((lx == 3 || lx == 8) && (lz == 5 || lz == 6) || (lz == 3 || lz == 8) && (lx == 5 || lx == 6));

    if (relY >= 1 && relY <= 3 && wall) {
        return window ? MAT_GLASS : MAT_PLANK;
    }
    if (relY == 4 && lx >= 2 && lx <= 9 && lz >= 2 && lz <= 9) {
        return MAT_ROOF;
    }
    if (relY == 5 && lx >= 3 && lx <= 8 && lz >= 3 && lz <= 8) {
        return MAT_ROOF;
    }
    if ((lx == 1 || lz == 1 || lx == 10 || lz == 10) && relY == 1 && hash12(c + 12.0) < 0.18) {
        return MAT_TORCH;
    }
    if (relY == 1 && lx >= 4 && lx <= 7 && (lz == 1 || lz == 10)) {
        return MAT_CROPS;
    }
    return MAT_AIR;
}

int villageSurfaceMaterial(ivec3 cell, vec4 state, int topY) {
    if (cell.y != topY || state.a < 0.30 || topY <= SEA_LEVEL) {
        return MAT_AIR;
    }
    vec2 c = vec2(float(cell.x), float(cell.z));
    vec2 local = mod(c + vec2(4096.0), 12.0);
    int lx = int(floor(local.x));
    int lz = int(floor(local.y));
    if (lx == 1 || lz == 1 || lx == 10 || lz == 10) {
        return MAT_PATH;
    }
    return MAT_AIR;
}

int materialAtCell(ivec3 cell, out vec4 stateOut, out int topY) {
    vec2 c = vec2(float(cell.x), float(cell.z));
    stateOut = sampleState(c);
    topY = terrainHeightBlocks(stateOut, c);

    if (cell.y <= topY) {
        if (isCaveAir(cell, stateOut, topY)) {
            if (cell.y <= -5 && (stateOut.a > 0.54 || hash13(vec3(float(cell.x), 17.0, float(cell.z))) > 0.86)) {
                return MAT_LAVA;
            }
            return MAT_AIR;
        }

        int surface = villageSurfaceMaterial(cell, stateOut, topY);
        if (surface != MAT_AIR) {
            return surface;
        }

        if (cell.y == topY) {
            if (topY <= SEA_LEVEL + 1 && stateOut.r > 0.36) {
                return MAT_SAND;
            }
            if (topY >= SNOW_LEVEL || stateOut.b > 0.90) {
                return MAT_SNOW;
            }
            return MAT_GRASS;
        }
        if (cell.y >= topY - 2) {
            return topY <= SEA_LEVEL + 1 ? MAT_SAND : MAT_DIRT;
        }

        float oreRoll = hash13(vec3(cell) + vec3(7.1, 3.7, 19.2));
        float oreMask = smoothstep(0.45, 0.92, stateOut.a) * (1.0 - smoothstep(float(topY - 4), float(topY), float(cell.y)));
        if (oreRoll < oreMask * 0.18) {
            return MAT_ORE;
        }
        return MAT_STONE;
    }

    if (cell.y <= SEA_LEVEL && topY < SEA_LEVEL) {
        return MAT_WATER;
    }

    int villageMat = villageStructureMaterial(cell);
    if (villageMat != MAT_AIR) {
        return villageMat;
    }

    int treeMat = treeCellMaterial(cell);
    if (treeMat != MAT_AIR) {
        return treeMat;
    }

    if (cell.y == topY + 1 && stateOut.a > 0.40 && hash12(c + vec2(7.0, 83.0)) < 0.025) {
        return MAT_TORCH;
    }
    return MAT_AIR;
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

void traceVoxel(vec3 ro, vec3 rd, int maxSteps, float maxDist, out VoxelHit hit) {
    hit.hit = false;
    hit.t = maxDist;
    hit.cell = ivec3(0);
    hit.normal = vec3(0.0);
    hit.material = MAT_AIR;
    hit.state = vec4(0.0);
    hit.topY = 0;

    vec3 gridPos = ro / BLOCK_SIZE;
    ivec3 cell = ivec3(floor(gridPos));
    ivec3 stepDir = ivec3(sign(rd));
    stepDir.x = stepDir.x == 0 ? 1 : stepDir.x;
    stepDir.y = stepDir.y == 0 ? 1 : stepDir.y;
    stepDir.z = stepDir.z == 0 ? 1 : stepDir.z;

    vec3 stepSign = vec3(float(stepDir.x), float(stepDir.y), float(stepDir.z));
    vec3 safeRd = stepSign * max(abs(rd), vec3(0.00001));
    vec3 boundary = (vec3(cell) + step(vec3(0.0), rd)) * BLOCK_SIZE;
    vec3 tMax = (boundary - ro) / safeRd;
    vec3 tDelta = BLOCK_SIZE / max(abs(rd), vec3(0.00001));

    float travel = 0.0;
    vec3 lastNormal = vec3(0.0);

    for (int i = 0; i < MAX_RAY_STEPS; i++) {
        if (i >= maxSteps || travel > maxDist) {
            break;
        }

        vec4 state;
        int topY;
        int material = materialAtCell(cell, state, topY);
        if (material != MAT_AIR) {
            hit.hit = true;
            hit.t = travel;
            hit.cell = cell;
            hit.normal = lastNormal;
            hit.material = material;
            hit.state = state;
            hit.topY = topY;
            return;
        }

        if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
            travel = tMax.x;
            tMax.x += tDelta.x;
            cell.x += stepDir.x;
            lastNormal = vec3(float(-stepDir.x), 0.0, 0.0);
        } else if (tMax.y <= tMax.z) {
            travel = tMax.y;
            tMax.y += tDelta.y;
            cell.y += stepDir.y;
            lastNormal = vec3(0.0, float(-stepDir.y), 0.0);
        } else {
            travel = tMax.z;
            tMax.z += tDelta.z;
            cell.z += stepDir.z;
            lastNormal = vec3(0.0, 0.0, float(-stepDir.z));
        }
    }
}

float voxelShadow(vec3 ro, vec3 rd, float maxDist) {
    vec3 gridPos = ro / BLOCK_SIZE;
    ivec3 cell = ivec3(floor(gridPos));
    ivec3 stepDir = ivec3(sign(rd));
    stepDir.x = stepDir.x == 0 ? 1 : stepDir.x;
    stepDir.y = stepDir.y == 0 ? 1 : stepDir.y;
    stepDir.z = stepDir.z == 0 ? 1 : stepDir.z;

    vec3 stepSign = vec3(float(stepDir.x), float(stepDir.y), float(stepDir.z));
    vec3 safeRd = stepSign * max(abs(rd), vec3(0.00001));
    vec3 boundary = (vec3(cell) + step(vec3(0.0), rd)) * BLOCK_SIZE;
    vec3 tMax = (boundary - ro) / safeRd;
    vec3 tDelta = BLOCK_SIZE / max(abs(rd), vec3(0.00001));

    float travel = 0.0;
    float shade = 1.0;
    for (int i = 0; i < MAX_SHADOW_STEPS; i++) {
        if (travel > maxDist) {
            break;
        }
        vec4 state;
        int topY;
        int material = materialAtCell(cell, state, topY);
        if (material == MAT_LEAVES) {
            shade *= 0.72;
        } else if (material != MAT_AIR && material != MAT_WATER && material != MAT_GLASS && material != MAT_TORCH && material != MAT_CROPS) {
            return 0.30;
        }

        if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
            travel = tMax.x;
            tMax.x += tDelta.x;
            cell.x += stepDir.x;
        } else if (tMax.y <= tMax.z) {
            travel = tMax.y;
            tMax.y += tDelta.y;
            cell.y += stepDir.y;
        } else {
            travel = tMax.z;
            tMax.z += tDelta.z;
            cell.z += stepDir.z;
        }
    }
    return clamp(shade, 0.30, 1.0);
}

vec2 faceUv(vec3 pos, vec3 normal) {
    vec3 lp = fract(pos / BLOCK_SIZE);
    if (abs(normal.y) > 0.5) {
        return lp.xz;
    }
    if (abs(normal.x) > 0.5) {
        return lp.zy;
    }
    return lp.xy;
}

float blockEdge(vec2 face) {
    float d = min(min(face.x, 1.0 - face.x), min(face.y, 1.0 - face.y));
    return 1.0 - smoothstep(0.020, 0.055, d);
}

float pixelNoise(ivec3 cell, vec2 face, float scale) {
    vec3 p = vec3(cell) + vec3(floor(face.x * scale), floor(face.y * scale), scale * 0.73);
    return hash13(p);
}

float squareSprite(vec3 rd, vec3 dir, float size) {
    vec3 helper = abs(dir.y) > 0.96 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(helper, dir));
    vec3 up = normalize(cross(dir, right));
    vec2 p = vec2(dot(rd, right), dot(rd, up));
    float core = 1.0 - smoothstep(size * 0.82, size, max(abs(p.x), abs(p.y)));
    return core * smoothstep(0.985, 0.999, dot(rd, dir));
}

vec3 skyColor(vec3 rd, vec3 sunDir, float dayAmount, float safeFx) {
    float horizon = smoothstep(-0.25, 0.85, rd.y);
    vec3 daySky = mix(vec3(0.48, 0.70, 0.96), vec3(0.78, 0.90, 1.00), horizon);
    vec3 nightSky = mix(vec3(0.015, 0.025, 0.065), vec3(0.055, 0.070, 0.135), horizon);
    vec3 sky = mix(nightSky, daySky, dayAmount);

    float sun = squareSprite(rd, sunDir, 0.085);
    sky += vec3(1.0, 0.83, 0.45) * sun * (0.8 + 1.5 * dayAmount) * safeFx;

    vec3 moonDir = normalize(-sunDir + vec3(0.0, 0.12, 0.0));
    float moon = squareSprite(rd, moonDir, 0.060);
    sky += vec3(0.62, 0.76, 1.0) * moon * (1.0 - dayAmount) * 0.85;

    if (rd.y > 0.02) {
        vec2 cloudUv = rd.xz / max(rd.y, 0.05) * 0.10 + vec2(time * 0.010, -time * 0.006);
        vec2 cell = floor(cloudUv * 10.0) / 10.0;
        float cloud = smoothstep(0.58, 0.76, fbm2(cell * 2.0 + 31.0));
        cloud *= smoothstep(0.03, 0.22, rd.y) * (1.0 - smoothstep(0.40, 0.92, rd.y));
        sky = mix(sky, vec3(0.94, 0.96, 0.98), cloud * (0.16 + dayAmount * 0.36));
    }

    float star = step(0.9975, hash12(floor(rd.xz * 240.0 / max(rd.y + 0.45, 0.1))));
    sky += vec3(0.80, 0.88, 1.0) * star * (1.0 - dayAmount) * smoothstep(0.08, 0.60, rd.y);
    return sky;
}

void main() {
    vec2 p = (gl_FragCoord.xy * 2.0 - resolution.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float tWorld = time * max(cameraSpeed, 0.05);

    float orbit = tWorld * 0.052;
    vec3 ro = vec3(sin(orbit) * 18.5, 7.7 + sin(tWorld * 0.17) * 1.2, cos(orbit) * 18.5);
    vec3 ta = vec3(sin(orbit + 0.65) * 4.2, 3.4 + sin(tWorld * 0.11) * 0.55, cos(orbit + 0.65) * 4.2);
    ro += cameraOffset;
    ta += cameraOffset * 0.25;

    mat3 cam = setCamera(ro, ta, 0.0);
    vec3 rd = normalize(cam * cameraInputRay(p, 1.62));

    float sunOrbit = time * 0.040 * max(cameraSpeed, 0.08) + 0.8;
    float dayAmount = smoothstep(-0.18, 0.55, sin(sunOrbit));
    vec3 sunDir = normalize(vec3(cos(sunOrbit) * 0.70, 0.25 + 0.70 * dayAmount, sin(sunOrbit) * 0.52));
    vec3 sky = skyColor(rd, sunDir, dayAmount, safeFx);

    VoxelHit hit;
    int maxSteps = clamp(raySteps, 48, MAX_RAY_STEPS);
    traceVoxel(ro, rd, maxSteps, FAR_DIST, hit);

    vec3 color = sky;
    if (hit.hit) {
        vec3 pos = ro + rd * hit.t;
        vec3 normal = length(hit.normal) < 0.5 ? vec3(0.0, 1.0, 0.0) : hit.normal;
        vec2 fuv = faceUv(pos, normal);
        float edge = blockEdge(fuv) * clamp(contourContrast, 0.0, 2.0);
        float speckle = pixelNoise(hit.cell, fuv, 5.0);
        float fine = pixelNoise(hit.cell, fuv + 0.37, 9.0);

        vec3 matColor = vec3(0.0);
        vec3 emission = vec3(0.0);
        float specular = 0.0;
        float alphaLike = 1.0;

        if (hit.material == MAT_GRASS) {
            vec3 grassA = vec3(0.16, 0.45, 0.10);
            vec3 grassB = vec3(0.46, 0.66, 0.18);
            vec3 dirt = vec3(0.36, 0.22, 0.10);
            float topFace = smoothstep(0.35, 0.85, normal.y);
            matColor = mix(dirt, mix(grassA, grassB, clamp(hit.state.g * 1.18 + speckle * 0.22, 0.0, 1.0)), topFace);
        } else if (hit.material == MAT_DIRT) {
            matColor = mix(vec3(0.28, 0.16, 0.07), vec3(0.49, 0.29, 0.13), speckle);
        } else if (hit.material == MAT_STONE) {
            float depth = smoothstep(-8.0, 8.0, float(hit.cell.y));
            matColor = mix(vec3(0.18, 0.19, 0.20), vec3(0.52, 0.53, 0.50), depth * 0.75 + speckle * 0.25);
        } else if (hit.material == MAT_WATER) {
            matColor = mix(vec3(0.02, 0.20, 0.42), vec3(0.10, 0.52, 0.82), clamp(hit.state.r + 0.25 * normal.y, 0.0, 1.0));
            specular = 0.75;
            alphaLike = 0.72;
            edge *= 0.35;
        } else if (hit.material == MAT_SAND) {
            matColor = mix(vec3(0.62, 0.52, 0.29), vec3(0.86, 0.77, 0.46), speckle);
        } else if (hit.material == MAT_SNOW) {
            matColor = mix(vec3(0.76, 0.84, 0.88), vec3(0.96, 0.98, 1.00), speckle * 0.65 + normal.y * 0.35);
            specular = 0.12;
        } else if (hit.material == MAT_LOG) {
            float stripe = step(0.5, fract((abs(normal.y) > 0.5 ? fuv.x : fuv.y) * 5.0 + speckle));
            matColor = mix(vec3(0.27, 0.14, 0.055), vec3(0.54, 0.33, 0.14), stripe * 0.75 + fine * 0.25);
        } else if (hit.material == MAT_LEAVES) {
            matColor = mix(vec3(0.07, 0.29, 0.08), vec3(0.30, 0.58, 0.14), clamp(hit.state.g + speckle * 0.25, 0.0, 1.0));
            alphaLike = 0.82;
        } else if (hit.material == MAT_PLANK) {
            float board = step(0.5, fract(fuv.y * 4.0 + float(hit.cell.x + hit.cell.z) * 0.11));
            matColor = mix(vec3(0.47, 0.27, 0.10), vec3(0.76, 0.52, 0.24), board * 0.55 + speckle * 0.45);
        } else if (hit.material == MAT_ROOF) {
            matColor = mix(vec3(0.38, 0.10, 0.055), vec3(0.72, 0.22, 0.10), speckle);
        } else if (hit.material == MAT_GLASS) {
            matColor = mix(vec3(0.40, 0.72, 0.88), vec3(0.86, 0.96, 1.00), normal.y * 0.15 + fine * 0.35);
            emission += vec3(0.06, 0.12, 0.16) * glow;
            specular = 0.55;
            alphaLike = 0.64;
        } else if (hit.material == MAT_TORCH) {
            matColor = vec3(0.37, 0.16, 0.045);
            float flicker = 0.78 + 0.22 * sin(time * 6.0 + hash13(vec3(hit.cell)) * 6.28318);
            emission += vec3(3.2, 1.35, 0.22) * flicker * (0.75 + glow * 0.55) * safeFx;
        } else if (hit.material == MAT_ORE) {
            vec3 stone = mix(vec3(0.24, 0.25, 0.25), vec3(0.48, 0.49, 0.47), speckle);
            vec3 ore = mix(vec3(0.08, 0.88, 0.82), vec3(1.0, 0.70, 0.18), hash12(vec2(float(hit.cell.x), float(hit.cell.z))));
            float fleck = step(0.62, fine) * (1.0 - smoothstep(0.82, 1.0, fuv.x * fuv.y));
            matColor = mix(stone, ore, 0.30 + 0.38 * fleck);
            emission += ore * fleck * (0.20 + glow * 0.18) * safeFx;
        } else if (hit.material == MAT_LAVA) {
            float lavaCell = 0.45 + 0.55 * sin(time * 1.8 + hash13(vec3(hit.cell)) * 6.28318);
            matColor = mix(vec3(0.70, 0.08, 0.00), vec3(1.0, 0.68, 0.05), lavaCell);
            emission += matColor * (1.6 + glow * 0.8) * safeFx;
        } else if (hit.material == MAT_PATH) {
            matColor = mix(vec3(0.42, 0.32, 0.18), vec3(0.71, 0.60, 0.34), speckle);
        } else if (hit.material == MAT_CROPS) {
            matColor = mix(vec3(0.18, 0.38, 0.07), vec3(0.88, 0.74, 0.22), clamp(hit.state.g + fine * 0.20, 0.0, 1.0));
        }

        matColor *= 1.0 - edge * 0.18;

        float shadow = voxelShadow(pos + normal * (BLOCK_SIZE * 0.22), sunDir, 18.0);
        float diffuse = max(dot(normal, sunDir), 0.0) * shadow;
        float ambient = 0.35 + 0.38 * max(normal.y, 0.0) + 0.12 * dayAmount;
        float bounce = 0.18 * max(-normal.y, 0.0);
        float rim = pow(clamp(1.0 + dot(normal, rd), 0.0, 1.0), 3.0);
        float spec = pow(max(dot(reflect(rd, normal), sunDir), 0.0), 34.0) * specular * shadow;

        vec3 lightColor = mix(vec3(0.55, 0.64, 0.90), vec3(1.0, 0.90, 0.72), dayAmount);
        vec3 lin = matColor * (lightColor * (0.38 + 1.48 * diffuse) + vec3(0.42, 0.52, 0.68) * ambient + vec3(0.25, 0.20, 0.16) * bounce);
        lin += vec3(0.9, 0.95, 1.0) * rim * 0.11;
        lin += vec3(1.0, 0.88, 0.62) * spec;
        lin += emission;

        // Water and glass keep some sky reflection without requiring a second ray.
        if (alphaLike < 0.99) {
            float fresnel = pow(clamp(1.0 + dot(normal, rd), 0.0, 1.0), 2.0);
            lin = mix(lin, sky + vec3(0.12, 0.18, 0.24) * dayAmount, (1.0 - alphaLike) * (0.35 + fresnel * 0.65));
        }

        float fog = 1.0 - exp(-0.00075 * hit.t * hit.t);
        float caveFog = smoothstep(2.5, 8.0, -float(hit.cell.y)) * 0.22;
        color = mix(lin, sky, clamp(fog + caveFog, 0.0, 0.92));
    }

    color += vec3(0.03, 0.05, 0.075) * sin(colorShift + time * 0.07) * 0.05 * safeFx;
    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(max(color, vec3(0.0)), vec3(1.0 / max(gamma, 0.20)));

    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.60 + 0.40 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.24);

    fragColor = vec4(color, 1.0);
}
"""


def _stamp_disk(target: np.ndarray, cx: int, cy: int, radius: int, value: float) -> None:
    height, width = target.shape
    y0 = max(cy - radius, 0)
    y1 = min(cy + radius + 1, height)
    x0 = max(cx - radius, 0)
    x1 = min(cx + radius + 1, width)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.meshgrid(
        np.arange(y0, y1, dtype=np.float32),
        np.arange(x0, x1, dtype=np.float32),
        indexing="ij",
    )
    dist = np.sqrt((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2)
    falloff = np.clip(1.0 - dist / max(float(radius), 1.0), 0.0, 1.0)
    target[y0:y1, x0:x1] = np.maximum(target[y0:y1, x0:x1], falloff * value)


def seed_field(width_px: int, height_px: int, tile_size: int) -> np.ndarray:
    tile_size = max(2, int(tile_size))
    tiles_x = max(1, int(np.ceil(width_px / tile_size)))
    tiles_y = max(1, int(np.ceil(height_px / tile_size)))

    rng = np.random.default_rng(20260615)
    tile_y, tile_x = np.meshgrid(
        np.arange(tiles_y, dtype=np.float32),
        np.arange(tiles_x, dtype=np.float32),
        indexing="ij",
    )
    x = tile_x / max(tiles_x - 1, 1)
    y = tile_y / max(tiles_y - 1, 1)

    continent = (
        0.47
        + 0.17 * np.sin(x * 7.4 + np.cos(y * 5.0) * 1.25)
        + 0.11 * np.cos(y * 8.8 - x * 1.4)
        + 0.07 * np.sin((x + y) * 15.5)
        + 0.05 * np.cos((x - y) * 23.0)
    ).astype(np.float32)
    height = continent + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.020

    biome_count = max(8, (tiles_x * tiles_y) // 2400)
    for _ in range(biome_count):
        cx = rng.uniform(0.0, tiles_x)
        cy = rng.uniform(0.0, tiles_y)
        rx = rng.uniform(max(7.0, tiles_x * 0.045), max(16.0, tiles_x * 0.18))
        ry = rng.uniform(max(7.0, tiles_y * 0.045), max(16.0, tiles_y * 0.18))
        distance = ((tile_x - cx) / rx) ** 2 + ((tile_y - cy) / ry) ** 2
        influence = np.clip(1.0 - distance, 0.0, 1.0)
        height += influence * rng.uniform(-0.12, 0.22)

    river_mask = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    for idx in range(3):
        phase = rng.uniform(0.0, np.pi * 2.0)
        freq_a = rng.uniform(1.4, 2.7)
        freq_b = rng.uniform(5.0, 8.5)
        width = rng.uniform(0.018, 0.034)
        centerline = 0.50 + 0.24 * np.sin(x * freq_a * np.pi + phase) + 0.09 * np.sin(x * freq_b * np.pi + phase * 0.47)
        if idx == 1:
            distance = np.abs(x - (0.52 + 0.20 * np.sin(y * freq_a * np.pi + phase)))
        else:
            distance = np.abs(y - centerline)
        river_mask = np.maximum(river_mask, np.exp(-((distance / width) ** 2)).astype(np.float32))

    height -= river_mask * 0.16
    height = np.clip(height, 0.0, 1.0).astype(np.float32)

    sea_level = 0.405
    water = np.clip((sea_level - height) / 0.19, 0.0, 1.0)
    shore = np.clip(1.0 - np.abs(height - sea_level) / 0.075, 0.0, 1.0)
    latitude = 1.0 - np.abs(y * 2.0 - 1.0)
    temperature = np.clip(0.28 + 0.72 * latitude - np.clip(height - 0.72, 0.0, 1.0) * 1.4, 0.0, 1.0)

    forest_noise = (
        0.44
        + 0.25 * np.sin(x * 18.0 + y * 5.0)
        + 0.17 * np.cos(y * 17.0 - x * 4.0)
        + 0.09 * np.sin((x - y) * 37.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.045
    )
    forest = np.clip(
        (1.0 - water)
        * (forest_noise + temperature * 0.24 + shore * 0.16 + river_mask * 0.20 - np.clip(height - 0.80, 0.0, 1.0) * 1.5),
        0.0,
        1.0,
    ).astype(np.float32)

    settlement = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    candidates = np.argwhere(
        (water < 0.28)
        & (shore > 0.20)
        & (forest > 0.22)
        & (height > sea_level + 0.025)
        & (height < 0.68)
    )
    if len(candidates) > 0:
        village_count = min(max(5, (tiles_x * tiles_y) // 2800), len(candidates))
        chosen = rng.choice(len(candidates), size=village_count, replace=False)
        for candidate_index in chosen:
            cy, cx = candidates[candidate_index]
            radius = int(rng.integers(4, 8))
            _stamp_disk(settlement, int(cx), int(cy), radius, rng.uniform(0.52, 0.90))

            # Lightly flatten village pads so houses read as intentional builds.
            y0 = max(cy - radius, 0)
            y1 = min(cy + radius + 1, tiles_y)
            x0 = max(cx - radius, 0)
            x1 = min(cx + radius + 1, tiles_x)
            yy, xx = np.meshgrid(
                np.arange(y0, y1, dtype=np.float32),
                np.arange(x0, x1, dtype=np.float32),
                indexing="ij",
            )
            dist = np.sqrt((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2)
            pad = np.clip(1.0 - dist / max(float(radius), 1.0), 0.0, 1.0)
            center_h = float(height[int(cy), int(cx)])
            height[y0:y1, x0:x1] = height[y0:y1, x0:x1] * (1.0 - pad * 0.22) + center_h * (pad * 0.22)

    cave_lattice = np.abs(np.sin(x * 30.0) * np.cos(y * 27.0))
    ore = np.clip(1.0 - cave_lattice * 3.0, 0.0, 1.0) * np.clip((height - 0.43) * 1.9, 0.0, 1.0)
    lava_faults = np.clip(1.0 - np.abs(np.sin(x * 19.0 + y * 11.0)) * 3.8, 0.0, 1.0) * np.clip(height - 0.56, 0.0, 1.0)
    light = np.clip(settlement + ore * 0.66 + lava_faults * 0.36 + rng.random((tiles_y, tiles_x), dtype=np.float32) * 0.018, 0.0, 1.0)

    moisture = np.clip(water * 0.82 + river_mask * 0.56 + shore * 0.16 + rng.random((tiles_y, tiles_x), dtype=np.float32) * 0.025, 0.0, 1.0)
    life = np.clip(forest * 0.80 + river_mask * 0.16 + settlement * 0.12 + (1.0 - water) * 0.08, 0.0, 1.0)

    tile_field = np.stack(
        [
            moisture.astype(np.float32),
            life.astype(np.float32),
            height.astype(np.float32),
            light.astype(np.float32),
        ],
        axis=-1,
    )

    field = np.repeat(np.repeat(tile_field, tile_size, axis=0), tile_size, axis=1)
    return field[:height_px, :width_px].copy()


SPEC = WorldSpec(
    id="minecraft-definitive-overworld-3d",
    display_name="Minecraft Definitive Overworld",
    window_title="Garage Life Lab - Minecraft Definitive Overworld",
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={
        "feed": 0.034,
        "kill": 0.058,
        "diff_u": 0.15,
        "diff_v": 0.075,
        "time_step": 0.85,
        "substeps": 10,
        "noise_strength": 0.010,
        "param_drift": 0.003,
        "ray_steps": 150,
        "fx_intensity": 1.18,
        "camera_speed": 0.74,
        "camera_move_speed": 10.0,
        "exposure": 1.24,
        "glow": 1.16,
        "gamma": 1.14,
        "contour_contrast": 1.06,
        "tile_size": 8,
    },
    preview_image=None,
    stability_notes=("default", "flagship"),
    hud_subtitle="DEFINITIVE BLOCK OVERWORLD",
    preview_palette=("#6db8ff", "#58a83a", "#2f6f24", "#7a4a20", "#6a6f6a", "#d9c16f", "#ff9d24"),
)
