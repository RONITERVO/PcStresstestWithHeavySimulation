"""Minecraft-inspired long-term stress-test world.

The state channels stay deterministic and readable:

R = water / river pressure / rain reserve
G = vegetation / canopy recovery
B = block elevation
A = village, road, torch, mine, and ore activity
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
uniform float tileSize;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    vec2 texel = 1.0 / max(resolution, vec2(1.0));
    vec4 c = texture(stateTex, uv);
    vec4 r = texture(stateTex, uv + vec2(texel.x, 0.0));
    vec4 l = texture(stateTex, uv - vec2(texel.x, 0.0));
    vec4 t = texture(stateTex, uv + vec2(0.0, texel.y));
    vec4 b = texture(stateTex, uv - vec2(0.0, texel.y));
    vec4 tr = texture(stateTex, uv + texel);
    vec4 tl = texture(stateTex, uv + vec2(-texel.x, texel.y));
    vec4 br = texture(stateTex, uv + vec2(texel.x, -texel.y));
    vec4 bl = texture(stateTex, uv - texel);

    vec4 lap = ((r + l + t + b) * 0.20 + (tr + tl + br + bl) * 0.05 - c) * laplaceScale;

    // R: bounded water cycle. It moves downhill, returns through rain, and never floods the whole map.
    vec2 heightGradient = vec2(r.b - l.b, t.b - b.b);
    vec2 waterGradient = vec2(r.r - l.r, t.r - b.r);
    float downhill = dot(heightGradient, waterGradient) * 0.38;
    float rain = smoothstep(
        0.72,
        0.98,
        sin(time * 0.030 + hash12(floor(uv * 13.0)) * 6.2831853) * 0.5 + 0.5
    );
    float riverRecharge = smoothstep(0.45, 0.82, c.r) * smoothstep(0.18, 0.58, c.g);
    float dr = diffU * lap.r + rain * 0.0038 + riverRecharge * 0.0016 - downhill - c.r * 0.00055;

    // G: vegetation recovers around water, roads/villages hold it back, and dry fire/torch pressure trims it.
    float wetGround = smoothstep(0.30, 0.78, c.r);
    float buildPressure = smoothstep(0.34, 0.92, c.a);
    float highCold = smoothstep(0.70, 0.96, c.b);
    float growth = wetGround * (1.0 - highCold * 0.55) * (1.0 - buildPressure * 0.70);
    float fireTrim = buildPressure * smoothstep(0.76, 1.0, c.g) * (0.0015 + kill * 0.0012);
    float dg = diffV * lap.g + growth * 0.0024 - fireTrim - max(0.0, 0.22 - c.r) * 0.0012;

    // B: terrain remains stable. Tiny erosion keeps the simulation alive without melting block silhouettes.
    float erosion = smoothstep(0.64, 1.0, c.r) * max(0.0, c.b - 0.32) * 0.00016;
    float db = lap.b * 0.00035 - erosion + parameterDrift * (hash12(uv * resolution + 19.0) - 0.5) * 0.00008;

    // A: village, torch, bridge, mine, and ore activity flickers locally and decays slowly.
    vec2 eventCell = floor(uv * resolution / max(tileSize, 2.0));
    float torchPulse = step(0.9975, hash12(eventCell + floor(time * 0.95)));
    float da = lap.a * 0.018 + torchPulse * 0.012 + buildPressure * 0.0009 - c.a * 0.0011;

    vec4 nextState = c + vec4(dr, dg, db, da) * max(dt, 0.01);
    nextState += (hash12(uv * resolution + floor(time * 0.4)) - 0.5) * noiseStrength * 0.00035;
    fragColor = clamp(nextState, 0.0, 1.0);
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
uniform float tileSize;

#define MAX_STEPS 160
#define MAX_DIST 78.0
#define SURF_DIST 0.0035
#define CELL 1.0
#define MAP_SCALE 0.0205
#define SEA_LEVEL 2.18
#define WORLD_BASE -3.25

const int MAT_AIR = 0;
const int MAT_GRASS = 1;
const int MAT_DIRT = 2;
const int MAT_STONE = 3;
const int MAT_SNOW = 4;
const int MAT_SAND = 5;
const int MAT_WATER = 6;
const int MAT_TRUNK = 7;
const int MAT_LEAVES = 8;
const int MAT_PLANK = 9;
const int MAT_ROOF = 10;
const int MAT_GLASS = 11;
const int MAT_TORCH = 12;
const int MAT_ORE = 13;
const int MAT_PATH = 14;
const int MAT_BEACON = 15;

struct HitInfo {
    float dist;
    int mat;
    vec4 state;
};

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash12(i);
    float b = hash12(i + vec2(1.0, 0.0));
    float c = hash12(i + vec2(0.0, 1.0));
    float d = hash12(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += noise2(p) * a;
        p = mat2(1.72, 1.13, -1.13, 1.72) * p + 13.1;
        a *= 0.5;
    }
    return v;
}

float boxSdf(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

vec4 sampleState(vec2 cell) {
    vec2 st = fract(cell * MAP_SCALE + vec2(0.371, 0.619));
    return texture(stateTex, st);
}

float terrainHeight(vec2 cell, vec4 s) {
    float ridge = abs(sin(cell.x * 0.105 + cos(cell.y * 0.075) * 1.7));
    float mesa = smoothstep(0.74, 0.95, s.b) * (0.28 + 0.22 * ridge);
    float rawH = 0.55 + s.b * 7.35 + mesa + (fbm(cell * 0.043) - 0.5) * 0.32;
    return floor(rawH / 0.42) * 0.42;
}

float roadMaskForCell(vec2 cell, vec4 s) {
    float spawn = 1.0 - smoothstep(20.0, 64.0, length(cell));
    float northRoad = 1.0 - smoothstep(0.34, 1.36, abs(cell.x - sin(cell.y * 0.080) * 3.8));
    float eastRoad = 1.0 - smoothstep(0.34, 1.25, abs(cell.y + 2.0 - cos(cell.x * 0.055) * 2.9));
    float ringRoad = 1.0 - smoothstep(0.0, 1.45, abs(length(cell) - 12.0));
    float road = max(max(northRoad, eastRoad), ringRoad * 0.35) * spawn;
    road = max(road, smoothstep(0.46, 0.88, s.a) * spawn * 0.45);
    return clamp(road, 0.0, 1.0);
}

float waterMaskForCell(vec2 cell, vec4 s, float h) {
    float river = smoothstep(0.53, 0.82, s.r);
    float ocean = smoothstep(0.34, 0.68, SEA_LEVEL + 0.10 - h);
    return clamp(max(river, ocean), 0.0, 1.0);
}

void takeShape(inout float bestD, inout int bestMat, float d, int mat) {
    if (d < bestD) {
        bestD = d;
        bestMat = mat;
    }
}

float pillarSdf(vec3 p, float radius, float halfHeight) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(radius, halfHeight);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

HitInfo mapWorld(vec3 p) {
    vec2 cell = floor(p.xz / CELL);
    vec2 center = (cell + vec2(0.5)) * CELL;
    vec2 local = p.xz - center;
    vec4 s = sampleState(cell);
    float h = terrainHeight(cell, s);
    float water = waterMaskForCell(cell, s, h);
    float road = roadMaskForCell(cell, s);

    float bestD = 999.0;
    int bestMat = MAT_AIR;

    float halfHeight = max(0.06, (h - WORLD_BASE) * 0.5);
    float terrainD = boxSdf(
        vec3(local.x, p.y - ((h + WORLD_BASE) * 0.5), local.y),
        vec3(0.492, halfHeight, 0.492)
    );

    int terrainMat = MAT_GRASS;
    if (road > 0.52 && water < 0.45 && h < 4.9) {
        terrainMat = MAT_PATH;
    } else if (water > 0.55 || h < SEA_LEVEL + 0.10) {
        terrainMat = MAT_SAND;
    } else if (h > 6.15) {
        terrainMat = MAT_SNOW;
    } else if (h > 4.85 || s.b > 0.78) {
        terrainMat = MAT_STONE;
    } else if (s.g < 0.20) {
        terrainMat = MAT_DIRT;
    }
    takeShape(bestD, bestMat, terrainD, terrainMat);

    if (water > 0.42) {
        float waterTop = SEA_LEVEL + 0.08 + sin(time * 0.45 + cell.x * 0.9 + cell.y * 0.7) * 0.018;
        float waterD = boxSdf(vec3(local.x, p.y - waterTop, local.y), vec3(0.515, 0.055, 0.515));
        takeShape(bestD, bestMat, waterD, MAT_WATER);
    }

    // Bridge planks use the same procedural roads, so river crossings stay readable.
    if (road > 0.55 && water > 0.40) {
        float bridgeD = boxSdf(vec3(local.x, p.y - (SEA_LEVEL + 0.20), local.y), vec3(0.53, 0.095, 0.53));
        takeShape(bestD, bestMat, bridgeD, MAT_PLANK);
    }

    // Forests: one compact block tree per 3x3 plot, never in water or on main roads.
    vec2 treePlot = floor(cell / 3.0);
    vec2 treeCenter = (treePlot * 3.0 + vec2(1.5)) * CELL;
    vec2 treeLocal = p.xz - treeCenter;
    vec4 treeState = sampleState(treePlot * 3.0 + vec2(1.5));
    float treeGround = terrainHeight(treePlot * 3.0 + vec2(1.5), treeState);
    float treeChance = hash12(treePlot + vec2(43.7, 8.1));
    float treeOK = smoothstep(0.42, 0.82, treeState.g)
        * (1.0 - waterMaskForCell(treePlot * 3.0 + vec2(1.5), treeState, treeGround))
        * (1.0 - roadMaskForCell(treePlot * 3.0 + vec2(1.5), treeState));
    if (treeOK > 0.30 && treeChance > 0.42) {
        float trunk = boxSdf(vec3(treeLocal.x, p.y - (treeGround + 0.58), treeLocal.y), vec3(0.16, 0.58, 0.16));
        float leaves1 = boxSdf(vec3(treeLocal.x, p.y - (treeGround + 1.45), treeLocal.y), vec3(0.76, 0.46, 0.76));
        float leaves2 = boxSdf(vec3(treeLocal.x, p.y - (treeGround + 2.02), treeLocal.y), vec3(0.48, 0.36, 0.48));
        takeShape(bestD, bestMat, trunk, MAT_TRUNK);
        takeShape(bestD, bestMat, leaves1, MAT_LEAVES);
        takeShape(bestD, bestMat, leaves2, MAT_LEAVES);
    }

    // Villages: one house per 8x8 plot around the spawn roads, with glass and torch detail.
    vec2 plot = floor((cell + vec2(32.0)) / 8.0);
    vec2 plotCenter = plot * 8.0 - vec2(32.0) + vec2(4.0);
    vec4 ps = sampleState(plotCenter);
    float ph = terrainHeight(plotCenter, ps);
    float village = (1.0 - smoothstep(18.0, 54.0, length(plotCenter))) * smoothstep(0.30, 0.82, ps.a);
    float plotHash = hash12(plot + vec2(11.0, 71.0));
    if (village > 0.13 && plotHash > 0.38 && waterMaskForCell(plotCenter, ps, ph) < 0.35) {
        vec2 houseLocal = p.xz - plotCenter;
        float body = boxSdf(vec3(houseLocal.x, p.y - (ph + 0.92), houseLocal.y), vec3(1.42, 0.86, 1.12));
        float roof = boxSdf(vec3(houseLocal.x, p.y - (ph + 1.78), houseLocal.y), vec3(1.62, 0.28, 1.30));
        float windowA = boxSdf(vec3(houseLocal.x - 1.43, p.y - (ph + 1.04), houseLocal.y - 0.44), vec3(0.035, 0.24, 0.25));
        float windowB = boxSdf(vec3(houseLocal.x + 1.43, p.y - (ph + 1.04), houseLocal.y + 0.44), vec3(0.035, 0.24, 0.25));
        float doorTorch = boxSdf(vec3(houseLocal.x - 0.78, p.y - (ph + 1.04), houseLocal.y - 1.18), vec3(0.055, 0.36, 0.055));
        takeShape(bestD, bestMat, body, MAT_PLANK);
        takeShape(bestD, bestMat, roof, MAT_ROOF);
        takeShape(bestD, bestMat, windowA, MAT_GLASS);
        takeShape(bestD, bestMat, windowB, MAT_GLASS);
        takeShape(bestD, bestMat, doorTorch, MAT_TORCH);
    }

    // Road torches read from a distance and add warm emission for the stress-test view.
    float torchSlot = step(0.935, 1.0 - abs(fract((cell.x + cell.y * 1.7) / 7.0) - 0.5) * 2.0);
    if (road > 0.58 && water < 0.35 && torchSlot > 0.5) {
        float post = boxSdf(vec3(local.x, p.y - (h + 0.36), local.y), vec3(0.055, 0.36, 0.055));
        float flame = boxSdf(vec3(local.x, p.y - (h + 0.78), local.y), vec3(0.13, 0.11, 0.13));
        takeShape(bestD, bestMat, post, MAT_TRUNK);
        takeShape(bestD, bestMat, flame, MAT_TORCH);
    }

    // Ore blocks and mine glints in mountains/caves.
    float oreChance = hash12(cell + vec2(91.0, 17.0));
    if (s.a > 0.38 && s.b > 0.57 && oreChance > 0.76) {
        float ore = boxSdf(vec3(local.x, p.y - (h + 0.13), local.y), vec3(0.23, 0.12, 0.23));
        takeShape(bestD, bestMat, ore, MAT_ORE);
    }

    // Spawn beacon: one obvious landmark for non-technical users and a stable camera target.
    vec2 spawnCell = vec2(0.0);
    vec4 spawnState = sampleState(spawnCell);
    float spawnH = terrainHeight(spawnCell, spawnState);
    float tower = boxSdf(vec3(p.x, p.y - (spawnH + 1.38), p.z), vec3(0.86, 1.38, 0.86));
    float battlementA = boxSdf(vec3(p.x, p.y - (spawnH + 2.92), p.z), vec3(1.08, 0.18, 1.08));
    float crystal = boxSdf(vec3(p.x, p.y - (spawnH + 3.30), p.z), vec3(0.30, 0.34, 0.30));
    takeShape(bestD, bestMat, tower, MAT_STONE);
    takeShape(bestD, bestMat, battlementA, MAT_STONE);
    takeShape(bestD, bestMat, crystal, MAT_BEACON);

    HitInfo hit;
    hit.dist = bestD;
    hit.mat = bestMat;
    hit.state = s;
    return hit;
}

float worldDistance(vec3 p) {
    HitInfo h = mapWorld(p);
    return h.dist;
}

vec3 calcNormal(vec3 p) {
    vec2 e = vec2(0.035, 0.0);
    return normalize(vec3(
        worldDistance(p + e.xyy) - worldDistance(p - e.xyy),
        worldDistance(p + e.yxy) - worldDistance(p - e.yxy),
        worldDistance(p + e.yyx) - worldDistance(p - e.yyx)
    ));
}

float softShadow(vec3 ro, vec3 rd, float mint, float maxt) {
    float res = 1.0;
    float t = mint;
    for (int i = 0; i < 34; i++) {
        float h = worldDistance(ro + rd * t);
        res = min(res, 8.0 * h / t);
        t += clamp(h, 0.035, 0.58);
        if (res < 0.01 || t > maxt) {
            break;
        }
    }
    return clamp(res, 0.0, 1.0);
}

float ambientOcclusion(vec3 p, vec3 n) {
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 5; i++) {
        float h = 0.04 + 0.13 * float(i);
        float d = worldDistance(p + n * h);
        occ += (h - d) * sca;
        sca *= 0.62;
    }
    return clamp(1.0 - occ * 1.8, 0.0, 1.0);
}

vec3 materialColor(int mat, vec4 s, vec3 p, vec3 n) {
    float face = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    float grain = noise2(floor(p.xz * 2.0) + float(mat) * 23.0) * 0.10;

    if (mat == MAT_GRASS) return mix(vec3(0.18, 0.43, 0.13), vec3(0.36, 0.64, 0.20), face) + grain;
    if (mat == MAT_DIRT) return vec3(0.38, 0.25, 0.14) + grain;
    if (mat == MAT_STONE) return vec3(0.43, 0.45, 0.43) + grain * 0.7;
    if (mat == MAT_SNOW) return vec3(0.78, 0.83, 0.84) + grain * 0.45;
    if (mat == MAT_SAND) return vec3(0.68, 0.58, 0.34) + grain * 0.7;
    if (mat == MAT_WATER) return vec3(0.08, 0.32, 0.58) + vec3(0.04, 0.10, 0.14) * sin(time + p.x * 1.7 + p.z * 2.1);
    if (mat == MAT_TRUNK) return vec3(0.34, 0.20, 0.10) + grain;
    if (mat == MAT_LEAVES) return mix(vec3(0.10, 0.36, 0.13), vec3(0.28, 0.56, 0.18), s.g) + grain;
    if (mat == MAT_PLANK) return vec3(0.55, 0.36, 0.18) + grain;
    if (mat == MAT_ROOF) return vec3(0.50, 0.18, 0.11) + grain * 0.5;
    if (mat == MAT_GLASS) return vec3(0.62, 0.86, 0.96);
    if (mat == MAT_TORCH) return vec3(1.0, 0.58, 0.18);
    if (mat == MAT_ORE) return mix(vec3(0.15, 0.62, 0.98), vec3(1.0, 0.74, 0.22), hash12(floor(p.xz)));
    if (mat == MAT_PATH) return vec3(0.50, 0.39, 0.22) + grain * 0.7;
    if (mat == MAT_BEACON) return vec3(0.38, 0.98, 0.92);
    return vec3(0.0);
}

vec3 skyColor(vec3 rd, vec3 lightDir, float rain) {
    float sun = pow(max(dot(rd, lightDir), 0.0), 480.0);
    vec3 horizon = mix(vec3(0.58, 0.76, 0.96), vec3(0.40, 0.48, 0.58), rain);
    vec3 zenith = mix(vec3(0.12, 0.36, 0.74), vec3(0.18, 0.22, 0.29), rain);
    vec3 col = mix(horizon, zenith, smoothstep(-0.05, 0.85, rd.y));
    col += sun * mix(vec3(1.0, 0.84, 0.55), vec3(0.75, 0.82, 0.95), rain);
    return col;
}

mat3 cameraBasis(vec3 ro, vec3 ta) {
    vec3 ww = normalize(ta - ro);
    vec3 uu = normalize(cross(ww, vec3(0.0, 1.0, 0.0)));
    vec3 vv = normalize(cross(uu, ww));
    return mat3(uu, vv, ww);
}

vec3 rotateYawPitch(vec3 rd, vec2 yawPitch) {
    float cy = cos(yawPitch.x);
    float sy = sin(yawPitch.x);
    rd.xz = mat2(cy, -sy, sy, cy) * rd.xz;
    float cp = cos(yawPitch.y);
    float sp = sin(yawPitch.y);
    rd.yz = mat2(cp, -sp, sp, cp) * rd.yz;
    return normalize(rd);
}

void main() {
    vec2 q = gl_FragCoord.xy / max(resolution, vec2(1.0));
    vec2 p = (gl_FragCoord.xy * 2.0 - resolution.xy) / max(resolution.y, 1.0);
    float safeFx = clamp(fxIntensity, 0.2, 1.8);
    float t = time * (0.042 + cameraSpeed * 0.014);

    float rain = smoothstep(0.68, 0.96, sin(time * 0.032 + 1.7) * 0.5 + 0.5) * (0.35 + 0.45 * safeFx);
    vec3 lightDir = normalize(vec3(0.42 + 0.16 * sin(t * 0.8), 0.78, 0.34 + 0.12 * cos(t)));

    vec3 target = vec3(0.0, 3.0, 0.0);
    vec3 ro = vec3(
        sin(t) * 25.0 + sin(t * 2.1) * 3.0,
        8.5 + sin(t * 0.7) * 1.7,
        cos(t) * 25.0
    );
    ro += cameraOffset;
    ro.y += cameraZoom * 1.8;
    float zoom = 1.10 + clamp(cameraZoom, -3.0, 4.0) * 0.08;

    mat3 cam = cameraBasis(ro, target + cameraOffset * 0.25);
    vec3 rd = normalize(cam * vec3(p * zoom, 1.75));
    rd = rotateYawPitch(rd, cameraYawPitch);

    vec3 color = skyColor(rd, lightDir, rain);
    float cloudBand = smoothstep(0.10, 0.78, rd.y);
    float clouds = fbm(rd.xz * 4.0 / max(0.18, rd.y + 0.35) + vec2(time * 0.018, 0.0));
    color = mix(color, vec3(0.88, 0.91, 0.93), smoothstep(0.55, 0.78, clouds) * cloudBand * (1.0 - rain * 0.55));

    float distTravel = 0.0;
    int hitMat = MAT_AIR;
    vec4 hitState = vec4(0.0);
    vec3 hitPos = ro;
    bool hit = false;
    int steps = clamp(raySteps, 32, MAX_STEPS);

    for (int i = 0; i < MAX_STEPS; i++) {
        if (i >= steps) {
            break;
        }
        hitPos = ro + rd * distTravel;
        HitInfo h = mapWorld(hitPos);
        if (h.dist < SURF_DIST) {
            hit = true;
            hitMat = h.mat;
            hitState = h.state;
            break;
        }
        distTravel += clamp(h.dist * 0.82, 0.018, 1.05);
        if (distTravel > MAX_DIST) {
            break;
        }
    }

    if (hit) {
        vec3 n = calcNormal(hitPos);
        float diff = max(dot(n, lightDir), 0.0);
        float sh = softShadow(hitPos + n * 0.05, lightDir, 0.08, 24.0);
        float ao = ambientOcclusion(hitPos, n);
        float sky = clamp(n.y * 0.55 + 0.55, 0.0, 1.0);
        float fresnel = pow(1.0 - max(dot(-rd, n), 0.0), 4.0);

        vec3 base = materialColor(hitMat, hitState, hitPos, n);
        vec3 emission = vec3(0.0);
        if (hitMat == MAT_TORCH) {
            emission += vec3(1.0, 0.45, 0.10) * (1.6 + glow * 1.2) * (0.86 + 0.14 * sin(time * 12.0));
        }
        if (hitMat == MAT_ORE) {
            emission += vec3(0.16, 0.50, 0.85) * glow * 0.75;
        }
        if (hitMat == MAT_BEACON) {
            emission += vec3(0.26, 1.0, 0.90) * (1.8 + glow * 1.8);
        }
        if (hitMat == MAT_GLASS) {
            base = mix(base, skyColor(reflect(rd, n), lightDir, rain), 0.35 + fresnel * 0.35);
            emission += vec3(0.05, 0.12, 0.16) * glow;
        }
        if (hitMat == MAT_WATER) {
            vec3 reflectedSky = skyColor(reflect(rd, n), lightDir, rain);
            base = mix(base, reflectedSky, 0.34 + fresnel * 0.38);
            emission += vec3(0.02, 0.07, 0.10) * glow;
        }

        vec2 blockLocal = abs(fract(hitPos.xz / CELL) - vec2(0.5));
        float edge = smoothstep(0.455, 0.500, max(blockLocal.x, blockLocal.y));
        base *= 1.0 - edge * 0.105 * clamp(contourContrast, 0.2, 2.0);

        vec3 lighting = vec3(0.0);
        lighting += vec3(1.00, 0.91, 0.72) * diff * sh * 1.75;
        lighting += vec3(0.34, 0.48, 0.68) * sky * ao * 0.70;
        lighting += vec3(0.95, 0.82, 0.58) * 0.10 * ao;
        color = base * lighting + emission;

        float torchField = smoothstep(0.40, 1.0, hitState.a) * (1.0 - smoothstep(0.0, 8.0, length(hitPos.xz)));
        color += vec3(1.0, 0.42, 0.12) * torchField * 0.12 * glow;

        float fog = 1.0 - exp(-0.00105 * distTravel * distTravel);
        fog = clamp(fog + rain * smoothstep(14.0, 62.0, distTravel) * 0.23, 0.0, 0.88);
        color = mix(color, skyColor(rd, lightDir, rain), fog);
    }

    // Beacon bloom is cheap and visible even before the ray hits it.
    float beaconRay = pow(max(dot(rd, normalize(vec3(0.0, 2.3, 0.0) - ro)), 0.0), 120.0);
    color += vec3(0.18, 0.92, 0.85) * beaconRay * glow * 1.6;

    if (rain > 0.04) {
        float streak = step(0.987, hash12(floor(vec2(q.x * 260.0 + time * 14.0, q.y * 110.0 - time * 38.0))));
        color = mix(color, vec3(0.67, 0.76, 0.84), streak * rain * 0.16 * smoothstep(-0.2, 0.55, rd.y));
    }

    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(max(color, vec3(0.0)), vec3(1.0 / max(gamma, 0.20)));
    color *= 0.62 + 0.38 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.25);

    fragColor = vec4(color, 1.0);
}
"""

WORLD_SEED = 20260615


def _box_blur(values: np.ndarray, passes: int = 3) -> np.ndarray:
    """Small wraparound blur using only NumPy, keeping startup deterministic and dependency-free."""

    result = values.astype(np.float32, copy=True)
    for _ in range(max(0, int(passes))):
        result = (
            result
            + np.roll(result, 1, axis=0)
            + np.roll(result, -1, axis=0)
            + np.roll(result, 1, axis=1)
            + np.roll(result, -1, axis=1)
            + np.roll(np.roll(result, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(result, 1, axis=0), -1, axis=1)
            + np.roll(np.roll(result, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(result, -1, axis=0), -1, axis=1)
        ) / 9.0
    return result


def _gaussian_lane(distance: np.ndarray, width: float) -> np.ndarray:
    return np.exp(-((distance / max(width, 1.0e-6)) ** 2))


def seed_field(width_px: int, height_px: int, tile_size: int) -> np.ndarray:
    """Build a deterministic, readable Overworld map at tile resolution then upscale to pixels."""

    tile_size = max(2, int(tile_size))
    tiles_x = max(1, int(np.ceil(width_px / tile_size)))
    tiles_y = max(1, int(np.ceil(height_px / tile_size)))
    rng = np.random.default_rng(WORLD_SEED)

    tile_y, tile_x = np.meshgrid(
        np.arange(tiles_y, dtype=np.float32),
        np.arange(tiles_x, dtype=np.float32),
        indexing="ij",
    )
    x = tile_x / max(tiles_x - 1, 1) * 2.0 - 1.0
    y = tile_y / max(tiles_y - 1, 1) * 2.0 - 1.0
    radius = np.sqrt((x * 0.92) ** 2 + (y * 1.06) ** 2)

    continent = np.clip(1.10 - radius, 0.0, 1.0)
    continent = continent ** 0.62
    low_noise = _box_blur(rng.random((tiles_y, tiles_x), dtype=np.float32), passes=9)
    mid_noise = _box_blur(rng.random((tiles_y, tiles_x), dtype=np.float32), passes=4)
    fine_noise = _box_blur(rng.random((tiles_y, tiles_x), dtype=np.float32), passes=1)

    river_center = 0.11 * np.sin(y * 5.4 + 0.7) + 0.045 * np.sin(y * 13.0)
    river = _gaussian_lane(x - river_center, 0.050)
    cross_river = _gaussian_lane(y + 0.10 + 0.04 * np.cos(x * 7.0), 0.040)
    river_network = np.clip(river + cross_river * 0.62, 0.0, 1.0)

    north_mountains = np.clip((y + 0.12) * 1.18 + low_noise * 0.85, 0.0, 1.0)
    west_ridge = _gaussian_lane(x + 0.62 - 0.10 * np.sin(y * 4.0), 0.16)
    east_cliffs = _gaussian_lane(x - 0.72 + 0.08 * np.cos(y * 6.0), 0.20)
    spawn_basin = np.exp(-((x / 0.26) ** 2 + ((y + 0.03) / 0.22) ** 2))

    height = (
        0.26
        + continent * 0.42
        + low_noise * 0.24
        + mid_noise * 0.12
        + fine_noise * 0.035
        + north_mountains * 0.22
        + west_ridge * 0.13
        + east_cliffs * 0.11
        - river_network * 0.18
        - spawn_basin * 0.08
    )
    height -= np.clip(radius - 0.82, 0.0, 1.0) * 0.36
    height = np.clip(height, 0.0, 1.0)

    sea_level = 0.39
    ocean = np.clip((sea_level - height) / 0.18, 0.0, 1.0)
    shore = np.clip(1.0 - np.abs(height - sea_level) / 0.085, 0.0, 1.0)
    road_north = _gaussian_lane(x - (0.10 * np.sin(y * 4.0)), 0.030) * np.clip(1.0 - np.abs(y) * 0.78, 0.0, 1.0)
    road_east = _gaussian_lane(y + 0.10 - 0.055 * np.cos(x * 5.0), 0.030) * np.clip(1.0 - np.abs(x) * 0.78, 0.0, 1.0)
    ring = _gaussian_lane(np.sqrt((x / 0.42) ** 2 + ((y + 0.03) / 0.35) ** 2) - 0.92, 0.045)
    roads = np.clip(road_north + road_east + ring * 0.45, 0.0, 1.0) * (1.0 - ocean)

    village_core = np.exp(-((x / 0.42) ** 2 + ((y + 0.04) / 0.36) ** 2))
    village_noise = _box_blur(rng.random((tiles_y, tiles_x), dtype=np.float32), passes=2)
    village_plots = (village_noise > 0.53).astype(np.float32) * village_core * (1.0 - ocean)
    torches = roads * (0.55 + 0.45 * (np.sin(tile_x * 0.57 + tile_y * 0.31) > 0.78))

    mountain_mines = np.clip((height - 0.58) * 2.2, 0.0, 1.0) * (west_ridge + east_cliffs + north_mountains * 0.65)
    ore_lattice = np.clip(1.0 - np.abs(np.sin(tile_x * 0.23) * np.cos(tile_y * 0.19)) * 2.7, 0.0, 1.0)
    ore = mountain_mines * ore_lattice * (0.45 + 0.55 * rng.random((tiles_y, tiles_x), dtype=np.float32))

    moisture = np.clip(
        ocean * 0.85
        + river_network * 0.72
        + shore * 0.26
        + (1.0 - np.abs(y)) * 0.09
        + low_noise * 0.12
        + rng.normal(0.0, 0.018, (tiles_y, tiles_x)).astype(np.float32),
        0.0,
        1.0,
    )
    vegetation = np.clip(
        (1.0 - ocean)
        * (0.18 + moisture * 0.62 + mid_noise * 0.42 - roads * 0.78 - np.clip(height - 0.76, 0.0, 1.0) * 1.30),
        0.0,
        1.0,
    )
    activity = np.clip(village_plots * 0.92 + roads * 0.56 + torches * 0.24 + ore * 0.64, 0.0, 1.0)

    field = np.stack(
        [
            moisture.astype(np.float32),
            vegetation.astype(np.float32),
            height.astype(np.float32),
            activity.astype(np.float32),
        ],
        axis=-1,
    )
    field = np.nan_to_num(field, nan=0.0, posinf=1.0, neginf=0.0)
    field = np.clip(field, 0.0, 1.0).astype(np.float32)
    pixels = np.repeat(np.repeat(field, tile_size, axis=0), tile_size, axis=1)
    return pixels[:height_px, :width_px].copy()


SPEC = WorldSpec(
    id="minecraft-overworld-prime-3d",
    display_name="Minecraft Overworld Prime",
    window_title="Garage Life Lab - Minecraft Overworld Prime",
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={
        "tile_size": 10,
        "substeps": 28,
        "ray_steps": 140,
        "fx_intensity": 1.24,
        "contour_contrast": 1.18,
        "camera_speed": 0.78,
        "glow": 1.30,
        "exposure": 1.36,
    },
    preview_image=None,
    stability_notes=("long-term", "Minecraft-inspired", "block-readable"),
    hud_subtitle="MINECRAFT OVERWORLD PRIME",
    preview_palette=("#78b7ff", "#2479bf", "#61b34f", "#2e6f2f", "#8d6a35", "#b9bfc3", "#ffb347", "#45f0df"),
)
