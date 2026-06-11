"""Long-term Minecraft collaboration world."""
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
    float downhillWater = dot(gradH, vec2(r.r - l.r, t.r - b.r)) * 0.55;
    float waterBelow = smoothstep(0.18, 0.66, c.r);
    float canopy = smoothstep(0.25, 0.92, c.g);
    float localNoise = (hash12(uv * resolution + floor(time * 0.42)) - 0.5) * noiseStrength;
    float weather = smoothstep(0.66, 0.98, sin(time * 0.031 + hash12(floor(uv * 11.0)) * 6.28318) * 0.5 + 0.5);

    float localFeed = feed * 0.50 + 0.018 + waterBelow * 0.011 + localNoise * 0.035;
    float localKill = kill * 0.48 + 0.025 - c.a * 0.006 + parameterDrift * 0.35;
    float reaction = c.r * c.g * c.g * (0.45 + canopy * 0.25);

    float dr = diffU * lapR - reaction + localFeed * (1.0 - c.r) - downhillWater + weather * 0.0022;
    float dg = diffV * lapG + reaction - (localFeed + localKill) * c.g + c.r * 0.0035 - c.a * 0.0015;
    float db = lapB * 0.0015 + (c.g - 0.40) * c.a * 0.00055 + localNoise * 0.00075;

    vec2 eventCell = floor(uv * resolution / 10.0);
    float torchSpark = step(0.9988, hash12(eventCell + floor(time * 0.85)));
    float da = lapA * 0.052 + canopy * 0.0007 - c.a * 0.0014 + torchSpark * 0.020;

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
    float raw = state.b * 7.1 + macro * 2.2 + cliff * 1.35;
    return floor(raw) * (BLOCK_SIZE * 0.52) + 0.48;
}

float waterPresence(vec4 state, float h, float sea) {
    float lowLand = 1.0 - smoothstep(sea - 0.06, sea + 0.18, h);
    float riverOrOcean = smoothstep(0.52, 0.86, state.r);
    return lowLand * riverOrOcean;
}

float blockGrid(vec3 p) {
    vec3 q = abs(fract(p / BLOCK_SIZE) - 0.5);
    float edge = min(min(q.x, q.y), q.z);
    return 1.0 - smoothstep(0.018, 0.055, edge);
}

float rainAmount() {
    return smoothstep(0.62, 0.92, fbm(vec2(time * 0.018, 23.4)));
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

    float treeMask = chance(cell + vec2(11.7, 31.0), 0.155) * smoothstep(0.34, 0.92, state.g) * aboveWater;
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
    }

    float villageMask = chance(cell + vec2(107.0, 13.0), 0.044) * smoothstep(0.50, 0.92, state.a) * aboveWater * lowEnoughForVillage;
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

    float torchMask = chance(cell + vec2(51.0, 91.0), 0.034) * smoothstep(0.28, 0.88, state.a) * aboveWater;
    if (torchMask > 0.5) {
        float torch = boxSdf(
            vec3(p.x - center2.x, p.y - (h + 0.38), p.z - center2.y),
            vec3(BLOCK_SIZE * 0.075, 0.38, BLOCK_SIZE * 0.075)
        );
        takeShape(bestD, bestMat, torch, MAT_TORCH);
    }

    float caveMask = chance(cell + vec2(199.0, 23.0), 0.028) * step(sea + 1.70, h);
    if (caveMask > 0.5) {
        vec2 rel = p.xz - center2;
        float cave = boxSdf(
            vec3(rel.x, p.y - (h - 0.12), rel.y),
            vec3(BLOCK_SIZE * 0.44, 0.56, BLOCK_SIZE * 0.10)
        );
        takeShape(bestD, bestMat, cave, MAT_CAVE);
    }

    float oreMask = chance(cell + vec2(17.0, 181.0), 0.080) * smoothstep(sea + 1.10, sea + 3.80, h) * smoothstep(0.36, 0.96, state.a);
    if (oreMask > 0.5) {
        float ore = boxSdf(
            vec3(p.x - center2.x, p.y - (h + 0.08), p.z - center2.y),
            vec3(BLOCK_SIZE * 0.18, 0.14, BLOCK_SIZE * 0.18)
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

vec3 skyColorForRay(vec3 rd, vec3 lightDir, float safeFx, float rain) {
    float skyRise = smoothstep(-0.18, 0.86, rd.y);
    vec3 clearLow = vec3(0.46, 0.68, 0.94);
    vec3 clearHigh = vec3(0.76, 0.91, 1.00);
    vec3 stormLow = vec3(0.32, 0.40, 0.46);
    vec3 stormHigh = vec3(0.55, 0.61, 0.66);
    vec3 sky = mix(mix(clearLow, clearHigh, skyRise), mix(stormLow, stormHigh, skyRise), rain * 0.72);

    vec3 sunRight = normalize(cross(lightDir, vec3(0.0, 1.0, 0.0)));
    vec3 sunUp = normalize(cross(sunRight, lightDir));
    vec2 sunUv = vec2(dot(rd, sunRight), dot(rd, sunUp));
    float sun = squareDisc(sunUv, 0.060, 0.018) * smoothstep(0.88, 0.995, dot(rd, lightDir));
    sky += vec3(1.0, 0.86, 0.40) * sun * (1.35 + safeFx) * (1.0 - rain * 0.75);

    float clouds = cloudMask(rd);
    sky = mix(sky, vec3(0.91, 0.96, 1.00), clouds * (0.54 + rain * 0.18));
    sky = mix(sky, vec3(0.44, 0.49, 0.55), clouds * rain * 0.46);
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
        vec3 pale = vec3(0.66, 0.73, 0.62);
        vec3 leaves = mix(oak, lush, clamp(state.g + cellShade * 0.20, 0.0, 1.0));
        return mix(leaves, pale, smoothstep(0.10, 0.28, state.r) * smoothstep(0.12, 0.34, state.g) * 0.22);
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

    float top = smoothstep(0.34, 0.76, nor.y);
    float altitude = smoothstep(3.80, 5.85, pos.y);
    float snow = smoothstep(5.15, 6.05, pos.y) * top;
    float dry = smoothstep(0.30, 0.72, (1.0 - state.r) * (1.0 - state.g));
    float stoneFace = altitude * (1.0 - top * 0.28);
    float shore = smoothstep(0.45, 0.72, state.r) * (1.0 - altitude);

    vec3 grass = mix(vec3(0.15, 0.42, 0.10), vec3(0.46, 0.68, 0.18), clamp(state.g * 1.15 + cellShade * 0.12, 0.0, 1.0));
    vec3 dirt = mix(vec3(0.32, 0.19, 0.08), vec3(0.49, 0.31, 0.14), cellShade);
    vec3 sand = vec3(0.77, 0.68, 0.40);
    vec3 stone = mix(vec3(0.34, 0.35, 0.33), vec3(0.58, 0.59, 0.56), cellShade);
    vec3 snowColor = vec3(0.88, 0.94, 0.98);

    vec3 terrain = mix(dirt, grass, top);
    terrain = mix(terrain, sand, max(shore, dry * (1.0 - state.g)) * (1.0 - altitude));
    terrain = mix(terrain, stone, stoneFace);
    terrain = mix(terrain, snowColor, snow);
    terrain *= 1.0 - grid * 0.14 * contourPower;
    return terrain;
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float contourPower = clamp(contourContrast, 0.0, 1.8);
    int maxRaySteps = clamp(raySteps, 32, MAX_STEPS);

    float rain = rainAmount();
    float camTime = time * 0.092 * max(cameraSpeed, 0.05);
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

    float sunDrift = sin(time * 0.020 + colorShift * 0.12) * 0.10;
    vec3 lightDir = normalize(vec3(0.54 + sunDrift, 0.75, -0.42));
    vec3 skyColor = skyColorForRay(rd, lightDir, safeFx, rain);

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

        if (currMat == MAT_TORCH || currMat == MAT_ORE || currState.a > 0.58) {
            vec3 glowCol = mix(vec3(1.0, 0.58, 0.15), vec3(0.22, 0.95, 0.95), step(0.72, hash12(floor(pos.xz / BLOCK_SIZE))));
            float pulse = 0.78 + 0.22 * sin(time * 4.0 + currState.a * 9.0);
            volumeGlow += glowCol * (currState.a + float(currMat == MAT_TORCH) * 1.35 + float(currMat == MAT_ORE) * 0.45) * pulse * (0.007 + glow * 0.004) / (1.0 + abs(h) * 7.5);
        }

        if (h < max(SURF_DIST, 0.00125 * t) || t > MAX_DIST) {
            matID = currMat;
            state = currState;
            break;
        }
        t += clamp(h, 0.024, 0.72);
    }

    vec3 color = skyColor;
    if (t < MAX_DIST) {
        vec3 pos = ro + rd * t;
        vec3 nor = calcNormal(pos);
        float sha = calcSoftShadow(pos + nor * 0.025, lightDir, 0.05, 14.0, 10.0);
        float ao = calcAO(pos, nor);
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
            matColor = mix(matColor, skyColorForRay(ref, lightDir, safeFx, rain), fresnel * 0.24);
            matColor += volumeGlow * 0.12;
        }

        vec3 lin = vec3(0.0);
        lin += 1.78 * dif * vec3(1.0, 0.92, 0.78) * ao;
        lin += 0.68 * skyLight * vec3(0.42, 0.55, 0.72) * ao;
        lin += 0.24 * fre * vec3(0.84, 0.92, 1.0);

        color = matColor * lin + emission;
        float fog = 1.0 - exp(-0.00115 * t * t);
        fog = clamp(fog + rain * smoothstep(20.0, 56.0, t) * 0.22, 0.0, 0.88);
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


WORLD_SEED = 20260611


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

    macro = (
        0.46
        + 0.16 * np.sin(x_norm * 7.2 + np.cos(y_norm * 5.6) * 1.35)
        + 0.12 * np.cos(y_norm * 9.1 - 0.7)
        + 0.07 * np.sin((x_norm + y_norm) * 15.0)
        + 0.05 * np.cos((x_norm - y_norm) * 25.0)
    )
    height = macro + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.020

    for _ in range(max(7, (tiles_x * tiles_y) // 3000)):
        blob = _ellipse(
            tile_x,
            tile_y,
            rng.uniform(0.0, tiles_x),
            rng.uniform(0.0, tiles_y),
            rng.uniform(max(6.0, tiles_x * 0.05), max(14.0, tiles_x * 0.20)),
            rng.uniform(max(6.0, tiles_y * 0.05), max(14.0, tiles_y * 0.20)),
        )
        height += blob * rng.uniform(-0.12, 0.22)

    mountain_ridges = np.abs(np.sin(x_norm * 22.0 + np.cos(y_norm * 12.0) * 2.4))
    height += np.clip(0.34 - mountain_ridges, 0.0, 1.0) * 0.15

    river_wave = np.abs(np.sin(x_norm * 17.0 + np.sin(y_norm * 9.5) * 2.7))
    river_mask = np.clip((0.17 - river_wave) / 0.17, 0.0, 1.0)
    height -= river_mask * 0.18
    height = np.clip(height, 0.0, 1.0)

    sea_level = 0.395
    ocean = np.clip((sea_level - height) / 0.16, 0.0, 1.0)
    shore = np.clip(1.0 - np.abs(height - sea_level) / 0.075, 0.0, 1.0)
    latitude = 1.0 - np.abs(y_norm * 2.0 - 1.0)

    moisture = np.clip(
        ocean * 0.84
        + river_mask * 0.58
        + shore * 0.24
        + latitude * 0.10
        + 0.08 * np.sin(x_norm * 8.0 - y_norm * 6.5)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.025,
        0.0,
        1.0,
    )

    forest_noise = (
        0.42
        + 0.25 * np.sin(x_norm * 18.0 + y_norm * 5.5)
        + 0.16 * np.cos(y_norm * 19.0 - x_norm * 4.2)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.040
    )
    vegetation = np.clip(
        (1.0 - ocean)
        * (
            forest_noise
            + moisture * 0.38
            + latitude * 0.16
            - np.clip(height - 0.78, 0.0, 1.0) * 1.45
            - np.clip(0.34 - moisture, 0.0, 1.0) * 0.60
        ),
        0.0,
        1.0,
    )

    village = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    village_candidates = np.argwhere(
        (ocean < 0.22)
        & (shore > 0.18)
        & (vegetation > 0.20)
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

    cave_lattice = np.abs(np.sin(x_norm * 33.0) * np.cos(y_norm * 29.0))
    ore = np.clip(1.0 - cave_lattice * 3.0, 0.0, 1.0) * np.clip((height - 0.46) * 1.9, 0.0, 1.0)
    mountain_light = np.clip((height - 0.74) * 2.0, 0.0, 1.0) * rng.random((tiles_y, tiles_x), dtype=np.float32) * 0.18
    light = np.clip(village + ore * 0.64 + mountain_light, 0.0, 1.0)

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
    id="minecraft-long-term-3d",
    display_name="Minecraft Long Term",
    window_title="Garage Life Lab - Minecraft Long Term",
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={
        "ray_steps": 132,
        "fx_intensity": 1.22,
        "contour_contrast": 1.12,
        "camera_speed": 0.82,
        "tile_size": 10,
    },
    preview_image="assets/world_previews/minecraft-long-term-3d.png",
    stability_notes=("new default", "long-term", "blocky 3D"),
    hud_subtitle="MINECRAFT LONG TERM",
    preview_palette=("#5bb8ff", "#92d85f", "#3f8c2d", "#8a612c", "#59615d", "#e3d187", "#ffbb45"),
)
