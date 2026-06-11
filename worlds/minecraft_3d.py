"""World definition for Minecraft-inspired Overworld."""
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
    // R: water table / river energy
    // G: grass, leaves, and farmable biomass
    // B: stepped terrain elevation
    // A: ore, torch, and village light energy
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
    float waterSlide = dot(gradH, vec2(r.r - l.r, t.r - b.r)) * 0.45;
    float localNoise = (hash12(uv * resolution + floor(time * 0.35)) - 0.5) * noiseStrength;
    float dayPulse = 0.5 + 0.5 * sin(time * 0.08 + hash12(floor(uv * 16.0)) * 6.28318);
    float rain = smoothstep(0.84, 1.0, dayPulse) * (0.55 + 0.45 * hash12(floor(uv * 10.0)));

    float localFeed = feed * 0.58 + 0.014 + c.r * 0.010 + localNoise * 0.05;
    float localKill = kill * 0.52 + 0.022 - c.a * 0.006 + parameterDrift * 0.4;
    float reaction = c.r * c.g * c.g * 0.55;

    float dr = diffU * lapR - reaction + localFeed * (1.0 - c.r) - waterSlide + rain * 0.0025;
    float dg = diffV * lapG + reaction - (localFeed + localKill) * c.g + c.r * 0.004 - c.a * 0.0018;
    float db = lapB * 0.002 + (c.g - 0.42) * 0.00055 + localNoise * 0.0009;

    vec2 lightCell = floor(uv * resolution / 12.0);
    float spark = step(0.9986, hash12(lightCell + floor(time * 0.7)));
    float da = lapA * 0.045 + c.g * 0.0008 - c.a * 0.0017 + spark * 0.026;

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
#define MAX_DIST 62.0
#define SURF_DIST 0.003
#define BLOCK_SIZE 0.76
#define MAP_SCALE 0.036

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
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
    mat2 rot = mat2(0.8, -0.6, 0.6, 0.8);
    for (int i = 0; i < 4; i++) {
        v += a * noise(p);
        p = rot * p * 2.03 + vec2(17.1, 4.7);
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

float terrainHeight(vec4 state, vec2 cell) {
    float macro = fbm(cell * 0.035 + vec2(2.0, 9.0));
    float ridges = 1.0 - abs(fbm(cell * 0.080) * 2.0 - 1.0);
    float raw = state.b * 6.2 + macro * 2.3 + ridges * 1.2;
    return floor(raw) * (BLOCK_SIZE * 0.52) + 0.55;
}

float waterPresence(vec4 state, float h, float sea) {
    float lowTerrain = 1.0 - smoothstep(sea - 0.04, sea + 0.16, h);
    float riverOrOcean = smoothstep(0.46, 0.80, state.r);
    return lowTerrain * riverOrOcean;
}

float map(in vec3 p, out int matID, out vec4 stateOut) {
    vec2 cell = floor(p.xz / BLOCK_SIZE);
    vec2 mapUV = (cell + 0.5) * BLOCK_SIZE * MAP_SCALE;
    vec4 state = textureLod(stateTex, mapUV, 0.0);
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float h = terrainHeight(state, cell);
    float sea = 1.18;

    float d = p.y - h;
    matID = 1;
    stateOut = state;

    float treeChance = hash12(cell + vec2(21.7, 8.9));
    float treeMask = step(treeChance, 0.11) * step(0.42, state.g) * step(sea + 0.18, h);
    if (treeMask > 0.5) {
        vec2 center = (cell + 0.5) * BLOCK_SIZE;
        float trunk = boxSdf(
            vec3(p.x - center.x, p.y - (h + 0.58), p.z - center.y),
            vec3(BLOCK_SIZE * 0.16, 0.60, BLOCK_SIZE * 0.16)
        );
        if (trunk < d) {
            d = trunk;
            matID = 3;
        }

        float leafShift = hash12(cell + vec2(3.3, 91.1)) - 0.5;
        float leaves = boxSdf(
            vec3(p.x - center.x - leafShift * 0.10, p.y - (h + 1.45), p.z - center.y + leafShift * 0.08),
            vec3(BLOCK_SIZE * 0.82, 0.78, BLOCK_SIZE * 0.82)
        );
        if (leaves < d) {
            d = leaves;
            matID = 4;
        }
    }

    float houseChance = hash12(cell + vec2(107.2, 13.5));
    float houseMask = step(houseChance, 0.035) * step(0.36, state.a) * step(sea + 0.08, h) * (1.0 - step(3.85, h));
    if (houseMask > 0.5) {
        vec2 center = (cell + 0.5) * BLOCK_SIZE;
        float hut = boxSdf(
            vec3(p.x - center.x, p.y - (h + 0.52), p.z - center.y),
            vec3(BLOCK_SIZE * 0.72, 0.52, BLOCK_SIZE * 0.62)
        );
        if (hut < d) {
            d = hut;
            matID = 5;
        }

        float roof = boxSdf(
            vec3(p.x - center.x, p.y - (h + 1.11), p.z - center.y),
            vec3(BLOCK_SIZE * 0.83, 0.20, BLOCK_SIZE * 0.73)
        );
        if (roof < d) {
            d = roof;
            matID = 6;
        }
    }

    float torchChance = hash12(cell + vec2(43.0, 71.0));
    float torchMask = step(torchChance, 0.022) * step(0.25, state.a) * step(sea + 0.05, h);
    if (torchMask > 0.5) {
        vec2 center = (cell + 0.5) * BLOCK_SIZE;
        float torch = boxSdf(
            vec3(p.x - center.x, p.y - (h + 0.38), p.z - center.y),
            vec3(BLOCK_SIZE * 0.08, 0.38, BLOCK_SIZE * 0.08)
        );
        if (torch < d) {
            d = torch;
            matID = 7;
        }
    }

    float waterWave = (floor(noise(cell * 0.16 + vec2(time * 0.07, -time * 0.04)) * 3.0) - 1.0) * 0.012 * safeFx;
    float waterMask = waterPresence(state, h, sea);
    if (waterMask > 0.04) {
        float dWater = p.y - (sea + waterWave);
        if (dWater < d) {
            d = dWater * 0.85;
            matID = 0;
        }
    }

    return d;
}

vec3 calcNormal(in vec3 p) {
    const vec2 e = vec2(0.035, 0.0);
    int mat; vec4 state;
    return normalize(vec3(
        map(p + e.xyy, mat, state) - map(p - e.xyy, mat, state),
        map(p + e.yxy, mat, state) - map(p - e.yxy, mat, state),
        map(p + e.yyx, mat, state) - map(p - e.yyx, mat, state)
    ));
}

float calcShadow(in vec3 ro, in vec3 rd) {
    float res = 1.0;
    float t = 0.06;
    int mat; vec4 state;
    for (int i = 0; i < 38; i++) {
        float h = map(ro + rd * t, mat, state);
        res = min(res, 10.0 * h / t);
        t += clamp(h, 0.035, 0.55);
        if (h < 0.001 || t > 14.0) {
            break;
        }
    }
    return clamp(res, 0.0, 1.0);
}

float blockGrid(vec3 p) {
    vec3 q = abs(fract(p / BLOCK_SIZE) - 0.5);
    float edge = min(min(q.x, q.y), q.z);
    return 1.0 - smoothstep(0.018, 0.060, edge);
}

vec3 skyColorForRay(vec3 rd, vec3 lightDir, float safeFx) {
    float skyRise = smoothstep(-0.25, 0.85, rd.y);
    vec3 sky = mix(vec3(0.43, 0.68, 0.96), vec3(0.70, 0.88, 1.00), skyRise);

    vec3 sunRight = normalize(cross(lightDir, vec3(0.0, 1.0, 0.0)));
    vec3 sunUp = normalize(cross(sunRight, lightDir));
    vec2 sunUv = vec2(dot(rd, sunRight), dot(rd, sunUp));
    float sunSquare = 1.0 - smoothstep(0.060, 0.094, max(abs(sunUv.x), abs(sunUv.y)));
    sunSquare *= smoothstep(0.92, 0.995, dot(rd, lightDir));
    sky += vec3(1.0, 0.88, 0.45) * sunSquare * (1.5 + safeFx);

    if (rd.y > 0.03) {
        vec2 cloudPos = rd.xz / max(rd.y, 0.05) * 2.3 + vec2(time * 0.018, -time * 0.007);
        vec2 cloudCell = floor(cloudPos * 4.0);
        float cloud = step(0.62, fbm(cloudCell * 0.18));
        float cloudBand = smoothstep(0.18, 0.42, rd.y) * (1.0 - smoothstep(0.70, 0.92, rd.y));
        sky = mix(sky, vec3(0.92, 0.96, 1.0), cloud * cloudBand * 0.55);
    }

    return sky;
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float contourPower = clamp(contourContrast, 0.0, 1.8);
    int maxRaySteps = clamp(raySteps, 32, MAX_STEPS);

    float camTime = time * 0.105 * max(cameraSpeed, 0.05);
    vec3 ro = vec3(
        camTime * 5.6 + sin(camTime * 0.31) * 3.0,
        6.6 + sin(camTime * 0.43) * 0.72,
        camTime * 4.8 + cos(camTime * 0.27) * 3.0
    );
    vec3 ta = vec3(ro.x + 5.7, 2.85 + sin(camTime * 0.40) * 0.34, ro.z + 5.2);
    ro += cameraOffset;
    ta += cameraOffset;

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.17) * 0.035);
    vec3 rd = ca * cameraInputRay(p, 2.0);

    vec3 lightDir = normalize(vec3(0.58, 0.72, -0.38));
    vec3 sky = skyColorForRay(rd, lightDir, safeFx);

    float t = 0.0;
    int matID = -1;
    vec4 state = vec4(0.0);
    vec3 glowAccum = vec3(0.0);

    for (int i = 0; i < MAX_STEPS; i++) {
        if (i >= maxRaySteps) {
            break;
        }
        vec3 pos = ro + rd * t;
        int currMat;
        vec4 currState;
        float h = map(pos, currMat, currState);
        if (currState.a > 0.38 || currMat == 7) {
            vec3 torchColor = vec3(1.0, 0.56, 0.16);
            float pulse = 0.75 + 0.25 * sin(time * 4.0 + currState.a * 12.0);
            glowAccum += torchColor * (currState.a + float(currMat == 7) * 1.4) * pulse * (0.008 + glow * 0.004) / (1.0 + abs(h) * 8.0);
        }
        if (h < max(SURF_DIST, 0.0014 * t) || t > MAX_DIST) {
            matID = currMat;
            state = currState;
            break;
        }
        t += clamp(h, 0.026, 0.72);
    }

    vec3 color = sky;
    if (t < MAX_DIST) {
        vec3 pos = ro + rd * t;
        vec3 nor = calcNormal(pos);
        float sha = calcShadow(pos + nor * 0.025, lightDir);
        float dif = clamp(dot(nor, lightDir), 0.0, 1.0) * sha;
        float amb = clamp(0.45 + 0.55 * nor.y, 0.0, 1.0);
        float fre = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 3.0);
        float grid = blockGrid(pos) * contourPower;

        vec3 matColor = vec3(0.0);
        vec3 emission = vec3(0.0);
        float sea = 1.18;

        if (matID == 0) {
            vec3 shallow = vec3(0.12, 0.52, 0.82);
            vec3 deep = vec3(0.02, 0.14, 0.34);
            float waterDepth = clamp((sea - terrainHeight(state, floor(pos.xz / BLOCK_SIZE))) * 0.42, 0.0, 1.0);
            matColor = mix(shallow, deep, waterDepth);
            matColor += vec3(0.28, 0.58, 0.86) * grid * 0.12;
            float spe = pow(clamp(dot(reflect(rd, nor), lightDir), 0.0, 1.0), 36.0) * sha;
            emission += vec3(0.8, 0.95, 1.0) * spe * (0.45 + safeFx * 0.25);
        } else if (matID == 3) {
            vec3 barkA = vec3(0.34, 0.18, 0.08);
            vec3 barkB = vec3(0.55, 0.32, 0.13);
            float bark = step(0.5, fract(pos.y * 3.8 + floor(pos.x / BLOCK_SIZE) * 0.37));
            matColor = mix(barkA, barkB, bark);
        } else if (matID == 4) {
            vec3 leafA = vec3(0.09, 0.36, 0.10);
            vec3 leafB = vec3(0.26, 0.58, 0.17);
            matColor = mix(leafA, leafB, clamp(state.g + noise(pos.xz * 2.1), 0.0, 1.0));
            matColor *= 1.0 - grid * 0.10;
        } else if (matID == 5) {
            vec3 plankA = vec3(0.55, 0.34, 0.15);
            vec3 plankB = vec3(0.72, 0.50, 0.24);
            matColor = mix(plankA, plankB, step(0.5, fract(pos.x * 2.0 + pos.y * 1.5)));
        } else if (matID == 6) {
            matColor = mix(vec3(0.36, 0.11, 0.07), vec3(0.62, 0.20, 0.11), noise(pos.xz * 3.0));
        } else if (matID == 7) {
            matColor = vec3(0.38, 0.18, 0.06);
            emission += vec3(2.8, 1.25, 0.24) * (1.0 + glow) * safeFx;
        } else {
            float top = smoothstep(0.36, 0.72, nor.y);
            float shore = 1.0 - smoothstep(sea + 0.08, sea + 0.68, pos.y);
            float highland = smoothstep(3.45, 4.95, pos.y);
            float snow = smoothstep(5.05, 5.90, pos.y) * top;

            vec3 grass = mix(vec3(0.13, 0.42, 0.10), vec3(0.45, 0.66, 0.18), clamp(state.g * 1.15, 0.0, 1.0));
            vec3 dirt = mix(vec3(0.33, 0.19, 0.08), vec3(0.46, 0.28, 0.13), noise(pos.xz * 2.0));
            vec3 sand = vec3(0.76, 0.66, 0.38);
            vec3 stone = mix(vec3(0.34, 0.35, 0.33), vec3(0.55, 0.56, 0.52), noise(pos.xz * 5.0));
            vec3 snowColor = vec3(0.88, 0.94, 0.98);

            matColor = mix(dirt, grass, top);
            matColor = mix(matColor, sand, shore * (1.0 - highland));
            matColor = mix(matColor, stone, highland * (1.0 - top * 0.35));
            matColor = mix(matColor, snowColor, snow);

            float ore = smoothstep(0.55, 0.95, state.a) * (1.0 - top) * step(2.25, pos.y);
            vec3 oreColor = mix(vec3(0.18, 0.92, 0.92), vec3(0.95, 0.70, 0.24), hash12(floor(pos.xz / BLOCK_SIZE)));
            matColor = mix(matColor, oreColor, ore * 0.38);
            emission += oreColor * ore * (0.16 + glow * 0.20) * safeFx;
        }

        matColor *= 1.0 - grid * 0.18;

        vec3 lin = vec3(0.0);
        lin += 1.72 * dif * vec3(1.0, 0.91, 0.78);
        lin += 0.62 * amb * vec3(0.42, 0.55, 0.72);
        lin += 0.26 * fre * vec3(0.85, 0.92, 1.0);

        color = matColor * lin + emission;
        float fog = 1.0 - exp(-0.0011 * t * t);
        color = mix(color, sky, fog);
    }

    color += glowAccum * (1.0 + safeFx * 0.7);
    color += vec3(0.03, 0.05, 0.08) * smoothstep(0.65, 1.0, rd.y) * sin(colorShift + time * 0.08) * 0.15;

    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(color, vec3(1.0 / max(gamma, 0.2)));

    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.58 + 0.42 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.22);

    fragColor = vec4(color, 1.0);
}
"""


def seed_field(width_px: int, height_px: int, tile_size: int) -> np.ndarray:
    tile_size = max(2, int(tile_size))
    tiles_x = max(1, int(np.ceil(width_px / tile_size)))
    tiles_y = max(1, int(np.ceil(height_px / tile_size)))

    rng = np.random.default_rng(19830517)
    tile_y, tile_x = np.meshgrid(
        np.arange(tiles_y, dtype=np.float32),
        np.arange(tiles_x, dtype=np.float32),
        indexing="ij",
    )
    x_norm = tile_x / max(tiles_x - 1, 1)
    y_norm = tile_y / max(tiles_y - 1, 1)

    continent = (
        0.44
        + 0.17 * np.sin(x_norm * 7.7 + np.cos(y_norm * 5.1) * 1.3)
        + 0.13 * np.cos(y_norm * 8.6 - 0.8)
        + 0.08 * np.sin((x_norm + y_norm) * 14.0)
        + 0.05 * np.cos((x_norm - y_norm) * 23.0)
    )
    height = continent + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.025

    biome_count = max(6, (tiles_x * tiles_y) // 3300)
    for _ in range(biome_count):
        cx = rng.uniform(0.0, tiles_x)
        cy = rng.uniform(0.0, tiles_y)
        rx = rng.uniform(max(6.0, tiles_x * 0.05), max(14.0, tiles_x * 0.20))
        ry = rng.uniform(max(6.0, tiles_y * 0.05), max(14.0, tiles_y * 0.20))
        distance = ((tile_x - cx) / rx) ** 2 + ((tile_y - cy) / ry) ** 2
        influence = np.clip(1.0 - distance, 0.0, 1.0)
        height += influence * rng.uniform(-0.13, 0.24)

    river_bands = np.abs(np.sin(x_norm * 16.0 + np.sin(y_norm * 10.0) * 2.6))
    river_mask = np.clip((0.18 - river_bands) / 0.18, 0.0, 1.0)
    height -= river_mask * 0.20
    height = np.clip(height, 0.0, 1.0)

    sea_level = 0.41
    water = np.clip((sea_level - height) / 0.18, 0.0, 1.0)
    shore = np.clip(1.0 - np.abs(height - sea_level) / 0.08, 0.0, 1.0)
    latitude = 1.0 - np.abs(y_norm * 2.0 - 1.0)
    forest_noise = (
        0.47
        + 0.26 * np.sin(x_norm * 18.0 + y_norm * 5.0)
        + 0.18 * np.cos(y_norm * 17.0 - x_norm * 4.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.05
    )
    forest = np.clip(
        (1.0 - water)
        * (forest_noise + latitude * 0.28 + shore * 0.20 - np.clip(height - 0.78, 0.0, 1.0) * 1.25),
        0.0,
        1.0,
    )

    settlement = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    candidates = np.argwhere(
        (water < 0.35)
        & (shore > 0.24)
        & (forest > 0.30)
        & (height > sea_level + 0.03)
        & (height < 0.67)
    )
    if len(candidates) > 0:
        village_count = min(max(5, (tiles_x * tiles_y) // 2400), len(candidates))
        for candidate_index in rng.choice(len(candidates), size=village_count, replace=False):
            cy, cx = candidates[candidate_index]
            radius = int(rng.integers(2, 5))
            y0 = max(cy - radius, 0)
            y1 = min(cy + radius + 1, tiles_y)
            x0 = max(cx - radius, 0)
            x1 = min(cx + radius + 1, tiles_x)
            patch_y, patch_x = np.meshgrid(
                np.arange(y0, y1, dtype=np.float32),
                np.arange(x0, x1, dtype=np.float32),
                indexing="ij",
            )
            distance = np.sqrt((patch_x - cx) ** 2 + (patch_y - cy) ** 2)
            influence = np.clip(1.0 - distance / max(radius + 0.5, 1.0), 0.0, 1.0)
            settlement[y0:y1, x0:x1] = np.maximum(
                settlement[y0:y1, x0:x1],
                influence * rng.uniform(0.45, 0.86),
            )

    cave_lattice = np.abs(np.sin(x_norm * 31.0) * np.cos(y_norm * 27.0))
    ore = np.clip(1.0 - cave_lattice * 3.1, 0.0, 1.0) * np.clip((height - 0.45) * 1.8, 0.0, 1.0)
    light = np.clip(settlement + ore * 0.72 + rng.random((tiles_y, tiles_x), dtype=np.float32) * 0.025, 0.0, 1.0)

    water_energy = np.clip(water * 0.82 + river_mask * 0.48 + shore * 0.16, 0.0, 1.0)
    grass_energy = np.clip(forest * 0.78 + shore * 0.10 + (1.0 - water) * 0.10, 0.0, 1.0)

    tile_field = np.stack(
        [
            water_energy.astype(np.float32),
            grass_energy.astype(np.float32),
            height.astype(np.float32),
            light.astype(np.float32),
        ],
        axis=-1,
    )

    field = np.repeat(np.repeat(tile_field, tile_size, axis=0), tile_size, axis=1)
    return field[:height_px, :width_px].copy()


SPEC = WorldSpec(
    id="minecraft-3d",
    display_name="Minecraft",
    window_title="Garage Life Lab - Minecraft Overworld",
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={
        "ray_steps": 118,
        "fx_intensity": 1.15,
        "contour_contrast": 1.05,
        "camera_speed": 0.9,
    },
    preview_image="assets/world_previews/minecraft-3d.png",
    stability_notes=("blocky 3D default", "safe"),
    hud_subtitle="MINECRAFT OVERWORLD",
    preview_palette=("#5fb8ff", "#74d36b", "#3f8b25", "#7a522c", "#575b58", "#e5d081", "#ffb23a"),
)
