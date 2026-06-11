"""Overdrive Amplified Minecraft world with raytraced reflections and 3D voxel carving."""
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
    // R: Superheated Magma (flows, diffuses)
    // G: Volatile Crystal Infection (spreads near magma, solidifies)
    // B: Terrain Elevation (geologic shift, cools from magma)
    // A: Bioluminescent Spores / Energy (bursts and travels along crystals)
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
    
    // Advection: Magma flows violently down steep gradients
    float magmaSlide = dot(gradH, vec2(r.r - l.r, t.r - b.r)) * 0.85;
    
    float localNoise = (hash12(uv * resolution + floor(time * 0.5)) - 0.5) * noiseStrength;
    
    // Crystal feeds on Magma, Magma is consumed
    float reaction = c.r * c.g * c.g * 0.65;
    
    // Magma slowly hardens into elevation if it gets too thin and cold
    float cooling = smoothstep(0.0, 0.4, 0.45 - c.r) * c.r * 0.015;

    float localFeed = feed * 0.45 + 0.02 + c.b * 0.01 + localNoise * 0.04;
    float localKill = kill * 0.55 + 0.02 - c.a * 0.008 + parameterDrift * 0.5;

    float dr = diffU * lapR - reaction + localFeed * (1.0 - c.r) - magmaSlide - cooling;
    float dg = diffV * lapG + reaction - (localFeed + localKill) * c.g + c.a * 0.005;
    float db = lapB * 0.0025 + cooling * 0.2 - magmaSlide * 0.05;

    // Energy sparks on mature crystals
    vec2 eventCell = floor(uv * resolution / 8.0);
    float spark = step(0.999, hash12(eventCell + floor(time * 1.2))) * smoothstep(0.5, 1.0, c.g);
    float da = lapA * 0.06 + c.g * 0.002 - c.a * 0.004 + spark * 0.04;

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

#define MAX_STEPS 180
#define MAX_DIST 75.0
#define SURF_DIST 0.002
#define BLOCK_SIZE 0.75
#define MAP_SCALE 0.035

const int MAT_OBSIDIAN = 1;
const int MAT_LAVA     = 2;
const int MAT_CRYSTAL  = 3;
const int MAT_CAVE     = 4;
const int MAT_SPORE    = 5;

// Hash and Noise
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float hash13(vec3 p3) {
    p3  = fract(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
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

float noise3d(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash13(p), hash13(p + vec3(1,0,0)), f.x),
            mix(hash13(p + vec3(0,1,0)), hash13(p + vec3(1,1,0)), f.x), f.y),
        mix(mix(hash13(p + vec3(0,0,1)), hash13(p + vec3(1,0,1)), f.x),
            mix(hash13(p + vec3(0,1,1)), hash13(p + vec3(1,1,1)), f.x), f.y), f.z
    );
}

float fbm(vec2 p) {
    float v = 0.0; float a = 0.5;
    mat2 rot = mat2(0.866, -0.5, 0.5, 0.866);
    for (int i = 0; i < 4; i++) {
        v += a * noise(p);
        p = rot * p * 2.02 + vec2(17.31, 91.73);
        a *= 0.5;
    }
    return v;
}

float fbm3d(vec3 p) {
    float f = 0.0; float amp = 0.5;
    for (int i = 0; i < 3; i++) {
        f += amp * noise3d(p);
        p *= 2.01; amp *= 0.5;
    }
    return f;
}

// Camera Setup
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

// Geometry
float boxSdf(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

vec4 worldState(vec2 cell) {
    return textureLod(stateTex, fract((cell + 0.5) * BLOCK_SIZE * MAP_SCALE), 0.0);
}

// Core SDF
float map(in vec3 p, out int matID, out vec4 stateOut) {
    vec3 cell3d = floor(p / BLOCK_SIZE);
    vec2 cell2d = cell3d.xz;
    vec4 state = worldState(cell2d);
    
    // Extreme amplified heightmap
    float macro = fbm(cell2d * 0.025 + vec2(13.0, 7.0));
    float ridges = pow(1.0 - abs(fbm(cell2d * 0.07) * 2.0 - 1.0), 2.0);
    float rawHeight = state.b * 16.0 + macro * 5.0 + ridges * 3.5;
    float blockHeight = floor(rawHeight) * BLOCK_SIZE;

    float bestD = p.y - blockHeight;
    int bestMat = MAT_OBSIDIAN;
    stateOut = state;

    // 3D Voxel Cave Carving (Amplified overhangs)
    float caveNoise = fbm3d(p * 0.18 + vec3(time * 0.015, 0.0, time * 0.01)) - 0.45;
    if (caveNoise > 0.0 && p.y < blockHeight - BLOCK_SIZE) {
        // Soften the SDF adjustment to avoid raymarching tears
        bestD = max(bestD, caveNoise * 0.85); 
        bestMat = MAT_CAVE;
    }

    // Magma Rivers
    float lavaLevel = 1.8 + smoothstep(0.4, 0.8, state.r) * 2.5;
    if (state.r > 0.15 && p.y < lavaLevel) {
        float lavaWave = (noise(cell2d * 0.2 + time * 0.2) - 0.5) * 0.05;
        float dLava = p.y - (floor(lavaLevel) * BLOCK_SIZE + lavaWave);
        if (dLava < bestD) {
            bestD = dLava * 0.8;
            bestMat = MAT_LAVA;
        }
    }

    // Spore / Crystal growths
    vec2 center = (cell2d + 0.5) * BLOCK_SIZE;
    float crystalMask = chance(cell2d + vec2(9.1, 8.2), 0.2) * smoothstep(0.3, 0.9, state.g);
    if (crystalMask > 0.5 && p.y > blockHeight) {
        float hCluster = mix(0.5, 2.5, hash12(cell2d));
        float crystal = boxSdf(
            vec3(p.x - center.x, p.y - (blockHeight + hCluster * 0.5), p.z - center.y),
            vec3(BLOCK_SIZE * 0.35, hCluster * 0.5, BLOCK_SIZE * 0.35)
        );
        if (crystal < bestD) {
            bestD = crystal;
            bestMat = MAT_CRYSTAL;
        }

        // Floating Spore cap
        if (chance(cell2d, 0.15) > 0.5) {
            float floatOffset = sin(time * 1.5 + hash12(cell2d) * 6.0) * 0.15;
            float spore = boxSdf(
                vec3(p.x - center.x, p.y - (blockHeight + hCluster + 1.2 + floatOffset), p.z - center.y),
                vec3(BLOCK_SIZE * 0.8, 0.3, BLOCK_SIZE * 0.8)
            );
            if (spore < bestD) {
                bestD = spore;
                bestMat = MAT_SPORE;
            }
        }
    }

    matID = bestMat;
    return bestD;
}

vec3 calcNormal(in vec3 p) {
    const vec2 e = vec2(0.025, 0.0);
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
    for (int i = 0; i < 40; i++) {
        float h = map(ro + rd * t, mat, state);
        res = min(res, k * h / t);
        t += clamp(h, 0.02, 0.6);
        if (res < 0.002 || t > tmax) break;
    }
    return clamp(res, 0.0, 1.0);
}

float calcAO(vec3 pos, vec3 nor) {
    float occ = 0.0;
    float sca = 1.0;
    int mat; vec4 state;
    for (int i = 0; i < 5; i++) {
        float h = 0.025 + 0.15 * float(i);
        float d = map(pos + nor * h, mat, state);
        occ += (h - d) * sca;
        sca *= 0.65;
    }
    return clamp(1.0 - 2.5 * occ, 0.0, 1.0);
}

// Visuals
float blockGrid(vec3 p) {
    vec3 q = abs(fract(p / BLOCK_SIZE) - 0.5);
    float edge = min(min(q.x, q.y), q.z);
    return 1.0 - smoothstep(0.015, 0.06, edge);
}

vec3 getSkyColor(vec3 rd, float safeFx) {
    vec3 deep = vec3(0.02, 0.03, 0.05);
    vec3 haze = vec3(0.35, 0.15, 0.1) * (1.0 + safeFx * 0.2);
    float horizon = smoothstep(-0.2, 0.5, rd.y);
    vec3 sky = mix(haze, deep, horizon);
    
    // Distant volumetric ash storms
    float storm = fbm(rd.xz * 3.0 + time * 0.05);
    sky += vec3(0.8, 0.3, 0.05) * smoothstep(0.6, 1.0, storm) * (1.0 - horizon) * 0.5 * safeFx;
    
    return sky;
}

vec3 getMaterialColor(int matID, vec3 pos, vec3 nor, vec3 rd, vec4 state, float grid, float safeFx, float contourPower, out vec3 emission) {
    emission = vec3(0.0);
    float cellHash = hash12(floor(pos.xz / BLOCK_SIZE) + floor(pos.y / BLOCK_SIZE) * 11.3);

    if (matID == MAT_LAVA) {
        float flow = fbm(pos.xz * 1.5 - time * 0.8);
        vec3 magmaHot = vec3(1.2, 0.9, 0.2);
        vec3 magmaCold = vec3(0.8, 0.2, 0.0);
        vec3 col = mix(magmaCold, magmaHot, smoothstep(0.3, 0.8, flow));
        emission = col * (0.8 + glow * 0.6) * safeFx;
        return col;
    }
    if (matID == MAT_CRYSTAL) {
        vec3 cA = vec3(0.05, 0.8, 0.9);
        vec3 cB = vec3(0.4, 0.1, 0.9);
        vec3 col = mix(cA, cB, cellHash);
        float pulse = 0.5 + 0.5 * sin(time * 3.0 + state.a * 10.0);
        emission = col * (state.a * 2.0 + pulse * 0.5) * safeFx * (0.5 + glow * 0.5);
        return col;
    }
    if (matID == MAT_SPORE) {
        vec3 col = vec3(0.1, 0.9, 0.4);
        emission = col * (1.0 + sin(time * 2.0 + pos.x)) * 0.8 * safeFx;
        return col;
    }
    if (matID == MAT_CAVE) {
        // Subterranean glowing moss
        vec3 col = vec3(0.05, 0.06, 0.08);
        float moss = smoothstep(0.7, 1.0, fbm3d(pos * 0.5));
        vec3 mossCol = vec3(0.1, 0.8, 0.5);
        emission = mossCol * moss * state.g * safeFx * 0.8;
        return mix(col, mossCol * 0.2, moss);
    }
    
    // MAT_OBSIDIAN
    vec3 darkRock = mix(vec3(0.03, 0.04, 0.05), vec3(0.08, 0.09, 0.11), cellHash);
    vec3 highlight = vec3(0.2, 0.15, 0.25);
    vec3 col = mix(darkRock, highlight, smoothstep(0.8, 1.0, state.b));
    col *= 1.0 - grid * 0.25 * contourPower;
    return col;
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float contourPower = clamp(contourContrast, 0.0, 1.8);
    int maxRaySteps = clamp(raySteps, 40, MAX_STEPS);

    float camTime = time * 0.12 * max(cameraSpeed, 0.05);
    vec3 ro = vec3(
        camTime * 6.5 + sin(camTime * 0.3) * 4.0,
        18.0 + sin(camTime * 0.25) * 6.0,
        camTime * 5.0 + cos(camTime * 0.2) * 4.0
    );
    vec3 ta = vec3(ro.x + 6.0, 12.0 + sin(camTime * 0.35) * 3.0, ro.z + 5.5);
    ro += cameraOffset;
    ta += cameraOffset;

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.15) * 0.05);
    vec3 rd = ca * cameraInputRay(p, 1.8);

    vec3 lightDir = normalize(vec3(0.4, 0.8, -0.5));
    vec3 skyColor = getSkyColor(rd, safeFx);

    float t = 0.0;
    int matID = -1;
    vec4 state = vec4(0.0);
    vec3 volumeGlow = vec3(0.0);

    // Primary Raymarch
    for (int i = 0; i < MAX_STEPS; i++) {
        if (i >= maxRaySteps) break;
        
        vec3 pos = ro + rd * t;
        int currMat; vec4 currState;
        float h = map(pos, currMat, currState);

        // Volumetric Embers & Spores Accumulation
        float emberZone = smoothstep(0.2, 0.8, currState.r) * smoothstep(12.0, 0.0, pos.y);
        if (emberZone > 0.0) {
            float ember = noise3d(pos * 4.0 - vec3(0.0, time * 1.5, 0.0));
            volumeGlow += vec3(1.0, 0.4, 0.05) * step(0.85, ember) * emberZone * safeFx * 0.18;
        }
        
        if (currMat == MAT_CRYSTAL || currState.a > 0.5) {
            vec3 glowCol = mix(vec3(0.1, 0.8, 0.9), vec3(0.6, 0.1, 0.9), hash12(floor(pos.xz / BLOCK_SIZE)));
            volumeGlow += glowCol * currState.a * (0.01 + glow * 0.005) / (1.0 + abs(h) * 6.0);
        }

        if (h < max(SURF_DIST, 0.0015 * t) || t > MAX_DIST) {
            matID = currMat;
            state = currState;
            break;
        }
        t += clamp(h, 0.02, 1.0);
    }

    vec3 color = skyColor;
    if (t < MAX_DIST) {
        vec3 pos = ro + rd * t;
        vec3 nor = calcNormal(pos);
        float sha = calcSoftShadow(pos + nor * 0.05, lightDir, 0.1, 16.0, 12.0);
        float ao = calcAO(pos, nor);
        float dif = clamp(dot(nor, lightDir), 0.0, 1.0) * sha;
        float skyLight = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
        float fre = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 4.0);
        float grid = blockGrid(pos);

        vec3 emission;
        vec3 matColor = getMaterialColor(matID, pos, nor, rd, state, grid, safeFx, contourPower, emission);

        // Screen-Space / Raytraced Reflections for sleek materials
        if (matID == MAT_LAVA || matID == MAT_CRYSTAL || matID == MAT_OBSIDIAN) {
            float refPower = (matID == MAT_LAVA) ? 0.35 : (matID == MAT_CRYSTAL) ? 0.6 : 0.15;
            refPower *= (1.0 - grid * 0.5); // Less reflective on grid edges
            
            vec3 refRd = reflect(rd, nor);
            vec3 refRo = pos + nor * 0.05;
            float rt = 0.0;
            int rMat = -1; vec4 rState;
            
            // Secondary Raymarch Loop (The GPU Melter)
            for (int j = 0; j < 32; j++) {
                float rh = map(refRo + refRd * rt, rMat, rState);
                if (rh < 0.01 || rt > 16.0) break;
                rt += clamp(rh, 0.03, 1.2);
            }
            
            if (rt < 16.0) {
                vec3 rPos = refRo + refRd * rt;
                vec3 rNor = calcNormal(rPos);
                vec3 rEmi;
                vec3 rCol = getMaterialColor(rMat, rPos, rNor, refRd, rState, blockGrid(rPos), safeFx, contourPower, rEmi);
                float rDif = clamp(dot(rNor, lightDir), 0.0, 1.0);
                // Fast shading for reflection
                vec3 rFinal = rCol * (rDif * 0.8 + 0.2) + rEmi;
                matColor = mix(matColor, rFinal, refPower * fre * 1.5);
            } else {
                matColor = mix(matColor, getSkyColor(refRd, safeFx), refPower * fre * 1.5);
            }
        }

        vec3 lin = vec3(0.0);
        lin += 1.6 * dif * vec3(1.0, 0.8, 0.6) * ao;
        lin += 0.8 * skyLight * vec3(0.2, 0.3, 0.4) * ao;
        lin += 0.4 * fre * vec3(1.0, 0.9, 0.8);

        color = matColor * lin + emission;
        
        // Heavy Volumetric Depth Fog
        float fog = 1.0 - exp(-0.0006 * t * t);
        color = mix(color, skyColor, clamp(fog, 0.0, 1.0));
    }

    color += volumeGlow * (1.0 + safeFx * 0.5);

    // Color Grading
    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14); // ACES tonemap
    color = pow(color, vec3(1.0 / max(gamma, 0.2)));

    // Vignette
    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.65 + 0.35 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.25);

    fragColor = vec4(color, 1.0);
}
"""

WORLD_SEED = 20260714

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

    # Dramatic, jagged geologic base
    macro = (
        0.5
        + 0.2 * np.sin(x_norm * 14.0 + np.cos(y_norm * 8.0) * 2.0)
        + 0.15 * np.cos(y_norm * 18.0 - x_norm * 6.0)
        + 0.1 * np.sin((x_norm + y_norm) * 32.0)
    )
    height = macro + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.03

    # Carve immense craters / caldera
    for _ in range(max(5, (tiles_x * tiles_y) // 4000)):
        blob = _ellipse(
            tile_x,
            tile_y,
            rng.uniform(0.0, tiles_x),
            rng.uniform(0.0, tiles_y),
            rng.uniform(max(10.0, tiles_x * 0.1), max(20.0, tiles_x * 0.3)),
            rng.uniform(max(10.0, tiles_y * 0.1), max(20.0, tiles_y * 0.3)),
        )
        height -= blob * rng.uniform(0.2, 0.4)

    # Spike mountains
    for _ in range(max(10, (tiles_x * tiles_y) // 2000)):
        blob = _ellipse(
            tile_x,
            tile_y,
            rng.uniform(0.0, tiles_x),
            rng.uniform(0.0, tiles_y),
            rng.uniform(3.0, tiles_x * 0.08),
            rng.uniform(3.0, tiles_y * 0.08),
        )
        height += blob * rng.uniform(0.3, 0.6)

    height = np.clip(height, 0.0, 1.0)

    # Magma flows naturally filling the lowlands and caldera
    magma = np.clip((0.35 - height) / 0.25, 0.0, 1.0)
    magma += np.clip(0.1 - np.abs(np.sin(x_norm * 25.0 + y_norm * 15.0)), 0.0, 1.0) * 0.2
    
    # Crystals grow on the shoreline of magma
    shore = np.clip(1.0 - np.abs(height - 0.35) / 0.1, 0.0, 1.0)
    crystals = np.clip(shore * 0.8 + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.1, 0.0, 1.0)

    # Energy clusters
    energy = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    candidates = np.argwhere((crystals > 0.4) & (height > 0.3))
    if len(candidates) > 0:
        cluster_count = min(max(8, (tiles_x * tiles_y) // 1500), len(candidates))
        for idx in rng.choice(len(candidates), size=cluster_count, replace=False):
            cy, cx = candidates[idx]
            radius = int(rng.integers(2, 6))
            y0, y1 = max(cy - radius, 0), min(cy + radius + 1, tiles_y)
            x0, x1 = max(cx - radius, 0), min(cx + radius + 1, tiles_x)
            patch_y, patch_x = np.meshgrid(
                np.arange(y0, y1, dtype=np.float32),
                np.arange(x0, x1, dtype=np.float32),
                indexing="ij",
            )
            dist = np.sqrt((patch_x - cx) ** 2 + (patch_y - cy) ** 2)
            influence = np.clip(1.0 - dist / max(radius + 0.5, 1.0), 0.0, 1.0)
            energy[y0:y1, x0:x1] = np.maximum(energy[y0:y1, x0:x1], influence * rng.uniform(0.6, 1.0))

    field = np.stack(
        [
            magma.astype(np.float32),
            crystals.astype(np.float32),
            height.astype(np.float32),
            energy.astype(np.float32),
        ],
        axis=-1,
    )
    field = np.nan_to_num(field, nan=0.0, posinf=1.0, neginf=0.0)
    field = np.clip(field, 0.0, 1.0).astype(np.float32)

    pixels = np.repeat(np.repeat(field, tile_size, axis=0), tile_size, axis=1)
    return pixels[:height_px, :width_px].copy()

SPEC = WorldSpec(
    id="minecraft-overdrive-3d",
    display_name="Minecraft Overdrive",
    window_title="Garage Life Lab - Minecraft Overdrive",
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={
        "ray_steps": 150,           # Extreme raymarching steps for 3D noise and reflections
        "fx_intensity": 1.45,       # Strong bloom and volumetric glow
        "contour_contrast": 1.2,
        "camera_speed": 0.85,
        "tile_size": 8,             # Fine fluid and geologic sim resolution
        "substeps": 24,             # Keep the sim hot and active
        "exposure": 1.35,
    },
    stability_notes=("extreme GPU thermal load", "raytraced SSR reflections", "3D voxel noise carving"),
    hud_subtitle="AMPLIFIED VOLUMETRIC STRESS",
    preview_palette=("#ff3c00", "#ff8800", "#111215", "#242730", "#0bd7e8", "#6a1be2", "#a6ff00"),
)
