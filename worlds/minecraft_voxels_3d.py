"""World definition for Minecraft Living Voxel."""
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
    // R: Water/Moisture (Fluid Advection)
    // G: Flora/Biomass (Reaction-Diffusion)
    // B: Terrain Elevation (Morphological)
    // A: Geothermal/Ore Energy (Static/Pulsing)
    vec4 c = texture(stateTex, uv);
    vec4 r = texture(stateTex, uv + vec2(texel.x, 0.0));
    vec4 l = texture(stateTex, uv - vec2(texel.x, 0.0));
    vec4 t = texture(stateTex, uv + vec2(0.0, texel.y));
    vec4 b = texture(stateTex, uv - vec2(0.0, texel.y));
    vec4 tr = texture(stateTex, uv + texel);
    vec4 tl = texture(stateTex, uv + vec2(-texel.x, texel.y));
    vec4 br = texture(stateTex, uv + vec2(texel.x, -texel.y));
    vec4 bl = texture(stateTex, uv - texel);

    // Laplacians
    float lapR = ((r.r + l.r + t.r + b.r) * 0.2 + (tr.r + tl.r + br.r + bl.r) * 0.05 - c.r) * laplaceScale;
    float lapG = ((r.g + l.g + t.g + b.g) * 0.2 + (tr.g + tl.g + br.g + bl.g) * 0.05 - c.g) * laplaceScale;
    float lapB = ((r.b + l.b + t.b + b.b) * 0.2 + (tr.b + tl.b + br.b + bl.b) * 0.05 - c.b) * laplaceScale;

    // Terrain Gradient for Advection
    vec2 gradH = vec2(r.b - l.b, t.b - b.b);
    
    // Water flows heavily down terrain gradients (Erosion flow)
    float flowStrength = 0.85;
    float advectR = dot(gradH, vec2(r.r - l.r, t.r - b.r)) * flowStrength;
    
    // Flora spreads subtly toward moisture
    vec2 gradR = vec2(r.r - l.r, t.r - b.r);
    float advectG = -dot(gradR, vec2(r.g - l.g, t.g - b.g)) * 0.3;

    // Stochastic elements and weather
    float localNoise = (hash12(uv * resolution + floor(time * 0.3)) - 0.5) * noiseStrength;
    float weatherPulse = smoothstep(0.8, 1.0, sin(time * 0.05 + hash12(floor(uv * 12.0)) * 6.283));
    
    // Reaction parameters
    float localFeed = feed * 0.55 + 0.015 + weatherPulse * 0.012 + localNoise * 0.05;
    float localKill = kill * 0.50 + 0.020 - c.a * 0.005 + parameterDrift * 0.4;
    
    // Gray-Scott Reaction: Flora (G) consumes Moisture (R) to grow
    float reaction = c.r * c.g * c.g * 0.65;

    // State Integration
    float dr = diffU * lapR - reaction + localFeed * (1.0 - c.r) - advectR + weatherPulse * 0.005;
    float dg = diffV * lapG + reaction - (localFeed + localKill) * c.g - advectG;
    
    // Morphological Terrain: 
    // - Fast water flow erodes terrain (canyons/rivers)
    // - High flora stabilizes and slightly builds soil
    float erosion = abs(advectR) * 0.08;
    float soilBuildup = c.g * 0.0015;
    float db = lapB * 0.002 - erosion + soilBuildup;

    // Energy/Ore channel flickers but mostly stays stable
    float spark = step(0.999, hash12(uv * resolution + time));
    float da = (c.a * -0.001) + spark * 0.05;

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
#define MAX_DIST 75.0
#define SURF_DIST 0.002
#define BLOCK_SIZE 0.8
#define MAP_SCALE 0.035

// Voxel Material IDs
const int MAT_WATER   = 0;
const int MAT_DIRT    = 1;
const int MAT_GRASS   = 2;
const int MAT_STONE   = 3;
const int MAT_SAND    = 4;
const int MAT_WOOD    = 5;
const int MAT_LEAVES  = 6;
const int MAT_ORE     = 7;
const int MAT_SNOW    = 8;

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
        p = rot * p * 2.0 + vec2(12.3, 4.5);
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

// Read map state
vec4 getState(vec2 cell) {
    return textureLod(stateTex, fract((cell + 0.5) * BLOCK_SIZE * MAP_SCALE), 0.0);
}

float getRawHeight(vec4 state, vec2 cell) {
    float macro = fbm(cell * 0.03 + vec2(9.1, 3.2));
    float peaks = pow(1.0 - abs(fbm(cell * 0.07) * 2.0 - 1.0), 2.0);
    return state.b * 8.5 + macro * 3.0 + peaks * 1.5;
}

float getTerrainHeight(vec4 state, vec2 cell) {
    return floor(getRawHeight(state, cell)) * (BLOCK_SIZE * 0.5) + 0.5;
}

// SDF Map Function
float map(in vec3 p, out int matID, out vec4 stateOut) {
    vec2 cell = floor(p.xz / BLOCK_SIZE);
    vec4 state = getState(cell);
    float h = getTerrainHeight(state, cell);
    float seaLevel = 1.8;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);

    float bestD = p.y - h;
    int bestMat = MAT_DIRT;
    
    // Determine base terrain material
    if (h >= 5.5) bestMat = MAT_SNOW;
    else if (h >= 4.0 || (h > 2.5 && state.g < 0.2)) bestMat = MAT_STONE;
    else if (h <= seaLevel + 0.1 && state.g < 0.3) bestMat = MAT_SAND;
    else if (state.g > 0.2) bestMat = MAT_GRASS;

    stateOut = state;

    // Water SDF
    float waterMask = smoothstep(0.4, 0.8, state.r) * step(h, seaLevel + 0.1);
    if (waterMask > 0.1) {
        float wave = (noise(p.xz * 1.5 + time * 0.8) - 0.5) * 0.04 * safeFx;
        float dWater = p.y - (seaLevel + wave);
        if (dWater < bestD) {
            bestD = dWater * 0.8;
            bestMat = MAT_WATER;
        }
    }

    // Flora/Trees SDF
    float treeChance = hash12(cell + vec2(13.4, 88.2));
    if (treeChance < 0.12 && state.g > 0.5 && h > seaLevel && h < 4.0) {
        vec2 center = (cell + 0.5) * BLOCK_SIZE;
        float heightVar = mix(0.8, 1.6, hash12(cell));
        
        // Trunk
        float dTrunk = boxSdf(
            vec3(p.x - center.x, p.y - (h + heightVar * 0.5), p.z - center.y),
            vec3(BLOCK_SIZE * 0.15, heightVar * 0.5, BLOCK_SIZE * 0.15)
        );
        if (dTrunk < bestD) {
            bestD = dTrunk;
            bestMat = MAT_WOOD;
        }

        // Leaves
        float dLeaves = boxSdf(
            vec3(p.x - center.x, p.y - (h + heightVar + 0.4), p.z - center.y),
            vec3(BLOCK_SIZE * 0.85, 0.6, BLOCK_SIZE * 0.85)
        );
        if (dLeaves < bestD) {
            bestD = dLeaves;
            bestMat = MAT_LEAVES;
        }
    }

    // Ores SDF (glowing blocks embedded in stone)
    if (state.a > 0.4 && h > 2.5 && h < 5.5) {
        float dOre = boxSdf(
            vec3(p.x - (cell.x + 0.5)*BLOCK_SIZE, p.y - (h + 0.05), p.z - (cell.y + 0.5)*BLOCK_SIZE),
            vec3(BLOCK_SIZE * 0.25, 0.2, BLOCK_SIZE * 0.25)
        );
        if (dOre < bestD) {
            bestD = dOre;
            bestMat = MAT_ORE;
        }
    }

    matID = bestMat;
    return bestD;
}

vec3 calcNormal(in vec3 p) {
    const vec2 e = vec2(0.03, 0.0);
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
    for (int i = 0; i < 32; i++) {
        float h = map(ro + rd * t, mat, state);
        res = min(res, k * h / t);
        t += clamp(h, 0.03, 0.6);
        if (res < 0.005 || t > tmax) break;
    }
    return clamp(res, 0.0, 1.0);
}

// Voxel-style Ambient Occlusion
float calcVoxelAO(vec3 pos, vec3 nor) {
    float occ = 0.0;
    float sca = 1.0;
    int mat; vec4 state;
    for (int i = 0; i < 5; i++) {
        float h = 0.03 + 0.15 * float(i);
        float d = map(pos + nor * h, mat, state);
        occ += (h - d) * sca;
        sca *= 0.65;
    }
    return clamp(1.0 - 2.5 * occ, 0.0, 1.0);
}

// Block grid lines
float blockGrid(vec3 p) {
    vec3 q = abs(fract(p / BLOCK_SIZE) - 0.5);
    float edge = min(min(q.x, q.y), q.z);
    return 1.0 - smoothstep(0.015, 0.05, edge);
}

// Dynamic Sky system
vec3 getSkyColor(vec3 rd, vec3 sunDir, float safeFx) {
    // Sun position dictates time of day colors
    float sunElev = sunDir.y;
    
    vec3 daySky = mix(vec3(0.3, 0.6, 0.9), vec3(0.1, 0.3, 0.7), smoothstep(0.0, 1.0, rd.y));
    vec3 sunsetSky = mix(vec3(0.9, 0.4, 0.2), vec3(0.4, 0.2, 0.5), smoothstep(0.0, 1.0, rd.y));
    vec3 nightSky = mix(vec3(0.02, 0.05, 0.1), vec3(0.0, 0.0, 0.02), smoothstep(0.0, 1.0, rd.y));
    
    vec3 sky = daySky;
    if (sunElev < 0.2) sky = mix(sunsetSky, daySky, smoothstep(-0.1, 0.2, sunElev));
    if (sunElev < -0.1) sky = mix(nightSky, sunsetSky, smoothstep(-0.3, -0.1, sunElev));

    // Sun disc
    float sunDot = dot(rd, sunDir);
    float sunDisc = smoothstep(0.995, 0.998, sunDot);
    sky += vec3(1.0, 0.9, 0.7) * sunDisc * (1.5 + safeFx);
    
    // Moon disc
    float moonDot = dot(rd, -sunDir);
    float moonDisc = smoothstep(0.997, 0.999, moonDot);
    sky += vec3(0.8, 0.9, 1.0) * moonDisc * 0.8;

    // Stars
    if (sunElev < 0.0) {
        float stars = pow(hash12(rd.xy * 250.0), 150.0);
        sky += stars * smoothstep(0.0, -0.2, sunElev) * (0.5 + 0.5*sin(time*2.0 + rd.x*100.0));
    }

    // Clouds
    if (rd.y > 0.05) {
        vec2 cloudPos = rd.xz / rd.y * 1.5 + vec2(time * 0.015, -time * 0.005);
        float cNoise = fbm(floor(cloudPos * 4.0) * 0.25);
        float cloudMask = smoothstep(0.55, 0.65, cNoise);
        vec3 cloudColor = mix(vec3(0.9, 0.95, 1.0), vec3(0.2, 0.1, 0.2), smoothstep(0.0, -0.2, sunElev));
        sky = mix(sky, cloudColor, cloudMask * smoothstep(0.0, 0.2, rd.y));
    }

    return sky;
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float contourPower = clamp(contourContrast, 0.0, 1.8);
    int maxRaySteps = clamp(raySteps, 32, MAX_STEPS);

    float camTime = time * 0.08 * max(cameraSpeed, 0.05);
    vec3 ro = vec3(
        camTime * 6.0 + sin(camTime * 0.3) * 2.5,
        7.5 + sin(camTime * 0.4) * 1.5,
        camTime * 5.0 + cos(camTime * 0.25) * 3.0
    );
    vec3 ta = vec3(ro.x + 5.5, 3.0 + sin(camTime * 0.3) * 0.5, ro.z + 5.5);
    ro += cameraOffset;
    ta += cameraOffset;

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.15) * 0.03);
    vec3 rd = ca * cameraInputRay(p, 2.0);

    // Day/Night Cycle
    float dayCycle = time * 0.02 + colorShift * 0.5;
    vec3 sunDir = normalize(vec3(cos(dayCycle), sin(dayCycle), -0.5));
    vec3 mainLight = sunDir.y > -0.1 ? sunDir : -sunDir; // Moonlight takes over

    vec3 skyColor = getSkyColor(rd, sunDir, safeFx);

    float t = 0.0;
    int matID = -1;
    vec4 state = vec4(0.0);
    vec3 volumeGlow = vec3(0.0);

    for (int i = 0; i < MAX_STEPS; i++) {
        if (i >= maxRaySteps) break;
        vec3 pos = ro + rd * t;
        int currMat; vec4 currState;
        float h = map(pos, currMat, currState);

        // Volumetric Ore Glow & Bioluminescence
        if (currMat == MAT_ORE || currState.a > 0.6) {
            vec3 glowCol = mix(vec3(1.0, 0.2, 0.1), vec3(0.1, 0.9, 1.0), hash12(floor(pos.xz / BLOCK_SIZE)));
            float pulse = 0.8 + 0.2 * sin(time * 3.0 + currState.a * 10.0);
            volumeGlow += glowCol * currState.a * pulse * (0.01 + glow * 0.005) * safeFx / (1.0 + abs(h) * 8.0);
        }

        if (h < max(SURF_DIST, 0.0015 * t) || t > MAX_DIST) {
            matID = currMat;
            state = currState;
            break;
        }
        t += clamp(h, 0.025, 0.8);
    }

    vec3 color = skyColor;

    if (t < MAX_DIST) {
        vec3 pos = ro + rd * t;
        vec3 nor = calcNormal(pos);
        
        float sha = calcSoftShadow(pos + nor * 0.02, mainLight, 0.05, 20.0, 12.0);
        float ao = calcVoxelAO(pos, nor);
        
        float dif = clamp(dot(nor, mainLight), 0.0, 1.0) * sha;
        float amb = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
        float fre = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 4.0);
        float grid = blockGrid(pos) * contourPower;
        
        // Light intensity drops at night
        float lightIntensity = smoothstep(-0.2, 0.2, sunDir.y) * 1.5 + 0.2; // 0.2 is moonlight
        vec3 lightColor = mix(vec3(0.4, 0.5, 0.8), vec3(1.0, 0.9, 0.8), smoothstep(-0.1, 0.2, sunDir.y));

        vec3 matColor = vec3(0.0);
        vec3 emission = vec3(0.0);
        float cellHash = hash12(floor(pos.xz / BLOCK_SIZE));

        if (matID == MAT_WATER) {
            vec3 shallow = vec3(0.1, 0.5, 0.7);
            vec3 deep = vec3(0.02, 0.1, 0.3);
            matColor = mix(shallow, deep, clamp(state.r, 0.0, 1.0));
            matColor += vec3(0.2, 0.4, 0.6) * grid * 0.15;
            
            // Specular
            vec3 ref = reflect(rd, nor);
            float spe = pow(clamp(dot(ref, mainLight), 0.0, 1.0), 64.0) * sha;
            emission += lightColor * spe * (0.8 + safeFx * 0.4);
            
            // Fresnel reflection of sky
            matColor = mix(matColor, getSkyColor(ref, sunDir, safeFx), fre * 0.6);
        }
        else if (matID == MAT_GRASS) {
            vec3 lush = vec3(0.2, 0.5, 0.15);
            vec3 dry = vec3(0.4, 0.5, 0.2);
            matColor = mix(dry, lush, state.g);
            matColor *= 1.0 - grid * 0.15;
        }
        else if (matID == MAT_DIRT) {
            matColor = mix(vec3(0.3, 0.18, 0.08), vec3(0.4, 0.25, 0.12), cellHash);
            matColor *= 1.0 - grid * 0.2;
        }
        else if (matID == MAT_STONE) {
            matColor = mix(vec3(0.35), vec3(0.45), cellHash);
            matColor *= 1.0 - grid * 0.25;
        }
        else if (matID == MAT_SAND) {
            matColor = vec3(0.75, 0.65, 0.4);
            matColor *= 1.0 - grid * 0.1;
        }
        else if (matID == MAT_SNOW) {
            matColor = vec3(0.9, 0.95, 0.98);
            matColor *= 1.0 - grid * 0.1;
            emission += vec3(0.1) * fre * safeFx; // Snow sparkle
        }
        else if (matID == MAT_WOOD) {
            matColor = mix(vec3(0.3, 0.15, 0.05), vec3(0.4, 0.2, 0.08), step(0.5, fract(pos.y * 4.0)));
        }
        else if (matID == MAT_LEAVES) {
            matColor = mix(vec3(0.1, 0.35, 0.1), vec3(0.2, 0.5, 0.15), cellHash);
            matColor *= 1.0 - grid * 0.1;
            // Subsurface scattering fake
            emission += matColor * clamp(dot(rd, mainLight), 0.0, 1.0) * 0.5 * lightIntensity; 
        }
        else if (matID == MAT_ORE) {
            vec3 oreCol = mix(vec3(1.0, 0.1, 0.1), vec3(0.1, 0.8, 1.0), cellHash);
            matColor = oreCol;
            emission += oreCol * (0.5 + glow) * safeFx;
        }

        vec3 lin = vec3(0.0);
        lin += 1.8 * dif * lightColor * ao * lightIntensity;
        lin += 0.6 * amb * vec3(0.3, 0.4, 0.6) * ao;
        lin += 0.3 * fre * lightColor * ao;

        color = matColor * lin + emission;
        
        // Atmospheric Perspective / Fog
        float fog = 1.0 - exp(-0.0008 * t * t);
        color = mix(color, skyColor, fog);
    }

    // Add volumetric scatter
    color += volumeGlow * (1.0 + safeFx * 0.5);

    // Post Processing: ACES Tonemap
    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    
    // Gamma
    color = pow(color, vec3(1.0 / max(gamma, 0.2)));

    // Vignette
    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.5 + 0.5 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.2);

    fragColor = vec4(color, 1.0);
}
"""

WORLD_SEED = 8847291

def seed_field(width_px: int, height_px: int, tile_size: int) -> np.ndarray:
    tile_size = max(2, int(tile_size))
    tiles_x = max(1, int(np.ceil(width_px / tile_size)))
    tiles_y = max(1, int(np.ceil(height_px / tile_size)))

    rng = np.random.default_rng(WORLD_SEED)
    tile_y, tile_x = np.meshgrid(
        np.arange(tiles_y, dtype=np.float32),
        np.arange(tiles_x, dtype=np.float32),
        indexing="ij"
    )
    x_norm = tile_x / max(tiles_x - 1, 1)
    y_norm = tile_y / max(tiles_y - 1, 1)

    # Base Continents
    height = (
        0.40
        + 0.18 * np.sin(x_norm * 6.5 + np.cos(y_norm * 4.2) * 1.5)
        + 0.14 * np.cos(y_norm * 8.3 - 0.5)
        + 0.10 * np.sin((x_norm + y_norm) * 12.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.02
    )

    # Add Mountain Ranges
    mountain_ridges = np.abs(np.sin(x_norm * 18.0 + np.cos(y_norm * 14.0) * 2.0))
    height += np.clip(0.3 - mountain_ridges, 0.0, 1.0) * 0.25

    # Carve Deep Rivers
    river_wave = np.abs(np.sin(x_norm * 15.0 + np.sin(y_norm * 11.0) * 3.0))
    river_mask = np.clip((0.15 - river_wave) / 0.15, 0.0, 1.0)
    height -= river_mask * 0.2
    
    height = np.clip(height, 0.0, 1.0)

    sea_level = 0.42
    ocean = (height < sea_level).astype(np.float32)
    coast = np.clip(1.0 - np.abs(height - sea_level) / 0.08, 0.0, 1.0)

    # Moisture distribution
    moisture = np.clip(
        0.15
        + ocean * 0.6
        + coast * 0.3
        + 0.1 * np.sin(x_norm * 10.0 - y_norm * 8.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.03,
        0.0,
        1.0
    )

    # Biomass clusters in moist, non-ocean areas
    biomass = np.clip(
        (1.0 - ocean)
        * (
            0.1
            + moisture * 0.7
            - np.clip(height - 0.65, 0.0, 1.0) * 1.5  # Trees don't grow on high peaks
            + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.05
        ),
        0.0,
        1.0
    )

    # Ores & Deep Energy Networks (Channel A)
    ore_veins = np.abs(np.sin(x_norm * 35.0) * np.cos(y_norm * 28.0))
    ore_energy = np.clip(1.0 - ore_veins * 3.5, 0.0, 1.0) * np.clip((height - 0.4) * 2.0, 0.0, 1.0)
    
    # Add scattered hot-spots
    hotspots = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    for _ in range(max(5, (tiles_x * tiles_y) // 2500)):
        cx, cy = rng.uniform(0, tiles_x), rng.uniform(0, tiles_y)
        radius = rng.uniform(2.0, 6.0)
        dist = np.sqrt((tile_x - cx)**2 + (tile_y - cy)**2)
        hotspots = np.maximum(hotspots, np.clip(1.0 - dist / radius, 0.0, 1.0) * rng.uniform(0.5, 1.0))
        
    energy = np.clip(ore_energy + hotspots * 0.8, 0.0, 1.0)

    # Final Channel Assembly
    tile_field = np.stack(
        [
            moisture.astype(np.float32),
            biomass.astype(np.float32),
            height.astype(np.float32),
            energy.astype(np.float32),
        ],
        axis=-1,
    )

    # Upscale to exact pixel dimensions
    field = np.repeat(np.repeat(tile_field, tile_size, axis=0), tile_size, axis=1)
    field = field[:height_px, :width_px].copy()
    return field

SPEC = WorldSpec(
    id='minecraft-living-voxel',
    display_name='Living Voxel World',
    window_title='Garage Life Lab - Living Voxel World',
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={
        "ray_steps": 144,
        "fx_intensity": 1.25,
        "contour_contrast": 1.15,
        "camera_speed": 0.85,
        "tile_size": 12,
        "substeps": 12,
    },
    preview_image='assets/world_previews/minecraft-long-term-3d.png',
    stability_notes=('dynamic terrain', 'cinematic lighting', 'voxel AO', 'day/night cycle'),
    hud_subtitle='LIVING VOXEL SIMULATION',
    preview_palette=('#0a1b2a', '#103842', '#2a6a3b', '#6b4e2a', '#666b68', '#f2e3a1', '#ff5a3a'),
)
