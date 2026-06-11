"""World definition for Minecraft Ultimate Voxel Edition."""
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

    // CHANNELS:
    // R: Moisture / Water coverage
    // G: Biomass / Forest canopy
    // B: Terrain Elevation (Static / Base)
    // A: Fire / Heat Energy (Cellular Automata)
    vec4 c = texture(stateTex, uv);
    vec4 r = texture(stateTex, uv + vec2(texel.x, 0.0));
    vec4 l = texture(stateTex, uv - vec2(texel.x, 0.0));
    vec4 t = texture(stateTex, uv + vec2(0.0, texel.y));
    vec4 b = texture(stateTex, uv - vec2(0.0, texel.y));
    vec4 tr = texture(stateTex, uv + texel);
    vec4 tl = texture(stateTex, uv + vec2(-texel.x, texel.y));
    vec4 br = texture(stateTex, uv + vec2(texel.x, -texel.y));
    vec4 bl = texture(stateTex, uv - texel);

    // Laplacians for diffusion
    float lapR = ((r.r + l.r + t.r + b.r) * 0.2 + (tr.r + tl.r + br.r + bl.r) * 0.05 - c.r) * laplaceScale;
    float lapG = ((r.g + l.g + t.g + b.g) * 0.2 + (tr.g + tl.g + br.g + bl.g) * 0.05 - c.g) * laplaceScale;

    // Terrain Gradient for Moisture Advection (Water flows downhill)
    vec2 gradH = vec2(r.b - l.b, t.b - b.b);
    float advectR = dot(gradH, vec2(r.r - l.r, t.r - b.r)) * 0.65;

    float localNoise = (hash12(uv * resolution + floor(time * 0.5)) - 0.5) * noiseStrength;
    
    // Growth Dynamics
    float localFeed = feed * 0.55 + 0.015 + localNoise * 0.04;
    float localKill = kill * 0.50 + 0.020 + parameterDrift * 0.3;
    float reaction = c.r * c.g * c.g * 0.55;

    // Integrate Moisture and Biomass
    float dr = diffU * lapR - reaction + localFeed * (1.0 - c.r) - advectR;
    float dg = diffV * lapG + reaction - (localFeed + localKill) * c.g;

    // --- EPIC WILDFIRE CELLULAR AUTOMATA ---
    float fire = c.a;
    float neighborsFire = r.a + l.a + t.a + b.a + tr.a + tl.a + br.a + bl.a;
    
    // 1. Spontaneous Ignition (Lightning/Lava sparks)
    float spark = step(0.9998, hash12(uv * resolution + time * 1.3));
    float ignite = spark * step(0.65, c.g); // Only dense forests catch fire
    
    // 2. Spread
    float spread = step(1.0, neighborsFire) * step(0.35, c.g) * step(0.4, hash12(uv + time));
    
    // Catch fire
    if (ignite > 0.0 || spread > 0.0) {
        fire = 1.0;
    }
    
    // Burn out and decay
    if (fire > 0.0) {
        fire -= 0.025 * dt; // Fire fades out
    }
    fire = clamp(fire, 0.0, 1.0);

    // Biomass is consumed heavily by active fire
    if (fire > 0.1) {
        dg -= 0.15 * dt;
    }
    
    // Very slow regeneration of terrain/soil via biomass
    float db = (c.g * 0.0005) * dt;

    fragColor = vec4(
        clamp(c.r + dr * dt, 0.0, 1.0),
        clamp(c.g + dg * dt, 0.0, 1.0),
        clamp(c.b + db, 0.0, 1.0), // Keep terrain mostly static
        fire
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
#define MAX_DIST 110.0
#define SURF_DIST 0.005
#define MAP_SCALE 0.035
#define SEA_LEVEL 10.0

// Material Definitions
const int MAT_WATER   = 0;
const int MAT_DIRT    = 1;
const int MAT_GRASS   = 2;
const int MAT_STONE   = 3;
const int MAT_SAND    = 4;
const int MAT_WOOD    = 5;
const int MAT_LEAVES  = 6;
const int MAT_SNOW    = 7;
const int MAT_FIRE    = 8;
const int MAT_BEDROCK = 9;

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
        p = rot * p * 2.0 + vec2(15.2, 3.1);
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

// Environment lookups
vec4 getState(vec2 cell) {
    return textureLod(stateTex, fract((cell + 0.5) * MAP_SCALE), 0.0);
}

float getTerrainHeight(vec4 state, vec2 cell) {
    float macro = fbm(cell * 0.02 + vec2(19.2, 3.7));
    float mountains = pow(1.0 - abs(fbm(cell * 0.05) * 2.0 - 1.0), 2.0);
    float rawHeight = state.b * 28.0 + macro * 8.0 + mountains * 4.0;
    return floor(rawHeight); // Perfect integer blocks
}

// Core SDF
float map(in vec3 p, out int matID, out vec4 stateOut) {
    vec2 cell = floor(p.xz);
    vec4 state = getState(cell);
    float h = getTerrainHeight(state, cell);
    
    float bestD = p.y - h;
    int bestMat = MAT_DIRT;
    stateOut = state;

    // Biome Logic
    if (h >= 24.0) bestMat = MAT_SNOW;
    else if (h >= 18.0 || (h > SEA_LEVEL + 2.0 && state.g < 0.2)) bestMat = MAT_STONE;
    else if (h <= SEA_LEVEL + 1.0 && state.g < 0.35) bestMat = MAT_SAND;
    else if (state.g > 0.35) bestMat = MAT_GRASS;

    // Water SDF
    if (h < SEA_LEVEL) {
        float dWater = p.y - (SEA_LEVEL - 0.1); // Slightly submerged
        if (dWater < bestD) {
            bestD = dWater;
            bestMat = MAT_WATER;
        }
    }

    // Tree SDF (Procedural Voxel Generation)
    if (state.g > 0.65 && h > SEA_LEVEL && h < 18.0 && state.a < 0.1) {
        float treeHash = hash12(cell + vec2(33.1, 8.9));
        if (treeHash < 0.06) {
            float trunkHeight = floor(mix(4.0, 7.0, hash12(cell)));
            // Trunk
            float dTrunk = boxSdf(p - vec3(cell.x + 0.5, h + trunkHeight * 0.5, cell.y + 0.5), vec3(0.3, trunkHeight * 0.5, 0.3));
            if (dTrunk < bestD) { bestD = dTrunk; bestMat = MAT_WOOD; }
            
            // Leaves (Blocky canopy)
            float dLeaves = boxSdf(p - vec3(cell.x + 0.5, h + trunkHeight + 1.0, cell.y + 0.5), vec3(2.0, 2.0, 2.0));
            if (dLeaves < bestD) { bestD = dLeaves; bestMat = MAT_LEAVES; }
        }
    }

    // Fire SDF (Dynamic from Cellular Automata)
    if (state.a > 0.1 && h >= SEA_LEVEL) {
        float fireHeight = h + 0.5;
        // If there's a tree here, fire is higher
        if (state.g > 0.65 && hash12(cell + vec2(33.1, 8.9)) < 0.06) fireHeight += 4.0; 
        
        float dFire = boxSdf(p - vec3(cell.x + 0.5, fireHeight, cell.y + 0.5), vec3(0.4, 0.4, 0.4));
        if (dFire < bestD) { bestD = dFire; bestMat = MAT_FIRE; }
    }
    
    // Bedrock bounds
    float dBed = p.y - (-2.0);
    if (dBed < bestD) { bestD = dBed; bestMat = MAT_BEDROCK; }

    matID = bestMat;
    
    // 0.4 multiplier acts as an under-relaxation step size to safely raymarch the discontinuous floor() function.
    return bestD * 0.4; 
}

// Snapped Voxel Normal Calculation
vec3 calcNormal(in vec3 p) {
    const vec2 e = vec2(0.02, 0.0);
    int mat; vec4 state;
    vec3 n = vec3(
        map(p + e.xyy, mat, state) - map(p - e.xyy, mat, state),
        map(p + e.yxy, mat, state) - map(p - e.yxy, mat, state),
        map(p + e.yyx, mat, state) - map(p - e.yyx, mat, state)
    );
    
    // Force absolute voxel grid axis alignment
    vec3 absN = abs(n);
    vec3 voxelN = vec3(0.0);
    if (absN.x > absN.y && absN.x > absN.z) voxelN.x = sign(n.x);
    else if (absN.y > absN.x && absN.y > absN.z) voxelN.y = sign(n.y);
    else voxelN.z = sign(n.z);
    
    return voxelN;
}

// Blocky Voxel Shadows
float calcVoxelShadow(in vec3 ro, in vec3 rd, float tmax) {
    float res = 1.0;
    float t = 0.1;
    int mat; vec4 state;
    for (int i = 0; i < 40; i++) {
        float h = map(ro + rd * t, mat, state);
        res = min(res, 12.0 * h / t);
        t += clamp(h, 0.05, 1.0);
        if (res < 0.01 || t > tmax) break;
    }
    return clamp(res, 0.0, 1.0);
}

// Blocky Ambient Occlusion
float calcVoxelAO(vec3 pos, vec3 nor) {
    float ao = 0.0;
    float weight = 1.0;
    int mat; vec4 st;
    for (int i = 1; i <= 4; i++) {
        float dist = float(i) * 0.3;
        float d = map(pos + nor * dist, mat, st);
        ao += (dist - d) * weight;
        weight *= 0.5;
    }
    return clamp(1.0 - ao * 1.5, 0.0, 1.0);
}

// Blocky Cloud Generator
float blockyClouds(vec3 rd, float timeOffset) {
    if (rd.y <= 0.0) return 0.0;
    vec2 cloudPlane = rd.xz * (40.0 / rd.y) + timeOffset;
    vec2 cloudCell = floor(cloudPlane * 0.2); // Big blocky clouds
    float cNoise = fbm(cloudCell * 0.15);
    return step(0.65, cNoise);
}

// Procedural 16x16 Voxel Textures
vec3 getVoxelTexture(int matID, vec3 pos, vec3 nor, vec4 state, float safeFx) {
    // Determine 2D UV on the block face
    vec2 uv;
    vec2 blockCell;
    if (abs(nor.y) > 0.5) {
        uv = fract(pos.xz);
        blockCell = floor(pos.xz);
    } else if (abs(nor.x) > 0.5) {
        uv = fract(pos.zy);
        blockCell = floor(pos.zy);
    } else {
        uv = fract(pos.xy);
        blockCell = floor(pos.xy);
    }
    
    // Pixelate UV to 16x16 grid
    vec2 pixelUv = floor(uv * 16.0) / 16.0;
    
    // Unique deterministic hash for this exact pixel on this exact block
    float pxHash = hash12(blockCell + pixelUv);
    
    vec3 color = vec3(1.0);

    if (matID == MAT_GRASS) {
        if (nor.y > 0.5) { // Top
            color = mix(vec3(0.28, 0.48, 0.15), vec3(0.22, 0.39, 0.12), pxHash);
        } else { // Sides
            if (uv.y > 0.75 || (uv.y > 0.5 && pxHash > 0.5)) { // Grass overhang
                color = mix(vec3(0.28, 0.48, 0.15), vec3(0.22, 0.39, 0.12), pxHash);
            } else { // Dirt underneath
                color = mix(vec3(0.42, 0.28, 0.15), vec3(0.35, 0.22, 0.10), pxHash);
            }
        }
    } 
    else if (matID == MAT_DIRT) {
        color = mix(vec3(0.42, 0.28, 0.15), vec3(0.35, 0.22, 0.10), pxHash);
    } 
    else if (matID == MAT_STONE || matID == MAT_BEDROCK) {
        color = mix(vec3(0.45, 0.45, 0.45), vec3(0.35, 0.35, 0.35), pxHash);
        if (matID == MAT_BEDROCK) color *= 0.3;
    } 
    else if (matID == MAT_SAND) {
        color = mix(vec3(0.84, 0.75, 0.52), vec3(0.78, 0.69, 0.47), pxHash);
    } 
    else if (matID == MAT_WOOD) {
        if (abs(nor.y) > 0.5) { // Rings
            float ring = step(0.5, fract(length(pixelUv - 0.5) * 6.0 + pxHash * 0.2));
            color = mix(vec3(0.54, 0.39, 0.22), vec3(0.44, 0.30, 0.16), ring);
        } else { // Bark
            float streak = step(0.5, fract(pixelUv.x * 5.0 + pxHash * 0.3));
            color = mix(vec3(0.29, 0.19, 0.11), vec3(0.22, 0.14, 0.08), streak);
        }
    } 
    else if (matID == MAT_LEAVES) {
        color = mix(vec3(0.18, 0.36, 0.12), vec3(0.12, 0.25, 0.08), pxHash);
        if (pxHash < 0.15) color *= 0.5; // Faked transparency holes
    } 
    else if (matID == MAT_SNOW) {
        color = mix(vec3(0.9, 0.95, 1.0), vec3(0.85, 0.9, 0.95), pxHash);
    } 
    else if (matID == MAT_WATER) {
        float wave = hash12(pixelUv + floor(time * 2.0));
        color = mix(vec3(0.15, 0.35, 0.8), vec3(0.1, 0.25, 0.7), wave);
    }
    else if (matID == MAT_FIRE) {
        float flicker = hash12(pixelUv + floor(time * 12.0));
        color = mix(vec3(1.0, 0.8, 0.1), vec3(1.0, 0.2, 0.0), uv.y + flicker * 0.4);
    }

    return color;
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    int maxRaySteps = clamp(raySteps, 32, MAX_STEPS);

    // Dynamic Time & Day/Night
    float dayTime = time * 0.04 + colorShift;
    vec3 sunDir = normalize(vec3(cos(dayTime), sin(dayTime), 0.4));
    vec3 moonDir = normalize(-sunDir);
    bool isDay = sunDir.y > 0.0;
    vec3 mainLight = isDay ? sunDir : moonDir;
    float lightIntensity = isDay ? smoothstep(-0.1, 0.2, sunDir.y) : 0.15 * smoothstep(-0.1, 0.2, moonDir.y);

    // Camera
    float camTime = time * 0.07 * max(cameraSpeed, 0.05);
    vec3 ro = vec3(
        camTime * 8.0 + sin(camTime * 0.2) * 5.0,
        28.0 + sin(camTime * 0.4) * 4.0,
        camTime * 6.0 + cos(camTime * 0.3) * 6.0
    );
    vec3 ta = vec3(ro.x + 6.0, ro.y - 4.0 + sin(camTime * 0.5), ro.z + 6.0);
    ro += cameraOffset;
    ta += cameraOffset;

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.1) * 0.04);
    vec3 rd = ca * cameraInputRay(p, 2.0);

    // Sky & Clouds
    vec3 daySky = mix(vec3(0.4, 0.65, 1.0), vec3(0.15, 0.4, 0.8), max(0.0, rd.y));
    vec3 nightSky = mix(vec3(0.02, 0.03, 0.08), vec3(0.0, 0.0, 0.02), max(0.0, rd.y));
    vec3 sunset = mix(vec3(0.9, 0.4, 0.2), vec3(0.5, 0.2, 0.4), max(0.0, rd.y));
    
    vec3 skyColor = mix(nightSky, daySky, smoothstep(-0.2, 0.2, sunDir.y));
    if (abs(sunDir.y) < 0.2) skyColor = mix(skyColor, sunset, 1.0 - abs(sunDir.y) * 5.0);

    // Blocky Sun & Moon
    if (max(abs(rd.x - sunDir.x), abs(rd.z - sunDir.z)) < 0.06 && abs(rd.y - sunDir.y) < 0.06) {
        skyColor += vec3(1.0, 0.9, 0.4) * (1.0 + safeFx);
    }
    if (max(abs(rd.x - moonDir.x), abs(rd.z - moonDir.z)) < 0.04 && abs(rd.y - moonDir.y) < 0.04) {
        skyColor += vec3(0.8, 0.9, 1.0);
    }

    // Stars
    if (!isDay) {
        float stars = pow(hash12(floor(rd.xy * 400.0)), 200.0);
        skyColor += stars * smoothstep(0.0, -0.2, sunDir.y);
    }

    // Blocky Clouds Overlay
    float cloudMask = blockyClouds(rd, time * 3.0);
    vec3 cloudColor = isDay ? vec3(1.0) : vec3(0.15);
    if (abs(sunDir.y) < 0.2) cloudColor = mix(cloudColor, vec3(1.0, 0.5, 0.3), 1.0 - abs(sunDir.y) * 5.0);
    skyColor = mix(skyColor, cloudColor, cloudMask * smoothstep(0.0, 0.2, rd.y));

    // Raymarching
    float t = 0.0;
    int matID = -1;
    vec4 state = vec4(0.0);
    vec3 volumeGlow = vec3(0.0);

    for (int i = 0; i < MAX_STEPS; i++) {
        if (i >= maxRaySteps) break;
        vec3 pos = ro + rd * t;
        int currMat; vec4 currState;
        float h = map(pos, currMat, currState);

        // Volumetric Fire Glow
        if (currMat == MAT_FIRE || currState.a > 0.1) {
            volumeGlow += vec3(1.0, 0.5, 0.1) * (0.015 * safeFx) / (1.0 + abs(h) * 5.0);
        }

        if (h < max(SURF_DIST, 0.002 * t) || t > MAX_DIST) {
            matID = currMat;
            state = currState;
            break;
        }
        t += clamp(h, 0.04, 1.2);
    }

    vec3 color = skyColor;

    if (t < MAX_DIST) {
        vec3 pos = ro + rd * t;
        vec3 nor = calcNormal(pos);
        
        // Lighting
        float sha = calcVoxelShadow(pos + nor * 0.05, mainLight, 25.0);
        
        // Cloud Shadows on terrain
        if (isDay && nor.y > 0.0) {
            float cShadow = blockyClouds(mainLight, time * 3.0 + pos.x * 0.001);
            sha *= mix(1.0, 0.4, cShadow);
        }

        float ao = calcVoxelAO(pos, nor);
        float dif = clamp(dot(nor, mainLight), 0.0, 1.0) * sha;
        float amb = clamp(0.4 + 0.6 * nor.y, 0.0, 1.0);
        
        // Texturing
        vec3 texColor = getVoxelTexture(matID, pos, nor, state, safeFx);
        vec3 emission = vec3(0.0);

        if (matID == MAT_FIRE) {
            emission = texColor * (1.5 + glow * 0.8) * safeFx;
            texColor = vec3(0.0); // Don't light the fire itself
        }
        else if (matID == MAT_WATER) {
            vec3 ref = reflect(rd, nor);
            float spe = pow(clamp(dot(ref, mainLight), 0.0, 1.0), 48.0) * sha;
            emission += vec3(0.9, 0.95, 1.0) * spe * (0.5 + safeFx * 0.5);
            // Blocky reflections
            vec2 refPlane = ref.xz * (40.0 / max(ref.y, 0.01));
            float refCloud = step(0.65, fbm(floor(refPlane * 0.2) * 0.15));
            texColor = mix(texColor, mix(skyColor, vec3(1.0), refCloud), 0.35);
        }

        // Fire local illumination on surrounding blocks
        float fireLight = clamp(state.a, 0.0, 1.0);
        vec3 fireIllum = vec3(1.0, 0.6, 0.2) * fireLight * 2.5 * safeFx;

        // Assembly
        vec3 lin = vec3(0.0);
        vec3 lightCol = isDay ? vec3(1.0, 0.95, 0.9) : vec3(0.3, 0.4, 0.6);
        lin += 1.8 * dif * lightCol * ao * lightIntensity;
        lin += 0.5 * amb * vec3(0.4, 0.5, 0.7) * ao;
        lin += fireIllum * ao;

        color = texColor * lin + emission;
        
        // Voxel Fog
        float fog = 1.0 - exp(-0.0003 * t * t);
        color = mix(color, skyColor, fog);
    }

    // Add accumulated volumetric fire glow
    color += volumeGlow * (1.2 + safeFx);

    // ACES Tonemapping
    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    
    // Gamma
    color = pow(color, vec3(1.0 / max(gamma, 0.2)));

    // Vignette
    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.5 + 0.5 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.25);

    fragColor = vec4(color, 1.0);
}
"""

WORLD_SEED = 7731429

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

    # Base Continent Shape
    height = (
        0.42
        + 0.22 * np.sin(x_norm * 5.5 + np.cos(y_norm * 3.2) * 1.8)
        + 0.15 * np.cos(y_norm * 7.3 - 0.9)
        + 0.12 * np.sin((x_norm + y_norm) * 9.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.025
    )

    # Majestic Mountain Ranges
    mountain_ridges = np.abs(np.sin(x_norm * 14.0 + np.cos(y_norm * 11.0) * 2.5))
    height += np.clip(0.35 - mountain_ridges, 0.0, 1.0) * 0.35

    # Canyons and Rivers
    river_wave = np.abs(np.sin(x_norm * 12.0 + np.sin(y_norm * 8.0) * 3.5))
    river_mask = np.clip((0.12 - river_wave) / 0.12, 0.0, 1.0)
    height -= river_mask * 0.25
    
    height = np.clip(height, 0.0, 1.0)

    # Sea Level threshold mapping
    sea_level_normalized = 0.40
    ocean = (height < sea_level_normalized).astype(np.float32)
    coast = np.clip(1.0 - np.abs(height - sea_level_normalized) / 0.06, 0.0, 1.0)

    # Moisture distribution (R Channel)
    moisture = np.clip(
        0.18
        + ocean * 0.65
        + coast * 0.35
        + 0.15 * np.sin(x_norm * 8.0 - y_norm * 7.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.04,
        0.0,
        1.0
    )

    # Biomass/Forest distribution (G Channel)
    # Forests thrive in moisture but avoid high peaks and oceans
    biomass = np.clip(
        (1.0 - ocean)
        * (
            0.15
            + moisture * 0.8
            - np.clip(height - 0.65, 0.0, 1.0) * 2.0
            + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.06
        ),
        0.0,
        1.0
    )

    # Fire / Energy (A Channel)
    # Start with isolated tiny sparks in dense forests to ignite the Cellular Automata immediately
    fire = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    dense_forests = np.argwhere((biomass > 0.8) & (height > sea_level_normalized + 0.05))
    if len(dense_forests) > 0:
        spark_count = max(4, (tiles_x * tiles_y) // 4000)
        spark_indices = rng.choice(len(dense_forests), size=spark_count, replace=False)
        for idx in spark_indices:
            cy, cx = dense_forests[idx]
            fire[cy, cx] = 1.0

    # Finalize state matrix
    tile_field = np.stack(
        [
            moisture.astype(np.float32),
            biomass.astype(np.float32),
            height.astype(np.float32),
            fire.astype(np.float32),
        ],
        axis=-1,
    )

    # Scale to full simulation size
    field = np.repeat(np.repeat(tile_field, tile_size, axis=0), tile_size, axis=1)
    field = field[:height_px, :width_px].copy()
    return field

SPEC = WorldSpec(
    id='minecraft-ultimate-voxel',
    display_name='Minecraft Ultimate Voxel',
    window_title='Garage Life Lab - Ultimate Voxel Edition',
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={
        "ray_steps": 160,
        "fx_intensity": 1.3,
        "contour_contrast": 1.2,
        "camera_speed": 0.8,
        "tile_size": 8,
        "substeps": 10,
    },
    preview_image='assets/world_previews/minecraft-long-term-3d.png',
    stability_notes=('extreme hardware stress', '16x16 procedural textures', 'cellular automata wildfires', 'true voxel lighting'),
    hud_subtitle='ULTIMATE VOXEL STRESS TEST',
    preview_palette=('#1E3D59', '#397A2B', '#4D3622', '#696969', '#E88C15', '#F5E6CC', '#D9381E'),
)
