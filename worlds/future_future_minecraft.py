"""World definition for Minecraft Voxel Stress."""
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
uniform float audioBass;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Minecraft Ecosystem & Geology Simulation
// R: Moisture / Water saturation (Fuels biomass, flows down)
// G: Biomass / Forests (Grows by consuming R)
// B: Terrain Height (LOCKED to prevent land running away)
// A: Magma / Tech Corruption (Destroys G, pulses with audio)
void main() {
    vec2 texel = 1.0 / resolution;

    vec4 c = texture(stateTex, uv);
    vec4 r = texture(stateTex, uv + vec2(texel.x, 0.0));
    vec4 l = texture(stateTex, uv - vec2(texel.x, 0.0));
    vec4 t = texture(stateTex, uv + vec2(0.0, texel.y));
    vec4 b = texture(stateTex, uv - vec2(0.0, texel.y));
    vec4 tr = texture(stateTex, uv + texel);
    vec4 tl = texture(stateTex, uv + vec2(-texel.x, texel.y));
    vec4 br = texture(stateTex, uv + vec2(texel.x, -texel.y));
    vec4 bl = texture(stateTex, uv - texel);

    // Laplacian for Reaction-Diffusion
    float lapR = (r.r + l.r + t.r + b.r) * 0.2 + (tr.r + tl.r + br.r + bl.r) * 0.05 - c.r;
    float lapG = (r.g + l.g + t.g + b.g) * 0.2 + (tr.g + tl.g + br.g + bl.g) * 0.05 - c.g;
    float lapA = (r.a + l.a + t.a + b.a) * 0.2 + (tr.a + tl.a + br.a + bl.a) * 0.05 - c.a;

    // Topographical Gradient for flow mapping
    vec2 gradB = vec2(r.b - l.b, t.b - b.b);
    
    // Advection: Moisture flows naturally down elevation gradients
    float advectR = dot(gradB, vec2(r.r - l.r, t.r - b.r)) * 1.5;

    float wetNoise = (hash(uv * resolution + floor(time * 0.2)) - 0.5) * noiseStrength;

    // Reaction-Diffusion ecosystem parameters
    // Lowlands (high c.b) collect more moisture organically
    float lowlandProximity = smoothstep(0.8, 0.2, c.b);
    float localFeed = feed + lowlandProximity * 0.02 + wetNoise * 0.01;
    
    // Altitude and Lava (A) kill off forests (G)
    float altitudeHarshness = smoothstep(0.6, 1.0, c.b) * 0.03;
    float localKill = kill + altitudeHarshness + (c.a * 0.1) + parameterDrift;

    float reaction = c.r * c.g * c.g;
    
    // Update Ecosystem
    float dr = (diffU * lapR * laplaceScale) - reaction + localFeed * (1.0 - c.r) - advectR;
    float dg = (diffV * lapG * laplaceScale) + reaction - (localFeed + localKill) * c.g;
    
    // Magma pulses, spreads slightly, cools down
    float da = (diffU * 0.5 * lapA * laplaceScale) - (c.a * 0.008) + (audioBass * 0.015 * c.a);

    // TERRAIN LOCK: By forcing db = 0.0, the gorgeous geological structures 
    // generated in Python NEVER run away, melt, or flatten. 
    // They act as the eternal canvas for the living ecosystem above.
    float db = 0.0;

    fragColor = vec4(
        clamp(c.r + dr * dt, 0.0, 1.0),
        clamp(c.g + dg * dt, 0.0, 1.0),
        clamp(c.b + db, 0.0, 1.0),
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
uniform float audioEnergy;
uniform float audioBass;
uniform float audioTreble;

// Constants
const float SEA_LEVEL = 64.0;
const float CLOUD_HEIGHT = 160.0;

// High-performance hash and 3D noise for voxel geometry
float hash(float n) { return fract(sin(n) * 43758.5453123); }

float hash31(vec3 p3) {
    p3  = fract(p3 * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float noise3D(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 57.0 + 113.0 * p.z;
    return mix(
        mix(mix(hash(n + 0.0), hash(n + 1.0), f.x),
            mix(hash(n + 57.0), hash(n + 58.0), f.x), f.y),
        mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
            mix(hash(n + 170.0), hash(n + 171.0), f.x), f.y), f.z
    );
}

mat3 setCamera(in vec3 ro, in vec3 ta, float cr) {
    vec3 cw = normalize(ta - ro);
    vec3 cp = vec3(sin(cr), cos(cr), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = normalize(cross(cu, cw));
    return mat3(cu, cv, cw);
}

mat2 rot(float a) {
    float s = sin(a), c = cos(a);
    return mat2(c, -s, s, c);
}

// Voxel Engine - Maps 3D coordinates to Block IDs
// 0:Air, 1:Bedrock, 2:Stone, 3:Dirt, 4:Grass, 5:Sand, 6:Water, 7:Lava, 8:Wood, 9:Leaves, 10:Tech, 11:Cloud
int getBlock(vec3 p, out vec4 stateOut) {
    if (p.y < 0.0) return 1;
    if (p.y > 256.0) return 0;
    
    // Volumetric Minecraft Clouds
    if (p.y >= CLOUD_HEIGHT && p.y <= CLOUD_HEIGHT + 4.0) {
        float cN = noise3D(p * 0.02 + vec3(time * 0.3, 0.0, time * 0.1));
        if (cN > 0.65) return 11;
    }

    vec2 mapUV = (floor(p.xz) + 0.5) * 0.002; // Map scaling
    vec4 s = textureLod(stateTex, mapUV, 0.0);
    stateOut = s;

    float baseH = floor(s.b * 160.0 + 20.0); // Epic Mountain Scale

    // Floating Islands & Overhangs (3D Extrusion)
    if (p.y > baseH) {
        float n3 = noise3D(p * 0.04 + vec3(100.0));
        if (n3 > 0.82 && p.y < baseH + 50.0) {
            if (p.y > baseH + 40.0) return 4; // Grass Top
            return 2; // Stone Body
        }
    } else {
        // Deep Cave Systems
        if (p.y > 5.0) {
            float cave = noise3D(p * 0.05) + noise3D(p * 0.1) * 0.5;
            if (cave > 0.85) return 0; // Air Pocket
        }
    }

    // Base Solid Terrain
    if (p.y <= baseH) {
        if (p.y < baseH - floor(s.r * 6.0 + 4.0)) return 2; // Stone
        
        if (s.a > 0.5 && p.y < baseH) return 7; // Subterranean Magma
        if (s.r < 0.2 && s.g < 0.2) return 5; // Desert Sand Biome
        if (baseH <= SEA_LEVEL + 2.0) return 5; // Coastal Beaches
        
        if (p.y == baseH) return 4; // Grass Surface
        return 3; // Dirt Sub-layer
    }

    // Oceans / Lakes
    if (p.y <= SEA_LEVEL) return 6; 

    // Procedural Forests (Driven by Simulation Biomass 'G')
    if (s.g > 0.4 && p.y > baseH && p.y < baseH + 20.0) {
        vec2 cell = floor(p.xz / 7.0);
        if (hash31(vec3(cell, 1.0)) > 0.55) {
            vec2 center = cell * 7.0 + vec2(3.5);
            float dist = max(abs(p.x - center.x), abs(p.z - center.y));
            float treeH = baseH + floor(hash31(vec3(cell, 2.0)) * 7.0) + 6.0;
            if (dist < 1.0 && p.y <= treeH) return 8; // Trunk
            if (dist < 3.0 && p.y >= treeH - 4.0 && p.y <= treeH + 1.0) {
                if (hash31(p) < 0.8) return 9; // Leaves
            }
        }
    }

    // Alien Obelisks (Driven by Simulation 'A')
    if (s.a > 0.7) {
        vec2 cell = floor(p.xz / 19.0);
        if (hash31(vec3(cell, 3.0)) > 0.75) {
            vec2 center = cell * 19.0 + vec2(9.5);
            float dist = max(abs(p.x - center.x), abs(p.z - center.y));
            float monoH = baseH + floor(s.a * 45.0);
            if (dist <= 2.0 && p.y <= monoH) {
                if (dist < 1.0 && p.y > baseH + 4.0 && p.y < monoH - 4.0) return 0; // Hollow
                return 10;
            }
        }
    }

    return 0; // Air
}

// Computes Minecraft-style voxel edge ambient occlusion
float calcVoxelAO(vec2 faceUv) {
    vec2 q = abs(faceUv - 0.5) * 2.0;
    float edge = max(q.x, q.y);
    return mix(1.0, 0.5, pow(edge, 5.0));
}

// Sweeping Cloud Shadows
float cloudShadow(vec3 pos, vec3 lightDir) {
    if (lightDir.y <= 0.0) return 1.0;
    float t = (CLOUD_HEIGHT - pos.y) / lightDir.y;
    if (t > 0.0) {
        vec3 cp = pos + lightDir * t;
        float n = noise3D(cp * 0.02 + vec3(time * 0.3, 0.0, time * 0.1));
        if (n > 0.65) return 0.5; // 50% Shadow
    }
    return 1.0;
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    int maxSteps = clamp(raySteps, 64, 400);

    // Dynamic Day/Night Cycle
    float tod = time * 0.06;
    vec3 sunDir = normalize(vec3(cos(tod), sin(tod), 0.4));
    vec3 moonDir = normalize(vec3(-cos(tod), -sin(tod), -0.4));
    bool isDay = sunDir.y > 0.0;
    vec3 lightDir = isDay ? sunDir : moonDir;
    float lightIntensity = isDay ? smoothstep(-0.1, 0.2, sunDir.y) : smoothstep(-0.1, 0.2, moonDir.y) * 0.4;

    // Smooth Cinematic Camera Flight
    float camTime = time * 0.2 * max(cameraSpeed, 0.05);
    vec3 ro = vec3(camTime * 18.0, 0.0, camTime * 14.0);
    
    // Prevent clipping into the massive terrain
    float terrainLimit = textureLod(stateTex, (floor(ro.xz) + 0.5) * 0.002, 0.0).b * 160.0 + 20.0;
    ro.y = max(max(terrainLimit, SEA_LEVEL) + 25.0 + sin(camTime * 0.7) * 8.0, SEA_LEVEL + 10.0);
    
    vec3 ta = vec3(ro.x + 15.0, ro.y - 12.0, ro.z + 15.0 + sin(camTime * 1.1) * 7.0);
    ro += cameraOffset;
    ta += cameraOffset;

    float zoomedLens = 1.8 * clamp(exp(cameraZoom), 0.35, 3.0);
    vec3 rd = normalize(vec3(p.xy, zoomedLens));
    rd.yz = rot(cameraYawPitch.y) * rd.yz;
    rd.xz = rot(cameraYawPitch.x) * rd.xz;
    mat3 ca = setCamera(ro, ta, sin(camTime * 0.2) * 0.08);
    rd = ca * rd;

    vec3 finalColor = vec3(0.0);
    vec3 rayAtten = vec3(1.0);
    
    // Recursive DDA pass implementation (1 bounce for stunning water reflections)
    for (int bounce = 0; bounce < 2; bounce++) {
        vec3 mapPos = floor(ro);
        vec3 rayStep = sign(rd);
        vec3 deltaDist = 1.0 / max(abs(rd), 1e-8);
        vec3 sideDist = (rayStep * (mapPos - ro) + (rayStep * 0.5 + 0.5)) * deltaDist;
        vec3 mask = vec3(0.0);

        int hitBlock = 0;
        vec4 hitState = vec4(0.0);
        float distT = 0.0;
        bool reflected = false;

        for (int i = 0; i < 400; i++) {
            if (i >= maxSteps) break;

            int b = getBlock(mapPos, hitState);
            
            if (b > 0) {
                if (b == 6 && bounce == 0) { // Hit Water on primary ray
                    vec3 tVec = (mapPos - ro + (1.0 - rayStep) * 0.5) / max(abs(rd), 1e-8);
                    distT = dot(tVec, mask);
                    vec3 pos = ro + rd * distT;
                    vec3 nor = vec3(0.0, 1.0, 0.0); // Flat water normal
                    
                    // Water Base Color
                    vec3 waterAlbedo = vec3(0.05, 0.2, 0.45) * safeFx;
                    float sunSpec = pow(max(0.0, dot(reflect(rd, nor), lightDir)), 200.0);
                    finalColor += rayAtten * (waterAlbedo + vec3(1.0, 0.9, 0.7) * sunSpec * lightIntensity);
                    
                    // Setup reflection ray
                    rayAtten *= vec3(0.7, 0.85, 1.0); // Tint the reflection
                    ro = pos + nor * 0.01;
                    rd = reflect(rd, nor);
                    // Add micro-waves
                    rd.xz += vec2(sin(pos.x * 2.0 + time), cos(pos.z * 2.0 + time)) * 0.02;
                    rd = normalize(rd);
                    reflected = true;
                    break;
                } else if (b == 6 && bounce == 1) {
                    // Ignore secondary water hits
                } else if (b == 11) { // Clouds are soft
                    finalColor += rayAtten * vec3(0.9, 0.95, 1.0) * 0.4;
                    rayAtten *= 0.6;
                } else if (b == 9) { // Leaves are translucent
                    finalColor += rayAtten * vec3(0.15, 0.4, 0.15) * 0.3;
                    rayAtten *= 0.7;
                } else {
                    hitBlock = b;
                    break;
                }
            }

            if (length(rayAtten) < 0.05) break;

            mask = step(sideDist.xyz, sideDist.yzx) * step(sideDist.xyz, sideDist.zxy);
            sideDist += mask * deltaDist;
            mapPos += mask * rayStep;
        }

        if (hitBlock == 0 && !reflected) {
            // Skybox Rendering
            vec3 skyColor;
            if (isDay) {
                skyColor = mix(vec3(0.4, 0.6, 0.9), vec3(0.1, 0.25, 0.6), max(0.0, rd.y));
                float sun = pow(max(0.0, dot(rd, lightDir)), 500.0);
                skyColor += vec3(1.0, 0.9, 0.7) * sun * safeFx * 2.0;
            } else {
                skyColor = mix(vec3(0.02, 0.04, 0.08), vec3(0.0), max(0.0, rd.y));
                if (rd.y > 0.0) { // Stars
                    float star = hash31(floor(rd * 300.0));
                    if (star > 0.99) skyColor += vec3(1.0) * (0.3 + 0.7 * sin(time * 2.0 + star * 20.0));
                }
                float moon = pow(max(0.0, dot(rd, lightDir)), 300.0);
                skyColor += vec3(0.6, 0.8, 1.0) * moon * safeFx;
            }
            finalColor += rayAtten * skyColor;
            break; // Sky hit, end bouncing
        }
        
        if (hitBlock > 0) {
            // Solid Hit Rendering
            vec3 tVec = (mapPos - ro + (1.0 - rayStep) * 0.5) / max(abs(rd), 1e-8);
            float t = dot(tVec, mask);
            vec3 pos = ro + rd * t;
            vec3 nor = -rayStep * mask;

            vec3 uvw = fract(pos);
            vec2 faceUv = (mask.x > 0.5) ? uvw.yz : (mask.y > 0.5 ? uvw.xz : uvw.xy);

            vec2 texUv = floor(faceUv * 16.0) / 16.0;
            float texNoise = hash31(vec3(floor(mapPos.xz) * 13.7, floor(mapPos.y) * 19.1) + texUv.xyx * 37.4);
            
            float blockAO = mix(1.0, calcVoxelAO(faceUv), contourContrast);

            vec3 albedo = vec3(1.0);
            vec3 emission = vec3(0.0);
            
            if (hitBlock == 1) albedo = vec3(0.1) + texNoise * 0.1;
            else if (hitBlock == 2) albedo = vec3(0.4) + texNoise * 0.1;
            else if (hitBlock == 3) albedo = vec3(0.42, 0.28, 0.18) + texNoise * 0.08;
            else if (hitBlock == 4) {
                if (nor.y > 0.5) albedo = vec3(0.25, 0.58, 0.25) + texNoise * 0.12;
                else if (faceUv.y > 0.75 + texNoise * 0.2) albedo = vec3(0.25, 0.58, 0.25) + texNoise * 0.12;
                else albedo = vec3(0.42, 0.28, 0.18) + texNoise * 0.08;
            }
            else if (hitBlock == 5) albedo = vec3(0.82, 0.78, 0.6) + texNoise * 0.05;
            else if (hitBlock == 7) {
                albedo = mix(vec3(0.9, 0.1, 0.0), vec3(1.0, 0.7, 0.0), texNoise);
                emission = albedo * (2.5 + audioBass * 4.0);
            }
            else if (hitBlock == 8) {
                if (abs(nor.y) > 0.5) albedo = vec3(0.5, 0.38, 0.2) + texNoise * 0.1;
                else albedo = vec3(0.3, 0.22, 0.15) + texNoise * 0.1;
            }
            else if (hitBlock == 10) {
                albedo = mix(vec3(0.0, 0.7, 1.0), vec3(0.8, 0.0, 1.0), hitState.a);
                float circuit = step(0.85, hash31(vec3(texUv, time * 0.05)));
                emission = albedo * circuit * (4.0 + audioTreble * 5.0) * safeFx;
                albedo += texNoise * 0.15;
            }

            // Raytraced Hard Shadows
            vec3 mapPosS = floor(pos + nor * 0.002);
            vec3 rdS = lightDir;
            vec3 rayStepS = sign(rdS);
            vec3 deltaDistS = 1.0 / max(abs(rdS), 1e-8);
            vec3 sideDistS = (rayStepS * (mapPosS - (pos + nor * 0.002)) + (rayStepS * 0.5 + 0.5)) * deltaDistS;
            vec3 maskS = vec3(0.0);
            float shadow = 1.0;
            
            for (int j = 0; j < 50; j++) {
                vec4 dummy;
                int bS = getBlock(mapPosS, dummy);
                if (bS > 0 && bS != 6 && bS != 9) {
                    shadow = (bS == 11) ? 0.5 : 0.1; // Cloud shadow vs Hard shadow
                    break;
                }
                maskS = step(sideDistS.xyz, sideDistS.yzx) * step(sideDistS.xyz, sideDistS.zxy);
                sideDistS += maskS * deltaDistS;
                mapPosS += maskS * rayStepS;
            }

            // Combine with Cloud Shadows
            shadow *= cloudShadow(pos, lightDir);

            // Lighting Model
            float diff = max(dot(nor, lightDir), 0.0);
            float sky = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
            
            vec3 lin = vec3(0.0);
            lin += 1.6 * diff * vec3(1.0, 0.95, 0.85) * shadow * lightIntensity;
            
            vec3 ambientSky = isDay ? vec3(0.3, 0.45, 0.6) : vec3(0.05, 0.08, 0.15);
            lin += 0.6 * sky * ambientSky;
            
            vec3 hitColor = albedo * lin * blockAO + emission * (1.0 + glow * 0.5);

            // Distance Atmospheric Fog
            float fogD = (bounce == 0) ? t : t + distT;
            float fog = 1.0 - exp(-0.00004 * fogD * fogD);
            vec3 fogColor = isDay ? mix(vec3(0.5, 0.6, 0.8), vec3(1.0, 0.9, 0.7), pow(max(0.0, dot(rd, lightDir)), 4.0)) : vec3(0.02, 0.04, 0.08);
            
            hitColor = mix(hitColor, fogColor, fog);
            finalColor += rayAtten * hitColor;
            
            break; // Stop bouncing after solid hit
        }
    }

    // ACES Tonemapping & Gamma
    finalColor *= exposure;
    finalColor = (finalColor * (2.51 * finalColor + 0.03)) / (finalColor * (2.43 * finalColor + 0.59) + 0.14);
    finalColor = pow(finalColor, vec3(1.0 / max(gamma, 0.2)));

    // Stylized Vignette
    vec2 q = gl_FragCoord.xy / resolution.xy;
    finalColor *= 0.5 + 0.5 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.2);

    fragColor = vec4(finalColor, 1.0);
}
"""

def seed_field(width_px: int, height_px: int, tile_size: int) -> np.ndarray:
    tile_size = max(2, int(tile_size))
    tiles_x = max(1, int(np.ceil(width_px / tile_size)))
    tiles_y = max(1, int(np.ceil(height_px / tile_size)))

    rng = np.random.default_rng(9999)
    tile_y, tile_x = np.meshgrid(
        np.arange(tiles_y, dtype=np.float32),
        np.arange(tiles_x, dtype=np.float32),
        indexing="ij",
    )
    x_norm = tile_x / max(tiles_x - 1, 1)
    y_norm = tile_y / max(tiles_y - 1, 1)

    def noise(nx, ny):
        return np.sin(nx) * np.cos(ny)

    # Majestic multi-frequency fractal noise
    n1 = noise(x_norm * 5.0, y_norm * 5.0)
    n2 = noise(x_norm * 13.0 + y_norm * 4.0, y_norm * 11.0 - x_norm * 3.0)
    n3 = noise(x_norm * 29.0, y_norm * 31.0)
    
    # Ridged multifractal for sweeping, dramatic mountain ranges
    ridges = 1.0 - np.abs(noise(x_norm * 8.0, y_norm * 8.0))
    ridges = ridges ** 2.5
    
    height = 0.20 + 0.20 * n1 + 0.10 * n2 + 0.05 * n3 + 0.40 * ridges
    
    # Introduce a colossal central volcano/caldera
    cx, cy = tiles_x * 0.5, tiles_y * 0.5
    dist = np.sqrt((tile_x - cx)**2 + (tile_y - cy)**2) / (max(tiles_x, tiles_y) * 0.4)
    volcano = np.clip(1.0 - dist * 2.0, 0.0, 1.0) ** 2.0
    crater = np.clip(1.0 - dist * 4.0, 0.0, 1.0)
    
    height += volcano * 0.45
    height -= crater * 0.40
    
    # Apply bounds
    height = np.clip(height, 0.0, 1.0)

    # Sea Level reference based on shader (64 / 256 approx = 0.25)
    sea_level = 0.25
    ocean = (height < sea_level).astype(np.float32)
    coast = np.clip(1.0 - np.abs(height - sea_level) / 0.04, 0.0, 1.0)

    # R: Moisture (Creates oases, rivers, and feeds biomass)
    moisture = np.clip(
        0.2 
        + ocean * 0.8 
        + coast * 0.5
        + 0.15 * noise(x_norm * 14.0, y_norm * 14.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.05,
        0.0,
        1.0,
    )

    # G: Biomass (Lush forests)
    biomass = np.clip(
        (1.0 - ocean)
        * (
            0.15
            + coast * 0.4
            + moisture * 0.5
            - np.clip(height - 0.6, 0.0, 1.0) * 1.8 # Treeline cutoff
        ),
        0.0,
        1.0,
    )

    # A: Magma / Obelisk Tech (Placed in high elevations and craters)
    magma = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    magma_mask = (height > 0.6) | (crater > 0.5)
    candidates = np.argwhere(magma_mask)
    if len(candidates) > 0:
        vent_count = min(max(8, (tiles_x * tiles_y) // 1800), len(candidates))
        vent_indices = rng.choice(len(candidates), size=vent_count, replace=False)
        for idx in vent_indices:
            cy, cx = candidates[idx]
            radius = int(rng.integers(4, 9))
            y0 = max(cy - radius, 0)
            y1 = min(cy + radius + 1, tiles_y)
            x0 = max(cx - radius, 0)
            x1 = min(cx + radius + 1, tiles_x)
            patch_y, patch_x = np.meshgrid(
                np.arange(y0, y1, dtype=np.float32),
                np.arange(x0, x1, dtype=np.float32),
                indexing="ij",
            )
            dist_p = np.sqrt((patch_x - cx) ** 2 + (patch_y - cy) ** 2)
            influence = np.clip(1.0 - dist_p / max(radius + 0.5, 1.0), 0.0, 1.0)
            magma[y0:y1, x0:x1] = np.maximum(
                magma[y0:y1, x0:x1],
                influence * rng.uniform(0.7, 1.0),
            )
            # Scorch the earth
            biomass[y0:y1, x0:x1] *= (1.0 - influence)

    tile_field = np.stack(
        [
            moisture.astype(np.float32),
            biomass.astype(np.float32),
            height.astype(np.float32),
            magma.astype(np.float32),
        ],
        axis=-1,
    )

    field = np.repeat(np.repeat(tile_field, tile_size, axis=0), tile_size, axis=1)
    field = field[:height_px, :width_px].copy()
    return field


SPEC = WorldSpec(
    id='minecraft-future-future-term-3d',
    display_name='Minecraft Ultimate Voxel test v2',
    window_title='Garage Life Lab - Minecraft Raytraced',
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={
        'feed': 0.038,        # Stable bio-growth
        'kill': 0.060,        
        'diff_u': 0.16,
        'diff_v': 0.08,
        'ray_steps': 200,     # Immense depth, 2-bounce reflection, volumetric clouds
        'fx_intensity': 1.1,
        'camera_speed': 0.8,
        'exposure': 1.4,      
        'contour_contrast': 0.85,
    },
    preview_image='assets/world_previews/minecraft-long-term-3d.png',
    stability_notes=('extreme gpu load', 'recursive dda', 'locked geology'),
    hud_subtitle='ULTIMATE VOXEL STRESS TEST',
    preview_palette=('#121212', '#332315', '#4ba34b', '#184b96', '#cfbc70', '#fc6203', '#ffffff'),
    uses_audio=True,
)