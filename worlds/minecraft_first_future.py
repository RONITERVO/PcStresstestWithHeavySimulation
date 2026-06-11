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
// R: Nutrients / Water (Flows downhill, erodes terrain)
// G: Biomass / Forests (Grows by consuming R)
// B: Terrain Height (Eroded by R, built up by A)
// A: Magma / Tech Corruption (Flows, destroys G, cools into terrain)
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

    // Standard Laplacian
    float lapR = (r.r + l.r + t.r + b.r) * 0.2 + (tr.r + tl.r + br.r + bl.r) * 0.05 - c.r;
    float lapG = (r.g + l.g + t.g + b.g) * 0.2 + (tr.g + tl.g + br.g + bl.g) * 0.05 - c.g;
    float lapA = (r.a + l.a + t.a + b.a) * 0.2 + (tr.a + tl.a + br.a + bl.a) * 0.05 - c.a;

    // Topographical Gradient
    vec2 gradB = vec2(r.b - l.b, t.b - b.b);

    // Advection: nutrients/water and magma flow downhill, but with bounded flux.
    float flowStrength = 0.55;
    float advectR = clamp(dot(gradB, vec2(r.r - l.r, t.r - b.r)) * flowStrength, -0.06, 0.06);
    float advectA = clamp(dot(gradB, vec2(r.a - l.a, t.a - b.a)) * flowStrength * 0.4, -0.04, 0.04);

    float wetNoise = (hash(uv * resolution + floor(time * 0.5)) - 0.5) * noiseStrength;

    // Gray-Scott Reaction (Biomass consumes Nutrients)
    float localFeed = feed + c.b * 0.01 + wetNoise * 0.03;
    float localKill = kill + parameterDrift + (c.a * 0.05); // Magma increases kill rate heavily

    float reaction = c.r * c.g * c.g;
    
    float dr = (diffU * lapR * laplaceScale) - reaction + localFeed * (1.0 - c.r) - advectR;
    float dg = (diffV * lapG * laplaceScale) + reaction - (localFeed + localKill) * c.g;
    
    // Magma cools slowly, pulses with bass, destroys biomass
    float da = (diffU * 0.5 * lapA * laplaceScale) - advectA - (c.a * 0.005) + (audioBass * 0.02 * c.a);

    // Active Geology: flowing water carves channels, standing moisture does not.
    float slope = length(gradB);
    float flowErosion = c.r * smoothstep(0.025, 0.16, slope);
    float aboveSea = smoothstep(0.34, 0.50, c.b);
    float rootProtection = mix(1.0, 0.12, smoothstep(0.18, 0.72, c.g));
    float erosion = flowErosion * aboveSea * rootProtection * 0.00018;

    // Sediment and roots rebuild low/flat areas so water cannot eat the whole map.
    float flatDeposition = c.r * (1.0 - smoothstep(0.04, 0.18, slope)) * smoothstep(0.16, 0.38, c.b) * 0.00008;
    float biomassBuild = c.g * 0.00008 * smoothstep(0.20, 0.75, c.b);
    float magmaBuild = c.a * 0.00025 * smoothstep(0.85, 0.35, c.b);
    float floorRecovery = max(0.0, 0.16 - c.b) * 0.012;
    float db = (magmaBuild + biomassBuild + flatDeposition + floorRecovery - erosion + wetNoise * 0.00002) * dt;

    fragColor = vec4(
        clamp(c.r + dr * dt, 0.0, 1.0),
        clamp(c.g + dg * dt, 0.0, 1.0),
        clamp(c.b + db, 0.08, 1.0),
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

// Clever hash for voxel texturing
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453123);
}

float hash3(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

const float WATER_LEVEL = 14.0;

// 3D Noise for massive cave systems
float noise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash3(i);
    float n100 = hash3(i + vec3(1,0,0));
    float n010 = hash3(i + vec3(0,1,0));
    float n110 = hash3(i + vec3(1,1,0));
    float n001 = hash3(i + vec3(0,0,1));
    float n101 = hash3(i + vec3(1,0,1));
    float n011 = hash3(i + vec3(0,1,1));
    float n111 = hash3(i + vec3(1,1,1));
    vec4 x1 = mix(vec4(n000, n010, n001, n011), vec4(n100, n110, n101, n111), f.x);
    vec2 y1 = mix(x1.xz, x1.yw, f.y);
    return mix(y1.x, y1.y, f.z);
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

// Voxel Engine - Maps 3D integer coordinates to Block IDs
// 0:Air, 1:Bedrock, 2:Stone, 3:Dirt, 4:Grass, 5:Sand, 6:Water, 7:Lava, 8:Wood, 9:Leaves, 10:Tech Crystal
int getBlock(vec3 p, out vec4 stateOut) {
    if (p.y < 0.0) return 1;
    if (p.y > 160.0) return 0;
    
    // Snap UV to block center to ensure exact columnar voxel geology
    vec2 cellXZ = floor(p.xz);
    vec2 mapUV = (cellXZ + 0.5) * 0.003;
    vec4 s = textureLod(stateTex, mapUV, 0.0);
    stateOut = s;

    float baseH = floor(s.b * 80.0 + 10.0);

    // Cave Generation (Extremely demanding 3D procedural noise)
    if (p.y < baseH && p.y > 4.0) {
        float n = noise3D(p * 0.04) + noise3D(p * 0.1) * 0.5;
        if (n > 0.85) return 0; // Cave Air
    }

    // Solid Terrain
    if (p.y <= baseH) {
        if (p.y < baseH - floor(s.r * 8.0 + 3.0)) return 2; // Stone
        if (p.y <= WATER_LEVEL + 3.0 && baseH <= WATER_LEVEL + 4.0) return 5; // Sand
        // Underground Lava lakes
        if (s.a > 0.4 && p.y < 12.0) return 7; 
        if (p.y == baseH && baseH > WATER_LEVEL + 2.0) return 4; // Grass
        return 3; // Dirt
    }

    // Sea Level
    if (baseH < WATER_LEVEL && p.y <= WATER_LEVEL) return 6;

    // Flora / Biomass Trees
    if (s.g > 0.45 && p.y > baseH && p.y < baseH + 18.0) {
        vec2 cell = floor(p.xz / 7.0);
        if (hash(cell * 13.1) > 0.6) {
            vec2 center = cell * 7.0 + vec2(3.5);
            float dist = max(abs(p.x - center.x), abs(p.z - center.y));
            float treeH = baseH + floor(hash(cell) * 6.0) + 5.0;
            if (dist < 1.0 && p.y <= treeH) return 8; // Wood
            if (dist < 3.0 && p.y > treeH - 3.0 && p.y <= treeH + 1.0) {
                if (hash3(p) < 0.8) return 9; // Leaves
            }
        }
    }

    // Alien Tech / Monoliths
    if (s.a > 0.5) {
        vec2 cell = floor(p.xz / 13.0);
        if (hash(cell * 7.7) > 0.85) {
            vec2 center = cell * 13.0 + vec2(6.5);
            float dist = max(abs(p.x - center.x), abs(p.z - center.y));
            float monoH = baseH + floor(s.a * 30.0);
            if (dist <= 2.0 && p.y <= monoH) {
                // Hollow pulsing cores
                if (dist < 1.0 && p.y > baseH + 2.0 && p.y < monoH - 2.0) return 0;
                return 10;
            }
        }
    }

    return 0;
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    int maxSteps = clamp(raySteps, 64, 300);

    // Cinematic Camera tracking terrain height
    float camTime = time * 0.2 * max(cameraSpeed, 0.05);
    vec3 ro = vec3(camTime * 15.0, 0.0, camTime * 12.0);
    
    float camH = textureLod(stateTex, (floor(ro.xz) + 0.5) * 0.003, 0.0).b * 80.0 + 10.0;
    ro.y = max(camH + 18.0 + sin(camTime)*5.0, 22.0);
    
    vec3 ta = vec3(ro.x + 12.0, ro.y - 8.0, ro.z + 12.0 + sin(camTime * 1.5) * 4.0);
    ro += cameraOffset;
    ta += cameraOffset;

    // Apply Zoom & FOV
    float zoomedLens = 2.0 * clamp(exp(cameraZoom), 0.35, 3.0);
    vec3 rd = normalize(vec3(p.xy, zoomedLens));
    rd.yz = rot(cameraYawPitch.y) * rd.yz;
    rd.xz = rot(cameraYawPitch.x) * rd.xz;
    mat3 ca = setCamera(ro, ta, sin(camTime * 0.3) * 0.05);
    rd = ca * rd;

    // Branchless DDA Setup
    vec3 mapPos = floor(ro);
    vec3 rayStep = sign(rd);
    vec3 deltaDist = 1.0 / max(abs(rd), 1e-8);
    vec3 sideDist = (rayStep * (mapPos - ro) + (rayStep * 0.5 + 0.5)) * deltaDist;
    vec3 mask = vec3(0.0);

    int hitBlock = 0;
    vec4 hitState = vec4(0.0);
    
    // Transparency accumulation
    vec3 throughput = vec3(1.0);
    vec3 transpColor = vec3(0.0);
    float godRays = 0.0;

    for (int i = 0; i < 300; i++) {
        if (i >= maxSteps) break;

        int b = getBlock(mapPos, hitState);
        
        if (b > 0) {
            if (b == 6) { // Water
                throughput *= exp(-vec3(0.8, 0.3, 0.1) * 0.8);
                transpColor += throughput * vec3(0.05, 0.2, 0.4) * 0.2;
            } else if (b == 9) { // Leaves
                throughput *= exp(-vec3(0.1, 0.6, 0.1) * 1.5);
                transpColor += throughput * vec3(0.1, 0.4, 0.1) * 0.3;
            } else {
                hitBlock = b;
                break;
            }
        } else {
            // Clever God Ray accumulation: If in air, check if we are below terrain height
            float localH = hitState.b * 80.0 + 10.0;
            if (mapPos.y > localH) {
                godRays += 0.015;
            }
        }

        if (length(throughput) < 0.05) break;

        mask = step(sideDist.xyz, sideDist.yzx) * step(sideDist.xyz, sideDist.zxy);
        sideDist += mask * deltaDist;
        mapPos += mask * rayStep;
    }

    // Environment Lighting
    vec3 lightDir = normalize(vec3(0.8, 0.9, -0.4));
    float skyRise = smoothstep(-0.2, 0.6, rd.y);
    vec3 skyColor = mix(vec3(0.05, 0.1, 0.2), vec3(0.2, 0.4, 0.8), skyRise);
    
    float sun = pow(max(0.0, dot(rd, lightDir)), 400.0);
    skyColor += vec3(1.0, 0.8, 0.6) * sun * safeFx * 2.0;
    
    // Audio-reactive Aurora
    float aurora = smoothstep(0.4, 0.98, rd.y + 0.2 * sin(p.x * 5.0 + time * 0.2));
    aurora *= 0.5 + 0.5 * sin(p.x * 15.0 + time * 0.5 + colorShift);
    skyColor += aurora * vec3(0.1, 0.9, 0.6) * safeFx * (0.5 + audioTreble * 2.5);

    vec3 color = skyColor;

    if (hitBlock > 0) {
        // Exact Hit Position
        vec3 tVec = (mapPos - ro + (1.0 - rayStep) * 0.5) / max(abs(rd), 1e-8);
        float t = dot(tVec, mask);
        vec3 pos = ro + rd * t;
        vec3 nor = -rayStep * mask;

        // Voxel UV for pixel art textures and edge AO
        vec3 uvw = fract(pos);
        vec2 faceUv;
        if (mask.x > 0.5) faceUv = uvw.yz;
        else if (mask.y > 0.5) faceUv = uvw.xz;
        else faceUv = uvw.xy;

        // Pixelate UV (16x16 resolution)
        vec2 texUv = floor(faceUv * 16.0) / 16.0;
        float texNoise = hash(floor(mapPos.xz)*17.1 + floor(mapPos.y)*13.3 + texUv * 41.7);
        
        // Voxel Edge AO (Classic bevel look)
        float edgeDist = min(min(faceUv.x, 1.0 - faceUv.x), min(faceUv.y, 1.0 - faceUv.y));
        float blockAO = smoothstep(0.0, 0.12, edgeDist) * 0.5 + 0.5;
        blockAO = mix(1.0, blockAO, contourContrast);

        // Albedo & Emission mapping
        vec3 albedo = vec3(1.0);
        vec3 emission = vec3(0.0);
        
        if (hitBlock == 1) albedo = vec3(0.08, 0.08, 0.09) + texNoise * 0.04;
        else if (hitBlock == 2) albedo = vec3(0.26, 0.27, 0.27) + texNoise * 0.06;
        else if (hitBlock == 3) albedo = vec3(0.34, 0.19, 0.08) + texNoise * 0.06;
        else if (hitBlock == 4) {
            if (nor.y > 0.5) albedo = vec3(0.08, 0.52, 0.10) + texNoise * 0.08;
            else if (faceUv.y > 0.58 + texNoise * 0.18) albedo = vec3(0.10, 0.42, 0.08) + texNoise * 0.08;
            else albedo = vec3(0.34, 0.19, 0.08) + texNoise * 0.06;
        }
        else if (hitBlock == 5) albedo = vec3(0.64, 0.55, 0.30) + texNoise * 0.05;
        else if (hitBlock == 7) {
            albedo = mix(vec3(1.0, 0.2, 0.0), vec3(1.0, 0.8, 0.0), texNoise);
            emission = albedo * (2.0 + audioBass * 3.0);
        }
        else if (hitBlock == 8) {
            if (abs(nor.y) > 0.5) albedo = vec3(0.6, 0.45, 0.25) + texNoise * 0.1;
            else albedo = vec3(0.35, 0.2, 0.1) + texNoise * 0.1;
        }
        else if (hitBlock == 10) {
            albedo = mix(vec3(0.0, 0.8, 1.0), vec3(0.8, 0.2, 1.0), hitState.a);
            // Pulsing circuit patterns
            float circuit = step(0.85, hash(texUv + time * 0.1));
            emission = albedo * circuit * (3.0 + audioBass * 5.0) * safeFx;
            albedo += texNoise * 0.2;
        }

        // Hard Voxel Shadows (Secondary DDA)
        vec3 mapPosS = floor(pos + nor * 0.001);
        vec3 rdS = lightDir;
        vec3 rayStepS = sign(rdS);
        vec3 deltaDistS = 1.0 / max(abs(rdS), 1e-8);
        vec3 sideDistS = (rayStepS * (mapPosS - (pos + nor * 0.001)) + (rayStepS * 0.5 + 0.5)) * deltaDistS;
        vec3 maskS = vec3(0.0);
        float shadow = 1.0;
        
        for (int j = 0; j < 40; j++) {
            vec4 dummy;
            int bS = getBlock(mapPosS, dummy);
            if (bS > 0 && bS != 6 && bS != 9) {
                shadow = 0.1; // In Shadow
                break;
            }
            maskS = step(sideDistS.xyz, sideDistS.yzx) * step(sideDistS.xyz, sideDistS.zxy);
            sideDistS += maskS * deltaDistS;
            mapPosS += maskS * rayStepS;
        }

        // Illumination
        float diff = max(dot(nor, lightDir), 0.0);
        float sky = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
        
        vec3 lin = vec3(0.0);
        lin += 2.2 * diff * vec3(1.0, 0.9, 0.7) * shadow; // Sun
        lin += 0.8 * sky * vec3(0.2, 0.35, 0.55); // Ambient Sky
        
        color = albedo * lin * blockAO + emission * (1.0 + glow * 0.5);

        // Depth Fog
        float fog = 1.0 - exp(-0.00008 * t * t);
        color = mix(color, skyColor, fog * 0.65);
    }

    // Apply accumulated transparency and god rays
    color = color * throughput + transpColor;
    
    vec3 godRayColor = vec3(1.0, 0.85, 0.7) * min(godRays * safeFx, 1.0) * (0.5 + 0.5 * max(dot(rd, lightDir), 0.0));
    color += godRayColor;

    // ACES Tonemapping & Gamma Correction
    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(color, vec3(1.0 / max(gamma, 0.2)));

    // Vignette
    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.5 + 0.5 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.25);

    fragColor = vec4(color, 1.0);
}
"""

def seed_field(width_px: int, height_px: int, tile_size: int) -> np.ndarray:
    tile_size = max(2, int(tile_size))
    tiles_x = max(1, int(np.ceil(width_px / tile_size)))
    tiles_y = max(1, int(np.ceil(height_px / tile_size)))

    rng = np.random.default_rng(7777)
    tile_y, tile_x = np.meshgrid(
        np.arange(tiles_y, dtype=np.float32),
        np.arange(tiles_x, dtype=np.float32),
        indexing="ij",
    )
    x_norm = tile_x / max(tiles_x - 1, 1)
    y_norm = tile_y / max(tiles_y - 1, 1)

    # Base Macro-Geology (Minecraft-style continents and oceans)
    height = (
        0.35
        + 0.25 * np.sin(x_norm * 4.1 + 0.5) * np.cos(y_norm * 3.7)
        + 0.15 * np.sin((x_norm + y_norm) * 12.0)
        + 0.10 * np.cos((x_norm - y_norm) * 19.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.02
    )

    # Inject Blocky Mountains / Plateaus
    mountain_count = max(6, (tiles_x * tiles_y) // 4000)
    for _ in range(mountain_count):
        cx = rng.uniform(0.0, tiles_x)
        cy = rng.uniform(0.0, tiles_y)
        rx = rng.uniform(max(10.0, tiles_x * 0.05), max(25.0, tiles_x * 0.15))
        ry = rng.uniform(max(10.0, tiles_y * 0.05), max(25.0, tiles_y * 0.15))
        distance = ((tile_x - cx) / rx) ** 2 + ((tile_y - cy) / ry) ** 2
        lift = np.clip(1.0 - distance, 0.0, 1.0)
        height += lift * rng.uniform(0.1, 0.4)

    height = np.clip(height, 0.0, 1.0)

    sea_level = 0.30
    ocean = (height < sea_level).astype(np.float32)
    coast = np.clip(1.0 - np.abs(height - sea_level) / 0.05, 0.0, 1.0)

    # R: Nutrients / Water saturation
    nutrients = np.clip(
        0.2 
        + ocean * 0.6 
        + coast * 0.3
        + 0.1 * np.sin(x_norm * 14.0 + y_norm * 9.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.05,
        0.0,
        1.0,
    )

    # G: Initial Biomass / Forests
    biomass = np.clip(
        (1.0 - ocean)
        * (
            0.1
            + coast * 0.5
            + nutrients * 0.4
            - np.clip(height - 0.6, 0.0, 1.0) * 1.5
        ),
        0.0,
        1.0,
    )

    # A: Magma Vents & Alien Tech corruption
    magma = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    magma_mask = (ocean < 0.5) & (height > 0.5)
    candidates = np.argwhere(magma_mask)
    if len(candidates) > 0:
        vent_count = min(max(5, (tiles_x * tiles_y) // 2500), len(candidates))
        vent_indices = rng.choice(len(candidates), size=vent_count, replace=False)
        for idx in vent_indices:
            cy, cx = candidates[idx]
            radius = int(rng.integers(3, 8))
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
            magma[y0:y1, x0:x1] = np.maximum(
                magma[y0:y1, x0:x1],
                influence * rng.uniform(0.7, 1.0),
            )
            # Clear biomass near magma
            biomass[y0:y1, x0:x1] *= (1.0 - influence)

    tile_field = np.stack(
        [
            nutrients.astype(np.float32),
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
    id='minecraft-first-future-3d',
    display_name='Minecraft First Future',
    window_title='Garage Life Lab - Minecraft First Future',
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={
        'feed': 0.038,
        'kill': 0.060,
        'diff_u': 0.16,
        'diff_v': 0.08,
        'time_step': 0.35,
        'ray_steps': 160,    # Deep DDA traversal for heavy stress
        'fx_intensity': 1.1,
        'camera_speed': 0.8,
        'contour_contrast': 0.85,
    },
    preview_image='assets/world_previews/minecraft-long-term-3d.png',
    stability_notes=('extreme gpu load', 'branchless dda', 'voxel gi'),
    hud_subtitle='VOXEL ENGINE STRESS TEST',
    preview_palette=('#1c1c1c', '#3b2513', '#418231', '#184b96', '#cfbc70', '#fc6203', '#2cdb8c'),
    uses_audio=True,
)
