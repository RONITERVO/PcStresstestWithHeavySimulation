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
// R: Nutrients / Moisture (Fuels biomass, pools in lowlands)
// G: Biomass / Forests (Grows by consuming R, stabilizes land)
// B: Terrain Height (Eroded by R, built by A, bounded safely)
// A: Magma / Corruption (Destroys G, cools into terrain)
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

    // Standard Laplacian (Stable Diffusion)
    float lapR = (r.r + l.r + t.r + b.r) * 0.2 + (tr.r + tl.r + br.r + bl.r) * 0.05 - c.r;
    float lapG = (r.g + l.g + t.g + b.g) * 0.2 + (tr.g + tl.g + br.g + bl.g) * 0.05 - c.g;
    float lapA = (r.a + l.a + t.a + b.a) * 0.2 + (tr.a + tl.a + br.a + bl.a) * 0.05 - c.a;

    float wetNoise = (hash(uv * resolution + floor(time * 0.2)) - 0.5) * noiseStrength;

    // Biome-driven Reaction-Diffusion (Absolutely Stable)
    // Moisture (R) collects more naturally near sea level (c.b = 0.3)
    float seaProximity = smoothstep(0.6, 0.3, c.b);
    float localFeed = feed + seaProximity * 0.015 + wetNoise * 0.01;
    
    // Biomass (G) dies faster at high altitudes or near magma
    float altitudeHarshness = smoothstep(0.5, 0.9, c.b) * 0.02;
    float localKill = kill + altitudeHarshness + (c.a * 0.08) + parameterDrift;

    float reaction = c.r * c.g * c.g;
    
    float dr = (diffU * lapR * laplaceScale) - reaction + localFeed * (1.0 - c.r);
    float dg = (diffV * lapG * laplaceScale) + reaction - (localFeed + localKill) * c.g;
    
    // Magma (A) diffuses slightly, cools over time, pulses with audio
    float da = (diffU * 0.5 * lapA * laplaceScale) - (c.a * 0.01) + (audioBass * 0.02 * c.a);

    // Controlled Geology Integration (No runaway flooding or spikes)
    // Terrain naturally wants to settle, Magma builds it up, Moisture slightly erodes it.
    float buildup = c.a * 0.01 * smoothstep(0.8, 0.4, c.b); // Cannot build past 0.8
    float erosion = c.r * 0.005 * smoothstep(0.35, 0.5, c.b) * (1.0 - c.g); // Only erodes above sea level, roots (G) stop erosion
    
    float db = (buildup - erosion) * dt;

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

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453123);
}

float hash3(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

// 3D Noise for massive, GPU-melting cave systems & clouds
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

// Global Sea Level Map
const float SEA_LEVEL = 32.0;

// Voxel Engine
// 0:Air, 1:Bedrock, 2:Stone, 3:Dirt, 4:Grass, 5:Sand, 6:Water, 7:Lava, 8:Wood, 9:Leaves, 10:Tech, 11:Clouds
int getBlock(vec3 p, out vec4 stateOut) {
    if (p.y < 0.0) return 1;
    if (p.y > 140.0) return 0;
    
    // Minecraft Clouds (Volumetric Stress)
    if (p.y >= 110.0 && p.y <= 114.0) {
        float cloudNoise = noise3D(p * 0.03 + vec3(time * 0.5, 0.0, time * 0.2));
        if (cloudNoise > 0.65) return 11;
    }

    vec2 mapUV = (floor(p.xz) + 0.5) * 0.003;
    vec4 s = textureLod(stateTex, mapUV, 0.0);
    stateOut = s;

    // Simulation mapping: 0.0 to 1.0 -> 10 to 90 blocks
    float baseH = floor(s.b * 80.0 + 10.0);

    // Procedural Caves
    if (p.y < baseH && p.y > 5.0) {
        float caveNoise = noise3D(p * 0.06) + noise3D(p * 0.12) * 0.5;
        if (caveNoise > 0.8) return 0; // Cave Air
    }

    // Solid Terrain
    if (p.y <= baseH) {
        if (p.y < baseH - floor(s.r * 5.0 + 3.0)) return 2; // Stone
        
        // Desert Biome
        if (s.r > 0.65 && s.g < 0.2) return 5; 
        
        // Magma Biome
        if (s.a > 0.4 && p.y < baseH) return 7; 
        
        // Beaches
        if (baseH <= SEA_LEVEL + 2.0) return 5; 
        
        if (p.y == baseH) return 4; // Grass
        return 3; // Dirt
    }

    // Oceans / Lakes
    if (p.y <= SEA_LEVEL) return 6; 

    // Forests / Biomass
    if (s.g > 0.4 && p.y > baseH && p.y < baseH + 18.0) {
        vec2 cell = floor(p.xz / 6.0);
        if (hash(cell * 17.3) > 0.6) {
            vec2 center = cell * 6.0 + vec2(3.0);
            float dist = max(abs(p.x - center.x), abs(p.z - center.y));
            float treeH = baseH + floor(hash(cell) * 5.0) + 5.0;
            if (dist < 1.0 && p.y <= treeH) return 8; // Trunk
            if (dist < 3.0 && p.y >= treeH - 3.0 && p.y <= treeH + 1.0) {
                if (hash3(p) < 0.75) return 9; // Leaves
            }
        }
    }

    // Alien Obelisks
    if (s.a > 0.6) {
        vec2 cell = floor(p.xz / 16.0);
        if (hash(cell * 9.1) > 0.8) {
            vec2 center = cell * 16.0 + vec2(8.0);
            float dist = max(abs(p.x - center.x), abs(p.z - center.y));
            float monoH = baseH + floor(s.a * 40.0);
            if (dist <= 2.0 && p.y <= monoH) {
                if (dist < 1.0 && p.y > baseH + 3.0 && p.y < monoH - 3.0) return 0; // Hollow Core
                return 10;
            }
        }
    }

    return 0; // Air
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    int maxSteps = clamp(raySteps, 64, 400);

    // Smooth Cinematic Camera
    float camTime = time * 0.2 * max(cameraSpeed, 0.05);
    vec3 ro = vec3(camTime * 14.0, 0.0, camTime * 11.0);
    
    // Ensure camera stays above the highest point (terrain or water)
    float localTerrainH = textureLod(stateTex, (floor(ro.xz) + 0.5) * 0.003, 0.0).b * 80.0 + 10.0;
    ro.y = max(max(localTerrainH, SEA_LEVEL) + 20.0 + sin(camTime * 0.8) * 5.0, SEA_LEVEL + 5.0);
    
    vec3 ta = vec3(ro.x + 12.0, ro.y - 12.0, ro.z + 12.0 + sin(camTime * 1.3) * 6.0);
    ro += cameraOffset;
    ta += cameraOffset;

    float zoomedLens = 1.8 * clamp(exp(cameraZoom), 0.35, 3.0);
    vec3 rd = normalize(vec3(p.xy, zoomedLens));
    rd.yz = rot(cameraYawPitch.y) * rd.yz;
    rd.xz = rot(cameraYawPitch.x) * rd.xz;
    mat3 ca = setCamera(ro, ta, sin(camTime * 0.25) * 0.05);
    rd = ca * rd;

    // Fast Branchless DDA Setup
    vec3 mapPos = floor(ro);
    vec3 rayStep = sign(rd);
    vec3 deltaDist = 1.0 / max(abs(rd), 1e-8);
    vec3 sideDist = (rayStep * (mapPos - ro) + (rayStep * 0.5 + 0.5)) * deltaDist;
    vec3 mask = vec3(0.0);

    int hitBlock = 0;
    vec4 hitState = vec4(0.0);
    
    // Transparency & Volume Accumulation
    vec3 throughput = vec3(1.0);
    vec3 transpColor = vec3(0.0);
    float godRays = 0.0;

    for (int i = 0; i < 400; i++) {
        if (i >= maxSteps) break;

        int b = getBlock(mapPos, hitState);
        
        if (b > 0) {
            if (b == 6) { // Water
                throughput *= exp(-vec3(0.8, 0.25, 0.05) * 0.6);
                transpColor += throughput * vec3(0.02, 0.15, 0.4) * 0.4;
            } else if (b == 9) { // Leaves
                throughput *= exp(-vec3(0.15, 0.5, 0.15) * 1.2);
                transpColor += throughput * vec3(0.1, 0.35, 0.1) * 0.3;
            } else {
                hitBlock = b;
                break;
            }
        } else {
            // God Ray Volume Tracing (Checks if currently below local terrain shadow)
            float shadowH = hitState.b * 80.0 + 10.0;
            if (mapPos.y > shadowH) {
                godRays += 0.015;
            }
        }

        if (length(throughput) < 0.05) break;

        mask = step(sideDist.xyz, sideDist.yzx) * step(sideDist.xyz, sideDist.zxy);
        sideDist += mask * deltaDist;
        mapPos += mask * rayStep;
    }

    // Illumination Setup
    vec3 lightDir = normalize(vec3(0.7, 0.85, -0.5));
    
    // Rich Sky Gradient
    float skyRise = smoothstep(-0.1, 0.5, rd.y);
    vec3 skyColor = mix(vec3(0.4, 0.6, 0.9), vec3(0.1, 0.3, 0.7), skyRise);
    
    // Sun Halo
    float sun = pow(max(0.0, dot(rd, lightDir)), 800.0);
    skyColor += vec3(1.0, 0.85, 0.6) * sun * safeFx * 1.5;

    vec3 color = skyColor;

    if (hitBlock > 0) {
        // Precise Intersection Point
        vec3 tVec = (mapPos - ro + (1.0 - rayStep) * 0.5) / max(abs(rd), 1e-8);
        float t = dot(tVec, mask);
        vec3 pos = ro + rd * t;
        vec3 nor = -rayStep * mask;

        // Voxel Face UV mapping
        vec3 uvw = fract(pos);
        vec2 faceUv;
        if (mask.x > 0.5) faceUv = uvw.yz;
        else if (mask.y > 0.5) faceUv = uvw.xz;
        else faceUv = uvw.xy;

        // Pixel-Art Noise Texturing (16x16 grid per block)
        vec2 texUv = floor(faceUv * 16.0) / 16.0;
        float texNoise = hash(floor(mapPos.xz)*13.7 + floor(mapPos.y)*19.1 + texUv * 37.4);
        
        // Voxel Edge Ambient Occlusion
        float edgeDist = min(min(faceUv.x, 1.0 - faceUv.x), min(faceUv.y, 1.0 - faceUv.y));
        float blockAO = smoothstep(0.0, 0.15, edgeDist) * 0.4 + 0.6;
        blockAO = mix(1.0, blockAO, contourContrast);

        // Materials
        vec3 albedo = vec3(1.0);
        vec3 emission = vec3(0.0);
        
        if (hitBlock == 1) albedo = vec3(0.1) + texNoise * 0.1;
        else if (hitBlock == 2) albedo = vec3(0.45) + texNoise * 0.1;
        else if (hitBlock == 3) albedo = vec3(0.47, 0.33, 0.22) + texNoise * 0.08;
        else if (hitBlock == 4) {
            if (nor.y > 0.5) albedo = vec3(0.29, 0.64, 0.29) + texNoise * 0.12;
            else if (faceUv.y > 0.7 + texNoise * 0.2) albedo = vec3(0.29, 0.64, 0.29) + texNoise * 0.12;
            else albedo = vec3(0.47, 0.33, 0.22) + texNoise * 0.08;
        }
        else if (hitBlock == 5) albedo = vec3(0.84, 0.81, 0.62) + texNoise * 0.05;
        else if (hitBlock == 7) {
            albedo = mix(vec3(1.0, 0.2, 0.0), vec3(1.0, 0.7, 0.0), texNoise);
            emission = albedo * (2.5 + audioBass * 3.0);
        }
        else if (hitBlock == 8) {
            if (abs(nor.y) > 0.5) albedo = vec3(0.55, 0.42, 0.25) + texNoise * 0.1;
            else albedo = vec3(0.36, 0.29, 0.20) + texNoise * 0.1;
        }
        else if (hitBlock == 10) {
            albedo = mix(vec3(0.1, 0.8, 1.0), vec3(0.8, 0.1, 1.0), hitState.a);
            float circuit = step(0.8, hash(texUv + time * 0.05));
            emission = albedo * circuit * (3.0 + audioTreble * 4.0) * safeFx;
            albedo += texNoise * 0.15;
        }
        else if (hitBlock == 11) { // Clouds
            albedo = vec3(0.95, 0.98, 1.0);
        }

        // Fast Secondary DDA for Hard Shadows
        vec3 mapPosS = floor(pos + nor * 0.002);
        vec3 rdS = lightDir;
        vec3 rayStepS = sign(rdS);
        vec3 deltaDistS = 1.0 / max(abs(rdS), 1e-8);
        vec3 sideDistS = (rayStepS * (mapPosS - (pos + nor * 0.002)) + (rayStepS * 0.5 + 0.5)) * deltaDistS;
        vec3 maskS = vec3(0.0);
        float shadow = 1.0;
        
        for (int j = 0; j < 45; j++) {
            vec4 dummy;
            int bS = getBlock(mapPosS, dummy);
            if (bS > 0 && bS != 6 && bS != 9) {
                shadow = (bS == 11) ? 0.4 : 0.15; // Clouds cast softer shadows
                break;
            }
            maskS = step(sideDistS.xyz, sideDistS.yzx) * step(sideDistS.xyz, sideDistS.zxy);
            sideDistS += maskS * deltaDistS;
            mapPosS += maskS * rayStepS;
        }

        // Lighting Model (Balanced to avoid ACES whiteout)
        float diff = max(dot(nor, lightDir), 0.0);
        float sky = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
        
        vec3 lin = vec3(0.0);
        lin += 1.4 * diff * vec3(1.0, 0.95, 0.85) * shadow; // Direct Sun
        lin += 0.5 * sky * vec3(0.4, 0.55, 0.7);            // Ambient Sky Light
        
        color = albedo * lin * blockAO + emission * (1.0 + glow * 0.5);

        // Distance Atmospheric Fog
        float fog = 1.0 - exp(-0.00008 * t * t);
        color = mix(color, skyColor, fog);
    }

    // Blend Volumes (Water, Leaves) and God Rays
    color = color * throughput + transpColor;
    
    vec3 godRayColor = vec3(1.0, 0.9, 0.75) * min(godRays * safeFx, 1.0) * (0.3 + 0.7 * max(dot(rd, lightDir), 0.0));
    color += godRayColor;

    // Premium Tonemapping (ACES) & Gamma
    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(color, vec3(1.0 / max(gamma, 0.2)));

    // Stylized Vignette
    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.5 + 0.5 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.2);

    fragColor = vec4(color, 1.0);
}
"""

def seed_field(width_px: int, height_px: int, tile_size: int) -> np.ndarray:
    tile_size = max(2, int(tile_size))
    tiles_x = max(1, int(np.ceil(width_px / tile_size)))
    tiles_y = max(1, int(np.ceil(height_px / tile_size)))

    rng = np.random.default_rng(2048)
    tile_y, tile_x = np.meshgrid(
        np.arange(tiles_y, dtype=np.float32),
        np.arange(tiles_x, dtype=np.float32),
        indexing="ij",
    )
    x_norm = tile_x / max(tiles_x - 1, 1)
    y_norm = tile_y / max(tiles_y - 1, 1)

    # Classic Minecraft Macro-Terrain Generation
    height = (
        0.30
        + 0.25 * np.sin(x_norm * 5.1 + 1.5) * np.cos(y_norm * 4.7)
        + 0.15 * np.sin((x_norm + y_norm) * 14.0)
        + 0.10 * np.cos((x_norm - y_norm) * 21.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.02
    )

    # Continent/Mountain Plateaus
    mountain_count = max(8, (tiles_x * tiles_y) // 2500)
    for _ in range(mountain_count):
        cx = rng.uniform(0.0, tiles_x)
        cy = rng.uniform(0.0, tiles_y)
        rx = rng.uniform(max(10.0, tiles_x * 0.05), max(30.0, tiles_x * 0.18))
        ry = rng.uniform(max(10.0, tiles_y * 0.05), max(30.0, tiles_y * 0.18))
        distance = ((tile_x - cx) / rx) ** 2 + ((tile_y - cy) / ry) ** 2
        lift = np.clip(1.0 - distance, 0.0, 1.0)
        height += lift * rng.uniform(0.1, 0.5)

    height = np.clip(height, 0.0, 1.0)

    # Sea Level reference in 0..1 scale (32 / 80 approx = 0.4)
    sea_level = 0.35
    ocean = (height < sea_level).astype(np.float32)
    coast = np.clip(1.0 - np.abs(height - sea_level) / 0.05, 0.0, 1.0)

    # R: Moisture (Dictates Forests vs Deserts)
    moisture = np.clip(
        0.2 
        + ocean * 0.8 
        + coast * 0.4
        + 0.15 * np.sin(x_norm * 11.0 + y_norm * 8.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.05,
        0.0,
        1.0,
    )

    # G: Biomass / Forests
    biomass = np.clip(
        (1.0 - ocean)
        * (
            0.1
            + coast * 0.3
            + moisture * 0.5
            - np.clip(height - 0.7, 0.0, 1.0) * 1.5
        ),
        0.0,
        1.0,
    )

    # A: Magma / Obelisk Tech
    magma = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    magma_mask = (ocean < 0.5) & (height > 0.5)
    candidates = np.argwhere(magma_mask)
    if len(candidates) > 0:
        vent_count = min(max(5, (tiles_x * tiles_y) // 2500), len(candidates))
        vent_indices = rng.choice(len(candidates), size=vent_count, replace=False)
        for idx in vent_indices:
            cy, cx = candidates[idx]
            radius = int(rng.integers(3, 7))
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
    id='minecraft-future-term-3d',
    display_name='Minecraft Voxel Stress test',
    window_title='Garage Life Lab - Minecraft Raytraced',
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={
        'feed': 0.038,        # Stable, organic growth parameters
        'kill': 0.060,        
        'diff_u': 0.16,
        'diff_v': 0.08,
        'ray_steps': 180,     # Incredible DDA depth for immense GPU load
        'fx_intensity': 1.1,
        'camera_speed': 0.8,
        'exposure': 1.3,      # Tuned for rich colors
        'contour_contrast': 0.85,
    },
    preview_image='assets/world_previews/minecraft-long-term-3d.png',
    stability_notes=('extreme gpu load', 'branchless dda', 'voxel gi'),
    hud_subtitle='VOXEL ENGINE STRESS TEST',
    preview_palette=('#1c1c1c', '#3b2513', '#4ba34b', '#24529c', '#d7d09e', '#fc6203', '#ffffff'),
    uses_audio=True,
)