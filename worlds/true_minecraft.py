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

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

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

    // Compute standard laplacian for R (Lava/Heat) and G (Biomass)
    float lapR = (r.r + l.r + t.r + b.r) * 0.2 + (tr.r + tl.r + br.r + bl.r) * 0.05 - c.r;
    float lapG = (r.g + l.g + t.g + b.g) * 0.2 + (tr.g + tl.g + br.g + bl.g) * 0.05 - c.g;

    // Terrain Gradient to direct flow
    vec2 gradH = vec2(r.b - l.b, t.b - b.b);
    
    // Advect heat downhill, advect biomass uphill slightly
    float advectR = dot(gradH, vec2(r.r - l.r, t.r - b.r)) * 0.9;
    float advectG = dot(-gradH, vec2(r.g - l.g, t.g - b.g)) * 0.2;

    float wetNoise = (hash(uv * resolution + floor(time * 0.2)) - 0.5) * noiseStrength;

    // Biome dynamics (Reaction-Diffusion)
    float localFeed = feed + c.a * 0.02 * sin(time * 0.05 + uv.x * 10.0) + wetNoise * 0.1;
    float localKill = kill + (1.0 - c.a) * 0.015 * cos(time * 0.08 + uv.y * 15.0) + parameterDrift;

    // Biomass consumes heat/nutrients
    float reaction = c.r * c.g * c.g;

    float dr = (diffU * lapR * laplaceScale) - reaction + localFeed * (1.0 - c.r) - advectR;
    float dg = (diffV * lapG * laplaceScale) + reaction - (localFeed + localKill) * c.g - advectG;

    // Active voxel terrain height manipulation
    // Biomass stabilizes and grows land, Heat melts and erodes land
    float dh = (c.g * 0.015 - c.r * 0.005 + wetNoise * 0.01) * dt * c.a;

    fragColor = vec4(
        clamp(c.r + dr * dt, 0.0, 1.0),
        clamp(c.g + dg * dt, 0.0, 1.0),
        clamp(c.b + dh * dt, 0.0, 1.0),
        c.a
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
    return fract(sin(dot(p, vec2(41.7, 289.1))) * 45758.5453);
}

float noise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    vec2 uv0 = (i.xy + vec2(37.0, 17.0) * i.z) + f.xy;
    vec2 uv1 = (i.xy + vec2(37.0, 17.0) * (i.z + 1.0)) + f.xy;
    float n0 = mix(hash(i.xy), hash(i.xy + vec2(1.0, 0.0)), f.x);
    // Simple 3D hash substitute for performance
    vec3 p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
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

// Map real voxel space to block IDs
int getBlock(vec3 pos, out vec4 stateOut) {
    if (pos.y < 0.0) return 1; // Unbreakable Bedrock
    if (pos.y > 100.0) return 0; // Sky Limit

    vec2 mapUV = pos.xz * 0.003;
    vec4 s = textureLod(stateTex, mapUV, 0.0);
    stateOut = s;

    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float baseH = floor(s.b * 45.0 + 10.0);

    // Procedural Caves
    if (pos.y < baseH && pos.y > 5.0) {
        vec3 cp = pos * 0.08;
        float n = sin(cp.x)*cos(cp.y)*sin(cp.z) + sin(cp.x*2.3)*cos(cp.z*2.1)*0.5;
        if (n > 0.45) return 0; // Air (Cave)
    }

    if (pos.y <= baseH) {
        if (pos.y < baseH - floor(s.r * 5.0 + 3.0)) return 1; // Stone
        if (pos.y <= 13.0 && baseH <= 15.0) return 5; // Sand
        if (s.r > 0.6 && pos.y < 8.0) return 6; // Lava in deep caves
        if (s.g > 0.45) return 3; // Grass
        return 2; // Dirt
    }

    // Water Level
    if (pos.y <= 12.0) return 4;

    // Organic Biome / Trees
    if (s.g > 0.5) {
        vec2 cell = floor(pos.xz / 6.0);
        if (hash(cell) > 0.7) {
            vec2 center = cell * 6.0 + vec2(3.0);
            float d = max(abs(pos.x - center.x), abs(pos.z - center.y));
            float treeH = baseH + floor(hash(cell + 1.0) * 5.0) + 4.0;
            
            if (d < 1.0 && pos.y <= treeH) return 8; // Wood Trunk
            if (d < 3.0 && pos.y > treeH - 3.0 && pos.y <= treeH + 1.0) return 7; // Leaves
        }
    }

    // Alien Monoliths / Tech structures
    if (s.a > 0.6) {
        vec2 cell = floor(pos.xz / 16.0);
        if (hash(cell + 8.0) > 0.92) {
            vec2 center = cell * 16.0 + vec2(8.0);
            float d = max(abs(pos.x - center.x), abs(pos.z - center.y));
            if (d < 2.0 && pos.y <= baseH + 18.0) return 9; // Crystal / Tech
        }
    }

    return 0; // Air
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    int safeRaySteps = clamp(raySteps, 32, 256);

    // Smooth cinematic camera
    float camTime = time * 0.15 * max(cameraSpeed, 0.05);
    vec3 ro = vec3(camTime * 12.0, 60.0 + sin(camTime * 0.5) * 8.0, camTime * 10.0);
    
    // Prevent camera from clipping underground
    float camH = textureLod(stateTex, ro.xz * 0.003, 0.0).b * 45.0 + 10.0;
    ro.y = max(ro.y, camH + 6.0);
    
    vec3 ta = vec3(ro.x + 8.0, ro.y - 4.0, ro.z + 8.0 + sin(camTime));
    ro += cameraOffset;
    ta += cameraOffset;

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.2) * 0.05);
    vec3 rd = ca * cameraInputRay(p, 2.0);

    // Voxel DDA Setup
    vec3 mapPos = floor(ro);
    vec3 rayStep = sign(rd + 1e-16);
    vec3 deltaDist = 1.0 / abs(rd + 1e-16);
    vec3 sideDist = (rayStep * (mapPos - ro) + (rayStep * 0.5 + 0.5)) * deltaDist;
    vec3 mask = vec3(0.0);

    int hitBlock = 0;
    vec4 hitState = vec4(0.0);
    float waterDepth = 0.0;
    float leafDepth = 0.0;
    
    for (int i = 0; i < 256; i++) {
        if (i >= safeRaySteps) break;
        
        vec4 state;
        int block = getBlock(mapPos, state);
        
        if (block == 4) { // Water
            waterDepth += 1.0;
        } else if (block == 7) { // Leaves
            leafDepth += 1.0;
        } else if (block > 0) {
            hitBlock = block;
            hitState = state;
            break;
        }
        
        mask = step(sideDist.xyz, sideDist.yzx) * step(sideDist.xyz, sideDist.zxy);
        sideDist += mask * deltaDist;
        mapPos += mask * rayStep;
    }

    vec3 lightDir = normalize(vec3(0.6, 0.8, -0.4));
    
    // Skybox with Auroras
    float skyRise = smoothstep(-0.2, 0.8, rd.y);
    vec3 skyColor = mix(vec3(0.02, 0.04, 0.08), vec3(0.1, 0.25, 0.45), skyRise);
    float sun = pow(max(0.0, dot(rd, lightDir)), 300.0);
    skyColor += vec3(1.0, 0.8, 0.5) * sun * safeFx;
    
    float aurora = smoothstep(0.6, 0.98, rd.y + 0.15 * sin(p.x * 4.0 + time * 0.3));
    aurora *= 0.5 + 0.5 * sin(p.x * 12.0 + time * 0.8 + colorShift);
    skyColor += aurora * vec3(0.1, 0.8, 0.5) * safeFx * (1.0 + audioTreble * 2.0);

    vec3 color = skyColor;

    if (hitBlock > 0) {
        vec3 tVec = (mapPos - ro + (1.0 - rayStep) * 0.5) / (rd + 1e-16);
        float t = dot(tVec, mask);
        vec3 pos = ro + rd * t;
        vec3 nor = -rayStep * mask;
        
        // Local face UV for texturing & AO
        vec3 uvw = fract(pos);
        vec2 faceUv;
        if (mask.x > 0.5) faceUv = uvw.yz;
        else if (mask.y > 0.5) faceUv = uvw.xz;
        else faceUv = uvw.xy;
        
        // Minecraft-style Block AO (Bevel Darkening)
        float borderDist = min(min(faceUv.x, 1.0 - faceUv.x), min(faceUv.y, 1.0 - faceUv.y));
        float blockAO = 0.5 + 0.5 * smoothstep(0.0, 0.15, borderDist);

        // Texturing
        vec3 albedo;
        float emissive = 0.0;
        float hsh = hash(mapPos.xz);
        
        if (hitBlock == 1) { // Stone
            albedo = vec3(0.35, 0.35, 0.4) * (0.8 + 0.2 * hsh);
        } else if (hitBlock == 2) { // Dirt
            albedo = vec3(0.4, 0.25, 0.15) * (0.8 + 0.2 * hsh);
        } else if (hitBlock == 3) { // Grass
            if (nor.y > 0.5) {
                albedo = vec3(0.25, 0.65, 0.2) * (0.8 + 0.2 * hsh);
            } else {
                albedo = vec3(0.4, 0.25, 0.15) * (0.8 + 0.2 * hsh);
                if (faceUv.y > 0.65 + hsh * 0.2) albedo = vec3(0.25, 0.65, 0.2); // Grass overlap
            }
        } else if (hitBlock == 5) { // Sand
            albedo = vec3(0.8, 0.75, 0.5) * (0.9 + 0.1 * hsh);
        } else if (hitBlock == 6) { // Lava
            albedo = mix(vec3(1.0, 0.2, 0.0), vec3(1.0, 0.8, 0.0), hsh);
            emissive = 1.5;
        } else if (hitBlock == 8) { // Wood
            if (abs(nor.y) > 0.5) albedo = vec3(0.5, 0.35, 0.15);
            else albedo = vec3(0.25, 0.15, 0.05);
        } else if (hitBlock == 9) { // Crystal Tech
            albedo = mix(vec3(0.0, 1.0, 0.8), vec3(1.0, 0.2, 0.8), hitState.r);
            emissive = 2.0 + audioBass * 3.0; // Audio reactive pulse
        }

        // Fast Voxel Shadows
        vec3 mapPosS = floor(pos + nor * 0.01);
        vec3 rdS = lightDir;
        vec3 rayStepS = sign(rdS + 1e-16);
        vec3 deltaDistS = 1.0 / abs(rdS + 1e-16);
        vec3 sideDistS = (rayStepS * (mapPosS - (pos + nor * 0.01)) + (rayStepS * 0.5 + 0.5)) * deltaDistS;
        vec3 maskS = vec3(0.0);
        float shadow = 1.0;
        
        for (int j = 0; j < 32; j++) {
            vec4 dummy;
            int b = getBlock(mapPosS, dummy);
            if (b > 0 && b != 4 && b != 7) {
                shadow = 0.15; // Hard shadow hit
                break;
            }
            maskS = step(sideDistS.xyz, sideDistS.yzx) * step(sideDistS.xyz, sideDistS.zxy);
            sideDistS += maskS * deltaDistS;
            mapPosS += maskS * rayStepS;
        }

        // Lighting Evaluation
        float diff = max(dot(nor, lightDir), 0.0);
        float sky = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
        
        vec3 lin = vec3(0.0);
        lin += 2.0 * diff * vec3(1.0, 0.9, 0.8) * shadow;
        lin += 0.6 * sky * vec3(0.2, 0.3, 0.5) * blockAO;
        lin += emissive * albedo * (1.0 + glow * 0.5) * safeFx;
        
        color = albedo * lin;
        
        // Distance Fog
        float fog = 1.0 - exp(-0.0005 * t * t);
        color = mix(color, skyColor, fog);
    }

    // Accumulate Volumetric Water and Leaves over the ray
    if (leafDepth > 0.0) {
        vec3 leafColor = vec3(0.15, 0.45, 0.2) * (1.0 + glow * 0.2 * hitState.g);
        color = mix(color, leafColor, 1.0 - exp(-leafDepth * 0.6));
    }
    if (waterDepth > 0.0) {
        vec3 waterColor = vec3(0.05, 0.25, 0.6) * safeFx;
        color = mix(color, waterColor, 1.0 - exp(-waterDepth * 0.3));
        // Add surface reflections to water conceptually
        color += vec3(0.1, 0.2, 0.3) * waterDepth * 0.05;
    }

    // ACES Tonemapping
    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(color, vec3(1.0 / max(gamma, 0.2)));

    // Stylized contour edge darkening
    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.5 + 0.5 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.25);

    fragColor = vec4(color, 1.0);
}
"""

def seed_field(width_px: int, height_px: int, tile_size: int) -> np.ndarray:
    tile_size = max(2, int(tile_size))
    tiles_x = max(1, int(np.ceil(width_px / tile_size)))
    tiles_y = max(1, int(np.ceil(height_px / tile_size)))

    rng = np.random.default_rng(4096)
    tile_y, tile_x = np.meshgrid(
        np.arange(tiles_y, dtype=np.float32),
        np.arange(tiles_x, dtype=np.float32),
        indexing="ij",
    )
    x_norm = tile_x / max(tiles_x - 1, 1)
    y_norm = tile_y / max(tiles_y - 1, 1)

    # Base Minecraft-style macro-terrain
    height = (
        0.35
        + 0.25 * np.sin(x_norm * 5.3 + 0.9)
        + 0.20 * np.cos(y_norm * 4.7 - 1.2)
        + 0.10 * np.sin((x_norm + y_norm) * 15.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.03
    )

    # Add blocky continents
    continent_count = max(8, (tiles_x * tiles_y) // 3000)
    for _ in range(continent_count):
        cx = rng.uniform(0.0, tiles_x)
        cy = rng.uniform(0.0, tiles_y)
        rx = rng.uniform(max(8.0, tiles_x * 0.05), max(20.0, tiles_x * 0.2))
        ry = rng.uniform(max(8.0, tiles_y * 0.05), max(20.0, tiles_y * 0.2))
        distance = ((tile_x - cx) / rx) ** 2 + ((tile_y - cy) / ry) ** 2
        lift = np.clip(1.0 - distance, 0.0, 1.0)
        height += lift * rng.uniform(0.15, 0.35)

    height = np.clip(height, 0.0, 1.0)

    # Biome distribution
    sea_level = 0.4
    ocean = (height < sea_level).astype(np.float32)
    coast = np.clip(1.0 - np.abs(height - sea_level) / 0.05, 0.0, 1.0)
    
    # Heat map (R) - Lava and deserts
    heat = np.clip(
        0.3 
        - ocean * 0.5
        + 0.2 * np.sin(x_norm * 8.0 - y_norm * 5.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.05,
        0.0,
        1.0,
    )

    # Biomass map (G) - Forests and Grass
    biomass = np.clip(
        (1.0 - ocean)
        * (
            0.1
            + coast * 0.3
            + (1.0 - heat) * 0.6
            - np.clip(height - 0.7, 0.0, 1.0) * 0.8
        ),
        0.0,
        1.0,
    )

    # Special Monoliths / Tech (A)
    settlement = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    candidate_mask = (ocean < 0.5) & (biomass > 0.4) & (height < 0.8)
    candidates = np.argwhere(candidate_mask)
    if len(candidates) > 0:
        city_count = min(max(10, (tiles_x * tiles_y) // 1500), len(candidates))
        city_indices = rng.choice(len(candidates), size=city_count, replace=False)
        for candidate_index in city_indices:
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
                influence * rng.uniform(0.6, 1.0),
            )

    tile_field = np.stack(
        [
            heat.astype(np.float32),
            biomass.astype(np.float32),
            height.astype(np.float32),
            settlement.astype(np.float32),
        ],
        axis=-1,
    )

    field = np.repeat(np.repeat(tile_field, tile_size, axis=0), tile_size, axis=1)
    field = field[:height_px, :width_px].copy()
    return field

SPEC = WorldSpec(
    id='minecraft-voxel-stress-3d',
    display_name='Minecraft Voxel Stress',
    window_title='Garage Life Lab - Minecraft Raytraced',
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={
        'feed': 0.045,
        'kill': 0.062,
        'diff_u': 0.14,
        'diff_v': 0.07,
        'ray_steps': 120,
        'fx_intensity': 1.1,
    },
    preview_image='assets/world_previews/minecraft-long-term-3d.png',
    stability_notes=('safe', 'heavy raymarching', 'voxel optimized'),
    hud_subtitle='VOXEL ENGINE STRESS TEST',
    preview_palette=('#383838', '#5e3a21', '#418231', '#184b96', '#cfbc70', '#e35112', '#2cdb8c'),
    uses_audio=True,
)
