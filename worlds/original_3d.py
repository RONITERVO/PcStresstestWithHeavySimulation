"""World definition for Original 3D."""
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

// Fluid advection based on terrain topography
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

    // Compute standard laplacian
    float lapU = (r.r + l.r + t.r + b.r) * 0.2 + (tr.r + tl.r + br.r + bl.r) * 0.05 - c.r;
    float lapV = (r.g + l.g + t.g + b.g) * 0.2 + (tr.g + tl.g + br.g + bl.g) * 0.05 - c.g;
    
    // Terrain Gradient (Channel B is height, A is biome/moisture)
    vec2 gradH = vec2(r.b - l.b, t.b - b.b);
    
    // Compute advection vectors (chemicals flow down height gradients)
    float flowStrength = 0.8;
    float advectU = dot(gradH, vec2(r.r - l.r, t.r - b.r)) * flowStrength;
    float advectV = dot(gradH, vec2(r.g - l.g, t.g - b.g)) * flowStrength;

    // Add local chaotic drift
    float localFeed = feed + c.a * 0.015 * sin(time * 0.1 + uv.x * 20.0);
    float localKill = kill + (1.0 - c.a) * 0.01 * cos(time * 0.15 + uv.y * 15.0);
    
    // Growth based on moisture
    float reaction = c.r * c.g * c.g;
    
    // Integration
    float du = (diffU * lapU * laplaceScale) - reaction + localFeed * (1.0 - c.r) - advectU;
    float dv = (diffV * lapV * laplaceScale) + reaction - (localFeed + localKill) * c.g - advectV;
    
    // Modify terrain height organically over time based on biomass (V)
    float dh = (c.g * 0.01 - 0.002) * dt * c.a;
    
    fragColor = vec4(
        clamp(c.r + du * dt, 0.0, 1.0),
        clamp(c.g + dv * dt, 0.0, 1.0),
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

// Raymarching camera and environment setup
mat3 setCamera(in vec3 ro, in vec3 ta, float cr) {
    vec3 cw = normalize(ta - ro);
    vec3 cp = vec3(sin(cr), cos(cr), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = normalize(cross(cu, cw));
    return mat3(cu, cv, cw);
}

// Terrain SDF derived from the 2D simulation texture
float map(in vec3 p, out float matID, out vec4 stateOut) {
    // Scale world to texture UVs
    vec2 mapUV = p.xz * 0.04;
    vec4 state = textureLod(stateTex, mapUV, 0.0);
    
    // Terrain height (b) and biomass extrusion (g)
    float baseHeight = state.b * 4.0;
    float biomassExtrusion = smoothstep(0.1, 0.8, state.g) * 2.5;
    float h = baseHeight + biomassExtrusion;
    
    // Add high frequency fractal noise for detail
    float detail = sin(p.x * 8.0) * cos(p.z * 8.3) * 0.05 + sin(p.x * 15.0) * cos(p.z * 14.1) * 0.02;
    h += detail * smoothstep(0.2, 0.8, state.b);
    
    float dTerrain = p.y - h;
    
    // Water plane
    float dWater = p.y - 1.8;
    
    if (dTerrain < dWater) {
        matID = 1.0; // Terrain/Organics
        stateOut = state;
        return dTerrain * 0.6; // Under-relax to prevent missed intersections on steep heightmaps
    } else {
        matID = 0.0; // Water
        stateOut = state;
        return dWater * 0.8;
    }
}

vec3 calcNormal(in vec3 p) {
    const float eps = 0.01;
    const vec2 h = vec2(eps, 0);
    float dummyMat; vec4 dummyState;
    return normalize(vec3(
        map(p + h.xyy, dummyMat, dummyState) - map(p - h.xyy, dummyMat, dummyState),
        map(p + h.yxy, dummyMat, dummyState) - map(p - h.yxy, dummyMat, dummyState),
        map(p + h.yyx, dummyMat, dummyState) - map(p - h.yyx, dummyMat, dummyState)
    ));
}

float calcShadow(in vec3 ro, in vec3 rd) {
    float res = 1.0;
    float t = 0.1;
    float dummyMat; vec4 dummyState;
    for (int i = 0; i < 32; i++) {
        float h = map(ro + rd * t, dummyMat, dummyState);
        res = min(res, 8.0 * h / t);
        t += clamp(h, 0.05, 0.5);
        if (h < 0.001 || t > 10.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    
    // Smooth cinematic camera movement
    float camTime = time * 0.15;
    vec3 ro = vec3(camTime * 5.0, 6.0 + sin(camTime * 0.5) * 1.5, camTime * 4.0);
    vec3 ta = vec3(ro.x + 4.0, 2.0, ro.z + 4.0 + sin(camTime));
    
    mat3 ca = setCamera(ro, ta, sin(camTime * 0.3) * 0.1);
    vec3 rd = ca * normalize(vec3(p.xy, 2.0));
    
    // Environment
    vec3 lightDir = normalize(vec3(0.8, 0.6, -0.4));
    vec3 skyColor = mix(vec3(0.01, 0.02, 0.05), vec3(0.1, 0.2, 0.3), pow(max(0.0, dot(rd, lightDir)), 2.0));
    
    // Raymarching
    float tMax = 50.0;
    float t = 0.0;
    float matID = -1.0;
    vec4 state = vec4(0.0);
    vec3 glow = vec3(0.0);
    
    for (int i = 0; i < 128; i++) {
        vec3 pos = ro + rd * t;
        float currMat; vec4 currState;
        float h = map(pos, currMat, currState);
        
        // Volumetric bioluminescence accumulation (V concentration)
        if (currMat == 1.0) {
            float bio = smoothstep(0.2, 0.8, currState.g);
            vec3 emColor = mix(vec3(0.0, 1.0, 0.8), vec3(1.0, 0.2, 0.5), currState.r);
            glow += emColor * bio * (0.015 / (1.0 + abs(h) * 10.0));
        }
        
        if (abs(h) < 0.002 * t || t > tMax) {
            matID = currMat;
            state = currState;
            break;
        }
        t += h;
    }
    
    vec3 color = skyColor;
    
    if (t < tMax) {
        vec3 pos = ro + rd * t;
        vec3 nor = calcNormal(pos);
        
        // Lighting
        float occ = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
        float sha = calcShadow(pos + nor * 0.01, lightDir);
        float dif = clamp(dot(nor, lightDir), 0.0, 1.0) * sha;
        float sky = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
        float fre = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 2.0);
        
        // Materials
        vec3 matColor;
        if (matID == 1.0) {
            // Terrain & Organics
            vec3 rock = vec3(0.1, 0.12, 0.15);
            vec3 sand = vec3(0.3, 0.25, 0.18);
            vec3 bio = mix(vec3(0.05, 0.2, 0.15), vec3(0.8, 1.0, 0.9), state.g);
            
            matColor = mix(rock, sand, smoothstep(0.4, 0.6, nor.y));
            matColor = mix(matColor, bio, smoothstep(0.1, 0.5, state.g));
            
            // Bioluminescence emission on surface
            matColor += mix(vec3(0.0, 0.8, 0.6), vec3(0.9, 0.1, 0.3), state.r) * pow(state.g, 3.0) * 2.0 * (1.0 + 0.5 * sin(time * 3.0 - pos.x));
        } else {
            // Water
            float depth = clamp((1.8 - state.b * 4.0) * 0.5, 0.0, 1.0);
            vec3 shallow = vec3(0.0, 0.4, 0.5);
            vec3 deep = vec3(0.0, 0.05, 0.15);
            matColor = mix(shallow, deep, depth);
            
            // Specular
            vec3 ref = reflect(rd, nor);
            float spe = pow(clamp(dot(ref, lightDir), 0.0, 1.0), 32.0) * sha;
            matColor += vec3(1.0) * spe * 0.5;
            
            // Reflected bioluminescence
            matColor += glow * 0.2;
        }
        
        vec3 lin = vec3(0.0);
        lin += 2.0 * dif * vec3(1.0, 0.9, 0.8);
        lin += 0.5 * sky * vec3(0.2, 0.3, 0.4) * occ;
        lin += 0.2 * fre * vec3(1.0);
        
        color = matColor * lin;
        
        // Add atmospheric fog
        float fog = 1.0 - exp(-0.001 * t * t);
        color = mix(color, skyColor, fog);
    }
    
    // Add accumulated volumetric glow
    color += glow;
    
    // ACES Tonemapping & Gamma Correction
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(color, vec3(1.0 / 2.2));
    
    // Vignetting
    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.5 + 0.5 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.25);
    
    fragColor = vec4(color, 1.0);
}
"""


def seed_field(width_px: int, height_px: int, tile_size: int) -> np.ndarray:
    tile_size = max(2, int(tile_size))
    tiles_x = max(1, int(np.ceil(width_px / tile_size)))
    tiles_y = max(1, int(np.ceil(height_px / tile_size)))

    rng = np.random.default_rng(2026)
    tile_y, tile_x = np.meshgrid(
        np.arange(tiles_y, dtype=np.float32),
        np.arange(tiles_x, dtype=np.float32),
        indexing="ij",
    )
    x_norm = tile_x / max(tiles_x - 1, 1)
    y_norm = tile_y / max(tiles_y - 1, 1)

    height = (
        0.45
        + 0.15 * np.sin(x_norm * 7.3 + 0.9)
        + 0.11 * np.cos(y_norm * 5.7 - 1.2)
        + 0.08 * np.sin((x_norm + y_norm) * 11.0)
        + 0.06 * np.cos((x_norm - y_norm) * 13.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.025
    )

    continent_count = max(5, (tiles_x * tiles_y) // 4000)
    for _ in range(continent_count):
        cx = rng.uniform(0.0, tiles_x)
        cy = rng.uniform(0.0, tiles_y)
        rx = rng.uniform(max(5.0, tiles_x * 0.04), max(12.0, tiles_x * 0.18))
        ry = rng.uniform(max(5.0, tiles_y * 0.04), max(12.0, tiles_y * 0.18))
        distance = ((tile_x - cx) / rx) ** 2 + ((tile_y - cy) / ry) ** 2
        lift = np.clip(1.0 - distance, 0.0, 1.0)
        height += lift * rng.uniform(0.10, 0.24)

    trench_count = max(4, continent_count // 2)
    for _ in range(trench_count):
        cx = rng.uniform(0.0, tiles_x)
        cy = rng.uniform(0.0, tiles_y)
        rx = rng.uniform(max(6.0, tiles_x * 0.05), max(14.0, tiles_x * 0.16))
        ry = rng.uniform(max(6.0, tiles_y * 0.05), max(14.0, tiles_y * 0.16))
        distance = ((tile_x - cx) / rx) ** 2 + ((tile_y - cy) / ry) ** 2
        carve = np.clip(1.0 - distance, 0.0, 1.0)
        height -= carve * rng.uniform(0.08, 0.18)

    ridge_bands = np.sin(x_norm * 21.0 + np.cos(y_norm * 9.0) * 2.3)
    height += np.clip(ridge_bands - 0.45, 0.0, 1.0) * 0.08
    height = np.clip(height, 0.0, 1.0)

    sea_level = 0.46
    ocean = (height < sea_level).astype(np.float32)
    coast = np.clip(1.0 - np.abs(height - sea_level) / 0.07, 0.0, 1.0)
    latitude = 1.0 - np.abs(y_norm * 2.0 - 1.0)

    moisture = np.clip(
        0.16
        + ocean * 0.52
        + coast * 0.22
        + latitude * 0.12
        + 0.08 * np.sin(x_norm * 9.0 - y_norm * 6.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.03,
        0.0,
        1.0,
    )

    biomass = np.clip(
        (1.0 - ocean)
        * (
            0.06
            + moisture * 0.62
            + latitude * 0.16
            - np.clip(height - 0.72, 0.0, 1.0) * 0.50
        ),
        0.0,
        1.0,
    )

    settlement = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    candidate_mask = (
        (ocean < 0.5)
        & (coast > 0.35)
        & (biomass > 0.28)
        & (height < 0.74)
    )
    candidates = np.argwhere(candidate_mask)
    if len(candidates) > 0:
        city_count = min(max(8, (tiles_x * tiles_y) // 1800), len(candidates))
        city_indices = rng.choice(len(candidates), size=city_count, replace=False)
        for candidate_index in city_indices:
            cy, cx = candidates[candidate_index]
            radius = int(rng.integers(1, 3))
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
                influence * rng.uniform(0.35, 0.78),
            )

    tile_field = np.stack(
        [
            height.astype(np.float32),
            moisture.astype(np.float32),
            biomass.astype(np.float32),
            settlement.astype(np.float32),
        ],
        axis=-1,
    )

    field = np.repeat(np.repeat(tile_field, tile_size, axis=0), tile_size, axis=1)
    field = field[:height_px, :width_px].copy()
    return field


SPEC = WorldSpec(
    id='original-3d',
    display_name='Original 3D',
    window_title='Garage Life Lab - 3D Bio-World',
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={},
    preview_image='assets/world_previews/original-3d.png',
    stability_notes=('safe', 'legacy raymarch'),
    hud_subtitle='3D VOLUMETRIC STRESS',
    preview_palette=('#031316', '#0a2a30', '#176a58', '#51bd72', '#bddf8b', '#ffc35d', '#f66f5e'),
)
