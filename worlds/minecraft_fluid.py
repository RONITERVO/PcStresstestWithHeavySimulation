"""World definition for Minecraft Fluid Dynamics."""
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
    // R: Glowing fluid / Lava / Magic Water
    // G: Organic biomass / Moss / Sculk
    // B: Terrain Elevation (slowly eroding/building)
    // A: Heat / Core Energy
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

    // Height gradient for advection (fluid flows downhill)
    vec2 gradH = vec2(r.b - l.b, t.b - b.b);
    
    // Advection vectors
    float flowStrength = 0.95;
    float advectR = dot(gradH, vec2(r.r - l.r, t.r - b.r)) * flowStrength;
    float advectG = dot(gradH, vec2(r.g - l.g, t.g - b.g)) * (flowStrength * 0.4); // Moss spreads downhill slower

    float envNoise = (hash12(uv * resolution + floor(time * 0.4)) - 0.5) * noiseStrength;

    // Localized chaotic drift driven by heat (A) and noise
    float localFeed = feed * 0.55 + 0.015 + c.a * 0.012 * sin(time * 0.15 + uv.x * 25.0) + envNoise * 0.12;
    float localKill = kill * 0.52 + 0.020 + (1.0 - c.a) * 0.008 * cos(time * 0.2 + uv.y * 20.0) + parameterDrift * 0.45;

    // Reaction: Biomass (G) consumes Fluid (R)
    float reaction = c.r * c.g * c.g * 0.65;

    // Integration step
    float dr = (diffU * lapR) - reaction + localFeed * (1.0 - c.r) - advectR + envNoise * 0.002;
    float dg = (diffV * lapG) + reaction - (localFeed + localKill) * c.g - advectG;
    
    // Organic terrain modification: height slowly shifts based on moss roots and heat erosion
    float dh = lapB * 0.002 + (c.g * 0.0015 - c.r * 0.0008 + envNoise * 0.001) * c.a;

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

#define MAX_STEPS 160
#define MAX_DIST 65.0
#define SURF_DIST 0.0025
#define BLOCK_SIZE 0.75
#define MAP_SCALE 0.035

const int MAT_STONE = 1;
const int MAT_FLUID = 2;
const int MAT_MOSS = 3;
const int MAT_ORE = 4;
const int MAT_OBSIDIAN = 5;

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

float blockGrid(vec3 p) {
    vec3 q = abs(fract(p / BLOCK_SIZE) - 0.5);
    float edge = min(min(q.x, q.y), q.z);
    return 1.0 - smoothstep(0.015, 0.055, edge);
}

float map(in vec3 p, out int matID, out vec4 stateOut) {
    vec2 cell = floor(p.xz / BLOCK_SIZE);
    vec2 mapUV = (cell + 0.5) * BLOCK_SIZE * MAP_SCALE;
    vec4 state = textureLod(stateTex, mapUV, 0.0);
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    
    float macro = fbm(cell * 0.04 + vec2(7.0, 11.0));
    float rawHeight = state.b * 8.5 + macro * 2.5;
    float h = floor(rawHeight) * (BLOCK_SIZE * 0.5) + 0.5;
    
    float d = p.y - h;
    matID = MAT_STONE;
    stateOut = state;
    
    vec2 center = (cell + 0.5) * BLOCK_SIZE;

    // Organic Moss/Sculk blocks growing on top
    float mossHeight = floor(state.g * 4.0) * (BLOCK_SIZE * 0.5);
    if (mossHeight > 0.1) {
        float mossD = boxSdf(
            vec3(p.x - center.x, p.y - (h + mossHeight * 0.5), p.z - center.y),
            vec3(BLOCK_SIZE * 0.48, mossHeight * 0.5, BLOCK_SIZE * 0.48)
        );
        if (mossD < d) {
            d = mossD;
            matID = MAT_MOSS;
        }
    }

    // Glowing fluid blocks flowing over terrain
    float fluidLevel = floor(state.r * 3.5) * (BLOCK_SIZE * 0.5);
    if (fluidLevel > 0.1) {
        // Fluid animates slightly vertically
        float wave = sin(time * 2.5 + cell.x * 1.5 + cell.y * 2.0) * 0.04 * safeFx;
        float fluidD = boxSdf(
            vec3(p.x - center.x, p.y - (h + fluidLevel * 0.5 + wave), p.z - center.y),
            vec3(BLOCK_SIZE * 0.46, fluidLevel * 0.5, BLOCK_SIZE * 0.46)
        );
        if (fluidD < d) {
            d = fluidD * 0.85; 
            matID = MAT_FLUID;
        }
    }

    // Cooling obsidian where fluid and moss mix
    float obsidianMask = step(0.4, state.r * state.g);
    if (obsidianMask > 0.5 && fluidLevel > 0.1) {
        if (matID == MAT_FLUID || matID == MAT_MOSS) {
            matID = MAT_OBSIDIAN;
        }
    }

    // Crystalline Energy Ores
    float oreMask = step(0.92, hash12(cell + vec2(42.0, 17.0))) * step(0.3, state.a);
    if (oreMask > 0.5) {
        float oreD = boxSdf(
            vec3(p.x - center.x, p.y - (h + BLOCK_SIZE * 0.25), p.z - center.y),
            vec3(BLOCK_SIZE * 0.25)
        );
        if (oreD < d) {
            d = oreD;
            matID = MAT_ORE;
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
    float t = 0.08;
    int mat; vec4 state;
    for (int i = 0; i < 35; i++) {
        float h = map(ro + rd * t, mat, state);
        res = min(res, 12.0 * h / t);
        t += clamp(h, 0.04, 0.6);
        if (h < 0.001 || t > 18.0) {
            break;
        }
    }
    return clamp(res, 0.0, 1.0);
}

float calcAO(vec3 pos, vec3 nor) {
    float occ = 0.0;
    float sca = 1.0;
    int mat; vec4 state;
    for (int i = 0; i < 4; i++) {
        float h = 0.03 + 0.15 * float(i);
        float d = map(pos + nor * h, mat, state);
        occ += (h - d) * sca;
        sca *= 0.75;
    }
    return clamp(1.0 - 2.0 * occ, 0.0, 1.0);
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float contourPower = clamp(contourContrast, 0.0, 1.8);
    int maxRaySteps = clamp(raySteps, 32, MAX_STEPS);

    float camTime = time * 0.12 * max(cameraSpeed, 0.05);
    vec3 ro = vec3(
        camTime * 6.0 + sin(camTime * 0.4) * 3.5,
        8.5 + sin(camTime * 0.5) * 1.5,
        camTime * 4.5 + cos(camTime * 0.35) * 3.5
    );
    vec3 ta = vec3(ro.x + 5.5, 3.0 + sin(camTime * 0.3) * 0.5, ro.z + 5.5);
    ro += cameraOffset;
    ta += cameraOffset;

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.2) * 0.04);
    vec3 rd = ca * cameraInputRay(p, 2.0);

    vec3 lightDir = normalize(vec3(0.65, 0.75, -0.45));
    float skyRise = smoothstep(-0.2, 0.8, rd.y);
    vec3 skyColor = mix(vec3(0.02, 0.03, 0.06), vec3(0.08, 0.14, 0.22), skyRise);
    
    float sun = pow(max(0.0, dot(rd, lightDir)), 150.0);
    skyColor += vec3(0.8, 0.9, 1.0) * sun * safeFx;
    
    float starGrid = step(0.995, hash12(floor(rd.xy * 300.0)));
    skyColor += vec3(1.0) * starGrid * (1.0 - skyRise);

    float t = 0.0;
    int matID = -1;
    vec4 state = vec4(0.0);
    vec3 volumeGlow = vec3(0.0);

    for (int i = 0; i < MAX_STEPS; i++) {
        if (i >= maxRaySteps) break;
        vec3 pos = ro + rd * t;
        int currMat; vec4 currState;
        float h = map(pos, currMat, currState);

        // Accumulate volumetric glow for fluid and ores
        if (currMat == MAT_FLUID || currMat == MAT_ORE) {
            vec3 glowCol = (currMat == MAT_FLUID) 
                ? mix(vec3(0.0, 0.8, 1.0), vec3(1.0, 0.2, 0.6), currState.r + sin(time+pos.x)*0.2)
                : mix(vec3(1.0, 0.8, 0.2), vec3(0.2, 0.9, 0.4), hash12(floor(pos.xz/BLOCK_SIZE)));
            
            float intensity = (currMat == MAT_FLUID) ? currState.r : currState.a * 1.5;
            float pulse = 1.0 + 0.3 * sin(time * 3.0 + pos.x * 2.0);
            volumeGlow += glowCol * intensity * pulse * (0.012 + glow * 0.005) * safeFx / (1.0 + abs(h) * 8.0);
        }

        if (h < max(SURF_DIST, 0.0015 * t) || t > MAX_DIST) {
            matID = currMat;
            state = currState;
            break;
        }
        t += clamp(h, 0.03, 0.75);
    }

    vec3 color = skyColor;

    if (t < MAX_DIST) {
        vec3 pos = ro + rd * t;
        vec3 nor = calcNormal(pos);
        float sha = calcShadow(pos + nor * 0.02, lightDir);
        float ao = calcAO(pos, nor);
        float dif = clamp(dot(nor, lightDir), 0.0, 1.0) * sha;
        float amb = clamp(0.4 + 0.6 * nor.y, 0.0, 1.0);
        float fre = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 3.0);
        float grid = blockGrid(pos) * contourPower;

        vec3 matColor = vec3(0.0);
        vec3 emission = vec3(0.0);
        float specPower = 0.0;

        float cellHash = hash12(floor(pos.xz / BLOCK_SIZE) + floor(pos.y / BLOCK_SIZE) * 11.3);

        if (matID == MAT_STONE) {
            vec3 darkStone = vec3(0.12, 0.13, 0.15);
            vec3 lightStone = vec3(0.25, 0.26, 0.28);
            matColor = mix(darkStone, lightStone, noise(pos.xz * 3.0 + pos.y * 2.0));
            matColor *= 1.0 - grid * 0.25;
            
            // Subtle glowing cracks based on heat (A)
            float cracks = step(0.85, noise(pos.xz * 12.0 + pos.y * 12.0));
            emission += vec3(1.0, 0.3, 0.1) * cracks * state.a * (0.5 + glow) * safeFx;
        } 
        else if (matID == MAT_FLUID) {
            vec3 fluidDeep = vec3(0.0, 0.2, 0.6);
            vec3 fluidSurface = mix(vec3(0.0, 0.8, 1.0), vec3(1.0, 0.1, 0.5), state.r);
            matColor = mix(fluidDeep, fluidSurface, clamp(state.r * 1.5, 0.0, 1.0));
            matColor *= 1.0 - grid * 0.1;
            specPower = 48.0;
            emission += fluidSurface * state.r * (0.6 + glow * 0.5) * safeFx;
        }
        else if (matID == MAT_MOSS) {
            vec3 mossA = vec3(0.05, 0.25, 0.18);
            vec3 mossB = vec3(0.15, 0.45, 0.35);
            matColor = mix(mossA, mossB, cellHash);
            matColor *= 1.0 - grid * 0.15;
            
            // Bioluminescent spores
            float spores = step(0.9, hash12(pos.xy * 15.0));
            emission += vec3(0.2, 1.0, 0.6) * spores * state.g * safeFx;
        }
        else if (matID == MAT_OBSIDIAN) {
            matColor = vec3(0.04, 0.02, 0.06);
            matColor *= 1.0 - grid * 0.3;
            specPower = 64.0;
            // Hot seams
            float seam = step(0.8, noise(pos.xz * 8.0));
            emission += vec3(0.8, 0.2, 0.8) * seam * state.r * safeFx;
        }
        else if (matID == MAT_ORE) {
            matColor = vec3(0.1, 0.1, 0.1);
            matColor *= 1.0 - grid * 0.2;
            vec3 oreCol = mix(vec3(1.0, 0.7, 0.1), vec3(0.1, 0.9, 0.4), cellHash);
            emission += oreCol * state.a * (1.2 + glow) * safeFx;
            specPower = 32.0;
        }

        vec3 lin = vec3(0.0);
        lin += 1.8 * dif * vec3(0.95, 0.9, 1.0) * ao;
        lin += 0.5 * amb * vec3(0.15, 0.25, 0.4) * ao;
        lin += 0.3 * fre * vec3(1.0);

        color = matColor * lin + emission;

        if (specPower > 0.0) {
            vec3 ref = reflect(rd, nor);
            float spe = pow(clamp(dot(ref, lightDir), 0.0, 1.0), specPower) * sha;
            color += vec3(1.0) * spe * (0.4 + safeFx * 0.2);
        }

        float fog = 1.0 - exp(-0.001 * t * t);
        color = mix(color, skyColor, fog);
    }

    color += volumeGlow * (0.9 + safeFx * 0.5);
    
    // Camera shake/aberration based on intense heat
    vec2 offset = vec2(sin(time * 15.0), cos(time * 19.0)) * 0.001 * state.a;
    color.r += offset.x;
    color.b -= offset.y;

    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(color, vec3(1.0 / max(gamma, 0.2)));

    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.55 + 0.45 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.25);

    fragColor = vec4(color, 1.0);
}
"""

def seed_field(width_px: int, height_px: int, tile_size: int) -> np.ndarray:
    tile_size = max(2, int(tile_size))
    tiles_x = max(1, int(np.ceil(width_px / tile_size)))
    tiles_y = max(1, int(np.ceil(height_px / tile_size)))

    rng = np.random.default_rng(733199)
    tile_y, tile_x = np.meshgrid(
        np.arange(tiles_y, dtype=np.float32),
        np.arange(tiles_x, dtype=np.float32),
        indexing="ij",
    )
    x_norm = tile_x / max(tiles_x - 1, 1)
    y_norm = tile_y / max(tiles_y - 1, 1)

    # Base elevation
    height = (
        0.35
        + 0.18 * np.sin(x_norm * 8.0 + np.cos(y_norm * 6.0) * 1.5)
        + 0.12 * np.cos(y_norm * 10.0 - 0.5)
        + 0.09 * np.sin((x_norm + y_norm) * 14.0)
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.02
    )

    # Add craters/calderas for fluid to pool in
    crater_count = max(6, (tiles_x * tiles_y) // 2500)
    for _ in range(crater_count):
        cx = rng.uniform(0.0, tiles_x)
        cy = rng.uniform(0.0, tiles_y)
        r = rng.uniform(max(8.0, tiles_x * 0.06), max(18.0, tiles_x * 0.15))
        distance = np.sqrt((tile_x - cx)**2 + (tile_y - cy)**2) / r
        crater_shape = np.clip(1.0 - distance, 0.0, 1.0)
        # Rim up, center down
        rim = np.sin(crater_shape * np.pi) * 0.15
        center = crater_shape**2 * 0.25
        height += rim - center

    height = np.clip(height, 0.0, 1.0)

    # Fluid (Magma/Magic Water) spawns in high craters and flows down
    fluid = np.zeros_like(height)
    high_spots = np.argwhere(height > 0.6)
    if len(high_spots) > 0:
        pool_count = min(crater_count, len(high_spots))
        for idx in rng.choice(len(high_spots), size=pool_count, replace=False):
            cy, cx = high_spots[idx]
            pr = rng.integers(3, 8)
            y0 = max(cy - pr, 0)
            y1 = min(cy + pr + 1, tiles_y)
            x0 = max(cx - pr, 0)
            x1 = min(cx + pr + 1, tiles_x)
            py, px = np.meshgrid(
                np.arange(y0, y1, dtype=np.float32),
                np.arange(x0, x1, dtype=np.float32),
                indexing="ij"
            )
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            fluid_mask = np.clip(1.0 - dist / max(pr, 1.0), 0.0, 1.0)
            fluid[y0:y1, x0:x1] = np.maximum(fluid[y0:y1, x0:x1], fluid_mask * rng.uniform(0.6, 1.0))

    # Biomass (Moss/Sculk) grows in valleys and mid-elevations
    biomass = np.clip(
        0.5 - np.abs(height - 0.4) * 2.0 
        + 0.1 * np.sin(x_norm * 25.0) 
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.05,
        0.0, 
        1.0
    )

    # Heat/Energy concentrated deep down or at fluid sources
    heat = np.clip(
        fluid * 0.8 + (1.0 - height) * 0.5 
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.05,
        0.0,
        1.0
    )

    tile_field = np.stack(
        [
            fluid.astype(np.float32),
            biomass.astype(np.float32),
            height.astype(np.float32),
            heat.astype(np.float32),
        ],
        axis=-1,
    )

    field = np.repeat(np.repeat(tile_field, tile_size, axis=0), tile_size, axis=1)
    return field[:height_px, :width_px].copy()

SPEC = WorldSpec(
    id="minecraft-fluid-3d",
    display_name="Minecraft Fluid Dynamics",
    window_title="Garage Life Lab - Minecraft Fluid",
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={
        "ray_steps": 128,
        "fx_intensity": 1.35,
        "contour_contrast": 1.15,
        "camera_speed": 0.85,
        "tile_size": 12,
    },
    stability_notes=("volumetric fluid", "heavy advection", "blocky 3D"),
    hud_subtitle="MINECRAFT FLUID DYNAMICS",
    preview_palette=("#0c0f12", "#1d2b38", "#0f5e55", "#2fa86d", "#1a75ff", "#e03870", "#ffaa1d"),
)