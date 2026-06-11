"""World definition for Neural Plane."""
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

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 getEventTarget(float id, float seed) {
    return vec2(hash12(vec2(id, seed)), hash12(vec2(id, seed + 1.0)));
}

void main() {
    vec2 texel = 1.0 / resolution;

    // Channels:
    // R (x): Substrate / Empty Grid (U)
    // G (y): Active Neural Nodes (V)
    // B (z): Crystal / Bismuth Matrix Elevation
    // A (w): Computational Heat / Energy Load

    vec4 c  = texture(stateTex, uv);
    vec4 r  = texture(stateTex, uv + vec2(texel.x, 0.0));
    vec4 l  = texture(stateTex, uv - vec2(texel.x, 0.0));
    vec4 t  = texture(stateTex, uv + vec2(0.0, texel.y));
    vec4 b_ = texture(stateTex, uv - vec2(0.0, texel.y));
    vec4 tr = texture(stateTex, uv + vec2(texel.x, texel.y));
    vec4 tl = texture(stateTex, uv + vec2(-texel.x, texel.y));
    vec4 br = texture(stateTex, uv + vec2(texel.x, -texel.y));
    vec4 bl = texture(stateTex, uv - texel);

    float lapU = (r.x + l.x + t.x + b_.x) * 0.2 + (tr.x + tl.x + br.x + bl.x) * 0.05 - c.x;
    float lapV = (r.y + l.y + t.y + b_.y) * 0.2 + (tr.y + tl.y + br.y + bl.y) * 0.05 - c.y;
    float lapH = (r.z + l.z + t.z + b_.z) * 0.2 + (tr.z + tl.z + br.z + bl.z) * 0.05 - c.z;
    float lapA = (r.w + l.w + t.w + b_.w) * 0.2 + (tr.w + tl.w + br.w + bl.w) * 0.05 - c.w;

    float reaction = c.x * c.y * c.y;

    // Matrix heat slows down structural feed, pushing network to branch
    float matrixFeed = feed - c.w * 0.012;
    float matrixKill = kill + c.w * 0.012;

    float du = (diffU * lapU * laplaceScale) - reaction + matrixFeed * (1.0 - c.x);
    float dv = (diffV * lapV * laplaceScale) + reaction - (matrixFeed + matrixKill) * c.y;

    // Matrix Height organically targets active node locations
    float targetH = smoothstep(0.25, 0.75, c.y); // Requires higher concentration to raise
    float dh = (targetH - c.z) * 0.05 + lapH * 0.2;

    // Computational Surges (Quantum tunnel events)
    float surgeInterval = 6.0;
    float surgeId = floor(time / surgeInterval);
    float surgeLocalTime = fract(time / surgeInterval) * surgeInterval;
    vec2 sTarget = getEventTarget(surgeId, 99.0);
    float sDist = length(uv - sTarget);

    float surgePulse = exp(-pow((surgeLocalTime - 1.0) * 15.0, 2.0));
    float surgeCore = 1.0 - smoothstep(0.0, 0.03, sDist);
    
    // Inject extreme heat and force nodes to activate
    float heatGain = surgePulse * surgeCore * 15.0;
    dv += surgePulse * surgeCore * 0.5;

    // Heat diffuses rapidly and is consumed by growth
    float dw = lapA * 0.85 + heatGain + reaction * 2.5 - c.w * 0.15;

    fragColor = vec4(
        clamp(c.x + du * dt, 0.0, 1.0),
        clamp(c.y + dv * dt, 0.0, 1.0),
        clamp(c.z + dh * dt, 0.0, 1.0),
        clamp(c.w + dw * dt, 0.0, 1.0)
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
uniform float cameraSpeed;
uniform float fxIntensity;
uniform int raySteps;

#define MAX_STEPS 160
#define MAX_DIST 40.0
#define SURF_DIST 0.002

mat3 setCamera(in vec3 ro, in vec3 ta, float cr) {
    vec3 cw = normalize(ta - ro);
    vec3 cp = vec3(sin(cr), cos(cr), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = normalize(cross(cu, cw));
    return mat3(cu, cv, cw);
}

float hash12(vec2 p) {
    vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i + vec2(0.0, 0.0)), hash12(i + vec2(1.0, 0.0)), f.x),
               mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), f.x), f.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    mat2 rot = mat2(0.866, -0.5, 0.5, 0.866);
    for (int i = 0; i < 4; i++) {
        v += a * noise(p);
        p = rot * p * 2.0 + vec2(100.0);
        a *= 0.5;
    }
    return v;
}

vec2 getEventTarget(float id, float seed) {
    return vec2(hash12(vec2(id, seed)), hash12(vec2(id, seed + 1.0)));
}

float map(in vec3 p, out int matID, out vec4 stateOut) {
    vec2 mapUV = p.xz * 0.06;
    vec4 state = textureLod(stateTex, mapUV, 0.0);

    // Quantize height to create Bismuth-like crystalline circuits. 
    // Smooth the step slightly to prevent SDF raymarching discontinuities.
    float hRaw = state.z * 2.5; 
    float levels = mix(4.0, 10.0, smoothstep(0.2, 1.5, fxIntensity));
    float q = hRaw * levels;
    float hStep = (floor(q) + smoothstep(0.0, 0.2, fract(q))) / levels * 1.5;

    // Tech grid gaps and trenches
    float gridX = abs(fract(p.x * 1.5) - 0.5) * 2.0;
    float gridZ = abs(fract(p.z * 1.5) - 0.5) * 2.0;
    float trench = 1.0 - smoothstep(0.02, 0.12, min(gridX, gridZ));
    
    // Deform terrain with trenches and structural steps
    float hTerrain = hStep - trench * 0.5 * smoothstep(0.2, 0.8, noise(p.xz * 3.0));
    float dTerrain = p.y - hTerrain;

    // Data Swarm / Quantum Plasma Volume
    float swarmH = state.w * 0.8 + state.y * 0.5 + fbm(p.xz * 1.5 - vec2(time * 0.4, 0.0)) * 0.4 - 0.2;
    float dSwarm = p.y - swarmH;

    if (dTerrain < dSwarm) {
        matID = 1; // Neural Bismuth Matrix
        stateOut = state;
        return dTerrain * 0.5;
    } else {
        matID = 0; // Data Swarm
        stateOut = state;
        return dSwarm * 0.7;
    }
}

vec3 calcNormal(in vec3 p) {
    const vec2 h = vec2(0.01, 0.0);
    int dMat; vec4 dState;
    return normalize(vec3(
        map(p + h.xyy, dMat, dState) - map(p - h.xyy, dMat, dState),
        map(p + h.yxy, dMat, dState) - map(p - h.yxy, dMat, dState),
        map(p + h.yyx, dMat, dState) - map(p - h.yyx, dMat, dState)
    ));
}

float calcShadow(in vec3 ro, in vec3 rd) {
    float res = 1.0;
    float t = 0.05;
    int dMat; vec4 dState;
    for (int i = 0; i < 32; i++) {
        float h = map(ro + rd * t, dMat, dState);
        res = min(res, 12.0 * h / t);
        t += clamp(h, 0.02, 0.5);
        if (h < 0.001 || t > 15.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float contourPower = clamp(contourContrast, 0.0, 1.6);
    int maxRaySteps = clamp(raySteps, 32, MAX_STEPS);

    float camTime = time * 0.08 * max(cameraSpeed, 0.01);
    // Lift camera baseline and look slightly down so it never dips below max terrain bounds
    vec3 ro = vec3(camTime * 4.0, 5.5 + sin(camTime * 0.3) * 1.2, camTime * 3.0);
    vec3 ta = ro + vec3(cos(camTime * 0.25), -0.6, sin(camTime * 0.25));

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.15) * 0.08);
    vec3 rd = ca * normalize(vec3(p.xy, 2.0));

    vec3 lightDir = normalize(vec3(0.5, 0.8, -0.6));

    // Deep void cyber-sky
    vec3 skyColor = vec3(0.01, 0.02, 0.04);
    float sun = pow(max(0.0, dot(rd, lightDir)), 120.0);
    skyColor += vec3(0.0, 0.4, 0.8) * sun * safeFx * 0.4;

    // Quantum Surge Flash
    float surgeInterval = 6.0;
    float surgeId = floor(time / surgeInterval);
    float surgeLocalTime = fract(time / surgeInterval) * surgeInterval;

    if (surgeLocalTime > 0.5 && surgeLocalTime < 2.0) {
        vec2 sTargetUV = getEventTarget(surgeId, 99.0);
        vec3 sTargetWorld = vec3(sTargetUV.x / 0.06, 0.0, sTargetUV.y / 0.06);
        vec3 toTarget = normalize(sTargetWorld - ro);
        float sProj = max(0.0, dot(rd, toTarget));
        
        float flash = exp(-pow(surgeLocalTime - 1.0, 2.0) * 80.0);
        skyColor += vec3(0.0, 1.0, 0.8) * pow(sProj, 250.0) * 6.0 * flash * safeFx;
        skyColor += vec3(1.0, 0.0, 0.6) * flash * 1.5 * safeFx;
    }

    float t = 0.0;
    int matID = -1;
    vec4 state = vec4(0.0);
    vec3 volumeGlow = vec3(0.0);

    for (int i = 0; i < MAX_STEPS; i++) {
        if (i >= maxRaySteps) break;
        vec3 pos = ro + rd * t;
        int currMat; vec4 currState;
        float h = map(pos, currMat, currState);

        if (currMat == 0) {
            float vDense = smoothstep(0.1, 0.8, currState.y);
            float wHot = smoothstep(0.1, 1.0, currState.w);
            vec3 swarmColor = mix(vec3(0.0, 0.7, 1.0), vec3(1.0, 0.0, 0.8), wHot);
            volumeGlow += swarmColor * (vDense + wHot * 0.5) * (0.04 * glow) * safeFx / (1.0 + abs(h) * 6.0);
        }

        if (h < SURF_DIST * (1.0 + t * 0.1) || t > MAX_DIST) {
            matID = currMat;
            state = currState;
            break;
        }
        t += clamp(h, 0.02, 0.8);
    }

    vec3 color = skyColor;

    if (t < MAX_DIST) {
        vec3 pos = ro + rd * t;
        vec3 nor = calcNormal(pos);

        float occ = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
        float sha = calcShadow(pos + nor * 0.02, lightDir);
        float dif = clamp(dot(nor, lightDir), 0.0, 1.0) * sha;
        float sky = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
        float fre = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 5.0);

        vec3 matColor;
        vec3 emission = vec3(0.0);

        if (matID == 1) {
            // Obsidian / Bismuth Matrix Base
            matColor = vec3(0.03, 0.04, 0.06);

            // Bismuth Iridescence
            vec3 iridescence = 0.5 + 0.5 * cos(6.28318 * (state.z * 1.5 + vec3(0.0, 0.33, 0.67)));
            matColor += iridescence * 0.15 * sha * safeFx;

            // Compute grid location for glowing circuit traces
            float gridX = abs(fract(pos.x * 1.5) - 0.5) * 2.0;
            float gridZ = abs(fract(pos.z * 1.5) - 0.5) * 2.0;
            float isTrench = 1.0 - smoothstep(0.02, 0.06, min(gridX, gridZ));
            float edgeGlow = smoothstep(0.08, 0.0, min(gridX, gridZ));
            float contourLine = 1.0 - smoothstep(
                0.0,
                0.055,
                abs(fract((pos.x * 0.35 + pos.z * 0.45 + pos.y * 1.8) * 5.0) - 0.5)
            );
            matColor += vec3(0.0, 0.45, 0.55) * contourLine * contourPower * 0.18;

            // Data Path Emissions
            vec3 pathGlow = mix(vec3(0.0, 0.8, 1.0), vec3(1.0, 0.2, 0.6), state.w);
            emission += pathGlow * isTrench * state.y * 3.0 * safeFx;
            emission += pathGlow * edgeGlow * state.w * 4.0 * safeFx;

            // Shiny matrix reflections
            float spec = pow(clamp(dot(reflect(rd, nor), lightDir), 0.0, 1.0), 32.0) * sha;
            emission += vec3(0.5, 0.8, 1.0) * spec * 0.5;

        } else {
            // Surface of the dense Quantum Swarm
            matColor = mix(vec3(0.0, 0.15, 0.25), vec3(0.6, 0.1, 0.4), state.y);
            emission += matColor * state.w * 1.5 * safeFx;
            
            float spec = pow(clamp(dot(reflect(rd, nor), lightDir), 0.0, 1.0), 8.0) * sha;
            emission += vec3(0.0, 0.6, 0.8) * spec * 0.4;
            emission += volumeGlow * 0.3;
        }

        vec3 lin = vec3(0.0);
        lin += 1.2 * dif * vec3(0.8, 0.9, 1.0);
        lin += 0.6 * sky * vec3(0.1, 0.2, 0.3) * occ;
        lin += 0.8 * fre * vec3(0.4, 0.8, 1.0) * occ;

        color = matColor * lin + emission;

        float fog = 1.0 - exp(-0.0015 * t * t);
        color = mix(color, skyColor, fog);
    }

    color += volumeGlow * (1.0 + safeFx * 0.5);

    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(color, vec3(1.0 / max(gamma, 0.1)));

    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.5 + 0.5 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.2);

    fragColor = vec4(color, 1.0);
}
"""


def seed_field(width_px: int, height_px: int, tile_size: int) -> np.ndarray:
    tile_size = max(2, int(tile_size))
    tiles_x = max(1, int(np.ceil(width_px / tile_size)))
    tiles_y = max(1, int(np.ceil(height_px / tile_size)))

    rng = np.random.default_rng(4242)

    reaction_u = np.ones((tiles_y, tiles_x), dtype=np.float32)
    reaction_v = np.zeros((tiles_y, tiles_x), dtype=np.float32)

    seed_count = max(5, (tiles_x * tiles_y) // 300)
    for _ in range(seed_count):
        cx = rng.integers(0, tiles_x)
        cy = rng.integers(0, tiles_y)
        radius = rng.integers(2, 8)
        y, x = np.ogrid[-cy:tiles_y-cy, -cx:tiles_x-cx]
        mask = x*x + y*y <= radius*radius
        reaction_v[mask] = 1.0
        reaction_u[mask] = 0.5

    height = rng.uniform(0.0, 0.2, size=(tiles_y, tiles_x)).astype(np.float32)
    heat = np.zeros((tiles_y, tiles_x), dtype=np.float32)

    tile_field = np.stack([reaction_u, reaction_v, height, heat], axis=-1)
    field = np.repeat(np.repeat(tile_field, tile_size, axis=0), tile_size, axis=1)
    field = field[:height_px, :width_px].copy()
    return field


SPEC = WorldSpec(
    id='neural-plane-3d',
    display_name='Neural Plane',
    window_title='Garage Life Lab - NEURAL MATRIX CONTAINMENT',
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={'exposure': 1.5, 'ray_steps': 120, 'fx_intensity': 1.2, 'tile_size': 10},
    preview_image='assets/world_previews/neural-plane-3d.png',
    stability_notes=('heavy raymarch', 'experimental'),
    hud_subtitle='NEURAL PLANE',
    preview_palette=('#050712', '#121635', '#24236a', '#4b35ad', '#1fd3c0', '#b6fff2', '#ff4fd8'),
)
