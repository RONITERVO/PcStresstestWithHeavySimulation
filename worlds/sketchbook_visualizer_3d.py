"""World definition for Sketchbook Visualizer."""
from __future__ import annotations

import numpy as np

from .spec import WorldSpec

SIM_FRAG_SHADER = r"""

#version 450
in vec2 uv;
out vec4 fragColor;
uniform sampler2D stateTex;
uniform sampler2D audioFft;

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
uniform float audioEnergy;
uniform float audioBass;
uniform float audioTreble;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

void main() {
    vec2 texel = 1.0 / resolution;
    // R: Ink/Graphite Density
    // G: Watercolor Wash Density
    // B: Terrain Elevation Mask
    // A: Audio Reactivity Buffer
    
    vec4 c = texture(stateTex, uv);
    vec4 r = texture(stateTex, uv + vec2(texel.x, 0.0));
    vec4 l = texture(stateTex, uv - vec2(texel.x, 0.0));
    vec4 t = texture(stateTex, uv + vec2(0.0, texel.y));
    vec4 b = texture(stateTex, uv - vec2(0.0, texel.y));
    vec4 tr = texture(stateTex, uv + texel);
    vec4 tl = texture(stateTex, uv + vec2(-texel.x, texel.y));
    vec4 br = texture(stateTex, uv + vec2(texel.x, -texel.y));
    vec4 bl = texture(stateTex, uv - texel);

    float lScale = laplaceScale * (1.0 + audioBass * 0.5);
    float lapU = ((r.r + l.r + t.r + b.r) * 0.2 + (tr.r + tl.r + br.r + bl.r) * 0.05 - c.r) * lScale;
    float lapV = ((r.g + l.g + t.g + b.g) * 0.2 + (tr.g + tl.g + br.g + bl.g) * 0.05 - c.g) * lScale;

    // Ink flows down terrain gradients
    vec2 gradH = vec2(r.b - l.b, t.b - b.b);
    float flowStrength = 0.5 + audioEnergy * 1.5;
    float advectU = dot(gradH, vec2(r.r - l.r, t.r - b.r)) * flowStrength;
    float advectV = dot(gradH, vec2(r.g - l.g, t.g - b.g)) * flowStrength;

    float wetNoise = (hash(uv * resolution + floor(time * 0.35)) - 0.5) * noiseStrength * (1.0 + audioTreble * 5.0);
    
    // Audio driven feed/kill injection
    float localFeed = feed + c.a * 0.015 * sin(time * 0.1 + uv.x * 20.0) + wetNoise * 0.18 + audioBass * 0.04;
    float localKill = kill + (1.0 - c.a) * 0.01 * cos(time * 0.15 + uv.y * 15.0) + parameterDrift * 0.6 - audioEnergy * 0.02;

    float reaction = c.r * c.g * c.g * (1.0 + audioTreble * 1.5);
    
    float du = (diffU * lapU) - reaction + localFeed * (1.0 - c.r) - advectU;
    float dv = (diffV * lapV) + reaction - (localFeed + localKill) * c.g - advectV;

    float dh = 0.0; // Static terrain for sketchy world
    float da = (audioEnergy * 0.1 - c.a * 0.05) * dt; 

    fragColor = vec4(
        clamp(c.r + du * dt, 0.0, 1.0),
        clamp(c.g + dv * dt, 0.0, 1.0),
        clamp(c.b + dh * dt, 0.0, 1.0),
        clamp(c.a + da * dt, 0.0, 1.0)
    );
}

"""

DISPLAY_FRAG_SHADER = r"""

#version 450
in vec2 uv;
out vec4 fragColor;

uniform sampler2D stateTex;
uniform sampler2D audioFft;
uniform sampler2D audioWave;

uniform vec2 resolution;
uniform float time;
uniform float exposure;
uniform float glow;
uniform float gamma;
uniform float contourContrast;
uniform float cameraSpeed;
uniform float fxIntensity;
uniform vec3 cameraOffset;
uniform vec2 cameraYawPitch;
uniform float cameraZoom;
uniform int raySteps;

uniform float audioEnergy;
uniform float audioBass;
uniform float audioTreble;

// --- Living Sketchbook Palette ---
const vec3 PAPER = vec3(0.957, 0.933, 0.882);
const vec3 INK_GRAPHITE = vec3(0.137, 0.118, 0.110);
const vec3 INK_BLUEPRINT = vec3(0.094, 0.294, 0.647);
const vec3 SUN_INK = vec3(0.922, 0.588, 0.176);
const vec3 WATER_WASH = vec3(0.431, 0.549, 0.627);

float hash(vec2 p) { return fract(sin(dot(p, vec2(41.7, 289.1))) * 45758.5453); }

float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

float fbm(vec2 p) {
    float v = 0.0; float a = 0.5;
    mat2 rot = mat2(0.866, -0.5, 0.5, 0.866);
    for (int i = 0; i < 4; i++) {
        v += a * noise(p);
        p = rot * p * 2.0 + vec2(100.0);
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

// Emulate Multiply Blend Mode
vec3 blendMultiply(vec3 base, vec3 blend, float opacity) {
    vec3 multiplied = base * blend;
    return mix(base, multiplied, opacity);
}

float getWaveHeight(vec2 p, float t) {
    float baseWave = sin(p.x * 0.5 + t) * cos(p.y * 0.3 + t * 0.8) * 0.2;
    float audioDisplacement = texture(audioWave, vec2(fract(p.x * 0.05 + p.y * 0.05 + t * 0.1), 0.5)).r;
    return baseWave + audioDisplacement * audioBass * 0.8;
}

float map(in vec3 p, out float matID, out vec4 stateOut) {
    vec2 mapUV = fract(p.xz * 0.02);
    vec4 state = textureLod(stateTex, mapUV, 0.0);
    
    // Stop-motion jitter time
    float frameTime = floor(time * 12.0) / 12.0;

    // Ocean plane (Base surface)
    float wave = getWaveHeight(p.xz, frameTime * fxIntensity);
    float dWater = p.y - (-1.0 + wave);

    // Island terrain
    float islandBase = state.b * 4.0; 
    float rough = fbm(p.xz * 2.0 + frameTime * 0.1) * 0.5;
    float hTerrain = -0.5 + islandBase + rough;
    float dTerrain = p.y - hTerrain;

    if (dTerrain < dWater && state.b > 0.1) {
        matID = 1.0; 
        stateOut = state; 
        return dTerrain * 0.6;
    } else {
        matID = 0.0; 
        stateOut = state; 
        return dWater * 0.8;
    }
}

vec3 calcNormal(in vec3 p) {
    vec2 e = vec2(0.02, 0.0); 
    float dummyMat; vec4 dummyState;
    return normalize(vec3(
        map(p + e.xyy, dummyMat, dummyState) - map(p - e.xyy, dummyMat, dummyState),
        map(p + e.yxy, dummyMat, dummyState) - map(p - e.yxy, dummyMat, dummyState),
        map(p + e.yyx, dummyMat, dummyState) - map(p - e.yyx, dummyMat, dummyState)
    ));
}

float calcShadow(in vec3 ro, in vec3 rd) {
    float res = 1.0; float t = 0.1; 
    float dMat; vec4 dSt;
    for (int i = 0; i < 20; i++) {
        float h = map(ro + rd * t, dMat, dSt);
        res = min(res, 8.0 * h / t); 
        t += clamp(h, 0.05, 0.5);
        if (h < 0.001 || t > 10.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

// Pencil Cross-Hatching
float getHatch(vec2 p, float lum) {
    float h = 0.0;
    float freq1 = 80.0;
    float freq2 = 85.0;
    
    // Stop motion jitter for lines
    vec2 jp = p + vec2(hash(p + floor(time*12.0)), hash(p - floor(time*12.0))) * 0.005;

    float line1 = sin(jp.x * freq1 + jp.y * freq1);
    float line2 = sin(jp.x * freq2 - jp.y * freq2);
    float line3 = sin(jp.x * freq1 * 1.5 + jp.y * freq1 * 0.5);

    if (lum < 0.8) h += smoothstep(0.0, 0.1, line1);
    if (lum < 0.5) h += smoothstep(0.0, 0.1, line2);
    if (lum < 0.2) h += smoothstep(0.0, 0.1, line3);

    return clamp(h, 0.0, 1.0);
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float frameTime = floor(time * 12.0) / 12.0;

    // Camera
    float camShake = noise(vec2(frameTime * 10.0, 0.0)) * audioEnergy * 0.1;
    float camTime = frameTime * 0.15 * max(cameraSpeed, 0.05);
    vec3 ro = vec3(camTime * 5.0, 4.0 + sin(camTime * 0.5) * 1.5 + camShake, camTime * 4.0);
    vec3 ta = vec3(ro.x + 4.8, 0.0, ro.z + 4.8 + sin(camTime));

    ro += cameraOffset;
    ta += cameraOffset;

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.3) * 0.1 + camShake);
    vec3 rd = ca * cameraInputRay(p, 1.8);

    vec3 lightDir = normalize(vec3(0.8, 0.6, -0.4));

    // Background: Paper Sky & Sketchy Sun
    vec3 color = PAPER;
    
    // Draw Sun
    float sunDot = dot(rd, lightDir);
    if (sunDot > 0.0) {
        vec2 sunUV = vec2(acos(rd.y), atan(rd.z, rd.x));
        float radial = sunUV.y * 10.0 + frameTime;
        float fftSample = texture(audioFft, vec2(fract(abs(sin(radial))), 0.5)).r;
        
        float sunRadius = 0.98 - fftSample * 0.01 - audioBass * 0.02;
        float sunShape = smoothstep(sunRadius - 0.005, sunRadius + 0.005, sunDot);
        
        // Ray scribbles
        float scribble = sin(radial * 15.0 + hash(vec2(frameTime)) * 2.0) * fftSample;
        if (sunDot > sunRadius - 0.05 + scribble * 0.05) {
            color = blendMultiply(color, SUN_INK, 0.8 * fxIntensity);
        }
    }

    float tMax = 35.0; 
    float t = 0.0; 
    float matID = -1.0;
    vec4 state = vec4(0.0); 

    int safeRaySteps = clamp(raySteps, 32, 160);
    for (int i = 0; i < 160; i++) {
        if (i >= safeRaySteps) break;
        vec3 pos = ro + rd * t;
        float currMat; vec4 currState;
        float h = map(pos, currMat, currState);

        if (h < max(0.003, 0.0015 * t) || t > tMax) {
            matID = currMat; state = currState; break;
        }
        t += clamp(h, 0.03, 0.8);
    }

    if (t < tMax) {
        vec3 pos = ro + rd * t;
        vec3 nor = calcNormal(pos);

        float sha = calcShadow(pos + nor * 0.02, lightDir);
        float dif = clamp(dot(nor, lightDir), 0.0, 1.0);
        float lum = dif * sha;

        // Outline contour detection
        float edge = smoothstep(0.0, 0.3 + contourContrast * 0.2, dot(nor, -rd));
        
        if (matID == 1.0) {
            // Island Terrain
            vec3 baseColor = PAPER;
            
            // Watercolor ink spread from simulation (R=Graphite, G=Watercolor wash)
            baseColor = blendMultiply(baseColor, WATER_WASH, state.g * 0.8);
            
            // Hatching shadows
            float hatch = getHatch(p, lum);
            baseColor = blendMultiply(baseColor, INK_GRAPHITE, hatch * 0.7);
            
            // Thick Ink Outlines
            baseColor = blendMultiply(baseColor, INK_GRAPHITE, (1.0 - edge));
            
            // Audio-reactive bloom strokes
            float energyStroke = smoothstep(0.5, 1.0, state.r) * audioEnergy;
            baseColor = blendMultiply(baseColor, SUN_INK, energyStroke);

            color = baseColor;
            
        } else {
            // Ocean
            vec3 baseColor = PAPER;
            
            // Blueprint wash
            float depth = clamp((pos.y - (-1.0)) * 2.0, 0.0, 1.0);
            baseColor = blendMultiply(baseColor, INK_BLUEPRINT, 0.4 + depth * 0.4 + audioTreble * 0.2);
            
            // Reflected hatching
            vec3 ref = reflect(rd, nor);
            float spe = pow(clamp(dot(ref, lightDir), 0.0, 1.0), 16.0);
            float hatchSpe = getHatch(p + vec2(pos.x, pos.z)*0.1, 1.0 - spe);
            
            baseColor = blendMultiply(baseColor, INK_GRAPHITE, hatchSpe * 0.5);
            baseColor = blendMultiply(baseColor, INK_BLUEPRINT, (1.0 - edge) * 0.5); // Softer water outlines

            color = baseColor;
        }

        // Distance fog fades to paper
        color = mix(color, PAPER, 1.0 - exp(-0.0025 * t * t));
    }

    // Post-Process: Paper Texture & Noise
    float grain = hash(p * resolution + frameTime) * 0.08;
    color -= grain;

    // Chromatic Aberration from Bass
    float caShift = audioBass * 0.015;
    vec2 caP = uv;
    color.r = mix(color.r, texture(stateTex, caP + vec2(caShift, 0)).r, caShift);
    color.b = mix(color.b, texture(stateTex, caP - vec2(caShift, 0)).b, caShift);

    // Exposure & Gamma
    color *= exposure;
    color = pow(color, vec3(1.0 / max(gamma, 0.2)));

    // Vignette
    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.6 + 0.4 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.15);

    fragColor = vec4(color, 1.0);
}

"""


def seed_field(width_px: int, height_px: int, tile_size: int) -> np.ndarray:
    w, h = width_px, height_px
    ts = max(2, int(tile_size))
    tx, ty = max(1, int(np.ceil(w / ts))), max(1, int(np.ceil(h / ts)))
    rng = np.random.default_rng(2026)
    ty_m, tx_m = np.meshgrid(
        np.arange(ty, dtype=np.float32),
        np.arange(tx, dtype=np.float32),
        indexing="ij",
    )

    height = np.zeros((ty, tx), dtype=np.float32)
    island_count = max(5, (tx * ty) // 2500)
    for _ in range(island_count):
        cx, cy = rng.uniform(0.0, tx), rng.uniform(0.0, ty)
        rx = rng.uniform(max(2.0, tx * 0.02), max(8.0, tx * 0.06))
        ry = rng.uniform(max(2.0, ty * 0.02), max(8.0, ty * 0.06))
        distance = np.clip(1.0 - (((tx_m - cx) / rx) ** 2 + ((ty_m - cy) / ry) ** 2), 0.0, 1.0)
        height = np.maximum(height, distance * rng.uniform(0.5, 1.0))

    ink = np.clip(height * 0.8 + rng.standard_normal((ty, tx), dtype=np.float32) * 0.05, 0.0, 1.0)
    wash = np.clip(height * 0.5 + 0.2 + rng.standard_normal((ty, tx), dtype=np.float32) * 0.02, 0.0, 1.0)
    audio_buffer = np.zeros((ty, tx), dtype=np.float32)
    tile_field = np.stack([
        ink.astype(np.float32),
        wash.astype(np.float32),
        height.astype(np.float32),
        audio_buffer,
    ], axis=-1)
    return np.repeat(np.repeat(tile_field, ts, axis=0), ts, axis=1)[:h, :w].copy()


SPEC = WorldSpec(
    id="sketchbook-visualizer-3d",
    display_name="Sketchbook Visualizer",
    window_title="Living Sketchbook - 3D Audio Visualizer",
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={"substeps": 8, "ray_steps": 96},
    preview_image="assets/world_previews/sketchbook-visualizer-3d.png",
    stability_notes=("audio reactive", "heavy raymarch", "experimental"),
    hud_subtitle="3D AUDIO VISUALIZER",
    preview_palette=("#f4eee1", "#23201f", "#1857a2", "#6e8ca0", "#e8952d", "#faf5e8"),
    uses_audio=True,
)
