"""World definition for Sketchbook Ink Islands."""
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

// Elegant procedural curl noise for fluid advection
vec2 hash2(vec2 p) {
    p = vec2(dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)));
    return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(dot(hash2(i + vec2(0.0,0.0)), f - vec2(0.0,0.0)),
                   dot(hash2(i + vec2(1.0,0.0)), f - vec2(1.0,0.0)), u.x),
               mix(dot(hash2(i + vec2(0.0,1.0)), f - vec2(0.0,1.0)),
                   dot(hash2(i + vec2(1.0,1.0)), f - vec2(1.0,1.0)), u.x), u.y);
}

vec2 curlNoise(vec2 p) {
    float e = 0.05;
    float dx = noise(p + vec2(e, 0.0)) - noise(p - vec2(e, 0.0));
    float dy = noise(p + vec2(0.0, e)) - noise(p - vec2(0.0, e));
    return vec2(dy, -dx);
}

void main() {
    vec2 texel = 1.0 / resolution;
    
    // Buffer Map:
    // R: Ink/Graphite Density (Gray-Scott U)
    // G: Watercolor Wash Density (Gray-Scott V)
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

    float lScale = laplaceScale * (1.0 + audioBass * 0.8);
    float lapU = ((r.r + l.r + t.r + b.r) * 0.2 + (tr.r + tl.r + br.r + bl.r) * 0.05 - c.r) * lScale;
    float lapV = ((r.g + l.g + t.g + b.g) * 0.2 + (tr.g + tl.g + br.g + bl.g) * 0.05 - c.g) * lScale;

    // Ink flows dynamically based on terrain gradients and curl noise
    vec2 gradH = vec2(r.b - l.b, t.b - b.b);
    vec2 curl = curlNoise(uv * 15.0 + time * 0.1) * (0.5 + audioTreble * 2.0);
    
    float flowStrength = 0.5 + audioEnergy * 2.0;
    float advectU = dot(gradH + curl * 0.1, vec2(r.r - l.r, t.r - b.r)) * flowStrength;
    float advectV = dot(gradH - curl * 0.1, vec2(r.g - l.g, t.g - b.g)) * flowStrength;

    float wetNoise = (hash(uv * resolution + floor(time * 0.35)) - 0.5) * noiseStrength * (1.0 + audioTreble * 8.0);
    
    float localFeed = feed + c.a * 0.02 * sin(time * 0.2 + uv.x * 30.0) + wetNoise * 0.2 + audioBass * 0.05;
    float localKill = kill + (1.0 - c.a) * 0.015 * cos(time * 0.25 + uv.y * 20.0) + parameterDrift * 0.8 - audioEnergy * 0.025;

    float reaction = c.r * c.g * c.g * (1.0 + audioTreble * 2.0);
    
    float du = (diffU * lapU) - reaction + localFeed * (1.0 - c.r) - advectU;
    float dv = (diffV * lapV) + reaction - (localFeed + localKill) * c.g - advectV;

    float da = (audioEnergy * 0.15 - c.a * 0.05) * dt; 

    fragColor = vec4(
        clamp(c.r + du * dt, 0.0, 1.0),
        clamp(c.g + dv * dt, 0.0, 1.0),
        c.b, // Static terrain
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

// --- Living Sketchbook Aesthetic Palette ---
const vec3 PAPER = vec3(0.965, 0.941, 0.898);
const vec3 INK_GRAPHITE = vec3(0.125, 0.110, 0.106);
const vec3 INK_BLUEPRINT = vec3(0.082, 0.275, 0.620);
const vec3 SUN_INK = vec3(0.941, 0.550, 0.141);
const vec3 WATER_WASH = vec3(0.400, 0.588, 0.655);

// --- Math & Noise ---
float hash(vec2 p) { return fract(sin(dot(p, vec2(41.7, 289.1))) * 45758.5453); }

vec2 hash2(vec2 p) {
    p = vec2(dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)));
    return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(dot(hash2(i + vec2(0.0,0.0)), f - vec2(0.0,0.0)),
                   dot(hash2(i + vec2(1.0,0.0)), f - vec2(1.0,0.0)), u.x),
               mix(dot(hash2(i + vec2(0.0,1.0)), f - vec2(0.0,1.0)),
                   dot(hash2(i + vec2(1.0,1.0)), f - vec2(1.0,1.0)), u.x), u.y);
}

float fbm(vec2 p) {
    float v = 0.0; float a = 0.5;
    mat2 rot = mat2(0.866, -0.5, 0.5, 0.866);
    for (int i = 0; i < 5; i++) {
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

// Emulate Photoshop Multiply Blend
vec3 blendMultiply(vec3 base, vec3 blend, float opacity) {
    return mix(base, base * blend, opacity);
}

// Emulate Color Burn
vec3 blendColorBurn(vec3 base, vec3 blend, float opacity) {
    vec3 burnt = 1.0 - (1.0 - base) / max(blend, 0.001);
    return mix(base, clamp(burnt, 0.0, 1.0), opacity);
}

float getWaveHeight(vec2 p, float t) {
    float baseWave = sin(p.x * 0.4 + t) * cos(p.y * 0.25 + t * 0.8) * 0.25;
    float audioDisplacement = texture(audioWave, vec2(fract(p.x * 0.03 + p.y * 0.03 + t * 0.1), 0.5)).r;
    return baseWave + audioDisplacement * audioBass * 1.2;
}

// Raymarching Distance Field
float map(in vec3 p, out float matID, out vec4 stateOut) {
    vec2 mapUV = fract(p.xz * 0.015); // Scale terrain mapping
    vec4 state = textureLod(stateTex, mapUV, 0.0);
    
    float frameTime = floor(time * 12.0) / 12.0;

    // Ocean surface
    float wave = getWaveHeight(p.xz, frameTime * fxIntensity);
    float dWater = p.y - (-1.0 + wave);

    // Island Terrain
    float islandBase = state.b * 4.5; 
    float rough = fbm(p.xz * 2.5 + frameTime * 0.05) * 0.4;
    float hTerrain = -0.5 + islandBase + rough;
    float dTerrain = p.y - hTerrain;

    if (dTerrain < dWater && state.b > 0.05) {
        matID = 1.0; 
        stateOut = state; 
        return dTerrain * 0.6;
    } else {
        matID = 0.0; 
        stateOut = state; 
        return dWater * 0.75;
    }
}

vec3 calcNormal(in vec3 p) {
    vec2 e = vec2(0.01, 0.0); 
    float dummyMat; vec4 dummyState;
    return normalize(vec3(
        map(p + e.xyy, dummyMat, dummyState) - map(p - e.xyy, dummyMat, dummyState),
        map(p + e.yxy, dummyMat, dummyState) - map(p - e.yxy, dummyMat, dummyState),
        map(p + e.yyx, dummyMat, dummyState) - map(p - e.yyx, dummyMat, dummyState)
    ));
}

// Elegant Soft Shadows
float calcSoftShadow(in vec3 ro, in vec3 rd, float mint, float tmax, float k) {
    float res = 1.0;
    float t = mint;
    float dummyMat; vec4 dummyState;
    for(int i = 0; i < 24; i++) {
        float h = map(ro + rd * t, dummyMat, dummyState);
        float s = clamp(8.0 * h / t, 0.0, 1.0);
        res = min(res, s);
        t += clamp(h, 0.02, 0.2);
        if(res < 0.004 || t > tmax) break;
    }
    return clamp(res, 0.0, 1.0);
}

// Ambient Occlusion for grounded sketch feel
float calcAO(in vec3 pos, in vec3 nor) {
    float occ = 0.0;
    float sca = 1.0;
    float dummyMat; vec4 dummyState;
    for(int i = 0; i < 5; i++) {
        float h = 0.01 + 0.12 * float(i) / 4.0;
        float d = map(pos + h * nor, dummyMat, dummyState);
        occ += (h - d) * sca;
        sca *= 0.95;
        if(occ > 0.35) break;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
}

// Artistic Cross-Hatching Textures
float getHatch(vec2 p, float lum) {
    float h = 0.0;
    float freq1 = 90.0;
    float freq2 = 100.0;
    
    vec2 jp = p + vec2(hash(p + floor(time*12.0)), hash(p - floor(time*12.0))) * 0.003;

    float line1 = sin(jp.x * freq1 + jp.y * freq1);
    float line2 = sin(jp.x * freq2 - jp.y * freq2);
    float line3 = sin(jp.x * freq1 * 1.5 + jp.y * freq1 * 0.5);

    if (lum < 0.7) h += smoothstep(0.1, 0.2, line1);
    if (lum < 0.4) h += smoothstep(0.1, 0.2, line2);
    if (lum < 0.2) h += smoothstep(0.1, 0.2, line3);

    return clamp(h, 0.0, 1.0);
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float frameTime = floor(time * 12.0) / 12.0;

    // Cinematic Camera
    float camShake = noise(vec2(frameTime * 15.0, 0.0)) * audioEnergy * 0.08;
    float camTime = frameTime * 0.12 * max(cameraSpeed, 0.05);
    vec3 ro = vec3(camTime * 6.0, 4.5 + sin(camTime * 0.4) * 2.0 + camShake, camTime * 5.0);
    vec3 ta = vec3(ro.x + 5.0, 0.0, ro.z + 5.0 + sin(camTime * 1.1));

    ro += cameraOffset;
    ta += cameraOffset;

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.25) * 0.15 + camShake);
    vec3 rd = ca * cameraInputRay(p, 1.6); // Wide lens

    vec3 lightDir = normalize(vec3(0.7, 0.5, -0.5));

    // Base Paper
    vec3 color = PAPER;
    
    // Volumetric Sketchy Sun
    float sunDot = dot(rd, lightDir);
    if (sunDot > 0.0) {
        vec2 sunUV = vec2(acos(rd.y), atan(rd.z, rd.x));
        float radial = sunUV.y * 12.0 + frameTime * 0.5;
        float fftSample = texture(audioFft, vec2(fract(abs(sin(radial))), 0.5)).r;
        
        float sunRadius = 0.985 - fftSample * 0.008 - audioBass * 0.015;
        
        float scribble = sin(radial * 20.0 + hash(vec2(frameTime)) * 4.0) * fftSample;
        if (sunDot > sunRadius - 0.06 + scribble * 0.04) {
            float sunGlow = smoothstep(sunRadius - 0.1, 1.0, sunDot);
            color = blendMultiply(color, SUN_INK, sunGlow * fxIntensity * 0.9);
        }
        
        // God rays / Watercolor bleeds in sky
        float rays = pow(sunDot, 12.0) * fbm(rd.xy * 8.0 - time * 0.1);
        color = blendColorBurn(color, SUN_INK, rays * audioEnergy * 0.8);
    }

    float tMax = 40.0; 
    float t = 0.0; 
    float matID = -1.0;
    vec4 state = vec4(0.0); 

    int safeRaySteps = clamp(raySteps, 32, 200);
    for (int i = 0; i < 200; i++) {
        if (i >= safeRaySteps) break;
        vec3 pos = ro + rd * t;
        float currMat; vec4 currState;
        float h = map(pos, currMat, currState);

        if (h < max(0.002, 0.001 * t) || t > tMax) {
            matID = currMat; state = currState; break;
        }
        t += clamp(h, 0.02, 0.8);
    }

    if (t < tMax) {
        vec3 pos = ro + rd * t;
        vec3 nor = calcNormal(pos);

        float sha = calcSoftShadow(pos + nor * 0.02, lightDir, 0.02, 10.0, 8.0);
        float ao = calcAO(pos, nor);
        float dif = clamp(dot(nor, lightDir), 0.0, 1.0);
        float lum = dif * sha * ao;

        // Depth & Normal based Outline Contour
        float edgeNormal = smoothstep(0.0, 0.35 + contourContrast * 0.25, dot(nor, -rd));
        float edgeDepth = exp(-0.05 * t);
        float edge = edgeNormal * edgeDepth;
        
        if (matID == 1.0) {
            // --- Island Terrain ---
            vec3 baseColor = PAPER;
            
            // Simulation textures dictating ink wash
            float wash = state.g * 0.85;
            baseColor = blendMultiply(baseColor, WATER_WASH, wash);
            
            // Cross-Hatch Shading
            float hatch = getHatch(p, lum);
            baseColor = blendMultiply(baseColor, INK_GRAPHITE, hatch * 0.8);
            
            // Thick Graphite Outlines
            baseColor = blendMultiply(baseColor, INK_GRAPHITE, (1.0 - edge) * 1.2);
            
            // Audio-reactive bloom strokes (Gray-Scott 'U' mapping to Sun Ink)
            float energyStroke = smoothstep(0.4, 1.0, state.r) * audioEnergy * 1.5;
            baseColor = blendColorBurn(baseColor, SUN_INK, energyStroke);

            color = baseColor;
            
        } else {
            // --- Blueprint Ocean ---
            vec3 baseColor = PAPER;
            
            float depth = clamp((pos.y - (-1.0)) * 1.5, 0.0, 1.0);
            baseColor = blendMultiply(baseColor, INK_BLUEPRINT, 0.35 + depth * 0.5 + audioTreble * 0.25);
            
            // Specular Reflections
            vec3 ref = reflect(rd, nor);
            float spe = pow(clamp(dot(ref, lightDir), 0.0, 1.0), 32.0);
            float fresnel = pow(1.0 - max(dot(nor, -rd), 0.0), 5.0);
            
            // Hatch specular
            float hatchSpe = getHatch(p + vec2(pos.x, pos.z) * 0.1, 1.0 - (spe + fresnel * 0.5));
            
            baseColor = blendMultiply(baseColor, INK_GRAPHITE, hatchSpe * 0.6);
            baseColor = blendMultiply(baseColor, INK_BLUEPRINT, (1.0 - edge) * 0.4); 

            color = baseColor;
        }

        // Distance watercolor fog fading to paper
        float fog = 1.0 - exp(-0.0015 * t * t);
        color = mix(color, PAPER, fog);
    }

    // Post-Process: Tactile Paper Grain
    float paperGrain = fbm(uv * resolution * 0.5) * 0.05 + hash(uv * resolution + frameTime) * 0.04;
    color -= paperGrain;

    // Chromatic Aberration from Kick Drum
    float caShift = audioBass * 0.012;
    if (caShift > 0.001) {
        vec2 caP = uv;
        color.r = mix(color.r, texture(stateTex, caP + vec2(caShift, 0)).r, caShift * 2.0);
        color.b = mix(color.b, texture(stateTex, caP - vec2(caShift, 0)).b, caShift * 2.0);
    }

    // Tonemapping & Gamma
    color *= exposure * glow;
    color = pow(color, vec3(1.0 / max(gamma, 0.2)));

    // Deep Vignette
    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.5 + 0.5 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.2);

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
    island_count = max(5, (tx * ty) // 3000)
    for _ in range(island_count):
        cx, cy = rng.uniform(0.0, tx), rng.uniform(0.0, ty)
        rx = rng.uniform(max(2.0, tx * 0.02), max(8.0, tx * 0.06))
        ry = rng.uniform(max(2.0, ty * 0.02), max(8.0, ty * 0.06))
        distance = np.clip(1.0 - (((tx_m - cx) / rx) ** 2 + ((ty_m - cy) / ry) ** 2), 0.0, 1.0)
        height = np.maximum(height, distance * rng.uniform(0.5, 1.0))

    ink = np.clip(height * 0.85 + rng.standard_normal((ty, tx), dtype=np.float32) * 0.08, 0.0, 1.0)
    wash = np.clip(height * 0.6 + 0.1 + rng.standard_normal((ty, tx), dtype=np.float32) * 0.05, 0.0, 1.0)
    audio_buffer = np.zeros((ty, tx), dtype=np.float32)
    tile_field = np.stack([
        ink.astype(np.float32),
        wash.astype(np.float32),
        height.astype(np.float32),
        audio_buffer,
    ], axis=-1)
    return np.repeat(np.repeat(tile_field, ts, axis=0), ts, axis=1)[:h, :w].copy()


SPEC = WorldSpec(
    id="sketchbook-ink-islands-3d",
    display_name="Sketchbook Ink Islands",
    window_title="Living Sketchbook - Ink Islands Audio Visualizer",
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={
        "feed": 0.040,
        "kill": 0.065,
        "diff_u": 0.20,
        "diff_v": 0.10,
        "substeps": 10,
        "noise_strength": 0.012,
        "param_drift": 0.005,
        "exposure": 1.30,
        "glow": 1.15,
        "gamma": 1.30,
        "contour_contrast": 0.85,
        "ray_steps": 128,
    },
    preview_image="assets/world_previews/sketchbook-ink-islands-3d.png",
    stability_notes=("audio reactive", "heavy raymarch", "experimental"),
    hud_subtitle="INK ISLANDS AUDIO VISUALIZER",
    preview_palette=("#f6f0e5", "#201c1b", "#15469e", "#6696a7", "#f08c24", "#fffaf0"),
    uses_audio=True,
)
