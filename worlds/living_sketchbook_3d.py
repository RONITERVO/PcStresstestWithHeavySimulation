"""World definition for Living Sketchbook."""
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
    vec4 c = texture(stateTex, uv);
    vec4 r = texture(stateTex, uv + vec2(texel.x, 0.0));
    vec4 l = texture(stateTex, uv - vec2(texel.x, 0.0));
    vec4 t = texture(stateTex, uv + vec2(0.0, texel.y));
    vec4 b = texture(stateTex, uv - vec2(0.0, texel.y));
    vec4 tr = texture(stateTex, uv + texel);
    vec4 tl = texture(stateTex, uv + vec2(-texel.x, texel.y));
    vec4 br = texture(stateTex, uv + vec2(texel.x, -texel.y));
    vec4 bl = texture(stateTex, uv - texel);

    float lScale = laplaceScale * (1.0 + audioTreble * 2.0);
    float lapU = ((r.r + l.r + t.r + b.r) * 0.2 + (tr.r + tl.r + br.r + bl.r) * 0.05 - c.r) * lScale;
    float lapV = ((r.g + l.g + t.g + b.g) * 0.2 + (tr.g + tl.g + br.g + bl.g) * 0.05 - c.g) * lScale;

    vec2 gradH = vec2(r.b - l.b, t.b - b.b);
    float flowStrength = 1.5 + audioEnergy * 4.0;
    float advectU = dot(gradH, vec2(r.r - l.r, t.r - b.r)) * flowStrength;
    float advectV = dot(gradH, vec2(r.g - l.g, t.g - b.g)) * flowStrength;

    float wetNoise = (hash(uv * resolution + floor(time * 0.35)) - 0.5) * noiseStrength * (1.0 + audioTreble * 5.0);
    float localFeed = feed + c.a * 0.02 * sin(time * 0.2 + uv.x * 15.0) + wetNoise * 0.2 + audioBass * 0.04;
    float localKill = kill + (1.0 - c.a) * 0.015 * cos(time * 0.1 + uv.y * 10.0) + parameterDrift * 0.5 - audioEnergy * 0.015;

    float reaction = c.r * c.g * c.g * (1.0 + audioBass * 1.5);
    
    float du = (diffU * lapU) - reaction + localFeed * (1.0 - c.r) - advectU;
    float dv = (diffV * lapV) + reaction - (localFeed + localKill) * c.g - advectV;

    float fftSample = texture(audioFft, vec2(uv.x, 0.5)).r;
    float dh = (c.g * 0.012 - 0.001 + wetNoise * 0.02 + fftSample * 0.01) * dt * c.a;

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
uniform sampler2D audioFft;
uniform sampler2D audioWave;

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

const vec3 COLOR_PAPER = vec3(0.957, 0.933, 0.882);
const vec3 COLOR_GRAPHITE = vec3(0.137, 0.118, 0.110);
const vec3 COLOR_BLUEPRINT = vec3(0.094, 0.294, 0.647);
const vec3 COLOR_SUN = vec3(0.922, 0.588, 0.176);
const vec3 COLOR_WASH = vec3(0.431, 0.549, 0.627);

float hash(vec2 p) { return fract(sin(dot(p, vec2(41.7, 289.1))) * 45758.5453); }

float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}

float fbm(vec2 p) {
    float f = 0.0; float amp = 0.5;
    for(int i=0; i<4; i++) {
        f += amp * noise(p); p *= 2.02; amp *= 0.5;
    }
    return f;
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

float map(in vec3 p, out float matID, out vec4 stateOut) {
    vec2 mapUV = p.xz * 0.05;
    vec4 state = textureLod(stateTex, mapUV, 0.0);

    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float waveDisturb = texture(audioWave, vec2(fract(p.x * 0.05 + time * 0.1), 0.5)).r * audioBass * 0.6;
    float baseHeight = state.b * mix(2.0, 3.5, safeFx);
    
    // Pigment clustering mapping to height
    float pigmentExtrusion = smoothstep(0.3, 0.9, state.g) * 1.5 * (1.0 + audioBass * 0.5);
    float h = baseHeight + pigmentExtrusion;

    float detail = fbm(p.xz * 2.4 + time * 0.02) * 0.08 * safeFx;
    h += detail * smoothstep(0.1, 0.7, state.b);

    float dTerrain = p.y - h;
    float inkRipples = texture(audioWave, vec2(fract(p.x * 0.1 + p.z * 0.05 - time * 0.3), 0.5)).r * audioTreble * 0.15;
    float dWater = p.y - 1.2 + inkRipples;

    if (dTerrain < dWater) {
        matID = 1.0; stateOut = state; return dTerrain * 0.65;
    } else {
        matID = 0.0; stateOut = state; return dWater * 0.8;
    }
}

vec3 calcNormal(in vec3 p) {
    vec2 e = vec2(0.02, 0); float dummyMat; vec4 dummyState;
    return normalize(vec3(
        map(p + e.xyy, dummyMat, dummyState) - map(p - e.xyy, dummyMat, dummyState),
        map(p + e.yxy, dummyMat, dummyState) - map(p - e.yxy, dummyMat, dummyState),
        map(p + e.yyx, dummyMat, dummyState) - map(p - e.yyx, dummyMat, dummyState)
    ));
}

float calcShadow(in vec3 ro, in vec3 rd) {
    float res = 1.0; float t = 0.1; float dMat; vec4 dSt;
    for (int i = 0; i < 30; i++) {
        float h = map(ro + rd * t, dMat, dSt);
        res = min(res, 8.0 * h / t); t += clamp(h, 0.02, 0.5);
        if (h < 0.001 || t > 10.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

float hatching(vec2 scr, float intensity) {
    float h = 0.0;
    scr *= min(resolution.x, resolution.y) * 0.4;
    float n = noise(scr * 0.1 + time * 2.0) * audioTreble * 2.0;
    if (intensity < 0.6) h += smoothstep(0.3, 0.7, sin(scr.x + scr.y + n));
    if (intensity < 0.4) h += smoothstep(0.3, 0.7, sin(scr.x - scr.y - n));
    if (intensity < 0.2) h += smoothstep(0.3, 0.7, sin(scr.x + n * 2.0));
    return clamp(h, 0.0, 1.0);
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    int safeRaySteps = clamp(raySteps, 32, 160);

    // Audio-driven camera jitter (handheld feel)
    float jitterSpeed = 12.0;
    vec2 camJitter = vec2(noise(vec2(time * jitterSpeed, 0.0)), noise(vec2(0.0, time * jitterSpeed))) - 0.5;
    camJitter *= audioEnergy * 0.08;

    float camTime = time * 0.1 * max(cameraSpeed, 0.05);
    vec3 ro = vec3(camTime * 5.0, 5.0 + sin(camTime * 0.5) * 0.5 + camJitter.y, camTime * 4.0);
    vec3 ta = vec3(ro.x + 4.0, 1.5, ro.z + 4.0 + sin(camTime));
    ro += cameraOffset;
    ta += cameraOffset;

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.3) * 0.05 + camJitter.x);
    vec3 rd = ca * cameraInputRay(p, 1.8 - audioBass * 0.1);

    vec3 lightDir = normalize(vec3(0.6, 0.5, -0.4));
    
    // Background: Paper sky with hand-drawn scribble sun
    vec3 skyColor = COLOR_PAPER;
    
    // Scribble Sun
    float sunAngle = atan(rd.y - 0.15, rd.x - 0.3);
    float sunDist = length(vec2(rd.x - 0.3, rd.y - 0.15));
    float scribbleDisturb = noise(vec2(sunAngle * 15.0, time * 3.0)) * 0.03 * (1.0 + audioBass * 2.0);
    float baseSunRad = 0.08 + audioBass * 0.05;
    float isSun = 1.0 - smoothstep(baseSunRad + scribbleDisturb, baseSunRad + scribbleDisturb + 0.005, sunDist);
    
    // Watercolor sun halo
    float sunHalo = smoothstep(0.3, 0.0, sunDist) * (0.4 + audioBass * 0.4);
    skyColor = mix(skyColor, COLOR_SUN, sunHalo * 0.5);
    skyColor = mix(skyColor, COLOR_SUN, isSun);
    
    // Cloudy watercolor washes in sky
    float skyWash = fbm(rd.xz * 4.0 / max(rd.y, 0.01) + time * 0.05) * audioEnergy;
    skyColor = mix(skyColor, COLOR_WASH, smoothstep(0.4, 0.8, skyWash) * 0.4);

    float tMax = 40.0; float t = 0.0; float matID = -1.0;
    vec4 state = vec4(0.0);

    for (int i = 0; i < 160; i++) {
        if (i >= safeRaySteps) break;
        vec3 pos = ro + rd * t;
        float currMat; vec4 currState;
        float h = map(pos, currMat, currState);

        if (h < max(0.002, 0.001 * t) || t > tMax) {
            matID = currMat; state = currState; break;
        }
        t += clamp(h, 0.02, 0.8);
    }

    vec3 color = skyColor;
    if (t < tMax) {
        vec3 pos = ro + rd * t;
        vec3 nor = calcNormal(pos);

        float sha = calcShadow(pos + nor * 0.01, lightDir);
        float dif = clamp(dot(nor, lightDir), 0.0, 1.0) * sha;
        float occ = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
        
        vec3 matColor;
        float lighting = dif * 0.8 + occ * 0.2;
        float hatchVal = hatching(gl_FragCoord.xy / resolution.xy, lighting);

        if (matID == 1.0) {
            // Terrain: Paper mapped with Graphite lines and ink concentrations
            float pigment = state.g;
            matColor = mix(COLOR_PAPER, COLOR_WASH * 1.5, smoothstep(0.1, 0.5, pigment));
            
            // Edge detection / outlines (Graphite)
            float contour = smoothstep(0.3, 0.7, 1.0 - abs(dot(nor, rd)));
            float highFreqEdge = smoothstep(0.02, 0.05, abs(fract(state.b * 15.0) - 0.5));
            float edgeMask = clamp(contour + (1.0 - highFreqEdge) * contourContrast, 0.0, 1.0);
            
            matColor = mix(matColor, COLOR_GRAPHITE, edgeMask);
            matColor = mix(matColor, COLOR_GRAPHITE, hatchVal * 0.6); // Apply crosshatch

        } else {
            // Water: Blueprint ink wash
            float depth = clamp((1.2 - state.b * 3.0) * 0.5, 0.0, 1.0);
            matColor = mix(COLOR_BLUEPRINT, COLOR_WASH, depth);
            
            // Reflected Scribble Sun
            vec3 ref = reflect(rd, nor);
            float refSunDist = length(vec2(ref.x - 0.3, ref.y - 0.15));
            float refSunWave = noise(vec2(pos.x * 5.0, time * 5.0)) * 0.05 * audioTreble;
            float isRefSun = 1.0 - smoothstep(0.08 + refSunWave, 0.1 + refSunWave, refSunDist);
            matColor = mix(matColor, COLOR_SUN, isRefSun * 0.7 * sha);

            matColor = mix(matColor, COLOR_GRAPHITE, hatchVal * 0.3);
        }

        // Atmospheric perspective - fades to paper
        float fog = 1.0 - exp(-0.002 * t * t);
        color = mix(matColor * (0.4 + lighting * 0.6), skyColor, fog);
    }

    // Global Paper Grain Post-Process
    float grain = fbm(gl_FragCoord.xy * 2.0);
    color *= 0.95 + 0.05 * grain;
    
    // Chromatic aberration / ink bleed from loud audio
    float caShift = audioEnergy * 0.005;
    color.r *= 1.0 - (texture(stateTex, uv + vec2(caShift, 0)).r * caShift);
    color.b *= 1.0 - (texture(stateTex, uv - vec2(caShift, 0)).b * caShift);

    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(color, vec3(1.0 / max(gamma, 0.2)));
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
    ty_m, tx_m = np.meshgrid(np.arange(ty, dtype=np.float32), np.arange(tx, dtype=np.float32), indexing="ij")
    xn, yn = tx_m / max(tx - 1, 1), ty_m / max(ty - 1, 1)

    ht = 0.45 + 0.15 * np.sin(xn * 7.3 + 0.9) + 0.11 * np.cos(yn * 5.7 - 1.2) + rng.standard_normal((ty, tx), dtype=np.float32) * 0.025
    continent_count = max(5, (tx * ty) // 4000)
    for _ in range(continent_count):
        cx, cy = rng.uniform(0.0, tx), rng.uniform(0.0, ty)
        rx = rng.uniform(max(5.0, tx * 0.04), max(12.0, tx * 0.18))
        ry = rng.uniform(max(5.0, ty * 0.04), max(12.0, ty * 0.18))
        ht += np.clip(1.0 - (((tx_m - cx) / rx) ** 2 + ((ty_m - cy) / ry) ** 2), 0.0, 1.0) * rng.uniform(0.10, 0.24)
    for _ in range(max(4, continent_count // 2)):
        cx, cy = rng.uniform(0.0, tx), rng.uniform(0.0, ty)
        rx = rng.uniform(max(6.0, tx * 0.05), max(14.0, tx * 0.16))
        ry = rng.uniform(max(6.0, ty * 0.05), max(14.0, ty * 0.16))
        ht -= np.clip(1.0 - (((tx_m - cx) / rx) ** 2 + ((ty_m - cy) / ry) ** 2), 0.0, 1.0) * rng.uniform(0.08, 0.18)

    ht = np.clip(ht + np.clip(np.sin(xn * 21.0 + np.cos(yn * 9.0) * 2.3) - 0.45, 0.0, 1.0) * 0.08, 0.0, 1.0)
    ocean = (ht < 0.46).astype(np.float32)
    coast = np.clip(1.0 - np.abs(ht - 0.46) / 0.07, 0.0, 1.0)
    latitude = 1.0 - np.abs(yn * 2.0 - 1.0)
    moist = np.clip(0.16 + ocean * 0.52 + coast * 0.22 + latitude * 0.12 + rng.standard_normal((ty, tx), dtype=np.float32) * 0.03, 0.0, 1.0)
    bio = np.clip((1.0 - ocean) * (0.06 + moist * 0.62 + latitude * 0.16 - np.clip(ht - 0.72, 0.0, 1.0) * 0.50), 0.0, 1.0)

    setl = np.zeros((ty, tx), dtype=np.float32)
    cands = np.argwhere((ocean < 0.5) & (coast > 0.35) & (bio > 0.28) & (ht < 0.74))
    if len(cands) > 0:
        city_count = min(max(8, (tx * ty) // 1800), len(cands))
        for idx in rng.choice(len(cands), size=city_count, replace=False):
            cy, cx = cands[idx]
            rad = int(rng.integers(1, 3))
            y0, y1 = max(cy - rad, 0), min(cy + rad + 1, ty)
            x0, x1 = max(cx - rad, 0), min(cx + rad + 1, tx)
            py, px = np.meshgrid(np.arange(y0, y1, dtype=np.float32), np.arange(x0, x1, dtype=np.float32), indexing="ij")
            influence = np.clip(1.0 - np.sqrt((px - cx) ** 2 + (py - cy) ** 2) / max(rad + 0.5, 1.0), 0.0, 1.0)
            setl[y0:y1, x0:x1] = np.maximum(setl[y0:y1, x0:x1], influence * rng.uniform(0.35, 0.78))

    tile_field = np.stack([
        np.clip(1.0 - bio * 0.36 + moist * 0.08, 0.0, 1.0).astype(np.float32),
        np.clip(bio * 0.82 + moist * 0.12, 0.0, 1.0).astype(np.float32),
        ht.astype(np.float32),
        setl.astype(np.float32),
    ], axis=-1)
    return np.repeat(np.repeat(tile_field, ts, axis=0), ts, axis=1)[:h, :w].copy()


SPEC = WorldSpec(
    id='living-sketchbook-3d',
    display_name='Living Sketchbook',
    window_title='Living Sketchbook - Volumetric Ink Audio Reactive',
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={'ray_steps': 110},
    preview_image='assets/world_previews/living-sketchbook-3d.png',
    stability_notes=('audio reactive', 'heavy raymarch', 'experimental'),
    hud_subtitle='3D VOLUMETRIC INK',
    preview_palette=('#f4eee1', '#d7cfc0', '#6e8ca0', '#1857a2', '#23201f', '#e8952d', '#faf5e8'),
    uses_audio=True,
)
