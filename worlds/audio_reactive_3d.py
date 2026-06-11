"""World definition for Audio Reactive 3D."""
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

    float lScale = laplaceScale * (1.0 + audioBass * 0.8);
    float lapU = ((r.r + l.r + t.r + b.r) * 0.2 + (tr.r + tl.r + br.r + bl.r) * 0.05 - c.r) * lScale;
    float lapV = ((r.g + l.g + t.g + b.g) * 0.2 + (tr.g + tl.g + br.g + bl.g) * 0.05 - c.g) * lScale;

    vec2 gradH = vec2(r.b - l.b, t.b - b.b);
    float flowStrength = 0.8 + audioEnergy * 2.0;
    float advectU = dot(gradH, vec2(r.r - l.r, t.r - b.r)) * flowStrength;
    float advectV = dot(gradH, vec2(r.g - l.g, t.g - b.g)) * flowStrength;

    float wetNoise = (hash(uv * resolution + floor(time * 0.35)) - 0.5) * noiseStrength * (1.0 + audioTreble * 15.0);
    float localFeed = feed + c.a * 0.015 * sin(time * 0.1 + uv.x * 20.0) + wetNoise * 0.18 + audioBass * 0.03;
    float localKill = kill + (1.0 - c.a) * 0.01 * cos(time * 0.15 + uv.y * 15.0) + parameterDrift * 0.6 - audioEnergy * 0.01;

    float reaction = c.r * c.g * c.g * (1.0 + audioTreble * 2.0);
    
    float du = (diffU * lapU) - reaction + localFeed * (1.0 - c.r) - advectU;
    float dv = (diffV * lapV) + reaction - (localFeed + localKill) * c.g - advectV;

    float fftSample = texture(audioFft, vec2(uv.x, 0.5)).r;
    float dh = (c.g * 0.01 - 0.002 + wetNoise * 0.015 + fftSample * 0.008) * dt * c.a;

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
uniform int raySteps;
uniform float audioEnergy;
uniform float audioBass;

float hash(vec2 p) { return fract(sin(dot(p, vec2(41.7, 289.1))) * 45758.5453); }

float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}

mat3 setCamera(in vec3 ro, in vec3 ta, float cr) {
    vec3 cw = normalize(ta - ro);
    vec3 cp = vec3(sin(cr), cos(cr), 0.0);
    vec3 cu = normalize(cross(cw, cp));
    vec3 cv = normalize(cross(cu, cw));
    return mat3(cu, cv, cw);
}

float map(in vec3 p, out float matID, out vec4 stateOut) {
    vec2 mapUV = p.xz * 0.04;
    vec4 state = textureLod(stateTex, mapUV, 0.0);

    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float waveDisturb = texture(audioWave, vec2(fract(p.x * 0.1 + time * 0.2), 0.5)).r * audioBass * 0.4;
    float baseHeight = state.b * mix(2.7, 3.9, safeFx) + waveDisturb;
    float biomassExtrusion = smoothstep(0.16, 0.86, state.g) * mix(0.85, 2.15, safeFx) * (1.0 + audioBass);
    float h = baseHeight + biomassExtrusion;

    float detail = sin(p.x * 8.0) * cos(p.z * 8.3) * 0.035 + sin(p.x * 15.0) * cos(p.z * 14.1) * 0.014;
    detail += (noise(p.xz * 2.4 + time * 0.02) - 0.5) * 0.045;
    h += detail * smoothstep(0.2, 0.8, state.b) * safeFx;

    float dTerrain = p.y - h;
    float dWater = p.y - 1.55 + sin(time * 2.0 + p.x * 4.0) * audioBass * 0.15;

    if (dTerrain < dWater) {
        matID = 1.0; stateOut = state; return dTerrain * 0.72;
    } else {
        matID = 0.0; stateOut = state; return dWater * 0.9;
    }
}

vec3 calcNormal(in vec3 p) {
    vec2 e = vec2(0.04, 0); float dummyMat; vec4 dummyState;
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
        res = min(res, 8.0 * h / t); t += clamp(h, 0.05, 0.75);
        if (h < 0.001 || t > 10.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

void main() {
    vec2 p = (-resolution.xy + 2.0 * gl_FragCoord.xy) / resolution.y;
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    int safeRaySteps = clamp(raySteps, 32, 160);

    float camShake = noise(vec2(time * 10.0, 0.0)) * audioEnergy * 0.05;
    float camTime = time * 0.15 * max(cameraSpeed, 0.05);
    vec3 ro = vec3(camTime * 5.0, 7.0 + sin(camTime * 0.5) * 1.2 + camShake, camTime * 4.0);
    vec3 ta = vec3(ro.x + 4.8, 2.1, ro.z + 4.8 + sin(camTime));

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.3) * 0.1 + camShake);
    vec3 rd = ca * normalize(vec3(p.xy, 2.0 - audioBass * 0.2));

    vec3 lightDir = normalize(vec3(0.8, 0.62, -0.4));
    float sun = pow(max(0.0, dot(rd, lightDir)), 220.0);
    float skyRise = smoothstep(-0.2, 0.9, rd.y);
    vec3 skyColor = mix(vec3(0.008, 0.014, 0.032), vec3(0.09, 0.18, 0.31), skyRise);
    skyColor += vec3(1.00, 0.72, 0.36) * sun * (0.8 + safeFx * 0.8);
    
    float auroraFft = texture(audioFft, vec2(abs(p.x), 0.5)).r;
    float aurora = smoothstep(0.74, 0.98, rd.y + 0.12 * sin(p.x * 3.0 + time * 0.22));
    aurora *= 0.5 + 0.5 * sin(p.x * 9.0 + time * 0.65 + colorShift) + auroraFft;
    skyColor += aurora * mix(vec3(0.08, 0.55, 0.46), vec3(0.8, 0.2, 0.5), audioBass) * safeFx;

    float tMax = 50.0; float t = 0.0; float matID = -1.0;
    vec4 state = vec4(0.0); vec3 volumeGlow = vec3(0.0);

    for (int i = 0; i < 160; i++) {
        if (i >= safeRaySteps) break;
        vec3 pos = ro + rd * t;
        float currMat; vec4 currState;
        float h = map(pos, currMat, currState);

        if (currMat == 1.0) {
            float bio = smoothstep(0.2, 0.8, currState.g);
            vec3 emColor = mix(vec3(0.0, 1.0, 0.8), vec3(1.0, 0.2, 0.5), currState.r + audioEnergy);
            volumeGlow += emColor * bio * (0.012 + glow * 0.003) * safeFx / (1.0 + abs(h) * 10.0) * (1.0 + audioBass * 3.0);
        }
        if (h < max(0.003, 0.0015 * t) || t > tMax) {
            matID = currMat; state = currState; break;
        }
        t += clamp(h, 0.035, 0.85);
    }

    vec3 color = skyColor;
    if (t < tMax) {
        vec3 pos = ro + rd * t;
        vec3 nor = calcNormal(pos);

        float occ = clamp(0.5 + 0.5 * nor.y, 0.0, 1.0);
        float sha = calcShadow(pos + nor * 0.01, lightDir);
        float dif = clamp(dot(nor, lightDir), 0.0, 1.0) * sha;
        float fre = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 2.0);

        vec3 matColor;
        if (matID == 1.0) {
            vec3 rock = vec3(0.1, 0.12, 0.15); vec3 sand = vec3(0.3, 0.25, 0.18);
            vec3 bio = mix(vec3(0.05, 0.2, 0.15), vec3(0.8, 1.0, 0.9), state.g);
            matColor = mix(rock, sand, smoothstep(0.4, 0.6, nor.y));
            matColor = mix(matColor, bio, smoothstep(0.1, 0.5, state.g));

            vec3 bioGlow = mix(vec3(0.0, 0.9, 0.7), vec3(1.0, 0.13, 0.42), state.r);
            float pulse = 1.0 + 0.5 * sin(time * 3.0 - pos.x + colorShift * 3.0) + audioBass * 2.0;
            matColor += bioGlow * pow(state.g, 3.0) * (1.6 + glow * 0.65) * pulse * safeFx;
            matColor += vec3(1.0, 0.72, 0.22) * pow(state.a, 2.0) * (1.0 + glow * 0.45) * safeFx;
            float contour = 1.0 - smoothstep(0.018, 0.045 + contourContrast * 0.03, abs(fract(state.b * 18.0) - 0.5));
            matColor += contour * vec3(0.08, 0.12, 0.11) * contourContrast;
        } else {
            float depth = clamp((1.55 - state.b * 3.7) * 0.5, 0.0, 1.0);
            matColor = mix(vec3(0.0, 0.4, 0.5), vec3(0.0, 0.05, 0.15), depth);
            matColor += vec3(0.05, 0.24, 0.22) * (0.5 + 0.5 * sin(pos.x * 3.0 + pos.z * 2.0 + time * 1.3)) * safeFx;
            vec3 ref = reflect(rd, nor);
            float spe = pow(clamp(dot(ref, lightDir), 0.0, 1.0), 32.0) * sha;
            matColor += vec3(1.0) * spe * 0.5 + volumeGlow * 0.18;
        }

        vec3 lin = 2.15 * dif * vec3(1.0, 0.88, 0.76) + 0.55 * clamp(0.5 + 0.5 * nor.y, 0.0, 1.0) * vec3(0.20, 0.32, 0.45) * occ + 0.2 * fre;
        color = mix(matColor * lin, skyColor, 1.0 - exp(-0.0016 * t * t));
    }

    color += volumeGlow * (0.8 + safeFx * 0.45);
    
    // Chromatic aberration from audio
    float caShift = audioBass * 0.015;
    color.r += texture(stateTex, uv + vec2(caShift, 0)).r * caShift;
    color.b += texture(stateTex, uv - vec2(caShift, 0)).b * caShift;

    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(color, vec3(1.0 / max(gamma, 0.2)));
    vec2 q = gl_FragCoord.xy / resolution.xy;
    color *= 0.5 + 0.5 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.25);

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
    for _ in range(max(5, (tx * ty) // 4000)):
        cx, cy = rng.uniform(0.0, tx), rng.uniform(0.0, ty)
        rx = rng.uniform(max(5.0, tx * 0.04), max(12.0, tx * 0.18))
        ry = rng.uniform(max(5.0, ty * 0.04), max(12.0, ty * 0.18))
        ht += np.clip(1.0 - (((tx_m - cx) / rx) ** 2 + ((ty_m - cy) / ry) ** 2), 0.0, 1.0) * rng.uniform(0.10, 0.24)
    for _ in range(max(4, max(5, (tx * ty) // 4000) // 2)):
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
    id='audio-reactive-3d',
    display_name='Audio Reactive 3D',
    window_title='Garage Life Lab - Audio-Reactive 3D Bio-World',
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={},
    preview_image='assets/world_previews/audio-reactive-3d.png',
    stability_notes=('audio reactive', 'heavy raymarch', 'experimental'),
    hud_subtitle='AUDIO REACTIVE 3D',
    preview_palette=('#040710', '#0b1d33', '#0f5266', '#14b8a6', '#4df5c8', '#b34cff', '#ff5ab3'),
    uses_audio=True,
)
