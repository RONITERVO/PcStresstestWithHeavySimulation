"""World definition for Sahara Sandstorm."""
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

// PRNG
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
    // R (x): Deep Moisture/Oasis Reserve (U)
    // G (y): Cacti & Desert Flora (V)
    // B (z): Topography/Sand Dunes
    // A (w): Geothermal Heat & Cosmic Energy

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

    vec2 gradH = vec2(r.z - l.z, t.z - b_.z);

    // Wind Advection: broad desert winds turn slowly over time.
    float windAngle = 0.35 + sin(time * 0.037) * 0.72 + sin(time * 0.013 + 1.7) * 0.38;
    vec2 windDir = normalize(vec2(cos(windAngle), sin(windAngle)));
    float flow = 0.8;
    float windErosion = dot(windDir, gradH) * flow;

    // Matter still naturally slips downhill slightly
    float advectU = dot(gradH, vec2(r.x - l.x, t.x - b_.x)) * flow * 0.2;
    float advectV = dot(gradH, vec2(r.y - l.y, t.y - b_.y)) * flow * 0.2;

    // Cosmic Events (Meteors hitting the desert)
    float eventInterval = 8.0;
    float eventId = floor(time / eventInterval);
    float eventLocalTime = fract(time / eventInterval) * eventInterval;
    vec2 mTarget = getEventTarget(eventId, 42.0);
    float mDist = length(uv - mTarget);

    float meteorStrike = 0.0;
    float cratering = 0.0;
    float shockwave = 0.0;

    float strikePulse = exp(-pow((eventLocalTime - 7.08) * 48.0, 2.0));
    float craterCore = 1.0 - smoothstep(0.0, 0.035, mDist);
    float heatCore = 1.0 - smoothstep(0.012, 0.080, mDist);
    float shockCore = 1.0 - smoothstep(0.040, 0.150, mDist);
    float ejectaRing = smoothstep(0.020, 0.040, mDist) * (1.0 - smoothstep(0.040, 0.065, mDist));

    // Deeper craters in sand, massive ejecta ring
    cratering = strikePulse * (-0.050 * craterCore + 0.025 * ejectaRing);
    meteorStrike = strikePulse * 0.060 * heatCore;
    // Shockwave devastates nearby cacti
    shockwave = -c.y * 0.030 * strikePulse * shockCore;

    // Geothermal Activity (Volcanoes)
    float geothermalDrift = (hash12(uv + time * 0.01) - 0.49) * 0.01;
    float heatGain = (c.w > 0.6 ? 0.005 : -0.001) + geothermalDrift;

    float eruptionH = 0.0;
    float eruptionHeat = 0.0;
    float eruptionKill = 0.0;

    vec2 eruptionCell = floor(uv * resolution / 4.0);
    float eruptionSeed = hash12(eruptionCell + floor(time * 0.5));
    float eruptionPulse = smoothstep(0.92, 1.0, c.w) * step(0.998, eruptionSeed);
    eruptionH = 0.015 * eruptionPulse;
    eruptionHeat = 0.050 * eruptionPulse;
    eruptionKill = -c.y * 0.025 * eruptionPulse;

    // Gray-Scott Reaction with ecological coupling
    float reaction = c.x * c.y * c.y;

    // Sahara Desert coupling: Moisture (c.x) is rare, cacti feed slowly, high heat kills
    float desertFeed = feed + (c.x * 0.01) - abs(c.w - 0.2) * 0.008;
    float desertKill = kill + c.w * 0.025;

    float du = (diffU * lapU * laplaceScale) - reaction + desertFeed * (1.0 - c.x) - advectU;
    float dv = (diffV * lapV * laplaceScale) + reaction - (desertFeed + desertKill) * c.y - advectV + shockwave + eruptionKill;

    // Topography shifts slowly via wind erosion and violently via meteor events
    float dh = lapH * 0.015 + (c.y * 0.001 - 0.0005) * dt + cratering + eruptionH - windErosion * dt * 0.05;

    // Heat diffusion and accumulation
    float dw = (0.25 * lapA) + heatGain + meteorStrike + eruptionHeat - (c.x * 0.003);

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
uniform vec3 cameraOffset;
uniform vec2 cameraYawPitch;
uniform float cameraZoom;
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

vec2 rotate2(vec2 v, float a) {
    float s = sin(a);
    float c = cos(a);
    return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

vec2 windField(vec2 xz, out float gust) {
    float regional = fbm(vec2(time * 0.021, 4.7)) * 6.28318;
    regional += sin(time * 0.039) * 0.95 + sin(time * 0.011 + 2.1) * 0.65;
    vec2 baseDir = normalize(vec2(cos(regional), sin(regional)));

    float shear = (fbm(xz * 0.018 + vec2(time * 0.026, -time * 0.011)) - 0.5) * 2.1;
    shear += (fbm(xz * 0.070 - baseDir * time * 0.045) - 0.5) * 0.85;
    vec2 localDir = normalize(rotate2(baseDir, shear));

    float gustBands = fbm(xz * 0.032 - localDir * time * 0.12);
    float gustFront = pow(max(sin(dot(xz, localDir) * 0.050 - time * 0.82 + fbm(xz * 0.010) * 6.0), 0.0), 5.0);
    gust = clamp(0.35 + gustBands * 1.15 + gustFront * 1.1, 0.2, 2.5);
    return localDir;
}

float asteroidDust(vec2 xz, float hTerrain, float safeFx, vec2 windDir, float gust, float eventId, float age, out float front, out float plume) {
    front = 0.0;
    plume = 0.0;
    if (age <= 0.0 || age > 6.0) return 0.0;

    vec2 impact = getEventTarget(eventId, 42.0) / 0.03;
    vec2 crossDir = vec2(-windDir.y, windDir.x);
    float radius = age * mix(4.5, 12.5, clamp(gust * 0.45, 0.0, 1.0));
    float width = 0.9 + age * (0.55 + gust * 0.28);
    float dist = length(xz - impact);
    float shock = exp(-pow((dist - radius) / max(width, 0.25), 2.0));

    vec2 advectedCenter = impact + windDir * (age * age * (2.2 + gust * 2.4));
    vec2 rel = xz - advectedCenter;
    float along = dot(rel, windDir);
    float cross = dot(rel, crossDir);
    float tail = smoothstep(-1.2, 4.5 + age * 1.6, along) * exp(-max(along, 0.0) / (8.0 + age * 6.0));
    float spread = exp(-pow(cross / (1.0 + age * (0.9 + gust * 0.35)), 2.0));
    float curl = fbm(xz * 0.22 - windDir * time * (0.45 + gust * 0.25) + vec2(eventId * 7.3));
    plume = tail * spread * (0.45 + curl * 0.75);

    float terrainPickup = smoothstep(0.18, 3.0, hTerrain);
    float fade = exp(-age * 0.18);
    front = shock * terrainPickup * fade;
    plume *= terrainPickup * fade;

    return (front * 3.4 + plume * 2.6) * safeFx;
}

float sandStormWave(vec2 xz, float hTerrain, float safeFx, out float crest, out float trough) {
    float gust;
    vec2 windDir = windField(xz, gust);
    vec2 crossDir = vec2(-windDir.y, windDir.x);
    float along = dot(xz, windDir);
    float cross = dot(xz, crossDir);
    float travel = along * 0.22 - time * (0.85 + gust * 0.95 + safeFx * 0.18);
    travel += fbm(vec2(cross * 0.045, along * 0.021) + windDir * time * 0.11) * 4.0;

    float cycle = sin(travel);
    crest = pow(max(cycle, 0.0), 3.0);
    trough = pow(max(-sin(travel - 0.45), 0.0), 2.0);

    float broadLift = (fbm(xz * 0.075 - windDir * time * 0.22) - 0.44) * (0.60 + gust * 0.42);
    float saltation = (fbm(xz * 1.7 - windDir * time * (1.5 + gust * 0.8)) - 0.5) * 0.16;
    float rollingFront = pow(max(sin(along * 0.070 - time * (0.45 + gust * 0.38) + fbm(xz * 0.018) * 6.0), 0.0), 7.0);

    float eventInterval = 8.0;
    float eventId = floor(time / eventInterval);
    float eventLocalTime = fract(time / eventInterval) * eventInterval;
    float impactOffset = 7.08;

    float frontA; float plumeA;
    float asteroidMain = asteroidDust(xz, hTerrain, safeFx, windDir, gust, eventId, eventLocalTime - impactOffset, frontA, plumeA);

    float frontB; float plumeB;
    float asteroidCarry = asteroidDust(xz, hTerrain, safeFx, windDir, gust, eventId - 1.0, eventLocalTime + eventInterval - impactOffset, frontB, plumeB);

    float asteroidStorm = asteroidMain + asteroidCarry * 0.75;
    float ambientStorm = crest * (0.55 + gust * 0.55) + rollingFront * (0.75 + gust * 0.38);
    float drawdown = trough * (0.20 + gust * 0.13);

    crest = clamp(max(max(crest * 0.55, rollingFront), max(frontA, frontB) + max(plumeA, plumeB) * 0.45), 0.0, 1.0);
    trough = clamp(gust * 0.34 + max(plumeA, plumeB) * 0.35, 0.0, 1.0);
    return broadLift + saltation + ambientStorm * safeFx - drawdown + asteroidStorm;
}

float map(in vec3 p, out int matID, out vec4 stateOut) {
    vec2 mapUV = p.xz * 0.03;
    vec4 state = textureLod(stateTex, mapUV, 0.0);

    // R: Deep Moisture, G: Cacti, B: Sand Height, A: Heat
    float safeFx = clamp(fxIntensity, 0.2, 1.6);
    float detailFx = smoothstep(0.2, 1.6, safeFx);

    // Terrain definition
    float baseH = state.z * mix(2.6, 3.8, detailFx);

    // Sand Dunes (Fractal ridges)
    float duneNoise = 1.0 - abs(fbm(p.xz * 0.8 + vec2(time * 0.02)) * 2.0 - 1.0);
    float duneRelief = pow(duneNoise, 1.8) * mix(0.4, 1.4, detailFx);

    // Cacti (Sparse, tall spiky displacements based on Biomass G)
    float cactiMask = smoothstep(0.4, 0.8, state.y);
    float spike = fbm(p.xz * 45.0);
    float cactiDetail = smoothstep(0.6, 1.0, spike) * cactiMask * 1.2 * detailFx;

    // Craters and Volcanic Peaks (Displacement based on Heat A)
    float rockyDetail = fbm(p.xz * 6.0) * mix(0.15, 0.45, detailFx) * smoothstep(0.3, 1.0, state.w);

    float continentLift = smoothstep(0.4, 0.7, fbm(p.xz * 0.05 + vec2(19.0, 7.0)));
    float microRelief = (fbm(p.xz * 2.5 + vec2(3.0)) - 0.5) * mix(0.04, 0.15, detailFx);

    float hTerrain = baseH + duneRelief + cactiDetail + rockyDetail + microRelief + continentLift * mix(0.3, 1.2, detailFx);
    float dTerrain = p.y - hTerrain;

    // Sandstorm replaces the ocean logic: sweeping waves of dense dust.
    float crest; float trough;
    float stormLift = sandStormWave(p.xz, hTerrain, safeFx, crest, trough);
    float hStorm = hTerrain + max(0.0, stormLift);

    float dStorm = p.y - hStorm;

    if (dTerrain < dStorm) {
        matID = 1; // Solid Terrain / Cacti
        stateOut = state;
        return dTerrain * 0.6; // Under-relax for terrain details
    } else {
        matID = 0; // Dense Sandstorm Volume
        stateOut = state;
        return dStorm * 0.9;
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

    float camTime = time * 0.12 * max(cameraSpeed, 0.01);
    vec3 ro = vec3(camTime * 6.0, 8.5 + sin(camTime * 0.4) * 1.5, camTime * 5.0);
    vec3 ta = vec3(ro.x + 6.0, 2.5, ro.z + 5.0 + sin(camTime * 0.8) * 2.0);
    ro += cameraOffset;
    ta += cameraOffset;

    mat3 ca = setCamera(ro, ta, sin(camTime * 0.25) * 0.12);
    vec3 rd = ca * cameraInputRay(p, 2.0);

    // Scorching desert sun
    vec3 lightDir = normalize(vec3(0.6, 0.7, -0.4));

    // Meteor Sky Event
    float eventInterval = 8.0;
    float eventId = floor(time / eventInterval);
    float eventLocalTime = fract(time / eventInterval) * eventInterval;

    // Dusty Sahara Sky
    vec3 skyColor = mix(vec3(0.70, 0.45, 0.25), vec3(0.95, 0.75, 0.55), rd.y * 0.5 + 0.5);
    float sun = pow(max(0.0, dot(rd, lightDir)), 80.0);
    skyColor += vec3(1.0, 0.9, 0.7) * sun * safeFx;

    // Meteor entry flash
    if (eventLocalTime > 6.0 && eventLocalTime < 7.5) {
        vec2 mTargetUV = getEventTarget(eventId, 42.0);
        vec3 mTargetWorld = vec3(mTargetUV.x / 0.03, 0.0, mTargetUV.y / 0.03);

        vec3 meteorStart = mTargetWorld + vec3(25.0, 50.0, -25.0);
        vec3 meteorEnd = mTargetWorld;

        float dropPhase = smoothstep(6.0, 7.08, eventLocalTime);
        vec3 mPos = mix(meteorStart, meteorEnd, dropPhase);

        vec3 mDir = normalize(meteorEnd - meteorStart);
        float mProj = max(0.0, dot(rd, normalize(mPos - ro)));

        float flash = exp(-pow(eventLocalTime - 7.08, 2.0) * 120.0);
        skyColor += vec3(1.0, 0.6, 0.2) * pow(mProj, 700.0) * 12.0 * (1.0 - flash);
        skyColor += vec3(1.0, 0.8, 0.5) * flash * 6.0 * safeFx; // Global flash over the desert
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

        // Volumetrics
        if (currMat == 1) {
            // Volcanic Lava / Smoldering Craters (A)
            float heat = smoothstep(0.7, 1.0, currState.w);
            vec3 lavaGlow = vec3(1.0, 0.35, 0.05);
            volumeGlow += lavaGlow * heat * (0.025 + glow * 0.01) * safeFx / (1.0 + abs(h) * 6.0);

            // Heat Haze / Dust near oases (R)
            float oasis = smoothstep(0.3, 0.8, currState.x);
            volumeGlow += vec3(0.7, 0.5, 0.3) * oasis * 0.01 * safeFx / (1.0 + abs(h) * 8.0);
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
        float fre = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 3.0);

        vec3 matColor;
        vec3 emission = vec3(0.0);

        if (matID == 1) {
            // Sahara Sand Base Terrain
            vec3 sandBase = vec3(0.78, 0.56, 0.35);
            vec3 sandShadow = vec3(0.48, 0.30, 0.18);
            matColor = mix(sandShadow, sandBase, smoothstep(0.2, 0.9, nor.y));

            // Cacti (Biomass G)
            vec3 flora = vec3(0.15, 0.42, 0.20);
            float isCactus = smoothstep(0.5, 0.9, state.y) * smoothstep(0.5, 1.0, fbm(pos.xz * 45.0));
            matColor = mix(matColor, flora, isCactus);

            // Burned Earth & Meteor Glass (A)
            vec3 obsidian = vec3(0.12, 0.08, 0.06);
            float heatGlass = smoothstep(0.6, 0.9, state.w);
            matColor = mix(matColor, obsidian, heatGlass);

            // Add shiny glass reflection inside craters
            float glassSpec = pow(clamp(dot(reflect(rd, nor), lightDir), 0.0, 1.0), 64.0) * heatGlass * sha;
            emission += vec3(1.0, 0.8, 0.6) * glassSpec * 1.5;

            // Wind ripples (contour mapping)
            float ripple = (1.0 - smoothstep(0.0, 0.06, abs(fract((pos.x * 0.6 + pos.z * 0.4 + pos.y * 1.5) * 6.0) - 0.5))) * contourPower;
            matColor += vec3(0.18, 0.10, 0.05) * ripple * 0.25;

            // Lava Emission (A)
            float lavaMask = smoothstep(0.85, 1.0, state.w);
            emission += vec3(2.5, 0.7, 0.1) * lavaMask * (1.0 + 0.3 * sin(time * 6.0 + pos.x)) * safeFx;

        } else {
            // Sandstorm Wave
            float waveCrest; float waveTrough;
            float approximateTerrain = state.z * 3.25 + 1.10;
            float stormLift = sandStormWave(pos.xz, approximateTerrain, safeFx, waveCrest, waveTrough);
            float stormDepth = max(0.18, max(stormLift, 0.0));
            float verticalFill = 1.0 - clamp((pos.y - approximateTerrain) / stormDepth, 0.0, 1.0);
            float stormDensity = clamp(verticalFill * 0.82 + waveTrough * 0.22 + waveCrest * 0.38, 0.0, 1.0);

            vec3 dustThick = vec3(0.42, 0.26, 0.14);
            vec3 dustLight = vec3(0.85, 0.65, 0.45);
            matColor = mix(dustThick, dustLight, stormDensity);

            float swirl = smoothstep(0.4, 0.8, fbm(pos.xz * 2.5 + vec2(time * 0.6, -time * 0.3)));
            matColor = mix(matColor, vec3(0.9, 0.75, 0.55), clamp(waveCrest * (0.4 + swirl * 0.6), 0.0, 0.8));

            vec3 ref = reflect(rd, nor);
            float spe = pow(clamp(dot(ref, lightDir), 0.0, 1.0), 8.0) * sha; // Rough, broad specular for dust
            emission += vec3(0.6, 0.5, 0.4) * spe * 0.3 * waveCrest;

            // Ambient scattering within the dust wave
            emission += vec3(0.8, 0.5, 0.3) * waveCrest * safeFx * 0.4;
            emission += volumeGlow * 0.4;
        }

        vec3 lin = vec3(0.0);
        lin += 1.8 * dif * vec3(1.0, 0.9, 0.8);
        lin += 0.8 * sky * vec3(0.5, 0.4, 0.3) * occ; // Warm ambient
        lin += 0.4 * fre * vec3(0.9, 0.8, 0.7) * occ;

        color = matColor * lin + emission;

        // Thick sandy fog in the distance
        float fog = 1.0 - exp(-0.0012 * t * t);
        color = mix(color, skyColor, fog);
    }

    color += volumeGlow * (1.0 + safeFx * 0.5);

    // Post-Processing: ACES Tonemapping
    color *= exposure;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = pow(color, vec3(1.0 / max(gamma, 0.1)));

    // Vignette
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
    tile_y, tile_x = np.meshgrid(
        np.arange(tiles_y, dtype=np.float32),
        np.arange(tiles_x, dtype=np.float32),
        indexing="ij",
    )
    x_norm = tile_x / max(tiles_x - 1, 1)
    y_norm = tile_y / max(tiles_y - 1, 1)

    dunes = 0.45 + 0.20 * np.sin(x_norm * 14.3 + np.cos(y_norm * 8.2) * 2.0)
    dunes += 0.12 * np.sin((y_norm + x_norm) * 9.5)

    height = np.clip(
        dunes
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.025,
        0.0, 1.0
    )

    continent_count = max(5, (tiles_x * tiles_y) // 4000)
    for _ in range(continent_count):
        cx = rng.uniform(0.0, tiles_x)
        cy = rng.uniform(0.0, tiles_y)
        rx = rng.uniform(max(5.0, tiles_x * 0.04), max(12.0, tiles_x * 0.18))
        ry = rng.uniform(max(5.0, tiles_y * 0.04), max(12.0, tiles_y * 0.18))
        distance = ((tile_x - cx) / rx) ** 2 + ((tile_y - cy) / ry) ** 2
        lift = np.clip(1.0 - distance, 0.0, 1.0)
        height += lift * rng.uniform(0.10, 0.30)

    trench_count = max(4, continent_count // 2)
    for _ in range(trench_count):
        cx = rng.uniform(0.0, tiles_x)
        cy = rng.uniform(0.0, tiles_y)
        rx = rng.uniform(max(6.0, tiles_x * 0.05), max(14.0, tiles_x * 0.16))
        ry = rng.uniform(max(6.0, tiles_y * 0.05), max(14.0, tiles_y * 0.16))
        distance = ((tile_x - cx) / rx) ** 2 + ((tile_y - cy) / ry) ** 2
        carve = np.clip(1.0 - distance, 0.0, 1.0)
        height -= carve * rng.uniform(0.10, 0.25)

    height = np.clip(height, 0.0, 1.0)

    sand_sea_level = 0.30
    deep_sand = (height < sand_sea_level).astype(np.float32)
    oasis_areas = np.clip(1.0 - np.abs(height - sand_sea_level) / 0.05, 0.0, 1.0)

    moisture = np.clip(
        0.05
        + deep_sand * 0.30
        + oasis_areas * 0.45
        + rng.standard_normal((tiles_y, tiles_x), dtype=np.float32) * 0.02,
        0.0,
        1.0,
    )

    biomass = np.clip(
        (1.0 - deep_sand)
        * (
            0.02
            + moisture * 0.50
            - np.clip(height - 0.70, 0.0, 1.0) * 0.60
        ),
        0.0,
        1.0,
    )

    geothermal_faults = np.clip(
        np.abs(np.sin(x_norm * 18.0 + np.sin(y_norm * 22.0)) * 0.5) * 2.0, 0.0, 1.0
    )
    heat = np.clip(1.0 - geothermal_faults * 3.0, 0.0, 1.0)

    reaction_u = np.clip(1.0 - biomass * 0.20 + moisture * 0.05, 0.0, 1.0)
    reaction_v = np.clip(biomass * 0.70 + moisture * 0.15, 0.0, 1.0)

    tile_field = np.stack(
        [
            reaction_u.astype(np.float32),
            reaction_v.astype(np.float32),
            height.astype(np.float32),
            heat.astype(np.float32),
        ],
        axis=-1,
    )

    field = np.repeat(np.repeat(tile_field, tile_size, axis=0), tile_size, axis=1)
    field = field[:height_px, :width_px].copy()
    return field


SPEC = WorldSpec(
    id='static-sandstorm-3d',
    display_name='Sahara Sandstorm',
    window_title='Garage Life Lab - 3D Sahara Bio-World',
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={},
    preview_image='assets/world_previews/static-sandstorm-3d.png',
    stability_notes=('heavy raymarch', 'safe'),
    hud_subtitle='SAHARA SANDSTORM',
    preview_palette=('#140d06', '#38220b', '#6c4215', '#b7792d', '#e3b75e', '#f4df9a', '#ffffff'),
)
