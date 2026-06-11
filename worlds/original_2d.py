"""World definition for Original 2D."""
from __future__ import annotations

import numpy as np

from .spec import WorldSpec

SIM_FRAG_SHADER = r"""
#version 450

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D stateTex;
uniform vec2 resolution;
uniform float tileSize;
uniform float feed;
uniform float kill;
uniform float diffU;
uniform float diffV;
uniform float dt;
uniform float laplaceScale;
uniform float noiseStrength;
uniform float parameterDrift;
uniform float time;

vec2 tileMetrics() {
    return vec2(max(tileSize, 2.0));
}

vec2 tileCount() {
    return max(vec2(1.0), ceil(resolution / tileMetrics()));
}

vec2 tileCenter(vec2 tileIndex) {
    vec2 count = tileCount();
    vec2 wrapped = mod(tileIndex + count, count);
    return ((wrapped + 0.5) * tileMetrics()) / resolution;
}

vec4 tileSample(vec2 tileIndex) {
    return texture(stateTex, tileCenter(tileIndex));
}

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 metrics = tileMetrics();
    vec2 grid = tileCount();
    vec2 tileIndex = floor((gl_FragCoord.xy - 0.5) / metrics);

    vec4 center = tileSample(tileIndex);
    vec4 north = tileSample(tileIndex + vec2(0.0, 1.0));
    vec4 south = tileSample(tileIndex + vec2(0.0, -1.0));
    vec4 east = tileSample(tileIndex + vec2(1.0, 0.0));
    vec4 west = tileSample(tileIndex + vec2(-1.0, 0.0));
    vec4 northeast = tileSample(tileIndex + vec2(1.0, 1.0));
    vec4 northwest = tileSample(tileIndex + vec2(-1.0, 1.0));
    vec4 southeast = tileSample(tileIndex + vec2(1.0, -1.0));
    vec4 southwest = tileSample(tileIndex + vec2(-1.0, -1.0));

    vec4 crossAvg = (north + south + east + west) * 0.25;
    vec4 cornerAvg = (northeast + northwest + southeast + southwest) * 0.25;
    vec4 neighborAvg = mix(crossAvg, cornerAvg, 0.35);

    float seaLevel = 0.46 + sin(time * 0.008 + parameterDrift * 20.0) * 0.008;
    float ocean = 1.0 - smoothstep(seaLevel - 0.02, seaLevel + 0.025, center.x);
    float land = 1.0 - ocean;
    float mountain = smoothstep(0.68, 0.92, center.x);

    float latitude = 1.0 - abs(((tileIndex.y + 0.5) / grid.y) * 2.0 - 1.0);
    float seasonSwing = 0.12 + parameterDrift * 18.0;
    float season = 0.5 + 0.5 * sin(time * 0.018 + latitude * 3.1);
    float temperature = clamp(
        0.18 +
        latitude * 0.60 +
        season * seasonSwing -
        mountain * 0.38 -
        max(center.x - seaLevel, 0.0) * 0.12,
        0.0,
        1.0
    );

    vec2 wind = normalize(vec2(
        sin(time * 0.017 + latitude * 4.0 + center.x * 6.0),
        cos(time * 0.019 - latitude * 5.0 + center.y * 4.0)
    ));
    vec4 upstream = tileSample(tileIndex - wind * 1.4);

    float wetNoise = (hash(tileIndex * 1.73 + floor(time * 0.3)) - 0.5) * noiseStrength;
    float evap = ocean * (0.03 + feed * 1.6 + temperature * 0.05);
    float rain = land * clamp(upstream.y * 0.10 + neighborAvg.y * 0.08 + mountain * 0.06, 0.0, 0.22);
    float moistureMix = 0.16 + diffV * 0.55 + laplaceScale * 0.04;
    float nextMoisture = center.y + dt * ((mix(neighborAvg.y, upstream.y, 0.45) - center.y) * moistureMix);
    nextMoisture += dt * (evap - rain - center.w * (0.010 + kill * 0.10) + wetNoise * 0.8);
    nextMoisture = clamp(nextMoisture, 0.0, 1.0);

    float fertility = land
        * (1.0 - mountain)
        * smoothstep(0.22, 0.74, nextMoisture)
        * smoothstep(0.14, 0.78, temperature)
        * (1.0 - smoothstep(0.72, 0.95, center.x));
    float dryness = smoothstep(0.20, 0.56, 1.0 - nextMoisture);
    float nextBiomass = center.z + dt * ((mix(neighborAvg.z, upstream.z, 0.35) - center.z) * 0.15);
    nextBiomass += dt * (fertility * (0.04 + nextMoisture * 0.04 + feed * 0.7));
    nextBiomass -= dt * (center.z * (0.018 + dryness * 0.05 + mountain * 0.04 + center.w * 0.045));
    nextBiomass += dt * (wetNoise * 0.25);
    nextBiomass = clamp(nextBiomass, 0.0, 1.0);

    float portBias = land * (1.0 - smoothstep(0.05, 0.15, abs(center.x - seaLevel)));
    float settlementSupport = land
        * (1.0 - mountain)
        * smoothstep(0.18, 0.68, nextBiomass)
        * smoothstep(0.18, 0.72, nextMoisture)
        * smoothstep(0.16, 0.72, temperature);
    float floodRisk = smoothstep(0.82, 1.0, nextMoisture) * 0.06;
    float settlementDrift = (hash(tileIndex + vec2(91.7, 47.2) + floor(time * 0.2)) - 0.5) * noiseStrength * 0.4;
    float nextSettlement = center.w + dt * ((neighborAvg.w - center.w) * 0.18);
    nextSettlement += dt * (settlementSupport * (0.035 + portBias * 0.03 + neighborAvg.w * 0.03));
    nextSettlement -= dt * (center.w * (0.025 + mountain * 0.05 + floodRisk + (1.0 - nextBiomass) * 0.05 + kill * 0.08));
    nextSettlement += dt * (portBias * 0.012 + settlementDrift);
    nextSettlement = clamp(nextSettlement, 0.0, 1.0);

    float terrainDrift = (hash(tileIndex + vec2(13.1, 71.9)) - 0.5) * noiseStrength * 0.01;
    float nextHeight = center.x + dt * ((neighborAvg.x - center.x) * (diffU * 0.018) + terrainDrift);
    nextHeight = clamp(nextHeight, 0.0, 1.0);

    fragColor = vec4(nextHeight, nextMoisture, nextBiomass, nextSettlement);
}
"""

DISPLAY_FRAG_SHADER = r"""
#version 450

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D stateTex;
uniform vec2 resolution;
uniform float tileSize;
uniform float exposure;
uniform float glow;
uniform float gamma;
uniform float contourContrast;
uniform float colorShift;
uniform float time;

vec2 tileMetrics() {
    return vec2(max(tileSize, 2.0));
}

vec2 tileCount() {
    return max(vec2(1.0), ceil(resolution / tileMetrics()));
}

vec2 tileCenter(vec2 tileIndex) {
    vec2 count = tileCount();
    vec2 wrapped = mod(tileIndex + count, count);
    return ((wrapped + 0.5) * tileMetrics()) / resolution;
}

vec4 tileSample(vec2 tileIndex) {
    return texture(stateTex, tileCenter(tileIndex));
}

float within(float value, float lo, float hi) {
    return step(lo, value) * step(value, hi);
}

float rect(vec2 p, vec2 lo, vec2 hi) {
    return within(p.x, lo.x, hi.x) * within(p.y, lo.y, hi.y);
}

float peak(vec2 p, float cx, float baseY, float halfWidth, float height) {
    float dx = abs(p.x - cx) / max(halfWidth, 0.001);
    float topY = baseY + height * (1.0 - dx);
    return step(dx, 1.0) * step(baseY, p.y) * step(p.y, topY);
}

void main() {
    vec2 metrics = tileMetrics();
    vec2 grid = tileCount();
    vec2 tileIndex = floor((gl_FragCoord.xy - 0.5) / metrics);
    vec2 local = fract((gl_FragCoord.xy - 0.5) / metrics);

    vec4 center = tileSample(tileIndex);
    vec4 north = tileSample(tileIndex + vec2(0.0, 1.0));
    vec4 south = tileSample(tileIndex + vec2(0.0, -1.0));
    vec4 east = tileSample(tileIndex + vec2(1.0, 0.0));
    vec4 west = tileSample(tileIndex + vec2(-1.0, 0.0));

    float seaLevel = 0.46 + sin(time * 0.008 + colorShift * 0.2) * 0.008;
    float water = 1.0 - smoothstep(seaLevel - 0.02, seaLevel + 0.025, center.x);
    float land = 1.0 - water;
    float shore = land * (1.0 - smoothstep(0.0, 0.05, abs(center.x - seaLevel)));
    float mountain = land * smoothstep(0.68, 0.92, center.x);

    float latitude = 1.0 - abs(((tileIndex.y + 0.5) / grid.y) * 2.0 - 1.0);
    float season = 0.5 + 0.5 * sin(time * 0.018 + latitude * 3.1);
    float temperature = clamp(
        0.18 +
        latitude * 0.60 +
        season * 0.18 -
        mountain * 0.40 -
        max(center.x - seaLevel, 0.0) * 0.12,
        0.0,
        1.0
    );
    float snow = mountain * smoothstep(0.42, 0.82, 1.0 - temperature);

    float humidity = clamp(center.y, 0.0, 1.0);
    float biomass = clamp(center.z, 0.0, 1.0);
    float settlement = clamp(center.w, 0.0, 1.0);

    float forest = land * (1.0 - mountain) * smoothstep(0.50, 0.82, humidity) * smoothstep(0.24, 0.72, biomass);
    float desert = land * (1.0 - mountain) * smoothstep(0.38, 0.82, temperature) * smoothstep(0.38, 0.82, 1.0 - humidity);
    float city = land * (1.0 - mountain) * smoothstep(0.26, 0.72, settlement + biomass * 0.20);
    float fields = clamp(land - mountain - forest - desert, 0.0, 1.0) * smoothstep(0.20, 0.60, biomass);

    float slope = (east.x - west.x) * 0.5 + (north.x - south.x) * 0.5;
    vec3 normal = normalize(vec3((west.x - east.x) * 5.0, (south.x - north.x) * 5.0, 1.0));
    float sunAngle = time * 0.022;
    float sunHeight = 0.15 + 0.85 * (0.5 + 0.5 * sin(time * 0.014));
    vec3 lightDir = normalize(vec3(cos(sunAngle), sin(sunAngle) * 0.35, sunHeight));
    float diffuse = 0.28 + 0.72 * max(dot(normal, lightDir), 0.0);
    float globalDay = smoothstep(0.18, 0.52, sunHeight);
    float ambient = mix(0.22, 0.58, globalDay);

    vec3 deepWater = vec3(0.04, 0.16, 0.34);
    vec3 shallowWater = vec3(0.10, 0.32, 0.50);
    vec3 coastColor = vec3(0.76, 0.70, 0.50);
    vec3 plainsColor = vec3(0.40, 0.58, 0.26);
    vec3 forestColor = vec3(0.14, 0.33, 0.16);
    vec3 desertColor = vec3(0.70, 0.58, 0.34);
    vec3 mountainColor = vec3(0.42, 0.40, 0.38);
    vec3 snowColor = vec3(0.90, 0.93, 0.96);
    vec3 cityColor = vec3(0.53, 0.56, 0.60);

    float waterDepth = smoothstep(seaLevel - 0.10, seaLevel + 0.01, center.x);
    vec3 color = mix(deepWater, shallowWater, waterDepth);
    vec3 landColor = mix(plainsColor, forestColor, forest);
    landColor = mix(landColor, desertColor, desert);
    landColor = mix(landColor, mountainColor, mountain);
    landColor = mix(landColor, cityColor, city * 0.75);
    landColor = mix(landColor, snowColor, snow);
    landColor = mix(landColor, coastColor, shore * 0.9);
    color = mix(color, landColor, land);
    color *= ambient + diffuse * 0.65;
    color += slope * 0.03;

    float waveA = 1.0 - smoothstep(0.04, 0.08, abs(local.y - (0.34 + 0.08 * sin(local.x * 6.283 + time * 1.2 + tileIndex.x * 0.4))));
    float waveB = 1.0 - smoothstep(0.04, 0.08, abs(local.y - (0.62 + 0.07 * sin(local.x * 6.283 + time * 1.4 + tileIndex.y * 0.5))));
    float waterGlyph = water * max(waveA, waveB) * 0.7;

    float forestGlyph = forest * (
        peak(local, 0.28, 0.22, 0.14, 0.36) +
        peak(local, 0.52, 0.18, 0.16, 0.42) +
        peak(local, 0.76, 0.24, 0.13, 0.34)
    );

    float mountainGlyph = mountain * (
        peak(local, 0.36, 0.18, 0.22, 0.52) +
        peak(local, 0.68, 0.18, 0.18, 0.42)
    );

    float desertGlyph = desert * max(
        1.0 - smoothstep(0.03, 0.07, abs(local.y - (0.36 + 0.06 * sin(local.x * 7.0 + tileIndex.y * 0.4)))),
        1.0 - smoothstep(0.03, 0.07, abs(local.y - (0.64 + 0.05 * sin(local.x * 6.0 + tileIndex.x * 0.5))))
    );

    float fieldGlyph = fields * (
        (1.0 - smoothstep(0.03, 0.07, abs(local.x - 0.33))) +
        (1.0 - smoothstep(0.03, 0.07, abs(local.x - 0.66))) +
        (1.0 - smoothstep(0.03, 0.07, abs(local.y - 0.50)))
    ) * 0.6;

    float cityGlyph = city * (
        rect(local, vec2(0.16, 0.18), vec2(0.30, 0.54)) +
        rect(local, vec2(0.38, 0.18), vec2(0.60, 0.72)) +
        rect(local, vec2(0.68, 0.18), vec2(0.82, 0.44))
    );

    vec3 glyphColor = vec3(0.0);
    glyphColor += waterGlyph * vec3(0.72, 0.87, 0.96);
    glyphColor += forestGlyph * vec3(0.75, 0.91, 0.74);
    glyphColor += mountainGlyph * vec3(0.83, 0.83, 0.86);
    glyphColor += desertGlyph * vec3(0.88, 0.80, 0.58);
    glyphColor += fieldGlyph * vec3(0.74, 0.86, 0.50);
    glyphColor += cityGlyph * vec3(0.90, 0.88, 0.78);
    color = mix(color, glyphColor + color * 0.45, clamp(max(max(waterGlyph, forestGlyph), max(max(mountainGlyph, desertGlyph), max(fieldGlyph, cityGlyph))), 0.0, 1.0));

    float cloudWave = 0.5 + 0.5 * sin(tileIndex.x * 0.55 + tileIndex.y * 0.37 + time * 0.7 + humidity * 3.0);
    float clouds = smoothstep(0.66, 0.92, humidity * 0.72 + cloudWave * 0.28) * (0.35 + 0.65 * land);
    color = mix(color, vec3(0.92, 0.95, 0.97), clouds * 0.32);

    float shoreSparkle = shore
        * (0.45 + 0.55 * sin(local.x * 9.0 + local.y * 5.0 + time * 1.8 + tileIndex.x * 0.17))
        * (0.35 + globalDay * 0.65);
    color += shoreSparkle * vec3(0.11, 0.18, 0.16);

    float nightLights = city * (1.0 - globalDay) * (0.45 + glow * 0.55);
    color += nightLights * vec3(1.00, 0.76, 0.35);
    float cityPulse = city * (0.55 + 0.45 * sin(time * 2.0 + tileIndex.x * 0.8 + tileIndex.y * 0.4));
    color += cityPulse * vec3(0.08, 0.06, 0.03) * (0.35 + glow * 0.25);

    float contourBand = abs(fract(center.x * 14.0) - 0.5);
    float contour = 1.0 - smoothstep(0.14, 0.22 + contourContrast * 0.12, contourBand);
    color += land * contour * vec3(0.08, 0.07, 0.05) * 0.28;

    float edge = min(min(local.x, local.y), min(1.0 - local.x, 1.0 - local.y));
    float gridLine = 1.0 - smoothstep(0.03, 0.10, edge);
    color = mix(color, color * 0.62, gridLine * 0.55);

    vec2 screenUv = gl_FragCoord.xy / resolution;
    float frameDistance = length((screenUv - 0.5) * vec2(resolution.x / resolution.y, 1.0));
    float centerLift = 1.0 - smoothstep(0.34, 0.78, frameDistance);
    color *= mix(0.78, 1.06, centerLift);
    color += vec3(0.015, 0.020, 0.024) * (1.0 - centerLift);

    color = vec3(1.0) - exp(-color * exposure);
    color = pow(color, vec3(1.0 / gamma));

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
    id='original-2d',
    display_name='Original 2D',
    window_title='Garage Life Lab - Tile World',
    sim_shader=SIM_FRAG_SHADER,
    display_shader=DISPLAY_FRAG_SHADER,
    seed_field=seed_field,
    default_overrides={'feed': 0.029, 'kill': 0.057, 'substeps': 12},
    preview_image='assets/world_previews/original-2d.png',
    stability_notes=('safe', 'legacy tile shader'),
    hud_subtitle='TILE WORLD STRESS',
    preview_palette=('#06101f', '#0b3143', '#106569', '#1db38b', '#80e0b5', '#ff7ba5', '#ffe083'),
)
