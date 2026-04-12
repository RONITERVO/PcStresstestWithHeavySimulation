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

    float nightLights = city * (1.0 - globalDay) * (0.45 + glow * 0.55);
    color += nightLights * vec3(1.00, 0.76, 0.35);

    float contourBand = abs(fract(center.x * 14.0) - 0.5);
    float contour = 1.0 - smoothstep(0.14, 0.22 + contourContrast * 0.12, contourBand);
    color += land * contour * vec3(0.08, 0.07, 0.05) * 0.28;

    float edge = min(min(local.x, local.y), min(1.0 - local.x, 1.0 - local.y));
    float gridLine = 1.0 - smoothstep(0.03, 0.10, edge);
    color = mix(color, color * 0.62, gridLine * 0.55);

    color = vec3(1.0) - exp(-color * exposure);
    color = pow(color, vec3(1.0 / gamma));

    fragColor = vec4(color, 1.0);
}
