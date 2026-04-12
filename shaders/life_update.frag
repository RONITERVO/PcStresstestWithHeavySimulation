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
