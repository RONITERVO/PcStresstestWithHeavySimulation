#version 450

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D stateTex;
uniform vec2 resolution;
uniform float exposure;
uniform float glow;
uniform float gamma;
uniform float contourContrast;
uniform float colorShift;
uniform float time;

vec4 gatherScale(vec2 uv, float radiusPx) {
    vec2 px = radiusPx / resolution;
    vec4 acc = texture(stateTex, uv) * 0.18;
    acc += texture(stateTex, uv + vec2(px.x, 0.0)) * 0.10;
    acc += texture(stateTex, uv - vec2(px.x, 0.0)) * 0.10;
    acc += texture(stateTex, uv + vec2(0.0, px.y)) * 0.10;
    acc += texture(stateTex, uv - vec2(0.0, px.y)) * 0.10;
    acc += texture(stateTex, uv + vec2(px.x, px.y)) * 0.06;
    acc += texture(stateTex, uv + vec2(-px.x, px.y)) * 0.06;
    acc += texture(stateTex, uv + vec2(px.x, -px.y)) * 0.06;
    acc += texture(stateTex, uv - vec2(px.x, px.y)) * 0.06;
    acc += texture(stateTex, uv + vec2(px.x * 1.7, px.y * 0.55)) * 0.045;
    acc += texture(stateTex, uv - vec2(px.x * 1.7, px.y * 0.55)) * 0.045;
    acc += texture(stateTex, uv + vec2(-px.x * 0.55, px.y * 1.7)) * 0.045;
    acc += texture(stateTex, uv - vec2(-px.x * 0.55, px.y * 1.7)) * 0.045;
    return acc;
}

float worldHeight(vec4 local, vec4 meso, vec4 macro, vec4 climate) {
    return clamp(
        macro.x * 0.82 +
        climate.w * 0.28 +
        meso.z * 0.14 -
        macro.y * 0.54 -
        local.y * 0.12,
        0.0,
        1.0
    );
}

void main() {
    vec4 local = texture(stateTex, v_uv);
    vec4 meso = gatherScale(v_uv, 8.0);
    vec4 macro = gatherScale(v_uv + vec2(local.w - 0.5, local.z - 0.5) * 0.04, 34.0);
    vec4 climate = gatherScale(v_uv + vec2(time * 0.0012, -time * 0.0008), 92.0);
    vec4 weather = gatherScale(
        v_uv + vec2(time * 0.0025, -time * 0.0018) + (macro.xy - 0.5) * 0.06,
        16.0
    );

    float height = worldHeight(local, meso, macro, climate);
    float seaLevel = 0.47 + sin(time * 0.015 + colorShift) * 0.01;
    float water = 1.0 - smoothstep(seaLevel - 0.025, seaLevel + 0.02, height);
    float land = 1.0 - water;
    float shore = land * (1.0 - smoothstep(0.0, 0.04, abs(height - seaLevel)));

    float humidity = clamp(meso.y * 0.55 + local.w * 0.35 + climate.z * 0.25, 0.0, 1.0);
    float fertility = clamp(
        local.y * 0.70 +
        meso.z * 0.50 +
        (1.0 - abs(height - seaLevel) * 2.2) * 0.18,
        0.0,
        1.0
    );
    float mountain = land * smoothstep(0.64, 0.94, height + meso.x * 0.10);
    float snow = mountain * smoothstep(0.78, 0.98, height + climate.w * 0.12);
    float forest = land * (1.0 - mountain) * smoothstep(0.34, 0.78, humidity) * smoothstep(0.22, 0.68, fertility);
    float desert = land * (1.0 - mountain) * smoothstep(0.45, 0.90, 1.0 - humidity) * smoothstep(0.36, 0.80, height);
    float grass = clamp(land - mountain - forest - desert, 0.0, 1.0);

    vec2 px = 1.0 / resolution;
    float gradX = texture(stateTex, v_uv + vec2(px.x * 4.0, 0.0)).x - texture(stateTex, v_uv - vec2(px.x * 4.0, 0.0)).x;
    float gradY = texture(stateTex, v_uv + vec2(0.0, px.y * 4.0)).x - texture(stateTex, v_uv - vec2(0.0, px.y * 4.0)).x;
    vec3 normal = normalize(vec3(-gradX * 5.0, -gradY * 5.0, 1.0));

    float sunAngle = time * 0.035;
    float sunHeight = 0.12 + 0.88 * (0.5 + 0.5 * sin(time * 0.02));
    vec3 lightDir = normalize(vec3(cos(sunAngle), 0.35 * sin(sunAngle * 0.6), sunHeight));
    float diffuse = 0.25 + 0.75 * max(dot(normal, lightDir), 0.0);
    float globalDay = smoothstep(0.20, 0.55, sunHeight);
    float ambient = mix(0.24, 0.55, globalDay);

    vec3 oceanAccent = 0.04 * cos(vec3(0.0, 0.9, 1.8) + colorShift);
    vec3 deepOcean = vec3(0.03, 0.12, 0.22) + oceanAccent * vec3(0.5, 0.7, 0.9);
    vec3 shallowOcean = vec3(0.08, 0.35, 0.42) + oceanAccent * vec3(0.4, 0.5, 0.7);
    vec3 shoreFoam = vec3(0.74, 0.84, 0.79);
    vec3 grassColor = vec3(0.27, 0.44, 0.18);
    vec3 forestColor = vec3(0.10, 0.27, 0.12);
    vec3 desertColor = vec3(0.63, 0.52, 0.29);
    vec3 mountainColor = vec3(0.42, 0.37, 0.32);
    vec3 snowColor = vec3(0.86, 0.89, 0.91);

    vec3 waterColor = mix(deepOcean, shallowOcean, shore + humidity * 0.12);
    vec3 landColor =
        grass * grassColor +
        forest * forestColor +
        desert * desertColor +
        mountain * mountainColor;
    landColor = mix(landColor, snowColor, snow);

    vec3 color = waterColor * water + landColor * land;
    color += shoreFoam * shore * 0.45;
    color *= ambient + diffuse * 0.65;

    float contourBand = abs(fract(height * 18.0) - 0.5);
    float contour = 1.0 - smoothstep(0.12, 0.18 + contourContrast * 0.14, contourBand);
    vec3 contourColor = mix(vec3(0.03, 0.05, 0.08), vec3(0.17, 0.14, 0.10), land);
    color += contourColor * contour * 0.22;

    float clouds = smoothstep(0.58, 0.88, weather.y + climate.w * 0.25 + local.z * 0.35);
    color = mix(color, vec3(0.93, 0.95, 0.97), clouds * 0.55);

    float settlement = land * (1.0 - mountain) * smoothstep(0.18, 0.58, local.z + meso.z * 0.6) * smoothstep(0.25, 0.75, fertility);
    float lights = settlement * (1.0 - globalDay) * (0.35 + glow * 0.65);
    color += lights * vec3(1.0, 0.72, 0.35);
    color += land * local.z * glow * vec3(0.10, 0.18, 0.05) * 0.30;

    float vignette = smoothstep(1.30, 0.25, dot(v_uv * 2.0 - 1.0, v_uv * 2.0 - 1.0));
    color *= 0.75 + 0.25 * vignette;

    color = vec3(1.0) - exp(-color * exposure);
    color = pow(color, vec3(1.0 / gamma));

    fragColor = vec4(color, 1.0);
}
