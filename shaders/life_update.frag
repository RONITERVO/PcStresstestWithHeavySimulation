#version 450

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D stateTex;
uniform vec2 resolution;
uniform float feed;
uniform float kill;
uniform float diffU;
uniform float diffV;
uniform float dt;
uniform float laplaceScale;
uniform float noiseStrength;
uniform float parameterDrift;
uniform float time;

vec2 laplaceState(vec2 uv) {
    vec2 px = 1.0 / resolution;
    vec2 center = texture(stateTex, uv).xy;
    vec2 sum = vec2(0.0);
    sum += texture(stateTex, uv + px * vec2(-1.0, 0.0)).xy * 0.2;
    sum += texture(stateTex, uv + px * vec2(1.0, 0.0)).xy * 0.2;
    sum += texture(stateTex, uv + px * vec2(0.0, -1.0)).xy * 0.2;
    sum += texture(stateTex, uv + px * vec2(0.0, 1.0)).xy * 0.2;
    sum += texture(stateTex, uv + px * vec2(-1.0, -1.0)).xy * 0.05;
    sum += texture(stateTex, uv + px * vec2(1.0, -1.0)).xy * 0.05;
    sum += texture(stateTex, uv + px * vec2(-1.0, 1.0)).xy * 0.05;
    sum += texture(stateTex, uv + px * vec2(1.0, 1.0)).xy * 0.05;
    return (sum - center) * laplaceScale;
}

float laplaceClimate(vec2 uv) {
    vec2 px = 1.0 / resolution;
    float center = texture(stateTex, uv).w;
    float sum = 0.0;
    sum += texture(stateTex, uv + px * vec2(-1.0, 0.0)).w * 0.2;
    sum += texture(stateTex, uv + px * vec2(1.0, 0.0)).w * 0.2;
    sum += texture(stateTex, uv + px * vec2(0.0, -1.0)).w * 0.2;
    sum += texture(stateTex, uv + px * vec2(0.0, 1.0)).w * 0.2;
    sum += texture(stateTex, uv + px * vec2(-1.0, -1.0)).w * 0.05;
    sum += texture(stateTex, uv + px * vec2(1.0, -1.0)).w * 0.05;
    sum += texture(stateTex, uv + px * vec2(-1.0, 1.0)).w * 0.05;
    sum += texture(stateTex, uv + px * vec2(1.0, 1.0)).w * 0.05;
    return (sum - center) * laplaceScale;
}

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec4 state = texture(stateTex, v_uv);
    vec2 uvState = state.xy;
    vec2 lap = laplaceState(v_uv);
    float climate = state.w;
    float climateLap = laplaceClimate(v_uv);

    float driftFeed = feed + sin(time * 0.05) * parameterDrift;
    float driftKill = kill + cos(time * 0.047) * parameterDrift;
    float localFeed = clamp(driftFeed + (climate - 0.5) * parameterDrift * 2.6, 0.0, 0.09);
    float localKill = clamp(driftKill - (climate - 0.5) * parameterDrift * 1.8, 0.0, 0.09);

    float u = uvState.x;
    float v = uvState.y;
    float uvv = u * v * v;

    float du = diffU * lap.x - uvv + localFeed * (1.0 - u);
    float dv = diffV * lap.y + uvv - (localFeed + localKill) * v;

    float n = (hash(v_uv * 4096.0 + vec2(time * 0.31, -time * 0.17)) - 0.5) * noiseStrength;
    du -= n * 0.35;
    dv += n;

    vec2 next = clamp(uvState + dt * vec2(du, dv), 0.0, 1.0);
    float activity = clamp(abs(next.x - u) + abs(next.y - v), 0.0, 1.0);
    float flux = mix(state.z, activity, 0.12);

    float nextClimate = climate + dt * (
        climateLap * 0.35 +
        (next.y - next.x) * 0.11 +
        (flux - climate) * 0.09 +
        n * 0.9
    );
    nextClimate = clamp(nextClimate, 0.0, 1.0);

    fragColor = vec4(next, flux, nextClimate);
}
