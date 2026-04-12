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

vec2 laplaceSample(vec2 uv) {
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
    vec2 lap = (sum - center) * laplaceScale;
    return lap;
}

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec4 state = texture(stateTex, v_uv);
    vec2 uvState = state.xy;
    vec2 lap = laplaceSample(v_uv);

    float driftFeed = feed + sin(time * 0.05) * parameterDrift;
    float driftKill = kill + cos(time * 0.047) * parameterDrift;

    float u = uvState.x;
    float v = uvState.y;
    float uvv = u * v * v;

    float du = diffU * lap.x - uvv + driftFeed * (1.0 - u);
    float dv = diffV * lap.y + uvv - (driftFeed + driftKill) * v;

    float n = (hash(v_uv * (time * 0.31 + 1.0)) - 0.5) * noiseStrength;
    dv += n;

    vec2 next = uvState + dt * vec2(du, dv);
    next = clamp(next, 0.0, 1.0);
    float activity = clamp(abs(next.x - u) + abs(next.y - v), 0.0, 1.0);
    fragColor = vec4(next, activity, 1.0);
}
