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

vec3 palette(float t) {
    return 0.55 + 0.45 * cos(vec3(0.0, 0.6, 1.1) + vec3(1.2, 2.0, 2.8) * t);
}

void main() {
    vec4 state = texture(stateTex, v_uv);
    float u = state.x;
    float v = state.y;
    float activity = state.z;

    float life = smoothstep(0.15, 0.85, v - u * 0.5);
    float contour = pow(abs(0.5 - u) * 2.0, contourContrast);

    vec3 base = palette(colorShift + life * 2.5 + time * 0.1);
    vec3 color = base * (0.4 + life * 1.7);
    color += vec3(contour) * 0.2;
    color += glow * vec3(activity * 0.8, activity, activity * 0.5);

    color = vec3(1.0) - exp(-color * exposure);
    color = pow(color, vec3(1.0 / gamma));

    fragColor = vec4(color, 1.0);
}
