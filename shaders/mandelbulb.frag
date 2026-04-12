#version 450

in vec2 v_uv;

uniform vec2 resolution;
uniform float time;
uniform vec3 cameraPos;
uniform vec3 cameraTarget;
uniform float exposure;
uniform float glowIntensity;
uniform float colorShift;
uniform int maxSteps;
uniform float maxDistance;
uniform float minDistance;
uniform float ambientOcclusionStrength;
uniform float power;

out vec4 fragColor;

const float PI = 3.14159265359;

mat3 cameraBasis(vec3 eye, vec3 target) {
    vec3 forward = normalize(target - eye);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = normalize(cross(right, forward));
    return mat3(right, up, -forward);
}

vec3 palette(float t) {
    return 0.5 + 0.5 * cos(vec3(0.0, 0.4, 0.8) + vec3(1.2, 0.9, 0.3) * t);
}

float mandelbulbDE(vec3 pos) {
    vec3 z = pos;
    float dr = 1.0;
    float r = 0.0;
    for (int i = 0; i < 64; ++i) {
        r = length(z);
        if (r > 4.0) {
            break;
        }
        float theta = acos(z.z / r);
        float phi = atan(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;
        float zr = pow(r, power);
        theta *= power;
        phi *= power;
        z = zr * vec3(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta)
            ) + pos;
    }
    return 0.5 * log(r) * r / abs(dr);
}

vec3 estimateNormal(vec3 p) {
    const vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        mandelbulbDE(p + e.xyy) - mandelbulbDE(p - e.xyy),
        mandelbulbDE(p + e.yxy) - mandelbulbDE(p - e.yxy),
        mandelbulbDE(p + e.yyx) - mandelbulbDE(p - e.yyx)
    ));
}

float softShadow(vec3 ro, vec3 rd, float k) {
    float res = 1.0;
    float t = 0.02;
    for (int i = 0; i < 64; ++i) {
        if (t > maxDistance)
            break;
        float h = mandelbulbDE(ro + rd * t);
        if (h < 0.001)
            return 0.0;
        res = min(res, k * h / t);
        t += clamp(h, 0.01, 0.25);
    }
    return clamp(res, 0.0, 1.0);
}

vec3 renderPixel(vec2 uvNorm) {
    vec2 fragCoord = vec2(uvNorm.x * resolution.x, uvNorm.y * resolution.y);
    vec2 uv = (fragCoord * 2.0 - resolution) / resolution.y;
    uv += vec2(sin(time * 0.13), cos(time * 0.17)) * 0.08;
    mat3 cam = cameraBasis(cameraPos, cameraTarget);
    vec3 rd = normalize(cam * vec3(uv, -1.5));

    float total = 0.0;
    float glow = 0.0;
    vec3 pos;

    bool hit = false;
    for (int i = 0; i < maxSteps; ++i) {
        pos = cameraPos + rd * total;
        float dist = mandelbulbDE(pos);
        glow += exp(-float(i) * 0.01) * 0.005;
        if (dist < minDistance) {
            hit = true;
            break;
        }
        total += dist * 0.9;
        if (total > maxDistance) {
            return vec3(0.0);
        }
    }

    if (!hit) {
        return vec3(0.0);
    }

    vec3 normal = estimateNormal(pos);
    vec3 lightDir = normalize(vec3(0.6, 0.8, -0.5));
    float diff = max(dot(normal, lightDir), 0.0);
    float shadow = softShadow(pos + normal * 0.02, lightDir, 12.0);
    float spec = pow(max(dot(reflect(-lightDir, normal), normalize(cameraPos - pos)), 0.0), 32.0);

    float ao = clamp(1.0 - ambientOcclusionStrength * total / maxDistance, 0.1, 1.0);
    vec3 base = palette(colorShift + total * 0.05);
    vec3 color = base * (0.4 + 1.6 * diff * shadow) + spec * 0.6;
    color += glow * glowIntensity;
    color *= ao;
    color = vec3(1.0) - exp(-color * exposure);
    return color;
}

void main() {
    vec3 col = renderPixel(v_uv);
    fragColor = vec4(col, 1.0);
}
