#version 450

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D displayTex;

void main() {
    fragColor = texture(displayTex, v_uv);
}
