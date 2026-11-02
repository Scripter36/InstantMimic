from __future__ import annotations

import colorsys
import logging
from typing import Any

import numpy as np
import torch
from cuda.bindings import runtime as cudart
from OpenGL import GL

from instantmimic.envs.controls.fast_muscle.tendon_activation_visualizer import (
    MuscleLineGLRenderSource,
)

logger = logging.getLogger(__name__)

_COMPOSITE_MODES = frozenset({"alpha", "max_activation", "unit_mean_alpha"})


_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in int a_body;
layout(location = 1) in vec3 a_local;
layout(location = 2) in int a_muscle;

uniform mat4 u_view_proj;
uniform samplerBuffer u_body_pos;
uniform samplerBuffer u_body_quat;
uniform samplerBuffer u_activation;
uniform isamplerBuffer u_endpoint_attachment_start;
uniform isamplerBuffer u_attachment_body;
uniform samplerBuffer u_attachment_local_weight;
uniform vec4 u_color_low;
uniform vec4 u_color_high;
uniform vec4 u_color_under;
uniform vec4 u_color_over;
uniform vec3 u_color_low_hsv;
uniform vec3 u_color_high_hsv;
uniform float u_hue_delta;
uniform float u_activation_lower;
uniform float u_activation_upper;
uniform int u_color_mode;
uniform int u_output_mode;
uniform int u_use_lbs;
uniform int u_use_world;
uniform mat4 u_projection;
uniform mat4 u_modelview;

out vec4 v_color;
out float v_activation_scalar;

vec3 rotate_wxyz(vec4 q, vec3 v) {
    vec3 qv = q.yzw;
    vec3 t = 2.0 * cross(qv, v);
    return v + q.x * t + cross(qv, t);
}

vec3 hsv_to_rgb(vec3 c) {
    vec4 k = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * mix(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec3 world = vec3(0.0);
    if (u_use_world != 0) {
        world = a_local;
    } else if (u_use_lbs != 0) {
        int begin = texelFetch(u_endpoint_attachment_start, gl_VertexID).r;
        int end = texelFetch(u_endpoint_attachment_start, gl_VertexID + 1).r;
        for (int attachment = begin; attachment < end; ++attachment) {
            int body = texelFetch(u_attachment_body, attachment).r;
            vec4 local_weight = texelFetch(u_attachment_local_weight, attachment);
            vec3 body_pos = texelFetch(u_body_pos, body).xyz;
            vec4 body_quat = texelFetch(u_body_quat, body);
            world += local_weight.w * (
                body_pos + rotate_wxyz(body_quat, local_weight.xyz)
            );
        }
    } else {
        vec3 body_pos = texelFetch(u_body_pos, a_body).xyz;
        vec4 body_quat = texelFetch(u_body_quat, a_body);
        world = body_pos + rotate_wxyz(body_quat, a_local);
    }
    gl_Position = u_projection * u_modelview * vec4(world, 1.0);

    float raw_activation = texelFetch(u_activation, a_muscle).r;
    float t = clamp(raw_activation, u_activation_lower, u_activation_upper);
    v_activation_scalar = t;

    vec4 color;
    if (u_color_mode == 1) {
        float hue = fract(u_color_low_hsv.x + u_hue_delta * t);
        float saturation = mix(u_color_low_hsv.y, u_color_high_hsv.y, t);
        float value = mix(u_color_low_hsv.z, u_color_high_hsv.z, t);
        float alpha = mix(u_color_low.a, u_color_high.a, t);
        color = vec4(hsv_to_rgb(vec3(hue, saturation, value)), alpha);
    } else {
        color = mix(u_color_low, u_color_high, t);
    }

    float under = max(-raw_activation, 0.0);
    float over = max(raw_activation - 1.0, 0.0);
    if (under > 0.0 && under >= over) {
        color = u_color_under;
    }
    if (over > under) {
        color = u_color_over;
    }
    v_color = color;
}
"""


_FRAGMENT_SHADER = """
#version 330 core

in vec4 v_color;
in float v_activation_scalar;
uniform int u_output_mode;
out vec4 frag_color;

void main() {
    if (u_output_mode == 1) {
        frag_color = vec4(v_activation_scalar, 0.0, 0.0, 1.0);
        return;
    }
    if (u_output_mode == 2) {
        frag_color = vec4(v_activation_scalar, 1.0, 0.0, 1.0);
        return;
    }
    frag_color = v_color;
}
"""


_COMPOSITE_VERTEX_SHADER = """
#version 330 core

out vec2 v_uv;

void main() {
    vec2 pos;
    if (gl_VertexID == 0) {
        pos = vec2(-1.0, -1.0);
    } else if (gl_VertexID == 1) {
        pos = vec2(3.0, -1.0);
    } else {
        pos = vec2(-1.0, 3.0);
    }
    v_uv = 0.5 * (pos + vec2(1.0));
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""


_COMPOSITE_FRAGMENT_SHADER = """
#version 330 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_activation_max;
uniform vec4 u_color_low;
uniform vec4 u_color_high;
uniform vec3 u_color_low_hsv;
uniform vec3 u_color_high_hsv;
uniform float u_hue_delta;
uniform float u_activation_lower;
uniform float u_activation_upper;
uniform int u_color_mode;
uniform int u_composite_input_mode;

vec3 hsv_to_rgb(vec3 c) {
    vec4 k = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * mix(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
}

void main() {
    float raw_activation;
    if (u_composite_input_mode == 1) {
        vec2 accum = texture(u_activation_max, v_uv).rg;
        if (accum.g <= 0.0) {
            discard;
        }
        raw_activation = accum.r / accum.g;
    } else {
        raw_activation = texture(u_activation_max, v_uv).r;
        if (raw_activation < 0.0) {
            discard;
        }
    }
    float t = clamp(raw_activation, u_activation_lower, u_activation_upper);

    if (u_color_mode == 1) {
        float hue = fract(u_color_low_hsv.x + u_hue_delta * t);
        float saturation = mix(u_color_low_hsv.y, u_color_high_hsv.y, t);
        float value = mix(u_color_low_hsv.z, u_color_high_hsv.z, t);
        float alpha = mix(u_color_low.a, u_color_high.a, t);
        frag_color = vec4(hsv_to_rgb(vec3(hue, saturation, value)), alpha);
    } else {
        frag_color = mix(u_color_low, u_color_high, t);
    }
}
"""


def _compile_shader(kind: int, source: str) -> int:
    shader = int(GL.glCreateShader(kind))
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if not status:
        info = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        GL.glDeleteShader(shader)
        raise RuntimeError(f"Failed to compile muscle-line shader: {info}")
    return shader


def _link_program(vertex_shader: int, fragment_shader: int) -> int:
    program = int(GL.glCreateProgram())
    GL.glAttachShader(program, vertex_shader)
    GL.glAttachShader(program, fragment_shader)
    GL.glLinkProgram(program)
    status = GL.glGetProgramiv(program, GL.GL_LINK_STATUS)
    if not status:
        info = GL.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
        GL.glDeleteProgram(program)
        raise RuntimeError(f"Failed to link muscle-line shader: {info}")
    return program


def _hsv_uniforms(
    low_rgba: np.ndarray, high_rgba: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    low_hsv = np.asarray(colorsys.rgb_to_hsv(*low_rgba[:3]), dtype=np.float32)
    high_hsv = np.asarray(colorsys.rgb_to_hsv(*high_rgba[:3]), dtype=np.float32)
    hue_delta = (float(high_hsv[0]) - float(low_hsv[0]) + 0.5) % 1.0 - 0.5
    return low_hsv, high_hsv, float(hue_delta)


def _effective_gl_line_width(source_width: float) -> float:
    return max(1.0, 0.5 * float(source_width))


def _normalize_activation_composite_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in _COMPOSITE_MODES:
        raise ValueError(
            "activation_line_composite_mode must be 'alpha', 'max_activation', "
            "or 'unit_mean_alpha'."
        )
    return normalized


class MuscleLineGLRenderer:
    def __init__(self, source: MuscleLineGLRenderSource) -> None:
        self._source = source
        self._vertex_count = int(source.vertex_count)
        self._uses_lbs = bool(getattr(source, "uses_lbs_endpoints", False))
        self._composite_mode = _normalize_activation_composite_mode(
            getattr(source, "composite_mode", "alpha")
        )
        self._warned_invalid_body = False
        self._warned_activation_interop = False
        self._cuda_activation_resource = None
        self._cuda_activation_disabled = False

        vertex_shader = _compile_shader(GL.GL_VERTEX_SHADER, _VERTEX_SHADER)
        fragment_shader = _compile_shader(GL.GL_FRAGMENT_SHADER, _FRAGMENT_SHADER)
        try:
            self._program = _link_program(vertex_shader, fragment_shader)
        finally:
            GL.glDeleteShader(vertex_shader)
            GL.glDeleteShader(fragment_shader)

        composite_vertex_shader = _compile_shader(
            GL.GL_VERTEX_SHADER,
            _COMPOSITE_VERTEX_SHADER,
        )
        composite_fragment_shader = _compile_shader(
            GL.GL_FRAGMENT_SHADER,
            _COMPOSITE_FRAGMENT_SHADER,
        )
        try:
            self._composite_program = _link_program(
                composite_vertex_shader,
                composite_fragment_shader,
            )
        finally:
            GL.glDeleteShader(composite_vertex_shader)
            GL.glDeleteShader(composite_fragment_shader)

        self._vao = int(GL.glGenVertexArrays(1))
        self._fullscreen_vao = int(GL.glGenVertexArrays(1))
        self._body_attr_buffer = int(GL.glGenBuffers(1))
        self._local_attr_buffer = int(GL.glGenBuffers(1))
        self._muscle_attr_buffer = int(GL.glGenBuffers(1))
        self._body_pos_buffer = int(GL.glGenBuffers(1))
        self._body_pos_texture = int(GL.glGenTextures(1))
        self._body_quat_buffer = int(GL.glGenBuffers(1))
        self._body_quat_texture = int(GL.glGenTextures(1))
        self._activation_buffer = int(GL.glGenBuffers(1))
        self._activation_texture = int(GL.glGenTextures(1))
        self._endpoint_attachment_start_buffer = 0
        self._endpoint_attachment_start_texture = 0
        self._attachment_body_buffer = 0
        self._attachment_body_texture = 0
        self._attachment_local_weight_buffer = 0
        self._attachment_local_weight_texture = 0
        if self._uses_lbs:
            self._endpoint_attachment_start_buffer = int(GL.glGenBuffers(1))
            self._endpoint_attachment_start_texture = int(GL.glGenTextures(1))
            self._attachment_body_buffer = int(GL.glGenBuffers(1))
            self._attachment_body_texture = int(GL.glGenTextures(1))
            self._attachment_local_weight_buffer = int(GL.glGenBuffers(1))
            self._attachment_local_weight_texture = int(GL.glGenTextures(1))
        self._body_capacity = 0
        self._max_activation_fbo = 0
        self._max_activation_texture = 0
        self._max_activation_depth_texture = 0
        self._max_activation_width = 0
        self._max_activation_height = 0
        self._unit_mean_fbo = 0
        self._unit_mean_texture = 0
        self._unit_mean_depth_texture = 0
        self._unit_mean_width = 0
        self._unit_mean_height = 0

        self._setup_static_attributes()
        self._setup_texture_buffer(
            buffer=self._body_pos_buffer,
            texture=self._body_pos_texture,
            byte_count=3 * 4,
            internal_format=GL.GL_RGB32F,
        )
        self._setup_texture_buffer(
            buffer=self._body_quat_buffer,
            texture=self._body_quat_texture,
            byte_count=4 * 4,
            internal_format=GL.GL_RGBA32F,
        )
        self._setup_texture_buffer(
            buffer=self._activation_buffer,
            texture=self._activation_texture,
            byte_count=max(1, int(source.muscle_count)) * 4,
            internal_format=GL.GL_R32F,
        )
        self._setup_lbs_endpoint_buffers()
        self._locations = {
            name: int(GL.glGetUniformLocation(self._program, name))
            for name in (
                "u_view_proj",
                "u_body_pos",
                "u_body_quat",
                "u_activation",
                "u_use_world",
                "u_endpoint_attachment_start",
                "u_attachment_body",
                "u_attachment_local_weight",
                "u_color_low",
                "u_color_high",
                "u_color_under",
                "u_color_over",
                "u_color_low_hsv",
                "u_color_high_hsv",
                "u_hue_delta",
                "u_activation_lower",
                "u_activation_upper",
                "u_color_mode",
                "u_output_mode",
                "u_use_lbs",
                "u_projection",
                "u_modelview",
            )
        }
        self._composite_locations = {
            name: int(GL.glGetUniformLocation(self._composite_program, name))
            for name in (
                "u_activation_max",
                "u_color_low",
                "u_color_high",
                "u_color_low_hsv",
                "u_color_high_hsv",
                "u_hue_delta",
                "u_activation_lower",
                "u_activation_upper",
                "u_color_mode",
                "u_composite_input_mode",
            )
        }

    def _setup_static_attributes(self) -> None:
        GL.glBindVertexArray(self._vao)

        body = np.ascontiguousarray(self._source.endpoint_body, dtype=np.int32)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._body_attr_buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, body.nbytes, body, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribIPointer(0, 1, GL.GL_INT, 0, None)

        local = np.ascontiguousarray(self._source.endpoint_local, dtype=np.float32)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._local_attr_buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, local.nbytes, local, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 0, None)

        muscle = np.ascontiguousarray(self._source.endpoint_muscle, dtype=np.int32)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._muscle_attr_buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, muscle.nbytes, muscle, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribIPointer(2, 1, GL.GL_INT, 0, None)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

    @staticmethod
    def _setup_static_texture_buffer(
        *,
        buffer: int,
        texture: int,
        array: np.ndarray,
        internal_format: int,
    ) -> None:
        values = np.ascontiguousarray(array)
        GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, buffer)
        GL.glBufferData(
            GL.GL_TEXTURE_BUFFER,
            max(1, int(values.nbytes)),
            values,
            GL.GL_STATIC_DRAW,
        )
        GL.glBindTexture(GL.GL_TEXTURE_BUFFER, texture)
        GL.glTexBuffer(GL.GL_TEXTURE_BUFFER, internal_format, buffer)
        GL.glBindTexture(GL.GL_TEXTURE_BUFFER, 0)
        GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, 0)

    def _setup_lbs_endpoint_buffers(self) -> None:
        if not self._uses_lbs:
            return
        self._setup_static_texture_buffer(
            buffer=self._endpoint_attachment_start_buffer,
            texture=self._endpoint_attachment_start_texture,
            array=np.asarray(
                self._source.endpoint_attachment_start,
                dtype=np.int32,
            ),
            internal_format=GL.GL_R32I,
        )
        self._setup_static_texture_buffer(
            buffer=self._attachment_body_buffer,
            texture=self._attachment_body_texture,
            array=np.asarray(
                self._source.endpoint_attachment_body,
                dtype=np.int32,
            ),
            internal_format=GL.GL_R32I,
        )
        self._setup_static_texture_buffer(
            buffer=self._attachment_local_weight_buffer,
            texture=self._attachment_local_weight_texture,
            array=np.asarray(
                self._source.endpoint_attachment_local_weight,
                dtype=np.float32,
            ).reshape(-1, 4),
            internal_format=GL.GL_RGBA32F,
        )

    @staticmethod
    def _setup_texture_buffer(
        *,
        buffer: int,
        texture: int,
        byte_count: int,
        internal_format: int,
    ) -> None:
        GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, buffer)
        GL.glBufferData(
            GL.GL_TEXTURE_BUFFER,
            max(1, int(byte_count)),
            None,
            GL.GL_DYNAMIC_DRAW,
        )
        GL.glBindTexture(GL.GL_TEXTURE_BUFFER, texture)
        GL.glTexBuffer(GL.GL_TEXTURE_BUFFER, internal_format, buffer)
        GL.glBindTexture(GL.GL_TEXTURE_BUFFER, 0)
        GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, 0)

    def _ensure_body_capacity(self, body_count: int) -> None:
        if body_count <= self._body_capacity:
            return
        self._body_capacity = int(body_count)
        self._setup_texture_buffer(
            buffer=self._body_pos_buffer,
            texture=self._body_pos_texture,
            byte_count=self._body_capacity * 3 * 4,
            internal_format=GL.GL_RGB32F,
        )
        self._setup_texture_buffer(
            buffer=self._body_quat_buffer,
            texture=self._body_quat_texture,
            byte_count=self._body_capacity * 4 * 4,
            internal_format=GL.GL_RGBA32F,
        )

    @staticmethod
    def _upload_texture_buffer(buffer: int, array: np.ndarray) -> None:
        values = np.ascontiguousarray(array, dtype=np.float32)
        GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, buffer)
        GL.glBufferSubData(GL.GL_TEXTURE_BUFFER, 0, values.nbytes, values)
        GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, 0)

    def _upload_activation_cpu(self, activation: torch.Tensor) -> None:
        values = activation.detach()
        if values.is_cuda:
            values = values.cpu()
        values_np = np.ascontiguousarray(values.numpy(), dtype=np.float32)
        self._upload_texture_buffer(self._activation_buffer, values_np)

    def _ensure_cuda_activation_resource(self) -> bool:
        if self._cuda_activation_disabled:
            return False
        if self._cuda_activation_resource is not None:
            return True
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
        err, resource = cudart.cudaGraphicsGLRegisterBuffer(
            int(self._activation_buffer),
            flags,
        )
        if err != cudart.cudaError_t.cudaSuccess:
            self._cuda_activation_disabled = True
            if not self._warned_activation_interop:
                logger.warning(
                    "CUDA/GL activation interop unavailable for muscle lines: %s",
                    err,
                )
                self._warned_activation_interop = True
            return False
        self._cuda_activation_resource = resource
        return True

    def _upload_activation_cuda(self, activation: torch.Tensor) -> bool:
        if not activation.is_cuda or not self._ensure_cuda_activation_resource():
            return False
        values = activation.contiguous()
        byte_count = int(values.numel() * values.element_size())
        with torch.cuda.device(values.device):
            (err,) = cudart.cudaGraphicsMapResources(
                1,
                self._cuda_activation_resource,
                0,
            )
            if err != cudart.cudaError_t.cudaSuccess:
                return False
            try:
                err, mapped_ptr, mapped_size = (
                    cudart.cudaGraphicsResourceGetMappedPointer(
                        self._cuda_activation_resource
                    )
                )
                if err != cudart.cudaError_t.cudaSuccess:
                    return False
                if int(mapped_size) < byte_count:
                    return False
                (err,) = cudart.cudaMemcpy(
                    int(mapped_ptr),
                    int(values.data_ptr()),
                    byte_count,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                )
                return err == cudart.cudaError_t.cudaSuccess
            finally:
                (err,) = cudart.cudaGraphicsUnmapResources(
                    1,
                    self._cuda_activation_resource,
                    0,
                )
                if (
                    err != cudart.cudaError_t.cudaSuccess
                    and not self._warned_activation_interop
                ):
                    logger.warning(
                        "Failed to unmap muscle-line activation GL buffer: %s",
                        err,
                    )
                    self._warned_activation_interop = True

    def _upload_activation(self) -> None:
        activation = self._source.resolve_activation_tensor()
        if activation.shape[0] != int(self._source.muscle_count):
            activation = torch.zeros((self._source.muscle_count,), dtype=torch.float32)
        elif activation.dtype != torch.float32:
            activation = activation.to(dtype=torch.float32)

        if self._upload_activation_cuda(activation):
            return
        self._upload_activation_cpu(activation)

    def _set_line_uniforms(self, *, output_mode: int) -> None:
        GL.glUniform1i(
            self._locations["u_use_world"],
            int(getattr(self._source, "endpoint_world_resolver", None) is not None),
        )
        projection = np.ascontiguousarray(
            GL.glGetFloatv(GL.GL_PROJECTION_MATRIX),
            dtype=np.float32,
        )
        modelview = np.ascontiguousarray(
            GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX),
            dtype=np.float32,
        )
        low_hsv, high_hsv, hue_delta = _hsv_uniforms(
            self._source.color_low_rgba,
            self._source.color_high_rgba,
        )

        GL.glUniformMatrix4fv(
            self._locations["u_projection"],
            1,
            GL.GL_FALSE,
            projection,
        )
        GL.glUniformMatrix4fv(
            self._locations["u_modelview"],
            1,
            GL.GL_FALSE,
            modelview,
        )
        GL.glUniform1i(self._locations["u_body_pos"], 0)
        GL.glUniform1i(self._locations["u_body_quat"], 1)
        GL.glUniform1i(self._locations["u_activation"], 2)
        uses_lbs = bool(getattr(self, "_uses_lbs", False))
        use_lbs_location = self._locations.get("u_use_lbs", -1)
        if use_lbs_location >= 0:
            GL.glUniform1i(use_lbs_location, 1 if uses_lbs else 0)
        if uses_lbs:
            start_location = self._locations.get("u_endpoint_attachment_start", -1)
            body_location = self._locations.get("u_attachment_body", -1)
            local_weight_location = self._locations.get(
                "u_attachment_local_weight",
                -1,
            )
            if start_location >= 0:
                GL.glUniform1i(start_location, 3)
            if body_location >= 0:
                GL.glUniform1i(body_location, 4)
            if local_weight_location >= 0:
                GL.glUniform1i(local_weight_location, 5)
        GL.glUniform4fv(
            self._locations["u_color_low"],
            1,
            np.asarray(self._source.color_low_rgba, dtype=np.float32),
        )
        GL.glUniform4fv(
            self._locations["u_color_high"],
            1,
            np.asarray(self._source.color_high_rgba, dtype=np.float32),
        )
        GL.glUniform4fv(
            self._locations["u_color_under"],
            1,
            np.asarray(self._source.color_under_rgba, dtype=np.float32),
        )
        GL.glUniform4fv(
            self._locations["u_color_over"],
            1,
            np.asarray(self._source.color_over_rgba, dtype=np.float32),
        )
        GL.glUniform3fv(self._locations["u_color_low_hsv"], 1, low_hsv)
        GL.glUniform3fv(self._locations["u_color_high_hsv"], 1, high_hsv)
        GL.glUniform1f(self._locations["u_hue_delta"], hue_delta)
        GL.glUniform1f(
            self._locations["u_activation_lower"],
            float(self._source.activation_lower_bound),
        )
        GL.glUniform1f(
            self._locations["u_activation_upper"],
            float(self._source.activation_upper_bound),
        )
        GL.glUniform1i(
            self._locations["u_color_mode"],
            1 if self._source.color_interpolation == "hsv" else 0,
        )
        GL.glUniform1i(self._locations["u_output_mode"], int(output_mode))

    def _set_composite_uniforms(self, *, input_mode: int = 0) -> None:
        low_hsv, high_hsv, hue_delta = _hsv_uniforms(
            self._source.color_low_rgba,
            self._source.color_high_rgba,
        )
        GL.glUniform1i(self._composite_locations["u_activation_max"], 0)
        GL.glUniform4fv(
            self._composite_locations["u_color_low"],
            1,
            np.asarray(self._source.color_low_rgba, dtype=np.float32),
        )
        GL.glUniform4fv(
            self._composite_locations["u_color_high"],
            1,
            np.asarray(self._source.color_high_rgba, dtype=np.float32),
        )
        GL.glUniform3fv(self._composite_locations["u_color_low_hsv"], 1, low_hsv)
        GL.glUniform3fv(self._composite_locations["u_color_high_hsv"], 1, high_hsv)
        GL.glUniform1f(self._composite_locations["u_hue_delta"], hue_delta)
        GL.glUniform1f(
            self._composite_locations["u_activation_lower"],
            float(self._source.activation_lower_bound),
        )
        GL.glUniform1f(
            self._composite_locations["u_activation_upper"],
            float(self._source.activation_upper_bound),
        )
        GL.glUniform1i(
            self._composite_locations["u_color_mode"],
            1 if self._source.color_interpolation == "hsv" else 0,
        )
        composite_input_location = self._composite_locations.get(
            "u_composite_input_mode",
            -1,
        )
        if composite_input_location >= 0:
            GL.glUniform1i(composite_input_location, int(input_mode))

    @staticmethod
    def _bind_texture(unit: int, texture: int) -> None:
        GL.glActiveTexture(GL.GL_TEXTURE0 + int(unit))
        GL.glBindTexture(GL.GL_TEXTURE_BUFFER, int(texture))

    @staticmethod
    def _bind_texture_2d(unit: int, texture: int) -> None:
        GL.glActiveTexture(GL.GL_TEXTURE0 + int(unit))
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(texture))

    def _delete_max_activation_targets(self) -> None:
        if self._max_activation_fbo:
            GL.glDeleteFramebuffers(1, [self._max_activation_fbo])
            self._max_activation_fbo = 0
        textures = [
            texture
            for texture in (
                self._max_activation_texture,
                self._max_activation_depth_texture,
            )
            if texture
        ]
        if textures:
            GL.glDeleteTextures(len(textures), textures)
        self._max_activation_texture = 0
        self._max_activation_depth_texture = 0
        self._max_activation_width = 0
        self._max_activation_height = 0

    def _delete_unit_mean_targets(self) -> None:
        if self._unit_mean_fbo:
            GL.glDeleteFramebuffers(1, [self._unit_mean_fbo])
            self._unit_mean_fbo = 0
        textures = [
            texture
            for texture in (
                self._unit_mean_texture,
                self._unit_mean_depth_texture,
            )
            if texture
        ]
        if textures:
            GL.glDeleteTextures(len(textures), textures)
        self._unit_mean_texture = 0
        self._unit_mean_depth_texture = 0
        self._unit_mean_width = 0
        self._unit_mean_height = 0

    @staticmethod
    def _configure_2d_texture(texture: int) -> None:
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(texture))
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

    def _ensure_max_activation_targets(self, width: int, height: int) -> None:
        width = int(width)
        height = int(height)
        if (
            self._max_activation_fbo
            and self._max_activation_width == width
            and self._max_activation_height == height
        ):
            return

        self._delete_max_activation_targets()
        self._max_activation_width = width
        self._max_activation_height = height
        self._max_activation_fbo = int(GL.glGenFramebuffers(1))
        self._max_activation_texture = int(GL.glGenTextures(1))
        self._max_activation_depth_texture = int(GL.glGenTextures(1))

        self._configure_2d_texture(self._max_activation_texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_R32F,
            width,
            height,
            0,
            GL.GL_RED,
            GL.GL_FLOAT,
            None,
        )

        self._configure_2d_texture(self._max_activation_depth_texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_DEPTH_COMPONENT32F,
            width,
            height,
            0,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_FLOAT,
            None,
        )

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._max_activation_fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            self._max_activation_texture,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_TEXTURE_2D,
            self._max_activation_depth_texture,
            0,
        )
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(
                f"Failed to create muscle-line max-activation framebuffer: {status}"
            )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _ensure_unit_mean_targets(self, width: int, height: int) -> None:
        width = int(width)
        height = int(height)
        if (
            self._unit_mean_fbo
            and self._unit_mean_width == width
            and self._unit_mean_height == height
        ):
            return

        self._delete_unit_mean_targets()
        self._unit_mean_width = width
        self._unit_mean_height = height
        self._unit_mean_fbo = int(GL.glGenFramebuffers(1))
        self._unit_mean_texture = int(GL.glGenTextures(1))
        self._unit_mean_depth_texture = int(GL.glGenTextures(1))

        self._configure_2d_texture(self._unit_mean_texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RG32F,
            width,
            height,
            0,
            GL.GL_RG,
            GL.GL_FLOAT,
            None,
        )

        self._configure_2d_texture(self._unit_mean_depth_texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_DEPTH_COMPONENT32F,
            width,
            height,
            0,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_FLOAT,
            None,
        )

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._unit_mean_fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            self._unit_mean_texture,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_TEXTURE_2D,
            self._unit_mean_depth_texture,
            0,
        )
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(
                f"Failed to create muscle-line unit-mean framebuffer: {status}"
            )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _bind_line_program(self, *, output_mode: int) -> None:
        GL.glUseProgram(self._program)
        self._bind_texture(0, self._body_pos_texture)
        self._bind_texture(1, self._body_quat_texture)
        self._bind_texture(2, self._activation_texture)
        if bool(getattr(self, "_uses_lbs", False)):
            self._bind_texture(3, self._endpoint_attachment_start_texture)
            self._bind_texture(4, self._attachment_body_texture)
            self._bind_texture(5, self._attachment_local_weight_texture)
        self._set_line_uniforms(output_mode=output_mode)
        GL.glBindVertexArray(self._vao)

    def _draw_alpha_lines(self, *, width: int, height: int, framebuffer: int) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(framebuffer))
        GL.glViewport(0, 0, int(width), int(height))
        GL.glEnable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glBlendEquationSeparate(GL.GL_FUNC_ADD, GL.GL_FUNC_ADD)
        GL.glBlendFuncSeparate(
            GL.GL_SRC_ALPHA,
            GL.GL_ONE_MINUS_SRC_ALPHA,
            GL.GL_ONE,
            GL.GL_ONE_MINUS_SRC_ALPHA,
        )
        GL.glLineWidth(_effective_gl_line_width(self._source.line_width))
        self._bind_line_program(output_mode=0)
        GL.glDrawArrays(GL.GL_LINES, 0, self._vertex_count)

    def _copy_depth_to_max_activation_target(
        self,
        *,
        width: int,
        height: int,
        framebuffer: int,
    ) -> None:
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, int(framebuffer))
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._max_activation_fbo)
        GL.glBlitFramebuffer(
            0,
            0,
            int(width),
            int(height),
            0,
            0,
            int(width),
            int(height),
            GL.GL_DEPTH_BUFFER_BIT,
            GL.GL_NEAREST,
        )

    def _copy_depth_to_unit_mean_target(
        self,
        *,
        width: int,
        height: int,
        framebuffer: int,
    ) -> None:
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, int(framebuffer))
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._unit_mean_fbo)
        GL.glBlitFramebuffer(
            0,
            0,
            int(width),
            int(height),
            0,
            0,
            int(width),
            int(height),
            GL.GL_DEPTH_BUFFER_BIT,
            GL.GL_NEAREST,
        )

    def _draw_max_activation_projection(
        self,
        *,
        width: int,
        height: int,
        framebuffer: int,
    ) -> None:
        self._ensure_max_activation_targets(width, height)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._max_activation_fbo)
        GL.glViewport(0, 0, int(width), int(height))
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glClearColor(-1.0, -1.0, -1.0, -1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        self._copy_depth_to_max_activation_target(
            width=width,
            height=height,
            framebuffer=framebuffer,
        )

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._max_activation_fbo)
        GL.glViewport(0, 0, int(width), int(height))
        GL.glEnable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glBlendEquationSeparate(GL.GL_MAX, GL.GL_MAX)
        GL.glBlendFuncSeparate(GL.GL_ONE, GL.GL_ONE, GL.GL_ONE, GL.GL_ONE)
        GL.glLineWidth(_effective_gl_line_width(self._source.line_width))
        self._bind_line_program(output_mode=1)
        GL.glDrawArrays(GL.GL_LINES, 0, self._vertex_count)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(framebuffer))
        GL.glViewport(0, 0, int(width), int(height))
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendEquationSeparate(GL.GL_FUNC_ADD, GL.GL_FUNC_ADD)
        GL.glBlendFuncSeparate(
            GL.GL_SRC_ALPHA,
            GL.GL_ONE_MINUS_SRC_ALPHA,
            GL.GL_ONE,
            GL.GL_ONE_MINUS_SRC_ALPHA,
        )
        GL.glUseProgram(self._composite_program)
        self._bind_texture_2d(0, self._max_activation_texture)
        self._set_composite_uniforms()
        GL.glBindVertexArray(self._fullscreen_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

    def _draw_unit_mean_alpha_projection(
        self,
        *,
        width: int,
        height: int,
        framebuffer: int,
    ) -> None:
        self._ensure_unit_mean_targets(width, height)
        self._copy_depth_to_unit_mean_target(
            width=width,
            height=height,
            framebuffer=framebuffer,
        )

        unit_vertex_start = np.asarray(
            getattr(self._source, "unit_vertex_start", np.asarray([0])),
            dtype=np.int32,
        ).reshape(-1)
        unit_vertex_count = np.asarray(
            getattr(
                self._source, "unit_vertex_count", np.asarray([self._vertex_count])
            ),
            dtype=np.int32,
        ).reshape(-1)
        unit_count = min(int(unit_vertex_start.size), int(unit_vertex_count.size))
        if unit_count <= 0:
            return

        GL.glLineWidth(_effective_gl_line_width(self._source.line_width))
        for unit_idx in range(unit_count):
            vertex_start = int(unit_vertex_start[unit_idx])
            vertex_count = int(unit_vertex_count[unit_idx])
            if vertex_count <= 0:
                continue

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._unit_mean_fbo)
            GL.glViewport(0, 0, int(width), int(height))
            GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_BLEND)
            GL.glDepthMask(GL.GL_FALSE)
            GL.glBlendEquationSeparate(GL.GL_FUNC_ADD, GL.GL_FUNC_ADD)
            GL.glBlendFuncSeparate(GL.GL_ONE, GL.GL_ONE, GL.GL_ONE, GL.GL_ONE)
            self._bind_line_program(output_mode=2)
            GL.glDrawArrays(GL.GL_LINES, vertex_start, vertex_count)

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(framebuffer))
            GL.glViewport(0, 0, int(width), int(height))
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendEquationSeparate(GL.GL_FUNC_ADD, GL.GL_FUNC_ADD)
            GL.glBlendFuncSeparate(
                GL.GL_SRC_ALPHA,
                GL.GL_ONE_MINUS_SRC_ALPHA,
                GL.GL_ONE,
                GL.GL_ONE_MINUS_SRC_ALPHA,
            )
            GL.glUseProgram(self._composite_program)
            self._bind_texture_2d(0, self._unit_mean_texture)
            self._set_composite_uniforms(input_mode=1)
            GL.glBindVertexArray(self._fullscreen_vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

    def draw(
        self,
        *,
        render_data: Any,
        scene: Any,
        width: int,
        height: int,
        framebuffer: int,
    ) -> None:
        if self._vertex_count <= 0:
            return
        body_count = int(np.asarray(render_data.xpos).shape[0])
        max_body = int(
            getattr(
                self._source,
                "max_referenced_body",
                int(self._source.endpoint_body.max(initial=-1)),
            )
        )
        if max_body >= body_count:
            if not self._warned_invalid_body:
                logger.warning(
                    "Skipping muscle-line GL render: endpoint body %d >= render nbody %d.",
                    max_body,
                    body_count,
                )
                self._warned_invalid_body = True
            return

        self._ensure_body_capacity(body_count)
        self._upload_texture_buffer(self._body_pos_buffer, render_data.xpos)
        self._upload_texture_buffer(self._body_quat_buffer, render_data.xquat)
        self._upload_activation()
        resolver = getattr(self._source, "endpoint_world_resolver", None)
        if resolver is not None:
            points = resolver(render_data)[0].detach().cpu().numpy()
            if points.shape != (self._vertex_count, 3):
                raise ValueError(
                    "Dynamic muscle endpoints do not match render topology."
                )
            points = np.ascontiguousarray(points, dtype=np.float32)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._local_attr_buffer)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, points.nbytes, points)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

        prev_program = int(GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM))
        prev_vao = int(GL.glGetIntegerv(GL.GL_VERTEX_ARRAY_BINDING))
        prev_active_texture = int(GL.glGetIntegerv(GL.GL_ACTIVE_TEXTURE))
        prev_array_buffer = int(GL.glGetIntegerv(GL.GL_ARRAY_BUFFER_BINDING))
        prev_texture_buffer = int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_BUFFER))
        prev_texture_2d = int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D))
        prev_read_framebuffer = int(GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING))
        prev_draw_framebuffer = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
        prev_blend_equation_rgb = int(GL.glGetIntegerv(GL.GL_BLEND_EQUATION_RGB))
        prev_blend_equation_alpha = int(GL.glGetIntegerv(GL.GL_BLEND_EQUATION_ALPHA))
        prev_blend_src_rgb = int(GL.glGetIntegerv(GL.GL_BLEND_SRC_RGB))
        prev_blend_dst_rgb = int(GL.glGetIntegerv(GL.GL_BLEND_DST_RGB))
        prev_blend_src_alpha = int(GL.glGetIntegerv(GL.GL_BLEND_SRC_ALPHA))
        prev_blend_dst_alpha = int(GL.glGetIntegerv(GL.GL_BLEND_DST_ALPHA))
        prev_line_width = float(GL.glGetFloatv(GL.GL_LINE_WIDTH))
        prev_clear_color = np.asarray(
            GL.glGetFloatv(GL.GL_COLOR_CLEAR_VALUE),
            dtype=np.float32,
        ).reshape(-1)
        if prev_clear_color.size < 4:
            prev_clear_color = np.zeros((4,), dtype=np.float32)
        prev_depth_writemask = bool(GL.glGetBooleanv(GL.GL_DEPTH_WRITEMASK))
        blend_enabled = bool(GL.glIsEnabled(GL.GL_BLEND))
        depth_test_enabled = bool(GL.glIsEnabled(GL.GL_DEPTH_TEST))

        try:
            composite_mode = getattr(self, "_composite_mode", "alpha")
            if composite_mode == "max_activation":
                self._draw_max_activation_projection(
                    width=width,
                    height=height,
                    framebuffer=framebuffer,
                )
            elif composite_mode == "unit_mean_alpha":
                self._draw_unit_mean_alpha_projection(
                    width=width,
                    height=height,
                    framebuffer=framebuffer,
                )
            else:
                self._draw_alpha_lines(
                    width=width,
                    height=height,
                    framebuffer=framebuffer,
                )
        finally:
            GL.glBindVertexArray(prev_vao)
            GL.glUseProgram(prev_program)
            GL.glLineWidth(prev_line_width)
            GL.glClearColor(*prev_clear_color[:4])
            GL.glDepthMask(GL.GL_TRUE if prev_depth_writemask else GL.GL_FALSE)
            GL.glBlendEquationSeparate(
                prev_blend_equation_rgb,
                prev_blend_equation_alpha,
            )
            GL.glBlendFuncSeparate(
                prev_blend_src_rgb,
                prev_blend_dst_rgb,
                prev_blend_src_alpha,
                prev_blend_dst_alpha,
            )
            if not blend_enabled:
                GL.glDisable(GL.GL_BLEND)
            if depth_test_enabled:
                GL.glEnable(GL.GL_DEPTH_TEST)
            else:
                GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glActiveTexture(prev_active_texture)
            GL.glBindTexture(GL.GL_TEXTURE_BUFFER, prev_texture_buffer)
            GL.glBindTexture(GL.GL_TEXTURE_2D, prev_texture_2d)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, prev_array_buffer)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, prev_read_framebuffer)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, prev_draw_framebuffer)

    def close(self) -> None:
        if self._cuda_activation_resource is not None:
            try:
                cudart.cudaGraphicsUnregisterResource(self._cuda_activation_resource)
            except Exception as exc:  # noqa: BLE001 - CUDA/GL cleanup is best-effort.
                logger.debug(
                    "CUDA activation resource unregister failed: %s",
                    exc,
                    exc_info=True,
                )
            self._cuda_activation_resource = None
        buffers = [
            self._body_attr_buffer,
            self._local_attr_buffer,
            self._muscle_attr_buffer,
            self._body_pos_buffer,
            self._body_quat_buffer,
            self._activation_buffer,
            self._endpoint_attachment_start_buffer,
            self._attachment_body_buffer,
            self._attachment_local_weight_buffer,
        ]
        textures = [
            self._body_pos_texture,
            self._body_quat_texture,
            self._activation_texture,
            self._endpoint_attachment_start_texture,
            self._attachment_body_texture,
            self._attachment_local_weight_texture,
        ]
        buffers = [buffer for buffer in buffers if buffer]
        textures = [texture for texture in textures if texture]
        GL.glDeleteBuffers(len(buffers), buffers)
        GL.glDeleteTextures(len(textures), textures)
        GL.glDeleteVertexArrays(1, [self._vao])
        GL.glDeleteVertexArrays(1, [self._fullscreen_vao])
        self._delete_max_activation_targets()
        self._delete_unit_mean_targets()
        GL.glDeleteProgram(self._program)
        GL.glDeleteProgram(self._composite_program)
