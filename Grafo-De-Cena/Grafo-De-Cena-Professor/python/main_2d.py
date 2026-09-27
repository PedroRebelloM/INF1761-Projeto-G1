from __future__ import annotations

from typing import Any
import time
import math
import numpy as np

import wgpu
from rendercanvas.glfw import RenderCanvas, loop

from camera2d import *
from colormaterial import *
from pointlight import *
from phongmaterial import *
from transform import *
from quad import *
from triangle import *
from node import *
from shader import *
from pipeline import *
from scene import *
from renderer import *
from engine import *
from shape import *
from state import *
from sampler import *
from texture import *
from textureset import *
import graphics_math as gm

canvas: RenderCanvas
device: wgpu.GPUDevice
context: Any
renderer: Renderer
camera: Camera2D
scene: Scene
last_t: float = 0.0

WORLD_SIZE = 10.0

R_SUN     = 2.20
R_MERCURY = 0.40
R_EARTH   = 0.70
R_MOON    = 0.30

ORBIT_MERCURY = 3.5
ORBIT_EARTH   = 6.0
ORBIT_MOON    = 1.6

OMEGA_MERCURY    =  48.0
OMEGA_EARTH      =  25.0
OMEGA_MOON       =  80.0
OMEGA_EARTH_SELF = 100.0

class Disk(Shape):
    def __init__(self, device: wgpu.GPUDevice, n: int = 64):
        coords = [0.0, 0.0]
        texcoords = [0.5, 0.5]
        for i in range(n + 1):
            theta = 2.0 * math.pi * i / n
            c = math.cos(theta)
            s = math.sin(theta)
            coords += [c, s]
            texcoords += [0.5 + 0.5 * c, 0.5 - 0.5 * s]

        indices = []
        for i in range(n):
            indices += [0, i + 1, i + 2]

        bcoord = np.array(coords, dtype='float32')
        btexcoord = np.array(texcoords, dtype='float32')
        bindex = np.array(indices, dtype='uint32')

        self.coord_vbo: wgpu.GPUBuffer = device.create_buffer_with_data(data=bcoord, usage=wgpu.BufferUsage.VERTEX)
        self.texcoord_vbo: wgpu.GPUBuffer = device.create_buffer_with_data(data=btexcoord, usage=wgpu.BufferUsage.VERTEX)
        self.ibo: wgpu.GPUBuffer = device.create_buffer_with_data(data=bindex, usage=wgpu.BufferUsage.INDEX)
        self.nind = len(indices)

    def draw(self, st: State) -> None:
        first_instance = st.get_shader().commit_matrix(st)
        st.render_pass.set_vertex_buffer(0, self.coord_vbo)
        st.render_pass.set_vertex_buffer(1, self.texcoord_vbo)
        st.render_pass.set_index_buffer(self.ibo, wgpu.IndexFormat.uint32)
        st.render_pass.draw_indexed(self.nind, 1, 0, 0, first_instance)


class SolarEngine(Engine):
    def __init__(self, trf_mercury_orbit: Transform, trf_earth_orbit: Transform,
                 trf_moon_orbit: Transform, trf_earth_shape: Transform):
        self.trf_mercury = trf_mercury_orbit
        self.trf_earth = trf_earth_orbit
        self.trf_moon = trf_moon_orbit
        self.trf_earth_shape = trf_earth_shape

        self.angle_mercury = 0.0
        self.angle_earth = 0.0
        self.angle_moon = 0.0
        self.angle_earth_self = 0.0

    def update(self, dt: float) -> None:
        self.angle_mercury = (self.angle_mercury + OMEGA_MERCURY * dt) % 360.0
        self.angle_earth = (self.angle_earth + OMEGA_EARTH * dt) % 360.0
        self.angle_moon = (self.angle_moon + OMEGA_MOON * dt) % 360.0
        self.angle_earth_self = (self.angle_earth_self + OMEGA_EARTH_SELF * dt) % 360.0

        self.trf_mercury.load_identity()
        self.trf_mercury.rotate(self.angle_mercury, 0, 0, 1)
        self.trf_mercury.translate(ORBIT_MERCURY, 0, 0)
        self.trf_mercury.rotate(-self.angle_mercury, 0, 0, 1)

        self.trf_earth.load_identity()
        self.trf_earth.rotate(self.angle_earth, 0, 0, 1)
        self.trf_earth.translate(ORBIT_EARTH, 0, 0)
        self.trf_earth.rotate(-self.angle_earth, 0, 0, 1)

        self.trf_moon.load_identity()
        self.trf_moon.rotate(self.angle_moon, 0, 0, 1)
        self.trf_moon.translate(ORBIT_MOON, 0, 0)
        self.trf_moon.rotate(-self.angle_moon, 0, 0, 1)

        self.trf_earth_shape.load_identity()
        self.trf_earth_shape.scale(R_EARTH, R_EARTH, 1)
        self.trf_earth_shape.rotate(self.angle_earth_self, 0, 0, 1)


class TiledQuad(Shape):
    def __init__(self, device: wgpu.GPUDevice, repeats: int = 5):
        grid = Grid(1, 1)
        self.nind = grid.index_count()
        self.coord_vbo = device.create_buffer_with_data(data=grid.get_coords(), usage=wgpu.BufferUsage.VERTEX)
        self.texcoord_vbo = device.create_buffer_with_data(data=grid.get_texcoords() * repeats, usage=wgpu.BufferUsage.VERTEX)
        self.ibo = device.create_buffer_with_data(data=grid.get_indices(), usage=wgpu.BufferUsage.INDEX)

    def draw(self, st: State) -> None:
        first_instance = st.get_shader().commit_matrix(st)
        st.render_pass.set_vertex_buffer(0, self.coord_vbo)
        st.render_pass.set_vertex_buffer(1, self.texcoord_vbo)
        st.render_pass.set_index_buffer(self.ibo, wgpu.IndexFormat.uint32)
        st.render_pass.draw_indexed(self.nind, 1, 0, 0, first_instance)


def initialize (device: wgpu.GPUDevice, target_format: str) -> None:
    global camera, scene

    camera = Camera2D(-WORLD_SIZE, WORLD_SIZE, -WORLD_SIZE, WORLD_SIZE)

    light = PointLight(
        0, 0, 50,
        space="world",
        light_ambient=gm.vec3(0.5, 0.5, 0.5),
        light_diffuse=gm.vec3(1.0, 1.0, 1.0),
        light_specular=gm.vec3(0.1, 0.1, 0.1)
    )

    shader = Shader(device, "../shaders/ilum_frag/textured.wgsl", light=light)
    shader.set_vertex_buffers([
        {"array_stride": 2 * 4, "step_mode": "vertex", "attributes": [{"format": "float32x2", "offset": 0, "var_name": "coord"}]},
        {"array_stride": 2 * 4, "step_mode": "vertex", "attributes": [{"format": "float32x2", "offset": 0, "var_name": "texcoord"}]},
    ])
    pipeline = Pipeline(shader, target_format, depth_stencil=None)

    def make_texset(filename: str) -> TextureSet:
        path = f"../images/{filename}"
        tex = Texture(device, "decal_texture", path)
        smp = Sampler(device, "decal_sampler", mag_filter="linear", min_filter="linear")
        tset = TextureSet([tex, smp])
        shader.add_texture_set(tset)
        return tset

    tset_space   = make_texset("stars.jpg")
    tset_sun     = make_texset("sun.jpg")
    tset_mercury = make_texset("mercury.jpg")
    tset_earth   = make_texset("earth.jpg")
    tset_moon    = make_texset("moon.jpg")

    mat_base = PhongMaterial(1.0, 1.0, 1.0)
    shader.add_material(mat_base)

    disk = Disk(device, n=64)
    space_quad = TiledQuad(device, repeats=5)

    root = Node(pipeline)

    trf_bg = Transform()
    trf_bg.scale(WORLD_SIZE * 10, WORLD_SIZE * 10, 1)
    trf_bg.translate(-0.5, -0.5, 0)
    bg_node = Node(trf=trf_bg, apps=[mat_base, tset_space], shps=[space_quad])
    root.add_node(bg_node)

    trf_sun = Transform()
    trf_sun.scale(R_SUN, R_SUN, 1)
    sun_shape = Node(trf=trf_sun, apps=[mat_base, tset_sun], shps=[disk])
    root.add_node(sun_shape)

    trf_merc_orbit = Transform()
    merc_orbit = Node(trf=trf_merc_orbit)
    trf_merc_shape = Transform()
    trf_merc_shape.scale(R_MERCURY, R_MERCURY, 1)
    merc_shape = Node(trf=trf_merc_shape, apps=[mat_base, tset_mercury], shps=[disk])
    merc_orbit.add_node(merc_shape)
    root.add_node(merc_orbit)

    trf_earth_orbit = Transform()
    earth_orbit = Node(trf=trf_earth_orbit)
    trf_earth_shape = Transform()
    earth_shape = Node(trf=trf_earth_shape, apps=[mat_base, tset_earth], shps=[disk])
    earth_orbit.add_node(earth_shape)
    root.add_node(earth_orbit)

    trf_moon_orbit = Transform()
    moon_orbit = Node(trf=trf_moon_orbit)
    trf_moon_shape = Transform()
    trf_moon_shape.scale(R_MOON, R_MOON, 1)
    moon_shape = Node(trf=trf_moon_shape, apps=[mat_base, tset_moon], shps=[disk])
    moon_orbit.add_node(moon_shape)
    earth_orbit.add_node(moon_orbit)

    global scene
    scene = Scene(root)
    engine = SolarEngine(trf_merc_orbit, trf_earth_orbit, trf_moon_orbit, trf_earth_shape)
    scene.add_engine(engine)

def update (dt: float) -> None:
    scene.update(dt)

def draw () -> None:
    global last_t
    t = time.perf_counter()
    update(t - last_t)
    last_t = t

    target_texture = context.get_current_texture()
    renderer.render(target_texture, scene, camera)

def on_key (event: Any) -> None:
    if event["key"] == "q":
        canvas.close()

def main () -> None:
    global canvas, device, context, renderer, last_t

    canvas = RenderCanvas(size=(700, 700), title="INF1761 - Sistema Solar", update_mode="continuous", max_fps=60)
    adapter = wgpu.gpu.request_adapter_sync()
    device = adapter.request_device_sync()
    context = canvas.get_context("wgpu")
    
    target_format = context.get_preferred_format(device.adapter)
    context.configure(device=device, format=target_format)

    renderer = Renderer(device, clear_value=(0.0, 0.0, 0.0, 1.0))

    initialize(device, target_format)

    canvas.add_event_handler(on_key, "key_down")
    last_t = time.perf_counter()
    canvas.request_draw(draw)
    loop.run()

if __name__ == "__main__":
    main()
