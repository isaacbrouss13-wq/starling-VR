//! What a headset adds to a frame, apart from the headset's API (starling-xr, through xr.rs): where
//! the player looks, how far the view leans with the head, each eye's view, and the copy of the
//! frame into an eye's image.
use super::view::{self, Fov, View};
use glam::{Quat, Vec3};

/// How much wider the frustum culled with before the frame is (a frame-old head pose), radians.
pub const CULL_MARGIN: f32 = 0.25;

/// How far the head may lean from where the player stands before the view stops following it,
/// metres: sideways, down and up.
pub const LEAN_SIDEWAYS: f32 = 0.35;
pub const LEAN_DOWN: f32 = 1.0;
pub const LEAN_UP: f32 = 0.3;

/// The look: the body's turn (vanilla yaw, degrees) the head turns within. The game turns the
/// body (a stick, a respawn's facing); the head turns the view within it.
#[derive(Clone, Copy, Debug)]
pub struct Look {
    body_yaw: f32,
    /// The yaw last handed to the game.
    last_set: Option<f32>,
    /// The head's rotation in the play area, last located.
    pub head: Option<Quat>,
}

impl Default for Look {
    fn default() -> Self {
        Self::new()
    }
}

fn wrap(degrees: f32) -> f32 {
    (degrees + 540.0).rem_euclid(360.0) - 180.0
}

impl Look {
    pub const fn new() -> Self {
        Self {
            body_yaw: 0.0,
            last_set: None,
            head: None,
        }
    }

    /// The play area's rotation in the world.
    pub fn body(&self) -> Quat {
        view::body_rotation(self.body_yaw)
    }

    /// The player's yaw and pitch from the head, given the yaw the game has now; `None` before
    /// the head is located.
    pub fn sync(&mut self, game_yaw: f32) -> Option<(f32, f32)> {
        let head = self.head?;
        match self.last_set {
            // A turn the game made since: the body turns by it.
            Some(last) => self.body_yaw += wrap(game_yaw - last),
            // The first frame: the body faces where the player does, less the head's turn.
            None => self.body_yaw = game_yaw - view::yaw_pitch(view::body_rotation(0.0) * head).0,
        }
        self.body_yaw = self.body_yaw.rem_euclid(360.0);
        let (yaw, pitch) = view::yaw_pitch(self.body() * head);
        self.last_set = Some(yaw);
        Some((yaw, pitch))
    }
}

/// The head's lean from `anchor` (where it stood when play began), kept within reach of where
/// the player stands.
pub fn lean(head: Vec3, anchor: Vec3) -> Vec3 {
    let d = head - anchor;
    let side = Vec3::new(d.x, 0.0, d.z);
    let side = if side.length() > LEAN_SIDEWAYS {
        side.normalize() * LEAN_SIDEWAYS
    } else {
        side
    };
    side + Vec3::Y * d.y.clamp(-LEAN_DOWN, LEAN_UP)
}

/// Copies the frame (sRGB-encoded colour in a linear format) into an eye's sRGB image: decoded
/// to linear, which the image's format encodes again.
pub struct Blit {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    cursor: wgpu::Buffer,
}

const BLIT_WGSL: &str = r"
@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var bilinear: sampler;
// Where a hand points (0..1 from the left and top), x negative for none.
@group(0) @binding(2) var<uniform> cursor: vec4<f32>;
struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: Out;
    o.pos = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    o.uv = uv;
    return o;
}
fn linear(c: vec3<f32>) -> vec3<f32> {
    let low = c / 12.92;
    let high = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, c <= vec3<f32>(0.04045));
}
@fragment fn fs(i: Out) -> @location(0) vec4<f32> {
    var c = linear(textureSample(frame, bilinear, i.uv).rgb);
    if (cursor.x >= 0.0) {
        // A dot, round on the screen whatever its shape: white, ringed dark to show on white.
        let size = vec2<f32>(textureDimensions(frame));
        let d = length((i.uv - cursor.xy) * size) / size.y;
        if (d < 0.009) {
            c = vec3<f32>(1.0);
        } else if (d < 0.013) {
            c = vec3<f32>(0.02);
        }
    }
    return vec4<f32>(c, 1.0);
}
";

impl Blit {
    pub fn new(device: &wgpu::Device, target: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Starling VR blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Starling VR blit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Starling VR blit"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Starling VR blit"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Starling VR blit"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let cursor = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Starling VR cursor"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            layout,
            sampler,
            cursor,
        }
    }

    /// Draws `frame` over all of `target`, with a dot where a hand points (`cursor`: 0..1 from
    /// the left and top).
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &wgpu::TextureView,
        target: &wgpu::TextureView,
        cursor: Option<[f32; 2]>,
    ) {
        let [x, y] = cursor.unwrap_or([-1.0, -1.0]);
        queue.write_buffer(&self.cursor, 0, bytemuck::cast_slice(&[x, y, 0.0f32, 0.0]));
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Starling VR blit"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(frame),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.cursor.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Starling VR blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Memory, for a headset's log (Starling VR): what the system has free and what the game holds,
/// from Linux's /proc (Android's too). A Quest shares its memory with the headset's own system,
/// and Android ends a game that takes too much without a word in the game's own log.
pub fn memory_note() -> String {
    let field = |path: &str, name: &str| -> Option<u64> {
        let text = std::fs::read_to_string(path).ok()?;
        let line = text.lines().find(|l| l.starts_with(name))?;
        line.split_whitespace().nth(1)?.parse().ok()
    };
    let mb = |kb: Option<u64>| kb.map_or_else(|| "?".to_owned(), |kb| format!("{} MB", kb / 1024));
    format!(
        "system free {}, of {}; game {} (GPU memory not counted)",
        mb(field("/proc/meminfo", "MemAvailable:")),
        mb(field("/proc/meminfo", "MemTotal:")),
        mb(field("/proc/self/status", "VmRSS:")),
    )
}

/// A place and a turn in the play area (metres; its floor at y = 0, -Z its forward).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub rotation: Quat,
    pub position: Vec3,
}

/// Each eye's camera-relative view and the frustum that holds both, for a head in the play area
/// turned by `body` in the world: the camera sits where the player's eyes are, the view leans with
/// the head from `anchor` (`lean`), and each eye looks from its own place through its lens.
pub fn eye_views(
    head: Pose,
    eyes: [(Pose, Fov); 2],
    anchor: Vec3,
    body: Quat,
    far: f32,
) -> ([View; 2], View) {
    let leaned = lean(head.position, anchor);
    let placed = eyes.map(|(p, f)| {
        (
            body * p.rotation,
            body * (p.position - head.position + leaned),
            f,
        )
    });
    let views = placed.map(|(q, o, f)| View::headset(q, o, f, far));
    let cull = view::stereo_cull(
        body * head.rotation,
        placed.map(|(_, o, f)| (o, f)),
        CULL_MARGIN,
        far,
    );
    (views, cull)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A head turned by vanilla yaw and pitch from the play area's forward.
    fn head(yaw: f32, pitch: f32) -> Quat {
        Quat::from_rotation_y(-yaw.to_radians()) * Quat::from_rotation_x(-pitch.to_radians())
    }

    fn near(a: f32, b: f32) -> bool {
        wrap(a - b).abs() < 1e-2
    }

    #[test]
    fn the_player_looks_where_the_head_does_and_keeps_its_facing_at_first() {
        let mut look = Look::new();
        assert_eq!(look.sync(30.0), None, "nothing before the head is found");
        look.head = Some(head(20.0, 10.0));
        let (yaw, pitch) = look.sync(130.0).unwrap();
        assert!(
            near(yaw, 130.0) && (pitch - 10.0).abs() < 1e-3,
            "{yaw} {pitch}"
        );
        // The head turns 15° more: so does the player.
        look.head = Some(head(35.0, -40.0));
        let (yaw, pitch) = look.sync(yaw).unwrap();
        assert!(
            near(yaw, 145.0) && (pitch + 40.0).abs() < 1e-3,
            "{yaw} {pitch}"
        );
    }

    #[test]
    fn a_turn_the_game_makes_turns_the_body() {
        let mut look = Look::new();
        look.head = Some(head(0.0, 0.0));
        let (yaw, _) = look.sync(0.0).unwrap();
        // A stick turns the player 45° right, across the wrap at 180°.
        let (yaw, _) = look.sync(yaw + 45.0).unwrap();
        assert!(near(yaw, 45.0));
        let (yaw, _) = look.sync(yaw).unwrap();
        assert!(near(yaw, 45.0), "and stays turned");
        look.head = Some(head(170.0, 0.0));
        let (yaw, _) = look.sync(yaw).unwrap();
        let (yaw, _) = look.sync(yaw + 30.0).unwrap();
        assert!(near(yaw, 245.0), "{yaw}");
    }

    #[test]
    fn the_blit_shader_is_valid_wgsl() {
        let module = naga::front::wgsl::parse_str(BLIT_WGSL).expect("parses");
        naga::valid::Validator::new(Default::default(), naga::valid::Capabilities::empty())
            .validate(&module)
            .expect("validates");
    }

    #[test]
    fn the_eyes_sit_about_the_camera_and_turn_with_the_body() {
        let fov = Fov {
            left: -0.9,
            right: 0.7,
            up: 0.8,
            down: -0.9,
        };
        let at = |x: f32| Pose {
            rotation: Quat::IDENTITY,
            position: Vec3::new(x, 1.6, 0.0),
        };
        let head = at(0.0);
        let body = view::body_rotation(90.0);
        let ([left, right], cull) = eye_views(
            head,
            [(at(-0.03), fov), (at(0.03), fov)],
            head.position,
            body,
            512.0,
        );
        // Facing west (yaw 90): the left eye sits to the south (+Z) of the camera.
        let eye_at = |v: &View| v.view.inverse().transform_point3(Vec3::ZERO);
        assert!(
            (eye_at(&left) - Vec3::new(0.0, 0.0, 0.03)).length() < 1e-5,
            "{}",
            eye_at(&left)
        );
        assert!((eye_at(&right) - Vec3::new(0.0, 0.0, -0.03)).length() < 1e-5);
        // Both look west, and so does the cull.
        let forward = |v: &View| v.view.inverse().transform_vector3(Vec3::NEG_Z);
        for v in [&left, &right, &cull] {
            assert!((forward(v) - Vec3::NEG_X).length() < 1e-5);
        }
    }

    #[test]
    fn leaning_follows_the_head_within_reach() {
        let anchor = Vec3::new(0.2, 1.6, -0.1);
        let small = Vec3::new(0.1, -0.05, 0.1);
        assert!((lean(anchor + small, anchor) - small).length() < 1e-6);
        let far = lean(anchor + Vec3::new(3.0, 0.0, 4.0), anchor);
        assert!((far - Vec3::new(0.6, 0.0, 0.8) * LEAN_SIDEWAYS).length() < 1e-5);
        assert!((lean(anchor - Vec3::Y * 5.0, anchor).y + LEAN_DOWN).abs() < 1e-6);
    }
}
