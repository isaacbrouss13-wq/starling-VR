//! The views a frame is drawn from. A flat screen has one; a headset has
//! one per eye. Every view shares the frame's camera position, so the
//! camera-relative world (section origins, entity offsets) is built once and
//! only each view's rotation and lens differ.

use glam::{Mat4, Quat, Vec3, Vec4};

/// The most views one frame draws.
const MAX_VIEWS: usize = 2;

/// A headset lens's edges as angles from the eye's forward axis, in
/// radians, the way OpenXR reports them (`XrFovf`): `left` and `down`
/// negative, `right` and `up` positive. A headset's lenses are usually
/// off-centre, so `left` and `right` differ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fov {
    pub left: f32,
    pub right: f32,
    pub up: f32,
    pub down: f32,
}

impl Fov {
    /// The four edges as tangents: `[left, right, up, down]`.
    fn tangents(self) -> [f32; 4] {
        [
            self.left.tan(),
            self.right.tan(),
            self.up.tan(),
            self.down.tan(),
        ]
    }
}

/// Reversed-Z, right-handed, off-centre perspective from edge tangents, the
/// same convention as `world_projection`: depth 1 at `near`, 0 at `far`.
fn off_center_projection([l, r, u, d]: [f32; 4], near: f32, far: f32) -> Mat4 {
    let depth = near / (far - near);
    Mat4::from_cols(
        Vec4::new(2.0 / (r - l), 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / (u - d), 0.0, 0.0),
        Vec4::new((r + l) / (r - l), (u + d) / (u - d), depth, -1.0),
        Vec4::new(0.0, 0.0, depth * far, 0.0),
    )
}

/// The rotation a headset's play area has in the world when the body faces
/// vanilla yaw `body_yaw` (degrees): the play area's forward (-Z) along that
/// yaw. The same turn `Camera.rotation` starts from (`camera::view_matrix`).
pub fn body_rotation(body_yaw: f32) -> Quat {
    Quat::from_rotation_y(std::f32::consts::PI - body_yaw.to_radians())
}

/// Vanilla yaw and pitch (degrees: yaw 0 south and growing to the west,
/// pitch positive down) of a rotation's forward (-Z) axis in the world.
pub fn yaw_pitch(rotation: Quat) -> (f32, f32) {
    let f = rotation * Vec3::NEG_Z;
    let yaw = (-f.x).atan2(f.z).to_degrees();
    let pitch = -f.y.clamp(-1.0, 1.0).asin().to_degrees();
    (yaw, pitch)
}

/// One eye's view of the camera-relative world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// Camera-relative world to eye: a rotation, plus the eye's offset from
    /// the camera on a headset.
    pub view: Mat4,
    /// Reversed-Z perspective: the near plane at depth 1, far at 0.
    pub projection: Mat4,
    /// Width over height of the image this view fills.
    pub aspect: f32,
    /// `tan(fov_y / 2)`, for shaders rebuilding view rays from NDC.
    pub tan_half_fov: f32,
}

impl View {
    /// The flat-screen view: vanilla's camera rotation and a symmetric
    /// vertical field of view.
    pub fn screen(yaw: f32, pitch: f32, fov_degrees: f32, aspect: f32, far: f32) -> Self {
        Self {
            view: crate::camera::view_matrix(yaw, pitch),
            projection: super::world_projection(fov_degrees, aspect, far),
            aspect,
            tan_half_fov: (fov_degrees.to_radians() / 2.0).tan(),
        }
    }

    /// A headset eye: `rotation` turns the eye's space into the world's,
    /// `offset` is where the eye sits from the camera (in blocks, one block a
    /// metre), and it looks through its own lens.
    ///
    /// `tan_half_fov` is the vertical half-extent's mean: exact for a lens
    /// centred vertically, close for one that isn't.
    pub fn headset(rotation: Quat, offset: Vec3, fov: Fov, far: f32) -> Self {
        let [l, r, u, d] = fov.tangents();
        Self {
            view: Mat4::from_quat(rotation.conjugate()) * Mat4::from_translation(-offset),
            projection: off_center_projection([l, r, u, d], super::NEAR_PLANE, far),
            aspect: (r - l) / (u - d),
            tan_half_fov: (u - d) / 2.0,
        }
    }

    /// Clip space from the camera-relative world, with `pose` (view bobbing,
    /// nausea spin) between the lens and the rotation.
    pub fn view_proj(&self, pose: Mat4) -> Mat4 {
        self.projection * pose * self.view
    }
}

/// Every view of a frame, and the one geometry is culled with.
#[derive(Clone, Copy, Debug)]
pub struct ViewSet {
    views: [View; MAX_VIEWS],
    len: usize,
    cull: View,
}

impl ViewSet {
    /// A frame drawn from `view` alone, culled with it.
    pub fn single(view: View) -> Self {
        Self {
            views: [view; MAX_VIEWS],
            len: 1,
            cull: view,
        }
    }

    /// One headset eye's frame: drawn from `view`, culled with `cull` (one
    /// frustum that holds both eyes, so both cull alike).
    pub fn eye(view: View, cull: View) -> Self {
        Self {
            views: [view; MAX_VIEWS],
            len: 1,
            cull,
        }
    }

    /// The view the HUD, hand and screen-space effects follow.
    pub fn primary(&self) -> &View {
        &self.views[0]
    }

    pub fn views(&self) -> &[View] {
        &self.views[..self.len]
    }

    /// The view whose frustum covers every other view's.
    pub fn cull(&self) -> &View {
        &self.cull
    }
}

/// One frustum that holds both eyes of a headset: their widest edges, seen
/// from behind them, far enough back that its edges clear each eye's own.
/// With the eyes `h` apart from their midpoint sideways, an edge of tangent
/// `t` clears both when the apex is `h / t` behind. Assumes the eyes face
/// the same way (`rotation`, the head's), as Quest's do; `margin` widens
/// every edge by that many radians, for a view a frame old (culling built
/// before the headset says where the eyes are).
#[cfg_attr(not(feature = "xr"), allow(dead_code))]
pub fn stereo_cull(rotation: Quat, eyes: [(Vec3, Fov); 2], margin: f32, far: f32) -> View {
    let widen = |f: Fov| Fov {
        left: (f.left - margin).max(-1.5),
        right: (f.right + margin).min(1.5),
        up: (f.up + margin).min(1.5),
        down: (f.down - margin).max(-1.5),
    };
    let [a, b] = eyes.map(|(_, f)| widen(f).tangents());
    let edges = [
        a[0].min(b[0]),
        a[1].max(b[1]),
        a[2].max(b[2]),
        a[3].min(b[3]),
    ];
    // The eyes' offsets in the head's own space.
    let local = eyes.map(|(o, _)| rotation.conjugate() * o);
    let mid = (local[0] + local[1]) / 2.0;
    let half = (local[0] - local[1]).abs() / 2.0;
    let sideways = half.x / (-edges[0]).min(edges[1]);
    let vertical = half.y / edges[2].min(-edges[3]);
    let back = sideways.max(vertical).max(0.0);
    // The head looks down -Z, so behind the eyes is +Z.
    let apex = rotation * (mid + Vec3::new(0.0, 0.0, back));
    View {
        view: Mat4::from_quat(rotation.conjugate()) * Mat4::from_translation(-apex),
        projection: off_center_projection(edges, super::NEAR_PLANE, far + back),
        aspect: (edges[1] - edges[0]) / (edges[2] - edges[3]),
        tan_half_fov: (edges[2] - edges[3]) / 2.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The formulas `Renderer` used before views existed, kept to pin the
    /// screen view to them bit for bit.
    fn legacy_view_proj(fov: f32, yaw: f32, pitch: f32, pose: Mat4, aspect: f32, far: f32) -> Mat4 {
        super::super::world_projection(fov, aspect, far)
            * pose
            * crate::camera::view_matrix(yaw, pitch)
    }

    fn poses() -> [Mat4; 3] {
        [
            Mat4::IDENTITY,
            Mat4::from_translation(glam::Vec3::new(0.013, -0.027, 0.0))
                * Mat4::from_rotation_z(0.021),
            Mat4::from_rotation_y(0.4) * Mat4::from_scale(glam::Vec3::new(1.1, 0.9, 1.0)),
        ]
    }

    #[test]
    fn the_screen_view_matches_the_legacy_projection_bit_for_bit() {
        for (fov, yaw, pitch, aspect, far) in [
            (70.0, 0.0, 0.0, 16.0 / 9.0, 512.0),
            (110.0, -137.25, 61.5, 21.0 / 9.0, 8192.0),
            (30.0, 359.9, -90.0, 4.0 / 3.0, 96.0),
            (82.333, 12.5, 7.75, 1.0, 1024.0),
        ] {
            let view = View::screen(yaw, pitch, fov, aspect, far);
            for pose in poses() {
                let old = legacy_view_proj(fov, yaw, pitch, pose, aspect, far);
                let new = view.view_proj(pose);
                assert_eq!(
                    old.to_cols_array().map(f32::to_bits),
                    new.to_cols_array().map(f32::to_bits),
                    "fov {fov} yaw {yaw} pitch {pitch}"
                );
            }
            assert_eq!(view.view, crate::camera::view_matrix(yaw, pitch));
            assert_eq!(
                view.tan_half_fov.to_bits(),
                (fov.to_radians() / 2.0).tan().to_bits()
            );
            assert_eq!(view.aspect.to_bits(), aspect.to_bits());
        }
    }

    #[test]
    fn a_single_view_set_draws_and_culls_with_its_view() {
        let view = View::screen(45.0, 10.0, 70.0, 16.0 / 9.0, 512.0);
        let set = ViewSet::single(view);
        assert_eq!(set.views(), &[view]);
        assert_eq!(*set.primary(), view);
        assert_eq!(*set.cull(), view);
    }

    /// Lenses shaped like a Quest 3's: each wider toward the far side of the
    /// nose, 63 mm between the eyes.
    fn quest_eyes(rotation: Quat) -> [(Vec3, Fov); 2] {
        let left = Fov {
            left: -0.942,
            right: 0.698,
            up: 0.768,
            down: -0.890,
        };
        let right = Fov {
            left: -left.right,
            right: -left.left,
            ..left
        };
        [
            (rotation * Vec3::new(-0.0315, 0.0, 0.0), left),
            (rotation * Vec3::new(0.0315, 0.0, 0.0), right),
        ]
    }

    fn close(a: Mat4, b: Mat4) -> bool {
        a.to_cols_array()
            .iter()
            .zip(b.to_cols_array())
            .all(|(x, y)| (x - y).abs() < 1e-5)
    }

    /// A head turned by vanilla yaw and pitch inside a play area turned by `body`.
    fn head(body: f32, yaw: f32, pitch: f32) -> Quat {
        body_rotation(body)
            * Quat::from_rotation_y(-yaw.to_radians())
            * Quat::from_rotation_x(-pitch.to_radians())
    }

    #[test]
    fn a_head_without_roll_sees_what_vanillas_camera_sees() {
        for (body, yaw, pitch) in [
            (0.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
            (-30.0, 75.0, 40.0),
            (170.0, 25.0, -80.0),
        ] {
            let rotation = head(body, yaw, pitch);
            let eye = View::headset(
                rotation,
                Vec3::ZERO,
                Fov {
                    left: -0.8,
                    right: 0.8,
                    up: 0.7,
                    down: -0.7,
                },
                512.0,
            );
            assert!(
                close(eye.view, crate::camera::view_matrix(body + yaw, pitch)),
                "{body} {yaw} {pitch}"
            );
            let (y, p) = yaw_pitch(rotation);
            let wrap = |a: f32| (a + 540.0).rem_euclid(360.0) - 180.0;
            assert!(
                wrap(y - (body + yaw)).abs() < 1e-3 && (p - pitch).abs() < 1e-3,
                "{y} {p}"
            );
        }
    }

    #[test]
    fn a_symmetric_lens_is_the_screens_projection() {
        let half = 35f32.to_radians();
        let aspect = 1.2f32;
        let side = (half.tan() * aspect).atan();
        let fov = Fov {
            left: -side,
            right: side,
            up: half,
            down: -half,
        };
        let eye = View::headset(Quat::IDENTITY, Vec3::ZERO, fov, 512.0);
        assert!(close(
            eye.projection,
            super::super::world_projection(70.0, aspect, 512.0)
        ));
    }

    #[test]
    fn an_eye_sees_from_its_offset() {
        // The left eye's view puts the camera 3 cm to its right.
        let eye = View::headset(
            Quat::IDENTITY,
            Vec3::new(-0.03, 0.0, 0.0),
            quest_eyes(Quat::IDENTITY)[0].1,
            512.0,
        );
        let camera = eye.view.transform_point3(Vec3::ZERO);
        assert!((camera - Vec3::new(0.03, 0.0, 0.0)).length() < 1e-6);
    }

    /// Whether a camera-relative point is inside a view's frustum.
    fn sees(view: &View, p: Vec3) -> bool {
        let c = view.view_proj(Mat4::IDENTITY) * p.extend(1.0);
        c.w > 0.0 && c.x.abs() <= c.w && c.y.abs() <= c.w && c.z >= 0.0 && c.z <= c.w
    }

    #[test]
    fn the_stereo_cull_holds_what_either_eye_sees() {
        let rotation = head(40.0, -20.0, 15.0);
        let eyes = quest_eyes(rotation);
        let cull = stereo_cull(rotation, eyes, 0.0, 512.0);
        let views = eyes.map(|(o, f)| View::headset(rotation, o, f, 512.0));
        let mut seen = 0;
        for i in 0..4000 {
            // Points spread around the head, near and far.
            let t = i as f32;
            let dir =
                Vec3::new((t * 0.37).sin(), (t * 0.23).cos(), (t * 0.11).sin() - 0.3).normalize();
            let p = dir * (0.2 + (t * 0.071).fract() * 300.0);
            if views.iter().any(|v| sees(v, p)) {
                seen += 1;
                assert!(sees(&cull, p), "{p}");
            }
        }
        assert!(seen > 200);
        // A margin only widens it.
        let wide = stereo_cull(rotation, eyes, 0.1, 512.0);
        let edge = rotation * Vec3::new(-1.6, 0.0, -1.0) * 10.0;
        assert!(!sees(&cull, edge) && sees(&wide, edge));
    }
}
