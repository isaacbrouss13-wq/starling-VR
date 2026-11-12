//! Starling VR's glue between the headset (starling-xr) and the game: the views a headset frame is
//! drawn from (`begin_frame`), where the player looks (`sync_look`) and the controllers.
//!
//! The player looks with the head: `sync_look` turns the game's yaw into the body's turn plus the
//! head's (a turn the game makes, by a stick or a respawn, turns the body) and the pitch into the
//! head's. The controllers reach the game as a pad.
use super::headset::{self, Look, Pose};
use super::view::{Fov, View};
use glam::Vec3;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

pub use starling_xr::{Controllers, buzz, controllers, exiting};
pub(super) use starling_xr::{
    Opened, SCREEN, SCREEN_PIXELS, acquire, end_empty, end_frame, end_frame_screen, hide_screen,
    open, release, swapchain_format,
};

static LOOK: Mutex<Look> = Mutex::new(Look::new());
/// The frame shows on the flat screen (`set_flat`), not in the eyes.
static FLAT: AtomicBool = AtomicBool::new(true);
/// Where the head stood when play began (or the play area was recentred): the view leans from here.
static ANCHOR: Mutex<Option<Vec3>> = Mutex::new(None);

/// Whether the game runs in the headset.
pub fn active() -> bool {
    starling_xr::active()
}

/// Whether the next frames show on a flat screen floating in front of the player (the title,
/// setup, loading and any menu: what isn't the world around them) or in the eyes. The game says
/// each frame.
pub fn set_flat(flat: bool) {
    FLAT.store(flat, Ordering::Relaxed);
}

pub(super) fn flat() -> bool {
    FLAT.load(Ordering::Relaxed)
}

/// The player's yaw and pitch for this frame from the head, given the yaw the game has now: `None`
/// before the headset has located the head.
pub fn sync_look(game_yaw: f32) -> Option<(f32, f32)> {
    LOOK.lock()
        .unwrap_or_else(|e| e.into_inner())
        .sync(game_yaw)
}

/// One located headset frame: each eye camera-relative, and the frustum that holds both.
pub(super) struct FrameViews {
    pub frame: starling_xr::Frame,
    pub eyes: [View; 2],
    pub cull: View,
}

fn pose(p: starling_xr::Pose) -> Pose {
    Pose {
        rotation: p.rotation,
        position: p.position,
    }
}

fn fov(l: starling_xr::Lens) -> Fov {
    Fov {
        left: l.left,
        right: l.right,
        up: l.up,
        down: l.down,
    }
}

/// Waits for the headset's next frame: `None` when there is none to draw.
pub(super) fn begin_frame(far: f32) -> Result<Option<FrameViews>, String> {
    let Some(frame) = starling_xr::begin_frame()? else {
        return Ok(None);
    };
    let head = pose(frame.head);
    let anchor = {
        let mut anchor = ANCHOR.lock().unwrap_or_else(|e| e.into_inner());
        if frame.recentred {
            *anchor = None;
        }
        *anchor.get_or_insert(head.position)
    };
    let body = {
        let mut look = LOOK.lock().unwrap_or_else(|e| e.into_inner());
        look.head = Some(head.rotation);
        look.body()
    };
    let eyes = frame.eyes.map(|(p, l)| (pose(p), fov(l)));
    let (eyes, cull) = headset::eye_views(head, eyes, anchor, body, far);
    Ok(Some(FrameViews { frame, eyes, cull }))
}
