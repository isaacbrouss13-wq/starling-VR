//! Starling VR: the player looks where the head does (`horizon_client::gfx::xr::sync_look`), each
//! frame after the mouse and the pad have turned it; and what isn't the world around the player
//! (the title, setup, loading, any menu) shows on a flat screen in front of them; a text box that
//! takes the focus brings up the on-screen keyboard there, its keys clicked with a hand.
use super::Game;

/// The keyboard came up for the field that has the focus (`xr_keyboard`): it comes up once a field.
#[cfg(feature = "xr")]
static FIELD_FOCUSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// A click went to the keyboard: its release does too (`xr_keyboard_event`).
#[cfg(all(target_os = "android", feature = "xr"))]
static CLICK_TAKEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl Game<'_> {
    /// In a headset, the on-screen keyboard comes up when a text box takes the focus and goes
    /// once it loses it (a headset has no keyboard of its own the game can show).
    pub(super) fn xr_keyboard(&mut self) {
        #[cfg(feature = "xr")]
        {
            use std::sync::atomic::Ordering;
            if !horizon_client::gfx::xr::active() {
                return;
            }
            let focused = self.text_field_focused();
            let was = FIELD_FOCUSED.swap(focused, Ordering::Relaxed);
            if focused && !was && self.pad.menu.osk.is_none() {
                self.pad.menu.osk = Some(horizon_client::osk::Keyboard::default());
            } else if !focused && was {
                self.pad.menu.osk = None;
            }
        }
    }

    /// A pointer event while the on-screen keyboard shows (a hand in a headset): over it, the key
    /// lights; a click on it types the key. True when the keyboard took the event.
    #[cfg(all(target_os = "android", feature = "xr"))]
    pub(super) fn xr_keyboard_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        use std::sync::atomic::Ordering;
        use winit::event::{ElementState, MouseButton, WindowEvent};
        let Some(scale) = self
            .renderer
            .as_ref()
            .map(|r| horizon_client::hud::HudBatch::new([r.config.width, r.config.height]).scale)
        else {
            return false;
        };
        let Some(keyboard) = self.pad.menu.osk.as_mut() else {
            CLICK_TAKEN.store(false, Ordering::Relaxed);
            return false;
        };
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                keyboard.hover(position.x as f32 / scale, position.y as f32 / scale);
                false
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                let (x, y) = self.cursor;
                match keyboard.click(x, y) {
                    Some(output) => {
                        CLICK_TAKEN.store(true, Ordering::Relaxed);
                        if let Some(output) = output {
                            self.pad_osk_output(output);
                        }
                        true
                    }
                    None => false,
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => CLICK_TAKEN.swap(false, Ordering::Relaxed),
            _ => false,
        }
    }

    /// Tells the headset whether this frame is the world around the player or a flat screen.
    pub(super) fn xr_screen_mode(&self) {
        #[cfg(feature = "xr")]
        horizon_client::gfx::xr::set_flat(
            self.player.is_none()
                || self.pad_in_menu()
                || self.integrated_starting.is_some()
                || self.pending_entry.is_some()
                || self.pack_reload.is_some(),
        );
    }

    pub(super) fn xr_look(&mut self) {
        #[cfg(feature = "xr")]
        if self.player.is_some()
            && let Some((yaw, pitch)) = horizon_client::gfx::xr::sync_look(self.yaw)
        {
            self.yaw = yaw;
            self.pitch = pitch;
            if let Some(m) = self.player_model.as_mut() {
                m.head_pitch = pitch;
            }
        }
    }
}
