//! Starling VR: the player looks where the head does (`horizon_client::gfx::xr::sync_look`), each
//! frame after the mouse and the pad have turned it; and what isn't the world around the player
//! (the title, setup, loading, any menu) shows on a flat screen in front of them.
use super::Game;

impl Game<'_> {
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
