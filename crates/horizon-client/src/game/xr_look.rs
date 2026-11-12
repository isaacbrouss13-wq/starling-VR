//! Starling VR: the player looks where the head does (`horizon_client::gfx::xr::sync_look`), each
//! frame after the mouse and the pad have turned it.
use super::Game;

impl Game<'_> {
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
