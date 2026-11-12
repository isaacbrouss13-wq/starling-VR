//! Starling VR: the game on a Meta Quest through OpenXR — the OpenXR and Vulkan calls the game's
//! crates, which forbid unsafe code, can't make themselves (as horizon-hdr's).
//!
//! OpenXR opens the Vulkan device (`open`): wgpu's own instance and device, created with the
//! extensions the headset's runtime asks for added through wgpu-hal's creation callbacks, so the
//! renderer keeps every feature and limit it asks for on a phone. Each headset frame the renderer
//! (horizon-client-render's `Renderer::render_xr`) calls `begin_frame` for where the eyes are,
//! draws each into the image `acquire` gives, `release`s it and `end_frame` shows both.
//!
//! Everything here runs on the game's thread; the state lives in one global.
use glam::{Quat, Vec3};
use std::ffi::{CStr, CString};
use std::sync::Mutex;
use std::time::Duration;

type Res<T> = Result<T, String>;

fn ctx<E: std::fmt::Debug>(what: &'static str) -> impl FnOnce(E) -> String {
    move |e| format!("{what}: {e:?}")
}

/// How much of the headset's recommended eye resolution the game draws at
/// (`STARLING_VR_SCALE`, 0.4 to 1.5): less for a steadier frame rate.
const DEFAULT_SCALE: f32 = 0.8;
/// The display refresh rate asked for, Hz (`STARLING_VR_HZ`).
const DEFAULT_HZ: f32 = 72.0;
/// A headset lens's edges as angles from the eye's forward axis, in radians (`XrFovf`): `left`
/// and `down` negative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lens {
    pub left: f32,
    pub right: f32,
    pub up: f32,
    pub down: f32,
}

/// A place and a turn in the play area (metres; its floor at y = 0, -Z its forward).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub rotation: Quat,
    pub position: Vec3,
}

/// The controllers, as the game reads a pad: Quest's buttons by place.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Controllers {
    pub a: bool,
    pub b: bool,
    pub x: bool,
    pub y: bool,
    /// The left controller's menu button.
    pub menu: bool,
    pub left_click: bool,
    pub right_click: bool,
    /// Sticks, right and up positive.
    pub left: [f32; 2],
    pub right: [f32; 2],
    /// Index triggers and grips, 0..1, left then right.
    pub triggers: [f32; 2],
    pub grips: [f32; 2],
}

/// One located headset frame.
pub struct Frame {
    time: openxr::Time,
    poses: [openxr::View; 2],
    /// Each eye, and the head.
    pub eyes: [(Pose, Lens); 2],
    pub head: Pose,
    /// The play area moved (the player recentred it): where the head stands is new.
    pub recentred: bool,
}

struct Eye {
    swapchain: openxr::Swapchain<openxr::Vulkan>,
    /// The swapchain's images, as sRGB render targets.
    images: Vec<wgpu::TextureView>,
    /// The image acquired for this frame.
    current: Option<usize>,
}

struct Input {
    set: openxr::ActionSet,
    left_stick: openxr::Action<openxr::Vector2f>,
    right_stick: openxr::Action<openxr::Vector2f>,
    buttons: [openxr::Action<bool>; 7],
    triggers: [openxr::Action<f32>; 2],
    grips: [openxr::Action<f32>; 2],
    haptics: [openxr::Action<openxr::Haptic>; 2],
}

struct Xr {
    instance: openxr::Instance,
    session: openxr::Session<openxr::Vulkan>,
    waiter: openxr::FrameWaiter,
    stream: openxr::FrameStream<openxr::Vulkan>,
    stage: openxr::Space,
    head_space: openxr::Space,
    eyes: [Eye; 2],
    swapchain_size: [u32; 2],
    input: Option<Input>,
    events: openxr::EventDataBuffer,
    running: bool,
    focused: bool,
    exiting: bool,
    recentred: bool,
}

// Safety: the OpenXR handles are used from the game's thread only (all of this module's calls
// come from the renderer and the game loop), behind the mutex.
unsafe impl Send for Xr {}

static XR: Mutex<Option<Xr>> = Mutex::new(None);
static CONTROLLERS: Mutex<Option<Controllers>> = Mutex::new(None);

fn env_f32(name: &str, default: f32, range: std::ops::RangeInclusive<f32>) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| range.contains(v))
        .unwrap_or(default)
}

/// Whether the game runs in the headset.
pub fn active() -> bool {
    XR.lock().unwrap_or_else(|e| e.into_inner()).is_some()
}

/// The controllers' state, for the game's pad, once the headset runs.
pub fn controllers() -> Option<Controllers> {
    *CONTROLLERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A short buzz in a hand (0 right, 1 left), for the pad's rumble.
pub fn buzz(hand: usize, amplitude: f32, millis: u32) {
    let guard = XR.lock().unwrap_or_else(|e| e.into_inner());
    let Some(xr) = guard.as_ref() else { return };
    let Some(input) = &xr.input else { return };
    let pulse = openxr::HapticVibration::new()
        .amplitude(amplitude.clamp(0.0, 1.0))
        .frequency(openxr::FREQUENCY_UNSPECIFIED)
        .duration(openxr::Duration::from_nanos(i64::from(millis) * 1_000_000));
    let _ = input.haptics[hand.min(1)].apply_feedback(&xr.session, openxr::Path::NULL, &pulse);
}

/// Whether the headset asked the game to end (the player quit from the system menu).
pub fn exiting() -> bool {
    XR.lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|x| x.exiting)
}

/// Leaks extension names: wgpu-hal wants `&'static CStr`, and these live as long as the process.
fn leak(names: impl IntoIterator<Item = String>) -> Vec<&'static CStr> {
    names
        .into_iter()
        .filter_map(|s| CString::new(s).ok())
        .map(|s| &*Box::leak(s.into_boxed_c_str()))
        .collect()
}

/// The device the headset's runtime renders with, and what the renderer needs to know about the
/// headset's images.
pub struct Opened {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// The size the game draws each eye at.
    pub size: [u32; 2],
}

/// Opens OpenXR and, through it, the Vulkan device; `features` picks the device's features from
/// the adapter, as the renderer does on a phone (gfx.rs `open_backend`).
pub fn open(features: impl Fn(&wgpu::Adapter) -> wgpu::Features) -> Res<Opened> {
    use wgpu::hal::api::Vulkan;
    #[cfg(target_os = "android")]
    let platform = {
        let android = ndk_context::android_context();
        // Safety: the activity's JavaVM and jobject, valid for the process's life.
        unsafe { openxr::AndroidPlatformInfo::new(android.vm(), android.context()) }
    };
    // Off Android (a desktop runtime, for development): nothing more to say.
    #[cfg(not(target_os = "android"))]
    let platform = ();
    // Safety: loads the loader packaged in the APK (libopenxr_loader.so).
    let entry = unsafe { openxr::Entry::load(&platform) }.map_err(ctx("OpenXR loader"))?;
    let available = entry
        .enumerate_extensions()
        .map_err(ctx("OpenXR extensions"))?;
    if !available.khr_vulkan_enable {
        return Err("the headset's runtime has no XR_KHR_vulkan_enable".into());
    }
    let mut wanted = openxr::ExtensionSet::default();
    wanted.khr_vulkan_enable = true;
    #[cfg(target_os = "android")]
    {
        wanted.khr_android_create_instance = available.khr_android_create_instance;
    }
    wanted.fb_display_refresh_rate = available.fb_display_refresh_rate;
    let app = openxr::ApplicationInfo {
        application_name: "Starling VR",
        application_version: 1,
        engine_name: "Horizon",
        engine_version: 1,
        ..Default::default()
    };
    let instance = entry
        .create_instance(&app, &wanted, &[], &platform)
        .map_err(ctx("OpenXR instance"))?;
    let system = instance
        .system(openxr::FormFactor::HEAD_MOUNTED_DISPLAY)
        .map_err(ctx("no headset"))?;
    if let Ok(p) = instance.system_properties(system) {
        log::info!(
            "Starling VR: headset {} (vendor {})",
            p.system_name,
            p.vendor_id
        );
    }
    // Asked before the instance by the spec, though only the device's version matters here.
    let _ = instance.graphics_requirements::<openxr::Vulkan>(system);
    let instance_extensions = leak(
        instance
            .vulkan_legacy_instance_extensions(system)
            .map_err(ctx("OpenXR's Vulkan instance extensions"))?
            .split_ascii_whitespace()
            .map(str::to_owned),
    );
    let device_extensions = leak(
        instance
            .vulkan_legacy_device_extensions(system)
            .map_err(ctx("OpenXR's Vulkan device extensions"))?
            .split_ascii_whitespace()
            .map(str::to_owned),
    );
    log::info!(
        "Starling VR: the runtime asks for instance extensions {instance_extensions:?}, device \
         extensions {device_extensions:?}"
    );

    // wgpu's Vulkan instance, with the runtime's extensions too.
    let flags = wgpu::InstanceFlags::default().with_env();
    let desc = wgpu::hal::InstanceDescriptor {
        name: "Starling VR",
        flags,
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        telemetry: None,
        display: None,
    };
    let add_instance = instance_extensions.clone();
    // Safety: wgpu-hal's own instance creation, with extensions added (never removed).
    let hal_instance = unsafe {
        wgpu::hal::vulkan::Instance::init_with_callback(
            &desc,
            Some(Box::new(
                move |args: wgpu::hal::vulkan::CreateInstanceCallbackArgs| {
                    for e in &add_instance {
                        if !args.extensions.contains(e) {
                            args.extensions.push(e);
                        }
                    }
                },
            )),
        )
    }
    .map_err(ctx("Vulkan instance"))?;
    let raw_instance = hal_instance.shared_instance().raw_instance().handle();
    // Safety: the instance just made, which the runtime's extensions are enabled on.
    let physical = unsafe {
        instance.vulkan_graphics_device(system, ash::vk::Handle::as_raw(raw_instance) as _)
    }
    .map_err(ctx("the headset's GPU"))?;
    let physical = <ash::vk::PhysicalDevice as ash::vk::Handle>::from_raw(physical as u64);
    // Safety: adapters of this instance.
    let exposed = unsafe { wgpu::hal::Instance::enumerate_adapters(&hal_instance, None) }
        .into_iter()
        .find(|a| a.adapter.raw_physical_device() == physical)
        .ok_or("wgpu can't use the headset's GPU")?;
    // Safety: the hal instance and adapter go to wgpu whole.
    let instance_wgpu = unsafe { wgpu::Instance::from_hal::<Vulkan>(hal_instance) };
    let adapter = unsafe { instance_wgpu.create_adapter_from_hal(exposed) };
    let info = adapter.get_info();
    log::info!("Starling VR: GPU {} ({:?})", info.name, info.backend);
    let features = features(&adapter);
    let limits = adapter.limits();
    let device_desc = wgpu::DeviceDescriptor {
        label: Some("Horizon device"),
        required_features: features,
        required_limits: limits.clone(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        ..Default::default()
    };
    let open = {
        // Safety: only read and opened from here; dropped before wgpu takes the adapter back.
        let hal = unsafe { adapter.as_hal::<Vulkan>() }.ok_or("not a Vulkan adapter")?;
        let add_device = device_extensions.clone();
        // Safety: what wgpu's own request_device does, with the runtime's extensions added.
        unsafe {
            hal.open_with_callback(
                features,
                &limits,
                &wgpu::MemoryHints::MemoryUsage,
                Some(Box::new(
                    move |args: wgpu::hal::vulkan::CreateDeviceCallbackArgs| {
                        for e in &add_device {
                            if !args.extensions.contains(e) {
                                args.extensions.push(e);
                            }
                        }
                    },
                )),
            )
        }
        .map_err(ctx("Vulkan device"))?
    };
    // Safety: opened from this adapter with exactly `device_desc`'s features.
    let (device, queue) = unsafe { adapter.create_device_from_hal(open, &device_desc) }
        .map_err(ctx("wgpu device"))?;

    // The session, on the device just made.
    let (raw_device, family, queue_index) = {
        // Safety: read-only access to the handles, which wgpu keeps alive.
        let hal = unsafe { device.as_hal::<Vulkan>() }.ok_or("not a Vulkan device")?;
        (
            hal.raw_device().handle(),
            hal.queue_family_index(),
            hal.queue_index(),
        )
    };
    // Safety: the instance, GPU and device the runtime asked for, with its extensions.
    let (session, waiter, stream) = unsafe {
        instance.create_session::<openxr::Vulkan>(
            system,
            &openxr::vulkan::SessionCreateInfo {
                instance: ash::vk::Handle::as_raw(raw_instance) as _,
                physical_device: ash::vk::Handle::as_raw(physical) as _,
                device: ash::vk::Handle::as_raw(raw_device) as _,
                queue_family_index: family,
                queue_index,
            },
        )
    }
    .map_err(ctx("OpenXR session"))?;
    if wanted.fb_display_refresh_rate
        && let Ok(rates) = session.enumerate_display_refresh_rates()
    {
        let want = env_f32("STARLING_VR_HZ", DEFAULT_HZ, 60.0..=120.0);
        let pick = rates
            .iter()
            .copied()
            .min_by(|a, b| (a - want).abs().total_cmp(&(b - want).abs()));
        if let Some(rate) = pick {
            let _ = session.request_display_refresh_rate(rate);
            log::info!("Starling VR: {rate} Hz (offered {rates:?})");
        }
    }

    // One swapchain per eye, in an sRGB format both sides have.
    let formats = session
        .enumerate_swapchain_formats()
        .map_err(ctx("swapchain formats"))?;
    let format = [
        ash::vk::Format::R8G8B8A8_SRGB,
        ash::vk::Format::B8G8R8A8_SRGB,
    ]
    .into_iter()
    .find(|f| formats.contains(&(f.as_raw() as u32)))
    .ok_or("the headset offers no 8-bit sRGB swapchain format")?;
    let wgpu_format = if format == ash::vk::Format::R8G8B8A8_SRGB {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Bgra8UnormSrgb
    };
    let configs = instance
        .enumerate_view_configuration_views(system, openxr::ViewConfigurationType::PRIMARY_STEREO)
        .map_err(ctx("eye resolution"))?;
    let view = configs.first().ok_or("no eye")?;
    let recommended = [
        view.recommended_image_rect_width,
        view.recommended_image_rect_height,
    ];
    let max_side = device.limits().max_texture_dimension_2d;
    let swapchain_size = recommended.map(|s| s.clamp(16, max_side));
    let scale = env_f32("STARLING_VR_SCALE", DEFAULT_SCALE, 0.4..=1.5);
    let size = recommended.map(|s| ((s as f32 * scale).round() as u32).clamp(16, max_side));
    log::info!(
        "Starling VR: eye images {}x{} ({wgpu_format:?}), drawn at {}x{} ({scale})",
        swapchain_size[0],
        swapchain_size[1],
        size[0],
        size[1]
    );
    let make_eye = || -> Res<Eye> {
        let swapchain = session
            .create_swapchain(&openxr::SwapchainCreateInfo {
                create_flags: openxr::SwapchainCreateFlags::EMPTY,
                usage_flags: openxr::SwapchainUsageFlags::COLOR_ATTACHMENT,
                format: format.as_raw() as u32,
                sample_count: 1,
                width: swapchain_size[0],
                height: swapchain_size[1],
                face_count: 1,
                array_size: 1,
                mip_count: 1,
            })
            .map_err(ctx("swapchain"))?;
        let extent = wgpu::Extent3d {
            width: swapchain_size[0],
            height: swapchain_size[1],
            depth_or_array_layers: 1,
        };
        let hal_desc = wgpu::hal::TextureDescriptor {
            label: Some("eye image"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::wgt::TextureUses::COLOR_TARGET,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("eye image"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };
        let hal = unsafe { device.as_hal::<Vulkan>() }.ok_or("not a Vulkan device")?;
        let mut images = Vec::new();
        for image in swapchain
            .enumerate_images()
            .map_err(ctx("swapchain images"))?
        {
            // The runtime owns the image: no drop callback, external memory.
            // Safety: an image of this device, made to `hal_desc`.
            let texture = unsafe {
                let raw = hal.texture_from_raw(
                    <ash::vk::Image as ash::vk::Handle>::from_raw(image),
                    &hal_desc,
                    Some(Box::new(|| {})),
                    wgpu::hal::vulkan::TextureMemory::External,
                );
                device.create_texture_from_hal::<Vulkan>(
                    raw,
                    &desc,
                    wgpu::wgt::TextureUses::UNINITIALIZED,
                )
            };
            images.push(texture.create_view(&Default::default()));
        }
        Ok(Eye {
            swapchain,
            images,
            current: None,
        })
    };
    let eyes = [make_eye()?, make_eye()?];
    // The play area, its floor at y = 0; a standing-only fallback.
    let stage = session
        .create_reference_space(openxr::ReferenceSpaceType::STAGE, openxr::Posef::IDENTITY)
        .or_else(|_| {
            let mut floor = openxr::Posef::IDENTITY;
            floor.position.y = -1.6;
            session.create_reference_space(openxr::ReferenceSpaceType::LOCAL, floor)
        })
        .map_err(ctx("play area"))?;
    let head_space = session
        .create_reference_space(openxr::ReferenceSpaceType::VIEW, openxr::Posef::IDENTITY)
        .map_err(ctx("head space"))?;
    let input = match make_input(&instance, &session) {
        Ok(input) => Some(input),
        Err(e) => {
            log::warn!("Starling VR: no controllers ({e})");
            None
        }
    };
    *XR.lock().unwrap_or_else(|e| e.into_inner()) = Some(Xr {
        instance,
        session,
        waiter,
        stream,
        stage,
        head_space,
        eyes,
        swapchain_size,
        input,
        events: openxr::EventDataBuffer::new(),
        running: false,
        focused: false,
        exiting: false,
        recentred: false,
    });
    Ok(Opened {
        instance: instance_wgpu,
        adapter,
        device,
        queue,
        size,
    })
}

fn make_input(
    instance: &openxr::Instance,
    session: &openxr::Session<openxr::Vulkan>,
) -> Res<Input> {
    let set = instance
        .create_action_set("game", "Game", 0)
        .map_err(ctx("action set"))?;
    let stick = |name: &str| {
        set.create_action::<openxr::Vector2f>(name, name, &[])
            .map_err(ctx("action"))
    };
    let button = |name: &str| {
        set.create_action::<bool>(name, name, &[])
            .map_err(ctx("action"))
    };
    let value = |name: &str| {
        set.create_action::<f32>(name, name, &[])
            .map_err(ctx("action"))
    };
    let haptic = |name: &str| {
        set.create_action::<openxr::Haptic>(name, name, &[])
            .map_err(ctx("action"))
    };
    let (left_stick, right_stick) = (stick("left_stick")?, stick("right_stick")?);
    let buttons = [
        button("a")?,
        button("b")?,
        button("x")?,
        button("y")?,
        button("menu")?,
        button("left_click")?,
        button("right_click")?,
    ];
    let triggers = [value("left_trigger")?, value("right_trigger")?];
    let grips = [value("left_grip")?, value("right_grip")?];
    let haptics = [haptic("buzz_right")?, haptic("buzz_left")?];
    let path = |p: &str| instance.string_to_path(p).map_err(ctx("path"));
    let l = |p: &str| path(&format!("/user/hand/left/{p}"));
    let r = |p: &str| path(&format!("/user/hand/right/{p}"));
    instance
        .suggest_interaction_profile_bindings(
            path("/interaction_profiles/oculus/touch_controller")?,
            &[
                openxr::Binding::new(&left_stick, l("input/thumbstick")?),
                openxr::Binding::new(&right_stick, r("input/thumbstick")?),
                openxr::Binding::new(&buttons[0], r("input/a/click")?),
                openxr::Binding::new(&buttons[1], r("input/b/click")?),
                openxr::Binding::new(&buttons[2], l("input/x/click")?),
                openxr::Binding::new(&buttons[3], l("input/y/click")?),
                openxr::Binding::new(&buttons[4], l("input/menu/click")?),
                openxr::Binding::new(&buttons[5], l("input/thumbstick/click")?),
                openxr::Binding::new(&buttons[6], r("input/thumbstick/click")?),
                openxr::Binding::new(&triggers[0], l("input/trigger/value")?),
                openxr::Binding::new(&triggers[1], r("input/trigger/value")?),
                openxr::Binding::new(&grips[0], l("input/squeeze/value")?),
                openxr::Binding::new(&grips[1], r("input/squeeze/value")?),
                openxr::Binding::new(&haptics[0], r("output/haptic")?),
                openxr::Binding::new(&haptics[1], l("output/haptic")?),
            ],
        )
        .map_err(ctx("controller bindings"))?;
    session
        .attach_action_sets(&[&set])
        .map_err(ctx("attach actions"))?;
    Ok(Input {
        set,
        left_stick,
        right_stick,
        buttons,
        triggers,
        grips,
        haptics,
    })
}

fn pose(pose: openxr::Posef) -> Pose {
    let o = pose.orientation;
    let p = pose.position;
    Pose {
        rotation: Quat::from_xyzw(o.x, o.y, o.z, o.w).normalize(),
        position: Vec3::new(p.x, p.y, p.z),
    }
}

fn lens(f: openxr::Fovf) -> Lens {
    Lens {
        left: f.angle_left,
        right: f.angle_right,
        up: f.angle_up,
        down: f.angle_down,
    }
}

impl Xr {
    /// Handles the runtime's events; false while there is no frame to draw.
    fn poll(&mut self) -> Res<bool> {
        while let Some(event) = self
            .instance
            .poll_event(&mut self.events)
            .map_err(ctx("OpenXR events"))?
        {
            use openxr::Event::*;
            match event {
                SessionStateChanged(e) => {
                    log::info!("Starling VR: session {:?}", e.state());
                    match e.state() {
                        openxr::SessionState::READY => {
                            self.session
                                .begin(openxr::ViewConfigurationType::PRIMARY_STEREO)
                                .map_err(ctx("begin session"))?;
                            self.running = true;
                        }
                        openxr::SessionState::STOPPING => {
                            self.session.end().map_err(ctx("end session"))?;
                            self.running = false;
                        }
                        openxr::SessionState::FOCUSED => self.focused = true,
                        openxr::SessionState::VISIBLE => self.focused = false,
                        openxr::SessionState::EXITING | openxr::SessionState::LOSS_PENDING => {
                            self.running = false;
                            self.exiting = true;
                        }
                        _ => {}
                    }
                }
                ReferenceSpaceChangePending(_) => self.recentred = true,
                InstanceLossPending(_) => {
                    self.running = false;
                    self.exiting = true;
                }
                _ => {}
            }
        }
        Ok(self.running)
    }

    fn read_controllers(&self) -> Option<Controllers> {
        let input = self.input.as_ref()?;
        if !self.focused {
            return Some(Controllers::default());
        }
        self.session.sync_actions(&[(&input.set).into()]).ok()?;
        let s = &self.session;
        let stick = |a: &openxr::Action<openxr::Vector2f>| {
            a.state(s, openxr::Path::NULL)
                .ok()
                .filter(|v| v.is_active)
                .map_or([0.0; 2], |v| [v.current_state.x, v.current_state.y])
        };
        let down = |a: &openxr::Action<bool>| {
            a.state(s, openxr::Path::NULL)
                .is_ok_and(|v| v.is_active && v.current_state)
        };
        let value = |a: &openxr::Action<f32>| {
            a.state(s, openxr::Path::NULL)
                .ok()
                .filter(|v| v.is_active)
                .map_or(0.0, |v| v.current_state)
        };
        let b = &input.buttons;
        Some(Controllers {
            a: down(&b[0]),
            b: down(&b[1]),
            x: down(&b[2]),
            y: down(&b[3]),
            menu: down(&b[4]),
            left_click: down(&b[5]),
            right_click: down(&b[6]),
            left: stick(&input.left_stick),
            right: stick(&input.right_stick),
            triggers: [value(&input.triggers[0]), value(&input.triggers[1])],
            grips: [value(&input.grips[0]), value(&input.grips[1])],
        })
    }
}

/// Waits for the headset's next frame and locates the eyes: `None` when there is none to draw
/// (the session not running, or the runtime skipping one).
pub fn begin_frame() -> Res<Option<Frame>> {
    let mut guard = XR.lock().unwrap_or_else(|e| e.into_inner());
    let Some(xr) = guard.as_mut() else {
        return Ok(None);
    };
    if !xr.poll()? {
        drop(guard);
        // No frame loop to pace the game: about one frame's time.
        std::thread::sleep(Duration::from_millis(14));
        return Ok(None);
    }
    let state = xr.waiter.wait().map_err(ctx("wait for the frame"))?;
    xr.stream.begin().map_err(ctx("begin the frame"))?;
    let time = state.predicted_display_time;
    let controllers = xr.read_controllers();
    *CONTROLLERS.lock().unwrap_or_else(|e| e.into_inner()) = controllers;
    if !state.should_render {
        xr.stream
            .end(time, openxr::EnvironmentBlendMode::OPAQUE, &[])
            .map_err(ctx("end the frame"))?;
        return Ok(None);
    }
    let (_, poses) = xr
        .session
        .locate_views(
            openxr::ViewConfigurationType::PRIMARY_STEREO,
            time,
            &xr.stage,
        )
        .map_err(ctx("locate the eyes"))?;
    let (Some(&left), Some(&right)) = (poses.first(), poses.get(1)) else {
        return Err(format!("{} eyes, not 2", poses.len()));
    };
    let eye = |v: openxr::View| (pose(v.pose), lens(v.fov));
    let eyes = [eye(left), eye(right)];
    let head = xr
        .head_space
        .locate(&xr.stage, time)
        .ok()
        .filter(|l| {
            l.location_flags.contains(
                openxr::SpaceLocationFlags::POSITION_VALID
                    | openxr::SpaceLocationFlags::ORIENTATION_VALID,
            )
        })
        .map(|l| pose(l.pose))
        .unwrap_or(Pose {
            rotation: eyes[0].0.rotation,
            position: (eyes[0].0.position + eyes[1].0.position) / 2.0,
        });
    Ok(Some(Frame {
        time,
        poses: [left, right],
        eyes,
        head,
        recentred: std::mem::take(&mut xr.recentred),
    }))
}

/// Acquires eye `eye`'s next image and returns it to draw into.
pub fn acquire(eye: usize) -> Res<wgpu::TextureView> {
    let mut guard = XR.lock().unwrap_or_else(|e| e.into_inner());
    let xr = guard.as_mut().ok_or("no headset")?;
    let e = &mut xr.eyes[eye];
    let index = e
        .swapchain
        .acquire_image()
        .map_err(ctx("acquire an eye image"))? as usize;
    e.swapchain
        .wait_image(openxr::Duration::INFINITE)
        .map_err(ctx("wait for an eye image"))?;
    e.current = Some(index);
    e.images
        .get(index)
        .cloned()
        .ok_or_else(|| "no such eye image".into())
}

/// Hands eye `eye`'s image back once the commands drawing it are submitted.
pub fn release(eye: usize) -> Res<()> {
    let mut guard = XR.lock().unwrap_or_else(|e| e.into_inner());
    let xr = guard.as_mut().ok_or("no headset")?;
    let e = &mut xr.eyes[eye];
    if e.current.take().is_some() {
        e.swapchain
            .release_image()
            .map_err(ctx("release an eye image"))?;
    }
    Ok(())
}

/// Shows the eyes' images.
pub fn end_frame(frame: &Frame) -> Res<()> {
    let mut guard = XR.lock().unwrap_or_else(|e| e.into_inner());
    let xr = guard.as_mut().ok_or("no headset")?;
    let [w, h] = xr.swapchain_size;
    let rect = openxr::Rect2Di {
        offset: openxr::Offset2Di { x: 0, y: 0 },
        extent: openxr::Extent2Di {
            width: w as i32,
            height: h as i32,
        },
    };
    let views = [0, 1].map(|i| {
        openxr::CompositionLayerProjectionView::new()
            .pose(frame.poses[i].pose)
            .fov(frame.poses[i].fov)
            .sub_image(
                openxr::SwapchainSubImage::new()
                    .swapchain(&xr.eyes[i].swapchain)
                    .image_array_index(0)
                    .image_rect(rect),
            )
    });
    let layer = openxr::CompositionLayerProjection::new()
        .space(&xr.stage)
        .views(&views);
    xr.stream
        .end(frame.time, openxr::EnvironmentBlendMode::OPAQUE, &[&layer])
        .map_err(ctx("end the frame"))
}

/// Ends a frame `begin_frame` began without showing anything (a frame that failed to draw).
pub fn end_empty(frame: &Frame) {
    let mut guard = XR.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(xr) = guard.as_mut() {
        for eye in &mut xr.eyes {
            if eye.current.take().is_some() {
                let _ = eye.swapchain.release_image();
            }
        }
        let _ = xr
            .stream
            .end(frame.time, openxr::EnvironmentBlendMode::OPAQUE, &[]);
    }
}

/// The swapchain's format, for the blit's pipeline.
pub fn swapchain_format() -> wgpu::TextureFormat {
    XR.lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|x| x.eyes[0].images.first())
        .map_or(wgpu::TextureFormat::Rgba8UnormSrgb, |v| {
            v.texture().format()
        })
}
