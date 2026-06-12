# Perfect Minecraft Living Ecosystem Plan

## Goal

Build a new flagship Garage Life Lab world that feels like a living Minecraft-like ecosystem while remaining a credible PC stress test. The world should stay block-readable, preserve the existing Python + ModernGL world contract, and add view-focused shader complexity instead of rewriting the engine.

## Current Architecture Evidence

- Worlds are data modules registered through `worlds/registry.py`. The registry imports each module, stores each module's `SPEC`, and exposes `iter_worlds()`, `world_ids()`, and `get_world()` (`worlds/registry.py:31`, `worlds/registry.py:33`, `worlds/registry.py:59`).
- The engine consumes a `WorldSpec` containing `id`, `display_name`, `window_title`, `sim_shader`, `display_shader`, `seed_field`, `default_overrides`, preview metadata, HUD subtitle, and optional audio (`worlds/spec.py:13`).
- Runtime defaults are merged with `world.default_overrides` before GPU programs are built (`engine.py:548`, `engine.py:583`, `engine.py:629`).
- Launcher presets pass explicit values for resolution, tile size, substeps, CPU workers, CPU matrix, glow, exposure, gamma, contour contrast, ray steps, FX intensity, camera speed, and thermal limits (`garage_life_presets.py:61`). Because `launch_command()` appends those preset args after `--world`, launcher runs bypass most per-world `default_overrides` unless the user clears or edits fields (`launcher.py:656`).
- The persistent simulation budget is exactly one ping-ponged RGBA32F texture per world. The engine creates two 4-channel float textures, seeds them from `seed_field()`, and validates the seed shape as `(height, width, 4)` (`engine.py:944`, `engine.py:973`).
- Simulation textures are currently linearly filtered and wrap on both axes (`engine.py:948`). That supports smooth analog fields, but it means exact persistent block IDs, hard biome IDs, inventories, pathfinding, mobs, and multi-layer chunk state are out of scope unless the engine grows additional discrete/storage textures.
- `raySteps` is clamped by the engine to 32 through 160 before reaching display shaders (`engine.py:996`). Higher command-line or world default values may still document intent in old worlds, but they are not effective without an engine change.
- Audio samplers are only initialized for worlds with `uses_audio=True` (`engine.py:597`). The target world should stay non-audio for the first implementation unless it explicitly opts into those uniforms.
- The engine already exposes the needed stress and safety controls: `--world`, `--substeps`, `--ray-steps`, `--fx-intensity`, `--tile-size`, `--cpu-workers`, `--cpu-matrix`, thermal hold, HUD, camera controls, and `--quit-after` (`engine.py:1261`, `engine.py:1269`, `engine.py:1279`, `engine.py:1286`, `engine.py:1292`, `engine.py:1295`).
- `scripts/smoke_worlds.ps1` discovers registered worlds and smoke-runs each with low resolution, `--tile-size 24`, `--substeps 1`, `--ray-steps 32`, `--fx-intensity 0.3`, no CPU workers, no thermal hold, no HUD, and `--quit-after 1.0` (`scripts/smoke_worlds.ps1:11`, `scripts/smoke_worlds.ps1:22`).

## Target Implementation Path

Add a new world module and register it alongside the existing Minecraft worlds.

- Target world id: `minecraft-perfect-ecosystem-3d`
- Display name: `Minecraft Perfect Ecosystem`
- Source module: `worlds/minecraft_perfect_ecosystem_3d.py`
- Registry placement: import it in `worlds/registry.py`, add its `SPEC` to `_WORLD_ORDER` ahead of `minecraft-long-term-3d`, then make it the default only after smoke and visual checks pass.
- Primary base: fork the shader structure of `minecraft-long-term-3d`, not the DDA experiments. It already has the strongest balanced combination of blocky raymarched SDFs, terrain-aware seed fields, villages, trees, water, weather, lighting, camera controls, and moderate defaults (`worlds/minecraft_long_term_3d.py:27`, `worlds/minecraft_long_term_3d.py:164`, `worlds/minecraft_long_term_3d.py:584`, `worlds/minecraft_long_term_3d.py:711`).
- Stress strategy: keep default settings survivable, but make the shader scale hard with existing `ray_steps`, `fx_intensity`, `substeps`, `tile_size`, and external CPU burners. Heavy visual paths can key off `fxIntensity` so smoke tests remain cheap.
- Launcher strategy: if this becomes the flagship/default, review `garage_life_presets.py` and launcher command preview behavior so Medium/High/Ultra presets launch the new world with intended effective settings instead of silently using generic preset values that flatten the world-specific tuning.

Do not replace the engine, add new global controls, add chunk storage, or port the recursive DDA stack wholesale in the first implementation. The engine already supports the needed lifecycle, controls, thermal hold, and launcher integration. The risk is in shader cost and stability, not in missing framework.

## State Texture Contract

Keep the four-channel field contract so the world can use the existing ping-pong simulation texture and `seed_field()` path.

- `R`: moisture, water pressure, river/lake/ocean energy.
- `G`: biomass, grass, tree canopy density, crop/groundcover recovery.
- `B`: terrain elevation and long-lived geology.
- `A`: heat, cave/ore/settlement energy, fire/light pulses.

This is the authoritative persistent ecosystem state for the first implementation. Richer concepts such as biome id, slope, temperature, fertility, plant age, species, and disturbance history should be derived deterministically from RGBA, world position, seed noise, and time. They should not be presented as separate persistent state until the engine has additional storage.

The simulation pass should remain a bounded local update:

- Reuse the 8-neighbor Laplacian pattern from `minecraft-long-term-3d` and `minecraft-living-voxel` (`worlds/minecraft_long_term_3d.py:44`, `worlds/minecraft_voxels_3d.py:44`).
- Add topographic moisture flow from height gradients (`worlds/minecraft_long_term_3d.py:50`, `worlds/minecraft_voxels_3d.py:55`, `worlds/original_3d.py:40`).
- Borrow climate budget ideas from `shaders/life_update.frag`: latitude/season temperature, evaporation, rain, moisture mixing, fertility, dryness, and settlement support (`shaders/life_update.frag:68`, `shaders/life_update.frag:85`, `shaders/life_update.frag:95`, `shaders/life_update.frag:109`).
- Keep terrain `B` mostly stable. Permit only slow erosion/soil recovery, bounded by a small multiplier. The future DDA worlds explicitly note locked terrain to avoid runaway geology (`worlds/future_future_minecraft.py:31`).
- Use `A` for heat, fire, torch/ore glow, and settlement cues. It should decay unless reinforced by heat sources, ores, villages, or fire.
- Use reaction-diffusion as one local living texture for moss, undergrowth, algae, fungal spread, or canopy pressure. Do not treat it as the whole ecosystem.

## Display Shader Architecture

Use raymarched SDF composition with block snapping, not full voxel DDA, for the first target world.

Core display pieces:

- `worldState(cell)`: sample `stateTex` from block cells, as in `minecraft-long-term-3d` (`worlds/minecraft_long_term_3d.py:179`).
- `terrainHeight(state, cell)`: combine seeded `B`, low-frequency macro noise, ridges, river cuts, and biome slope rules, then snap to block tiers (`worlds/minecraft_long_term_3d.py:183`).
- `map(p, out matID, out stateOut)`: compose terrain, water, trees, shrubs, grass blocks, villages, caves, ores, fire, and optional fauna silhouettes with `boxSdf()` and a small `takeShape()` helper (`worlds/minecraft_long_term_3d.py:164`, `worlds/minecraft_long_term_3d.py:187`).
- Lighting: keep soft shadow, AO, block-grid contours, sky/clouds, volumetric glow, ACES tonemap, and live camera uniform sync (`worlds/minecraft_long_term_3d.py:279`, `worlds/minecraft_long_term_3d.py:295`, `worlds/minecraft_long_term_3d.py:516`).
- Add heavy mode features only when `fxIntensity` is high: short secondary reflection ray on water/crystal/ore, denser volumetric glow, extra cloud/leaf/fog samples, and higher procedural texture detail. `minecraft-overdrive-3d` proves the secondary raymarch can melt the GPU, but it must be capped and gated (`worlds/Final_minecraft_3D.py:452`).

Avoid using the recursive DDA worlds as the base. They are valuable for ideas such as block normals, day/night lighting, and camera altitude, but their shaders request large internal step budgets while the engine currently sends at most 160 effective `raySteps`; the recursive pass is a much larger stability risk for this shader-first SDF architecture (`engine.py:996`, `worlds/future_minecraft.py:232`, `worlds/future_future_minecraft.py:254`, `worlds/future_future_minecraft.py:286`).

## Source World Inventory And Feature Map

| World | File | Current role | Reuse or avoid |
| --- | --- | --- | --- |
| `minecraft-long-term-3d` | `worlds/minecraft_long_term_3d.py` | Current default; balanced SDF Minecraft world | Base implementation. Reuse state contract, seed field, block SDF objects, villages, weather, water reflection, AO/shadows, volumetric glow, and moderate defaults. |
| `minecraft-3d` | `worlds/minecraft_3d.py` | Original Overworld-style SDF scene | Reuse simpler terrain-aware seed concepts, river bands, forest/village placement, and material palette as a fallback readability reference. |
| `minecraft-living-voxel` | `worlds/minecraft_voxels_3d.py` | Dynamic terrain/ecology variant | Reuse stronger moisture advection, biomass spread, erosion/soil recovery, day/night cycle, voxel AO, and forest/water material behavior. |
| `minecraft-fluid-3d` | `worlds/minecraft_fluid.py` | Volumetric fluid and moss world | Reuse bounded fluid/moss/heat coupling, stacked fluid blocks, obsidian/intersection material, and strong volumetric glow patterns. |
| `minecraft-ultimate-voxel` | `worlds/minecraft_voxelv2_3d.py` | Extreme voxel texture/fire world | Reuse 16x16 procedural block-face texture idea, snapped normals, wildfire cellular automata, blocky sun/moon/clouds, and fire consumption. Avoid its very high height scale as the default. |
| `minecraft-overdrive-3d` | `worlds/Final_minecraft_3D.py` | Extreme GPU thermal load | Reuse 3D cave carving, ash/ember/spore glow, crystals, and capped secondary reflection as opt-in heavy paths. Avoid making its reflection pass unconditional. |
| `minecraft-voxel-stress-3d` | `worlds/true_minecraft.py` | Safe heavy voxel-optimized stress world | Reuse volumetric leaf/water depth idea and audio-uniform compatibility only if audio becomes a later goal. |
| `minecraft-first-future-3d` | `worlds/minecraft_first_future.py` | Branchless DDA stress experiment | Reuse bounded terrain flow comments and big-cave ideas. Avoid using it as the base due to large DDA step budgets. |
| `minecraft-future-term-3d` | `worlds/future_minecraft.py` | DDA/large terrain stress test | Reuse camera-above-terrain logic, massive cave/cloud concepts, and biomass-driven forests. Avoid recursive/large DDA as first target path. |
| `minecraft-future-future-term-3d` | `worlds/future_future_minecraft.py` | Recursive DDA reflection test | Reuse the locked-geology principle and water-reflection concept. Avoid two-bounce DDA as default. |
| `original-3d` | `worlds/original_3d.py` | Legacy raymarch living terrain | Reuse topographic reaction-diffusion and biomass extrusion as the biological core. |
| `original-tuned-3d` | `worlds/original_tuned_3d.py` | Tuned legacy raymarch | Use as a stability/palette baseline if new growth becomes too noisy. |
| `original-2d` | `worlds/original_2d.py` | Legacy 2D tile shader | Use only for reaction-diffusion parameter sanity checks. |
| `audio-reactive-3d` | `worlds/audio_reactive_3d.py` | Audio-driven living feedback | Reuse volumetric glow accumulation and optional audio response later. Do not require audio for the target world. |
| `living-sketchbook-3d` | `worlds/living_sketchbook_3d.py` | Audio/living ink world | Reuse high-energy advection ideas and organic feedback, not the sketchbook rendering style. |
| `sketchbook-visualizer-3d` | `worlds/sketchbook_visualizer_3d.py` | Audio visualizer composition | Reuse only composition ideas, fog rhythm, and silhouette restraint. |
| `sketchbook-ink-islands-3d` | `worlds/sketchbook_ink_islands_3d.py` | Ink-island raymarch | Reuse soft-shadow/AO structure and distant watercolor fog ideas only if the new world needs more atmosphere. |
| `static-sandstorm-3d` | `worlds/static_sandstorm_3d.py` | Atmospheric event world | Reuse storm/fog volume discipline, event pulse timing, and heat-glass material ideas. |
| `tsunami-land-3d` | `worlds/tsunami_land_3d.py` | Dynamic water/event world | Reuse shore-aware water runup/drawdown and event visual staging, not tsunami behavior as the default ecology. |
| `muddy-asteroid-planet-3d` | `worlds/muddy_asteroid_planet_3d.py` | Heat/moisture/biomass planet | Reuse ecological coupling among moisture, biomass, height, heat, canopy displacement, and bioluminescence. |
| `neural-plane-3d` | `worlds/neural_plane_3d.py` | Heavy abstract raymarch | Reuse only performance-tuned raymarch atmosphere if needed; it is less Minecraft-readable. |
| Shared life shaders | `shaders/life_update.frag`, `shaders/life_display.frag` | 2D life/climate reference | Reuse temperature, rainfall, fertility, dryness, settlement support, and derived biome-display ideas as math references, translated into the Minecraft RGBA channel contract. |

## Seed Field Plan

Start from `minecraft-long-term-3d.seed_field()` and expand it into explicit biome layers:

- Generate macro elevation, mountain ridges, river cuts, sea/ocean mask, shore, latitude, moisture, forest density, villages, caves, ore/light, and heat/fire seeds.
- Replace pure sine-band rivers with terrain-derived or terrain-following flow masks where practical: downhill routing from the seeded height field, basin/lake fill, shore wetlands, floodplains, infiltration, evaporation, and rainfall. If a sine/noise guide is used for performance, it should be warped by elevation and slope so it reads as drainage, not a periodic stripe.
- Add biome classification as derived math rather than a fifth texture channel. Example classes: ocean, river, beach, wetland, plains, forest, dryland/desert, mountain, snow/highland, village/clearing, cave/ore, hot/fire scar.
- Make plant distribution terrain-aware: trees require non-ocean, enough moisture, enough biomass, reasonable altitude, slope stability, and fire below a threshold. Shrubs/grass can appear in lower biomass zones.
- Keep settlements as sparse patches near shore/river, moderate altitude, low water, and moderate vegetation, as `minecraft-long-term-3d` already seeds villages from candidate masks (`worlds/minecraft_long_term_3d.py:664`).
- Use deterministic `WORLD_SEED` and `np.random.default_rng()` so screenshot comparisons and smoke behavior are reproducible.

## Ecosystem Behavior Plan

The target is not full ecological simulation. It should create believable visible cause/effect inside the four-channel texture:

- Moisture flows downhill and recovers through rainfall/weather. It concentrates rivers, wetlands, and shore plant life.
- Biomass grows where moisture, temperature, terrain, and disturbance are suitable, spreads slowly to neighbors, dies back in dry/overheated/flooded cells, and stabilizes soil.
- Heat/fire appears from sparse lightning/ore/lava sources, spreads only through suitable biomass, consumes biomass, decays, and leaves darker/scarred terrain or exposed stone.
- Terrain changes are slow and bounded: water erodes steep wet slopes slightly, biomass rebuilds soil slightly, fire/heat can scar but not flatten the world.
- At least two visible feedback loops should be present: roots/canopy reduce erosion, canopy alters moisture retention, fire consumes biomass, flooding suppresses or shifts vegetation, or biomass recovery recolonizes scars.
- Display shader should make these states legible: wet valleys with water blocks and reeds, dense canopy on humid lowlands, sparse dry grass in arid highlands, snow/stone above altitude, caves/ores in mountains, torches/villages in moderate biomes, and occasional fire/glow events.

## Stress Test Design

GPU stress should come from view-dependent shader work:

- Raymarch at up to the current effective engine cap of 160 `raySteps`, with default `ray_steps` around 144 to 160 and smoke-test clamp down to 32. Higher stress should come from shader work gated by `fxIntensity`, smaller `tile_size`, more `substeps`, resolution, and CPU burners, not from assuming more than 160 effective ray steps.
- Per-hit `map()` should include enough SDF candidates to be expensive but bounded: terrain, water, 2 to 3 tree parts, shrubs/grass clumps, village blocks, caves, ore/fire, and optional crystal/fireflies.
- AO and soft shadows should stay in fixed short loops. Do not add unbounded per-block neighbor search.
- Volumetric glow and fog should accumulate in the primary loop, scaled by `fxIntensity`.
- Secondary reflections should be short, conditional, and only active for water/crystal/ore when `fxIntensity` is high. They are a stress feature, not a correctness requirement.
- CPU stress remains delegated to existing `--cpu-workers` and `--cpu-matrix` burners (`engine.py:1104`, `engine.py:1115`).

## Realism Gaps To Close

- Current trees are mostly isolated cuboids. The new world needs more varied plant strata: grass/reeds, shrubs, small trees, canopy blocks, and dead/burned variants.
- Current water is mostly presence/level, not watershed-aware. Rivers and shore wetlands need to read from seeded river masks and evolved moisture together.
- Current terrain changes can either be too subtle or run away. The new plan needs strict `B` update bounds and visual evidence that the world remains stable over minutes.
- Current villages are visual markers. They should read as clearings with torches, planks, and paths, placed by terrain-aware masks rather than random glow only.
- Current fire is strongest in `minecraft-ultimate-voxel`, but should connect to biomass and leave visible ecological aftermath without consuming the whole map.
- Fauna is absent. The first pass can use tiny blocky moving silhouettes, fireflies, or fish/birds as shader-only visual life, with no extra simulation channel.
- Biome transitions need better material mapping: grass to dirt/sand/stone/snow should follow moisture, altitude, heat, and top-face normal.
- Persistent plant age, exact species, discrete block IDs, mobs, inventories, and chunk-level hydrology remain unrealized under the current single RGBA texture. They should be called future engine work, not hidden inside the first world module.

## Main Risks

- Shader compile or runtime failure from adding too many branches, helper functions, or GLSL constructs. Mitigation: keep one module, run one-world smoke early, then full smoke.
- Raymarch artifacts from discontinuous `floor()` height and many block SDF candidates. Mitigation: under-relax distances where discontinuities are strong, use conservative `SURF_DIST`, and cap candidate complexity.
- Blurred categorical state from linear texture filtering and repeat wrapping. Mitigation: keep persistent channels analog, derive hard categories from sampled values and deterministic hash, and avoid persistent exact block IDs.
- Runaway simulation in `B`, `G`, or `A`. Mitigation: clamp all channels, tiny terrain deltas, decay fire/heat, and smoke with elevated substeps.
- Default too heavy for launcher users. Mitigation: defaults should be heavy but not overdrive-class; reserve meltdown paths for `fxIntensity`, effective 160-step raymarching, small `tile_size`, more `substeps`, higher resolution, and CPU presets.
- World defaults hidden by launcher presets. Mitigation: verify direct CLI defaults separately from launcher preset runs, and update presets/preview copy if the new default requires different effective values.
- Smoke test too slow or flaky. Mitigation: all expensive loops obey `raySteps` or short fixed caps; no code path should assume high resolution or audio.
- Stress profile becomes an unbounded machine-freeze profile. Mitigation: keep documented stress commands inside launcher/engine caps, use thermal hold, bound CPU worker/matrix choices, and use a fixed-duration run with an external timeout during validation.
- Minecraft readability lost to atmospheric effects. Mitigation: block grid, snapped/stepped terrain, pixelated textures, square sun/clouds, and cuboid vegetation remain first-order visuals.
- Ecosystem not legible. Mitigation: visual states must make moisture, biomass, heat/fire, altitude, villages, and caves distinguishable at normal camera distance.

## Acceptance Criteria

Architecture and integration:

- New module `worlds/minecraft_perfect_ecosystem_3d.py` defines `SIM_FRAG_SHADER`, `DISPLAY_FRAG_SHADER`, `seed_field()`, and `SPEC`.
- New `SPEC.id` is `minecraft-perfect-ecosystem-3d` and `display_name` is `Minecraft Perfect Ecosystem`.
- `worlds/registry.py` imports and registers the world without breaking `python main.py --list-worlds`.
- The default may change only after the new world passes smoke and a visual review.
- The new world should get a preview image or at least a tuned preview palette, and README/launcher world highlights should be updated when the world becomes a flagship/default entry.
- Launcher command preview for the new world should show the intended effective Medium/High/Ultra values, or the presets should be adjusted so world-specific tuning is not bypassed.
- If the shader samples `audioFft` or `audioWave`, its `SPEC` must set `uses_audio=True`; otherwise it must not reference audio samplers.

Visual and ecosystem behavior:

- First 10 seconds show a readable Minecraft-like world with blocky terrain, water, trees, biome variation, caves/ores or torches, sky/clouds, and living motion.
- Moisture, biomass, terrain, and heat/light channels are visibly connected to materials and growth.
- At least five derived biomes are readable from camera height, including water/shore, river or wetland, forest/grassland, dryland or fire-scar, and mountain/stone/snow.
- Trees and plant cover are terrain-aware, not random uniform scatter.
- Plant cover visibly changes through growth, spread, death, or disturbance recovery during a short run.
- Rivers and lakes read as connected or basin-shaped terrain features, not periodic sine bands.
- At least two feedback loops are visible or inspectable through channel behavior: roots/canopy reduce erosion, canopy alters moisture retention, fire consumes biomass, flooding suppresses vegetation, or biomass recolonizes disturbed ground.
- Fire/heat/glow events are bounded and do not erase the map.

Stress and safety:

- Low-cost smoke succeeds for the new world:
  `python main.py --world minecraft-perfect-ecosystem-3d -wnd glfw --width 320 --height 180 --tile-size 24 --substeps 1 --ray-steps 32 --fx-intensity 0.3 --cpu-workers 0 --no-thermal-hold --no-hud --quit-after 1.0`
- Targeted smoke succeeds for 3 to 5 seconds with a hard external timeout around the process:
  `.\scripts\smoke_worlds.ps1 -World minecraft-perfect-ecosystem-3d -Width 640 -Height 360 -TileSize 24 -Substeps 1 -RaySteps 32 -FxIntensity 0.3 -CpuWorkers 0 -QuitAfter 3 -TimeoutSeconds 30 -ReportPath logs\validation\minecraft-perfect-ecosystem-targeted.md`
- Full registry smoke succeeds through `scripts/smoke_worlds.ps1`.
- Heavy launch remains credible with the current effective 160-step raymarch cap, high `fx_intensity`, smaller `tile_size`, more `substeps`, higher resolution, and `--cpu-workers`/`--cpu-matrix`, while thermal hold remains available. This is a manual opt-in stress sanity profile, not part of scheduled lightweight validation:
  `python -B main.py --world minecraft-perfect-ecosystem-3d -wnd glfw --width 1920 --height 1080 --tile-size 8 --substeps 24 --ray-steps 160 --fx-intensity 1.3 --cpu-workers 8 --cpu-matrix 1024 --max-gpu-temp 80 --max-cpu-temp 88 --quit-after 300`
- Stress validation should note shader compile errors, blank/NaN frames, unexpected thermal holds, startup hangs, and sustained responsiveness, not just pass/fail exit status.
- No unbounded loops, no dependency on audio unless `uses_audio=True`, no new engine flags required.

## Next Implementation Slice

1. Create `worlds/minecraft_perfect_ecosystem_3d.py` by forking `minecraft_long_term_3d.py`.
2. Add stronger ecosystem simulation from `minecraft_voxels_3d.py`, `minecraft_fluid.py`, `minecraft_voxelv2_3d.py`, and `muddy_asteroid_planet_3d.py`.
3. Add block-face texture and plant/fauna detail selectively from `minecraft_voxelv2_3d.py` while keeping SDF raymarching as the renderer.
4. Register the new world after the extreme Minecraft experiments and before `minecraft-long-term-3d`.
5. Verify direct CLI defaults and launcher preset command previews; adjust presets or default decision if the launcher masks the world tuning.
6. Add preview metadata and update README highlights if the world becomes the flagship/default.
7. Run the one-world smoke, targeted smoke with an external timeout, full smoke, then a fixed-duration high-load run with CPU workers.
