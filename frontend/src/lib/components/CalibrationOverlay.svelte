<script lang="ts">
  import Button from "$lib/components/ui/button/button.svelte";
  import { RotateCcw, Check } from "lucide-svelte";
  import {
    CALIBRATION_LABLES,
    CALIBRATION_COLORS,
    MAX_CALIBRATION_POINTS,
    toDisplayCoords,
    type CalibrationPoint,
  } from "$lib/services/calibration";

  type Props = {
    points: CalibrationPoint[];
    videoElement: HTMLVideoElement;
    onclick: (e: MouseEvent) => void;
    onreset: () => void;
    onconfirm: () => void;
  };

  let { points, videoElement, onclick, onreset, onconfirm }: Props = $props();

  function displayCoords(point: CalibrationPoint) {
    return toDisplayCoords(point, videoElement);
  }

  // SVG positions of the 4 paint-box calibration points on the mini court diagram (viewBox 0 0 50 47)
  // FIBA paint box: x=[5.05,9.95]m (of 15m), y=[0,5.8]m (of 14m)
  // Mapped to SVG: x = fiba_x/15*50, y = fiba_y/14*47 (baseline at top, y=2 offset for border)
  const REFERENCE_POINTS = [
    { cx: 16.8, cy: 2    },  // 1: Baseline Left Paint Corner  (5.05, 0)
    { cx: 33.2, cy: 2    },  // 2: Baseline Right Paint Corner (9.95, 0)
    { cx: 16.8, cy: 21.5 },  // 3: FT Line Left Corner         (5.05, 5.8)
    { cx: 33.2, cy: 21.5 },  // 4: FT Line Right Corner        (9.95, 5.8)
  ];
</script>

<style>
  @keyframes pulse-ring {
    0%   { stroke-width: 0.5; stroke-opacity: 1; }
    100% { stroke-width: 3;   stroke-opacity: 0; }
  }
  .pulse-ring {
    animation: pulse-ring 1.2s ease-out infinite;
  }
</style>

<!-- svelte-ignore a11y_no_static_element_interactions, a11y_click_events_have_key_events-->
<div class="absolute inset-0 cursor-crosshair" {onclick}>
  <svg class="absolute inset-0 w-full h-full" style="pointer-events: none;">
    <!-- Baseline paint segment: point 0 → point 1 -->
    {#if points.length >= 2}
      {@const p0 = displayCoords(points[0])}
      {@const p1 = displayCoords(points[1])}
      <line x1={p0.x} y1={p0.y} x2={p1.x} y2={p1.y} stroke="#eab308" stroke-width="2" opacity="0.7" />
    {/if}
    <!-- Left paint edge: point 0 → point 2 -->
    {#if points.length >= 3}
      {@const p0 = displayCoords(points[0])}
      {@const p2 = displayCoords(points[2])}
      <line x1={p0.x} y1={p0.y} x2={p2.x} y2={p2.y} stroke="#94a3b8" stroke-width="2" opacity="0.7" />
    {/if}
    <!-- Right paint edge: point 1 → point 3 and FT line: point 2 → point 3 -->
    {#if points.length >= 4}
      {@const p1 = displayCoords(points[1])}
      {@const p2 = displayCoords(points[2])}
      {@const p3 = displayCoords(points[3])}
      <line x1={p1.x} y1={p1.y} x2={p3.x} y2={p3.y} stroke="#94a3b8" stroke-width="2" opacity="0.7" />
      <line x1={p2.x} y1={p2.y} x2={p3.x} y2={p3.y} stroke="#3b82f6" stroke-width="2" opacity="0.7" />
    {/if}

    {#each points as point, i}
      {@const dp = displayCoords(point)}
      <circle cx={dp.x} cy={dp.y} r="8" fill={CALIBRATION_COLORS[i]} opacity="0.9" stroke="white" stroke-width="1.5" />
      <text x={dp.x} y={dp.y + 1} text-anchor="middle" dominant-baseline="middle" fill="black" font-size="10" font-weight="bold"
        >{i + 1}</text
      >
      <text x={dp.x + 14} y={dp.y + 4} fill="white" font-size="11" font-weight="600" style="text-shadow: 0 1px 3px rgba(0,0,0,0.8);"
        >{CALIBRATION_LABLES[i]}</text
      >
    {/each}
  </svg>

  <!-- Reference guide: mini court diagram bottom-right -->
  <div class="absolute bottom-4 right-4 pointer-events-none w-44 bg-black/80 rounded-lg p-2">
    <svg viewBox="0 0 50 47" class="w-full" style="display:block;">
      <g stroke="#4b5563" stroke-width="0.8" fill="none">
        <!-- Court outline -->
        <rect x="1" y="2" width="48" height="44" />
        <!-- Paint box -->
        <rect x="16.8" y="2" width="16.4" height="19.6" />
        <!-- FT circle -->
        <circle cx="25" cy="21.6" r="6" />
        <!-- 3PT arc -->
        <line x1="3" y1="2" x2="3" y2="12" />
        <line x1="47" y1="2" x2="47" y2="12" />
        <path d="M 3 12 A 22.5 22.5 0 0 1 47 12" />
        <!-- Basket -->
        <circle cx="25" cy="7.3" r="0.75" stroke="#6b7280" />
        <line x1="22" y1="5" x2="28" y2="5" />
      </g>
      {#each REFERENCE_POINTS as pt, i}
        {#if i === points.length && points.length < MAX_CALIBRATION_POINTS}
          <circle cx={pt.cx} cy={pt.cy} r="3.5" fill="none" stroke={CALIBRATION_COLORS[i]} class="pulse-ring" />
        {/if}
        <circle cx={pt.cx} cy={pt.cy} r="2" fill={CALIBRATION_COLORS[i]} opacity={i < points.length ? 0.4 : 1} />
        <text
          x={pt.cx}
          y={pt.cy + 0.4}
          text-anchor="middle"
          dominant-baseline="middle"
          fill="white"
          font-size="1.8"
          font-weight="bold"
          opacity={i < points.length ? 0.4 : 1}
        >{i + 1}</text>
      {/each}
    </svg>
    {#if points.length < MAX_CALIBRATION_POINTS}
      <p class="text-xs text-center mt-1 truncate" style="color: {CALIBRATION_COLORS[points.length]}">
        {points.length + 1}. {CALIBRATION_LABLES[points.length]}
      </p>
    {:else}
      <p class="text-xs text-center mt-1 text-green-400">All 4 points placed</p>
    {/if}
  </div>

  <div
    class="absolute top-4 left-1/2 -translate-x-1/2 bg-black/80 rounded-lg px-4 py-2 text-center pointer-events-auto flex flex-col items-center gap-2"
  >
    {#if points.length < MAX_CALIBRATION_POINTS}
      <p class="text-sm font-medium text-white pointer-events-none">
        Click point {points.length + 1}/{MAX_CALIBRATION_POINTS}:
        <span class="text-blue-400">{CALIBRATION_LABLES[points.length]}</span>
      </p>
    {:else}
      <p class="text-sm font-medium text-green-400 pointer-events-none">All 4 points placed. Confirm or reset.</p>
    {/if}
    <div class="flex gap-3">
      <Button
        onclick={(e: MouseEvent) => {
          e.stopPropagation();
          onreset();
        }}
        variant="secondary"
        size="sm"
      >
        <RotateCcw class="w-4 h-4 mr-1" /> Reset
      </Button>
      {#if points.length === MAX_CALIBRATION_POINTS}
        <Button
          onclick={(e: MouseEvent) => {
            e.stopPropagation();
            onconfirm();
          }}
          size="sm"
          class="bg-green-600 hover:bg-green-700"
        >
          <Check class="w-4 h-4 mr-1" /> Confirm Calibration
        </Button>
      {/if}
    </div>
  </div>
</div>
