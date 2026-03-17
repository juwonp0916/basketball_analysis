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
</script>

<!-- svelte-ignore a11y_no_static_element_interactions, a11y_click_events_have_key_events-->
<div class="absolute inset-0 cursor-crosshair" {onclick}>
  <svg class="absolute inset-0 w-full h-full" style="pointer-events: none;">
    {#if points.length >= 2}
      {#each [[0, 1], [1, 2], [2, 3]] as [a, b]}
        {#if points.length > b}
          {@const pa = displayCoords(points[a])}
          {@const pb = displayCoords(points[b])}
          <line
            x1={pa.x}
            y1={pa.y}
            x2={pb.x}
            y2={pb.y}
            stroke={b <= 1 ? "#eab308" : b === 2 ? "#22c55e" : "#eab308"}
            stroke-width="2"
            opacity="0.7"
          />
        {/if}
      {/each}
    {/if}
    {#if points.length >= 6}
      {@const p4 = displayCoords(points[4])}
      {@const p5 = displayCoords(points[5])}
      {@const p1 = displayCoords(points[1])}
      {@const p2 = displayCoords(points[2])}
      <line x1={p4.x} y1={p4.y} x2={p5.x} y2={p5.y} stroke="#d946ef" stroke-width="2" opacity="0.7" />
      <line x1={p1.x} y1={p1.y} x2={p4.x} y2={p4.y} stroke="#22c55e" stroke-width="2" opacity="0.7" />
      <line x1={p2.x} y1={p2.y} x2={p5.x} y2={p5.y} stroke="#22c55e" stroke-width="2" opacity="0.7" />
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

  <div
    class="absolute top-4 left-1/2 -translate-x-1/2 bg-black/80 rounded-lg px-4 py-2 text-center pointer-events-auto flex flex-col items-center gap-2"
  >
    {#if points.length < MAX_CALIBRATION_POINTS}
      <p class="text-sm font-medium text-white pointer-events-none">
        Click point {points.length + 1}/{MAX_CALIBRATION_POINTS}:
        <span class="text-blue-400">{CALIBRATION_LABLES[points.length]}</span>
      </p>
    {:else}
      <p class="text-sm font-medium text-green-400 pointer-events-none">All 6 points placed. Confirm or reset.</p>
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
