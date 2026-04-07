<script lang="ts">
  import * as Card from "$lib/components/ui/card/index.js";
  import { Chart, Svg, Points } from "layerchart";

  type ShotPoint = { id: number; x: number; y: number; result: "made" | "missed" };

  type Props = {
    shotPoints: ShotPoint[];
    selectedShotId: number | null;
    hideHeader?: boolean;
  };

  let { shotPoints, selectedShotId, hideHeader = false }: Props = $props();

  const chartPadding = { top: 20, bottom: 20, left: 20, right: 20 };
  let containerWidth = $state(0);
  let containerHeight = $state(0);
  let innerWidth = $derived(Math.max(0, containerWidth - chartPadding.left - chartPadding.right));
  let innerHeight = $derived(Math.max(0, containerHeight - chartPadding.top - chartPadding.bottom));
</script>

<Card.Root class="bg-[#1a1d24] border-gray-800 flex flex-col h-[600px] {hideHeader ? 'h-full border-0 bg-transparent shadow-none' : ''}">
  {#if !hideHeader}
    <Card.Header class="pb-2">
      <div class="flex bg-[#0f1116] rounded-lg p-1 w-full">
        <button class="flex-1 py-1 text-sm font-medium rounded bg-[#1a1d24] text-white shadow">All</button>
        <button class="flex-1 py-1 text-sm font-medium text-gray-400 hover:text-white">Made</button>
        <button class="flex-1 py-1 text-sm font-medium text-gray-400 hover:text-white">Missed</button>
        <button class="flex-1 py-1 text-sm font-medium text-gray-400 hover:text-white">3PT</button>
        <button class="flex-1 py-1 text-sm font-medium text-gray-400 hover:text-white">2PT</button>
      </div>
    </Card.Header>
  {/if}
  <Card.Content class="flex-1 relative flex items-center justify-center {hideHeader ? 'p-0' : 'p-4'} min-h-0">
    <div class="relative max-h-full max-w-full" style="aspect-ratio: 50 / 47; height: 100%; width: auto;" bind:clientWidth={containerWidth} bind:clientHeight={containerHeight}>
      <Chart data={shotPoints} x="x" y="y" xDomain={[0, 50]} yDomain={[47, 0]} padding={chartPadding}>
        <Svg>
          <svg width={innerWidth} height={innerHeight} viewBox="0 0 50 47" preserveAspectRatio="none" style="overflow: visible;">
            <g class="court-lines" stroke="#374151" stroke-width="0.5" fill="none">
              <rect x="0" y="0" width="50" height="47" />

              <!-- Hoop (moved up) -->
              <circle cx="25" cy="3" r="0.75" stroke="#ef4444" />
              <line x1="22" y1="1.5" x2="28" y2="1.5" />

              <!-- Paint -->
              <rect x="17" y="0" width="16" height="19" />

              <!-- Free throw -->
              <path d="M 31 19 A 6 6 0 0 1 19 19" />

              <!-- Restricted -->
              <path d="M 21 4 A 4 4 0 0 0 29 4" />

              <!-- Corner 3 -->
              <line x1="3" y1="0" x2="3" y2="10" />
              <line x1="47" y1="0" x2="47" y2="10" />

              <!-- FIXED 3PT ARC (now curves downward) -->
              <path d="M 3 10 A 22.75 22.75 0 0 0 47 10" />
            </g>
          </svg>

          <Points>
            {#snippet children({ points })}
              {#each points as point}
                {#if selectedShotId === point.data.id}
                  <circle cx={point.x} cy={point.y} r="5" fill="none" stroke="white" stroke-width="0.6" class="animate-pulse" />
                {/if}
                <circle
                  cx={point.x}
                  cy={point.y}
                  r={selectedShotId === point.data.id ? "3.5" : "3"}
                  fill={point.data.result === "made" ? "#10b981" : "#ef4444"}
                  stroke="none"
                  class="transition-all duration-300"
                />
              {/each}
            {/snippet}
          </Points>
        </Svg>
      </Chart>
    </div>
  </Card.Content>
</Card.Root>
