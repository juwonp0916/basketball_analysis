<script lang="ts">
  import * as Card from "$lib/components/ui/card/index.js";
  import { Chart, Svg, Points } from "layerchart";

  type ShotPoint = { id: number; x: number; y: number; result: "made" | "missed" };

  type Props = {
    shotPoints: ShotPoint[];
    selectedShotId: number | null;
  };

  let { shotPoints, selectedShotId }: Props = $props();

  const chartPadding = { top: 20, bottom: 20, left: 20, right: 20 };
  let containerWidth = $state(0);
  let containerHeight = $state(0);
  let innerWidth = $derived(Math.max(0, containerWidth - chartPadding.left - chartPadding.right));
  let innerHeight = $derived(Math.max(0, containerHeight - chartPadding.top - chartPadding.bottom));
</script>

<Card.Root class="bg-[#1a1d24] border-gray-800 flex flex-col h-[600px]">
  <Card.Header class="pb-2">
    <div class="flex bg-[#0f1116] rounded-lg p-1 w-full">
      <button class="flex-1 py-1 text-sm font-medium rounded bg-[#1a1d24] text-white shadow">All</button>
      <button class="flex-1 py-1 text-sm font-medium text-gray-400 hover:text-white">Made</button>
      <button class="flex-1 py-1 text-sm font-medium text-gray-400 hover:text-white">Missed</button>
      <button class="flex-1 py-1 text-sm font-medium text-gray-400 hover:text-white">3PT</button>
      <button class="flex-1 py-1 text-sm font-medium text-gray-400 hover:text-white">2PT</button>
    </div>
  </Card.Header>
  <Card.Content class="flex-1 relative p-4 min-h-0">
    <div class="w-full h-full relative" bind:clientWidth={containerWidth} bind:clientHeight={containerHeight}>
      <Chart data={shotPoints} x="x" y="y" xDomain={[0, 50]} yDomain={[47, 0]} padding={chartPadding}>
        <Svg>
          <svg width={innerWidth} height={innerHeight} viewBox="0 0 50 47" preserveAspectRatio="none" style="overflow: visible;">
            <g class="court-lines" stroke="#374151" stroke-width="0.5" fill="none">
              <rect x="0" y="0" width="50" height="47" vector-effect="non-scaling-stroke" />
              <rect x="17" y="0" width="16" height="19" vector-effect="non-scaling-stroke" />
              <circle cx="25" cy="19" r="6" vector-effect="non-scaling-stroke" />
              <path d="M 3 0 L 3 14 Q 25 43 47 14 L 47 0" vector-effect="non-scaling-stroke" />
              <circle cx="25" cy="5.25" r="0.75" stroke="#ef4444" vector-effect="non-scaling-stroke" />
              <line x1="22" y1="4" x2="28" y2="4" vector-effect="non-scaling-stroke" />
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
