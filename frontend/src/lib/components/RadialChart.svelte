<script lang="ts">
  import * as Card from "$lib/components/ui/card/index.js";
  import * as Chart from "$lib/components/ui/chart/index.js";
  import { ArcChart, Text } from "layerchart";
  import TrendingUpIcon from "@lucide/svelte/icons/trending-up";

  // TODO: Create valid data interface
  interface Data {
    player: string;
    shots_succs: {
      twoPt: number;
      threePt: number;
    };
    total_shots: number;
    color: string;
  }

  interface Props {
    shotData: Data;
  }

  const { shotData }: Props = $props();

  const chartConfig = {
    visitors: { label: "Visitors" },
    safari: { label: "Safari", color: "var(--chart-2)" },
  } satisfies Chart.ChartConfig;
</script>

<Card.Root>
  <Card.Header class="items-center">
    <Card.Title>Shots Completed</Card.Title>
    <Card.Description>Showing total visitors for the last 6 months</Card.Description>
  </Card.Header>
  <Card.Content class="flex-1">
    <Chart.Container config={chartConfig} class="mx-auto aspect-square max-h-[250px]">
      <ArcChart
        label="browser"
        value="shots_succs"
        outerRadius={-2}
        innerRadius={-14}
        padding={20}
        range={[0, -360]}
        maxValue={shotData.total_shots}
        cornerRadius={20}
        series={[
          {
            key: shotData.player,
            color: shotData.color,
            data: [shotData],
          },
        ]}
        props={{
          arc: { track: { fill: "var(--muted)" }, motion: "tween" },
          tooltip: { context: { hideDelay: 350 } },
        }}
        tooltip={false}
      >
        {#snippet belowMarks()}
          <circle cx="0" cy="0" r="60" class="fill-background" />
        {/snippet}

        {#snippet aboveMarks()}
          <Text
            value={String(shotData.shots_succs) + "/" + String(shotData.total_shots)}
            textAnchor="middle"
            verticalAnchor="middle"
            class="fill-foreground text-4xl! font-bold"
            dy={3}
          />
          <Text value="Succeeded" textAnchor="middle" verticalAnchor="middle" class="fill-muted-foreground!" dy={22} />
        {/snippet}
      </ArcChart>
    </Chart.Container>
  </Card.Content>
  <Card.Footer class="flex-col gap-2 text-sm">
    <div class="flex items-center gap-2 font-medium leading-none">
      Trending up by 5.2% this month <TrendingUpIcon class="size-4" />
    </div>
    <div class="text-muted-foreground flex items-center gap-2 leading-none">January - June 2024</div>
  </Card.Footer>
</Card.Root>
