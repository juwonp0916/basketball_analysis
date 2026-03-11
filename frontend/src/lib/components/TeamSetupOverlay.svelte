<script lang="ts">
  import Button from "$lib/components/ui/button/button.svelte";
  import { Check } from "lucide-svelte";
  import { AVAILABLE_COLORS } from "$lib/services/team-colors";

  type Props = {
    team0Color: string;
    team1Color: string;
    team0Name: string;
    team1Name: string;
    onconfirm: () => void;
    onskip: () => void;
  };

  let {
    team0Color = $bindable(),
    team1Color = $bindable(),
    team0Name = $bindable(),
    team1Name = $bindable(),
    onconfirm,
    onskip,
  }: Props = $props();

  let colorsMatch = $derived(team0Color === team1Color);
</script>

<div class="absolute inset-0 bg-black/70 flex items-center justify-center">
  <div class="bg-[#1a1d24] rounded-xl p-6 max-w-md w-full mx-4 border border-gray-700">
    <h3 class="text-xl font-bold text-white mb-4 text-center">Team Setup</h3>
    <p class="text-gray-400 text-sm mb-6 text-center">Set team names and select jersey colors to enable team-based shot tracking.</p>

    <div class="space-y-4 mb-6">
      <div class="grid grid-cols-2 gap-4">
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2" for="team0-name">Team 1 Name</label>
          <input
            id="team0-name"
            type="text"
            bind:value={team0Name}
            placeholder="Team 1"
            class="w-full bg-[#0f1116] border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-500 flex-1"
          />
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2" for="team0-color">Team 1 Jersey Color</label>
          <select
            id="team0-color"
            bind:value={team0Color}
            class="w-full bg-[#0f1116] border border-gray-600 rounded-lg px-4 py-2 text-white capitalize"
          >
            {#each AVAILABLE_COLORS as color}
              <option value={color} class="capitalize">{color}</option>
            {/each}
          </select>
        </div>
      </div>
      <div class="grid grid-cols-2 gap-4">
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2" for="team1-name">Team 2 Name</label>
          <input
            id="team1-name"
            type="text"
            bind:value={team1Name}
            placeholder="Team 2"
            class="w-full bg-[#0f1116] border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-500 flex-1"
          />
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2" for="team1-color">Team 2 Jersey Color</label>
          <select
            id="team1-color"
            bind:value={team1Color}
            class="w-full bg-[#0f1116] border border-gray-600 rounded-lg px-4 py-2 text-white capitalize"
          >
            {#each AVAILABLE_COLORS as color}
              <option value={color} class="capitalize">{color}</option>
            {/each}
          </select>
        </div>
      </div>
    </div>

    {#if colorsMatch}
      <p class="text-red-400 text-sm mb-4 text-center">Please select different colors for each team.</p>
    {/if}

    <div class="flex gap-3">
      <Button onclick={onskip} variant="secondary" class="flex-1">Skip</Button>
      <Button onclick={onconfirm} class="flex-1 bg-green-600 hover:bg-green-700" disabled={colorsMatch}>
        <Check class="w-4 h-4 mr-2" /> Confirm
      </Button>
    </div>
  </div>
</div>
