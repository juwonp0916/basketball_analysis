<script lang="ts">
  import { onDestroy } from "svelte";
  import Button from "$lib/components/ui/button/button.svelte";
  import * as Card from "$lib/components/ui/card/index.js";
  import { Settings, Square, Check, X, Play, Video, Crosshair, Activity, List, Pause } from "lucide-svelte";
  import CalibrationOverlay from "$lib/components/CalibrationOverlay.svelte";
  import ShotChart from "$lib/components/ShotChart.svelte";
  import { createAnalysisSession } from "$lib/utils/webrtc-utils";
  import { videoClickToCalibrationPoint, submitCalibration, type CalibrationPoint } from "$lib/services/calibration";
  import { BACKEND_URL } from "$lib";
  import { formatMs } from "$lib/utils/format";
  import type { Shot, GameStats } from "$lib/types";

  // --- DOM refs ---
  let videoElement = $state<HTMLVideoElement | null>(null);
  let stream = $state<MediaStream | null>(null);
  let playbackSpeed = $state<number>(1.0);
  let isPaused = $state<boolean>(true);

  // --- Mode flags ---
  let simulationMode = $state<boolean>(false);
  let calibrationMode = $state(false);
  let calibrationPoints = $state<CalibrationPoint[]>([]);
  let isCalibrated = $state(false);
  let team0Color = $state("#ff0000");
  let team1Color = $state("#0000ff");
  let team0Name = $state("Team 1");
  let team1Name = $state("Team 2");
  let teamsConfigured = $state(false);

  // --- Analysis state ---
  let session = $state<ReturnType<typeof createAnalysisSession> | null>(null);
  let logs = $state<Shot[]>([]);
  let shots = $state<Shot[]>([]);
  let selectedShotId = $state<number | null>(null);
  let activeTab = $state<"all" | "team1" | "team2" | "comparison">("all");
  let eventLogFilter = $state<"all" | "team1" | "team2">("all");

  let globalStats = $state<GameStats>({
    totalShots: { made: 0, total: 0 },
    percentages: { fieldGoal: 0.0, twoPoint: 0.0, threePoint: 0.0 },
  });

  type ShotPoint = { id: number; x: number; y: number; result: Shot["result"]; team?: string };
  const shotPoints = $derived<ShotPoint[]>(shots.map((s) => ({ id: s.id, x: s.coord.x, y: s.coord.y, result: s.result, team: s.team })));

  // Computed local stats derived from current shots mapped to team 0 and 1
  function calculateStats(shotList: Shot[]): GameStats {
    let made = 0,
      total = 0,
      twoMade = 0,
      twoTotal = 0,
      threeMade = 0,
      threeTotal = 0;
    for (const sh of shotList) {
      total++;
      if (sh.result === "made") made++;
      if (sh.type === "2pt") {
        twoTotal++;
        if (sh.result === "made") twoMade++;
      }
      if (sh.type === "3pt") {
        threeTotal++;
        if (sh.result === "made") threeMade++;
      }
    }
    return {
      totalShots: { made, total },
      percentages: {
        fieldGoal: total > 0 ? (made / total) * 100 : 0.0,
        twoPoint: twoTotal > 0 ? (twoMade / twoTotal) * 100 : 0.0,
        threePoint: threeTotal > 0 ? (threeMade / threeTotal) * 100 : 0.0,
      },
    };
  }

  const team0Shots = $derived(shots.filter((s) => s.team === team0Name));
  const team1Shots = $derived(shots.filter((s) => s.team === team1Name));
  const team0Stats = $derived(calculateStats(team0Shots));
  const team1Stats = $derived(calculateStats(team1Shots));

  let currentStats = $derived.by(() => {
    switch (activeTab) {
      case "team1":
        return team0Stats;
      case "team2":
        return team1Stats;
      case "all":
      default:
        return calculateStats(shots);
    }
  });

  // --- Camera ---
  async function handleStartCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (videoElement) videoElement.srcObject = stream;
    } catch (error) {
      console.error("Error accessing camera:", error);
      alert("Could not access the camera.");
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      if (videoElement) videoElement.srcObject = null;
      stream = null;
    }
  }

  // --- Simulation ---
  async function handleStartSimulation() {
    if (!videoElement) return;
    simulationMode = true;
    videoElement.crossOrigin = "anonymous";
    videoElement.src = `${BACKEND_URL}/video/video6.mp4`;
    await new Promise<void>((r) => videoElement!.addEventListener("canplay", () => r(), { once: true }));
    videoElement.playbackRate = playbackSpeed;
    isPaused = true;
    await videoElement.play();
    stream = (videoElement as any).captureStream();
    videoElement.pause();
  }

  
  function togglePlayPause() {
    if (!videoElement) return;
    if (videoElement.paused) {
      videoElement.play();
      isPaused = false;
    } else {
      videoElement.pause();
      isPaused = true;
    }
  }

  function setPlaybackSpeed(speed: number) {
    if (!videoElement) return;
    playbackSpeed = speed;
    videoElement.playbackRate = speed;
  }

  // --- Calibration handlers ---
  function handleCalibrationClick(e: MouseEvent) {
    if (!videoElement || calibrationPoints.length >= 6) return;
    calibrationPoints = [...calibrationPoints, videoClickToCalibrationPoint(e, videoElement)];
  }

  async function confirmCalibration() {
    if (!videoElement || calibrationPoints.length !== 6) return;
    try {
      const data = await submitCalibration(calibrationPoints, videoElement.videoWidth, videoElement.videoHeight);
      if (data.success) {
        isCalibrated = true;
        calibrationMode = false;
        startAnalysisWebrtc();
      } else {
        alert(`Calibration failed: ${data.error || "Unknown error"}`);
      }
    } catch {
      alert("Could not send calibration to server.");
    }
  }

  // --- Analysis ---
  function handleStartAnalysis() {
    if (!stream) {
      alert("Start camera or simulation first.");
      return;
    }
    if (!isCalibrated) {
      enterCalibrationMode();
    } else {
      startAnalysisWebrtc();
    }
  }

  async function startAnalysisWebrtc() {
    if (!stream) return;
    try {
      session = createAnalysisSession(
        stream,
        {
          onStatsUpdate: (s) => (globalStats = s),
          onShotDetected: (shot) => {
            logs = [shot, ...logs].slice(0, 50);
            shots = [...shots, shot].slice(-200);
          },
          onTeamColorsCalibrated: (team0: string, team1: string) => {
            team0Color = team0;
            team1Color = team1;
            teamsConfigured = true;
          },
        },
        { backendUrl: BACKEND_URL, simulationMode, isCalibrated },
      );
      await session.start();
      if (videoElement?.paused) videoElement.play();
    } catch (error) {
      console.error("Failed to start analysis:", error);
      alert("Could not establish a connection with the server.");
    }
  }

  function requestStopAnalysis() {
    if (confirm("Are you sure you want to stop the ongoing analysis?")) {
      stopAnalysis();
    }
  }

  function stopAnalysis() {
    session?.stop();
    session = null;
    if (simulationMode) {
      if (videoElement) {
        videoElement.pause();
        videoElement.src = "";
      }
      simulationMode = false;
      stream = null;
    } else {
      stopCamera();
    }
    calibrationMode = false;
    calibrationPoints = [];
    isCalibrated = false;
    teamsConfigured = false;
  }

  function enterCalibrationMode() {
    if (!videoElement) return;
    videoElement.pause();
    calibrationMode = true;
    calibrationPoints = [];
    isCalibrated = false;
  }

  onDestroy(() => {
    stopCamera();
    stopAnalysis();
  });

  let isRecording = $state(false);
  let recordingDuration = $state("00:12:45");
</script>

<!-- The template remains the same structure but with extracted components -->
<div class="min-h-screen bg-[#0f1116] text-white font-sans p-6">
  <header class="flex justify-between items-center mb-6">
    <div class="flex items-center gap-2">
      <div class="w-6 h-6 bg-blue-600 rounded-full"></div>
      <h1 class="text-xl font-bold">LiveShot Analytics</h1>
    </div>
    <div class="flex items-center gap-4">
      <div class="flex items-center gap-2 bg-[#1a1d24] px-3 py-1 rounded-full">
        <div class="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
        <span class="text-sm font-medium text-gray-300">LIVE</span>
      </div>
      <Button variant="ghost" size="icon" class="text-gray-400 hover:text-white">
        <Settings class="w-5 h-5" />
      </Button>
    </div>
  </header>

  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
    <div class="lg:col-span-2 flex flex-col gap-6">
      <!-- Video Player -->
      <div class="relative w-full aspect-video bg-black rounded-xl overflow-hidden border border-gray-800 shadow-2xl">
        {#if !stream}
          <div class="absolute inset-0 flex flex-col items-center justify-center text-gray-500">
            <Video class="w-12 h-12 mb-4 opacity-50" />
            <p class="mb-4">Camera not started</p>
            <Button onclick={handleStartCamera} variant="secondary">Start Camera</Button>
          </div>
        {/if}
        <!-- svelte-ignore a11y_media_has_caption -->
        <video bind:this={videoElement} class="w-full h-full object-cover" autoplay playsinline muted crossorigin="anonymous"></video>

        {#if calibrationMode && videoElement}
          <CalibrationOverlay
            points={calibrationPoints}
            {videoElement}
            onclick={handleCalibrationClick}
            onreset={() => (calibrationPoints = [])}
            onconfirm={confirmCalibration}
          />
        {/if}
      </div>

      <!-- Recording Controls -->
      <div class="mt-auto">
      <Card.Root class="bg-[#1a1d24] border-gray-800">
        <Card.Content class="flex items-center justify-between p-4">
          <div class="flex items-center gap-4">
            <div class="w-10 h-10 rounded-full bg-red-500/20 flex items-center justify-center">
              <div class="w-4 h-4 bg-red-500 rounded-full {isRecording ? 'animate-pulse' : ''}"></div>
            </div>
            <div>
              <h3 class="font-bold text-white">{simulationMode ? "Simulation Mode" : "Recording in Progress"}</h3>
              <p class="text-sm text-gray-400">
                {#if simulationMode && !isCalibrated}Calibration required
                {:else if simulationMode && isCalibrated}Ready for analysis {teamsConfigured
                    ? `(Teams: ${team0Name} vs ${team1Name})`
                    : "(Auto-detecting teams...)"}
                {:else}Duration: {recordingDuration}{/if}
              </p>
            </div>
          </div>
          <div class="flex gap-3">
            {#if simulationMode && session?.isConnected()}
              <div class="flex items-center gap-2 mr-2">
                <Button size="icon" variant="ghost" onclick={togglePlayPause}>
                  {#if isPaused}<Play class="w-4 h-4" />{:else}<Pause class="w-4 h-4" />{/if}
                </Button>
                <select 
                  class="bg-[#0f1116] border border-gray-700 rounded px-2 py-1 text-sm text-white"
                  bind:value={playbackSpeed}
                  onchange={(e) => setPlaybackSpeed(parseFloat(e.currentTarget.value))}
                >
                  <option value="0.25">0.25x</option>
                  <option value="0.5">0.5x</option>
                  <option value="1">1x</option>
                  <option value="1.5">1.5x</option>
                  <option value="2">2x</option>
                </select>
              </div>
            {/if}
            {#if !stream}
              <Button disabled variant="secondary">Start Camera First</Button>
            {:else if !session?.isConnected()}
              <Button 
                onclick={handleStartAnalysis} 
                variant="secondary"
                disabled={calibrationMode}
                class={calibrationMode ? "opacity-50 cursor-not-allowed" : ""}
                title={calibrationMode ? "Complete calibration before starting analysis" : ""}
              >
                <Play class="w-4 h-4 mr-2" /> {calibrationMode ? "Calibrating..." : "Start Analysis"}
              </Button>
            {:else}
              <Button onclick={requestStopAnalysis} variant="destructive">
                <Square class="w-4 h-4 mr-2" /> Stop
              </Button>
            {/if}
          </div>
        </Card.Content>
      </Card.Root>
      </div>

      
    </div>

    <!-- Right Column -->
    <div class="flex flex-col gap-6 justify-between h-full">
      <div class="flex flex-col gap-6">
              {#if !session?.isConnected()}
          <!-- Empty State for Stats -->
          <div class="flex flex-col items-center justify-center py-12 px-4 rounded-xl border-2 border-dashed border-gray-700 bg-[#1a1d24]/50">
            <Activity class="w-10 h-10 text-gray-600 mb-3" />
            <p class="text-gray-500 font-medium text-sm">Statistics</p>
            <p class="text-gray-600 text-xs mt-1">Start analysis to view shot statistics</p>
          </div>
        {:else}
{#if teamsConfigured}
        <div class="flex bg-[#0f1116] rounded-xl p-1.5 border border-gray-800 shadow-xl overflow-hidden shadow-black/50">
          <button
            class="flex-1 py-2 px-4 text-sm font-bold flex items-center justify-center gap-2 rounded-lg transition-all duration-300 {activeTab ===
            'all'
              ? 'bg-[#1a1d24] text-white shadow-md'
              : 'text-gray-400 hover:text-white hover:bg-white/5'}"
            onclick={() => (activeTab = "all")}>All</button
          >
          <button
            class="flex-1 py-2 px-4 text-sm font-bold flex items-center justify-center gap-2 rounded-lg transition-all duration-300 {activeTab ===
            'team1'
              ? 'bg-[#1a1d24] text-white shadow-md'
              : 'text-gray-400 hover:text-white hover:bg-white/5'}"
            onclick={() => (activeTab = "team1")}
            ><div class="w-2 h-2 rounded-full" style="background-color: {team0Color}"></div>
            {team0Name}</button
          >
          <button
            class="flex-1 py-2 px-4 text-sm font-bold flex items-center justify-center gap-2 rounded-lg transition-all duration-300 {activeTab ===
            'team2'
              ? 'bg-[#1a1d24] text-white shadow-md'
              : 'text-gray-400 hover:text-white hover:bg-white/5'}"
            onclick={() => (activeTab = "team2")}
            ><div class="w-2 h-2 rounded-full" style="background-color: {team1Color}"></div>
            {team1Name}</button
          >
          <button
            class="flex-1 py-2 px-4 text-sm font-bold flex items-center justify-center gap-2 rounded-lg transition-all duration-300 {activeTab ===
            'comparison'
              ? 'bg-indigo-600/20 text-indigo-400 border border-indigo-500/30'
              : 'text-gray-400 hover:text-white hover:bg-white/5'}"
            onclick={() => (activeTab = "comparison")}>Comparison</button
          >
        </div>
      {/if}

      {#if activeTab !== "comparison"}
        <div class="grid grid-cols-2 gap-4">
          {#each [{ label: "Total Shots", value: `${currentStats.totalShots.made}/${currentStats.totalShots.total}` }, { label: "Field Goal %", value: `${Number.isFinite(currentStats.percentages.fieldGoal) ? currentStats.percentages.fieldGoal.toFixed(1) : "0.0"}%` }, { label: "2-Point %", value: `${Number.isFinite(currentStats.percentages.twoPoint) ? currentStats.percentages.twoPoint.toFixed(1) : "0.0"}%` }, { label: "3-Point %", value: `${Number.isFinite(currentStats.percentages.threePoint) ? currentStats.percentages.threePoint.toFixed(1) : "0.0"}%` }] as stat}
            <Card.Root class="bg-[#1a1d24] border-gray-800">
              <Card.Content class="p-4">
                <p class="text-gray-400 text-sm mb-1">{stat.label}</p>
                <p class="text-2xl font-bold text-white">{stat.value}</p>
              </Card.Content>
            </Card.Root>
          {/each}
        </div>

      {:else}
        <!-- Comparison View -->
        <Card.Root class="bg-[#1a1d24] border-gray-800 flex flex-col pt-4 min-h-0">
          <Card.Header class="pb-4 pt-0">
            <Card.Title class="text-center text-sm text-gray-400 font-bold uppercase tracking-widest">Head to Head</Card.Title>
          </Card.Header>
          <Card.Content class="space-y-6 flex-1 flex flex-col px-4 text-sm pb-4 min-h-0 overflow-y-auto">
            <div class="space-y-4">
              <!-- Total Shots -->
              <div class="flex items-center justify-between text-base">
                <div class="font-bold text-white text-left flex-1 bg-[#0f1116] px-4 py-3 rounded-lg border border-gray-800">
                  {team0Stats.totalShots.made}/{team0Stats.totalShots.total}
                </div>
                <div class="text-gray-500 font-medium px-4 text-xs">VS</div>
                <div class="font-bold text-white text-right flex-1 bg-[#0f1116] px-4 py-3 rounded-lg border border-gray-800">
                  {team1Stats.totalShots.made}/{team1Stats.totalShots.total}
                </div>
              </div>
              <div class="text-center text-gray-500 text-xs font-semibold uppercase -mt-11 translate-y-2 pointer-events-none">
                Total Shots
              </div>

              <!-- Field Goal % -->
              <div class="flex items-center justify-between text-base pt-2">
                <div class="font-bold text-white text-left flex-1 bg-[#0f1116] px-4 py-3 rounded-lg border border-gray-800">
                  {Number.isFinite(team0Stats.percentages.fieldGoal) ? team0Stats.percentages.fieldGoal.toFixed(1) : "0.0"}%
                </div>
                <div class="text-gray-500 font-medium px-4 text-xs">VS</div>
                <div class="font-bold text-white text-right flex-1 bg-[#0f1116] px-4 py-3 rounded-lg border border-gray-800">
                  {Number.isFinite(team1Stats.percentages.fieldGoal) ? team1Stats.percentages.fieldGoal.toFixed(1) : "0.0"}%
                </div>
              </div>
              <div class="text-center text-gray-500 text-xs font-semibold uppercase -mt-11 translate-y-2 pointer-events-none">
                Field Goal %
              </div>

              <!-- 2 Point % -->
              <div class="flex items-center justify-between text-base pt-2">
                <div class="font-bold text-white text-left flex-1 bg-[#0f1116] px-4 py-3 rounded-lg border border-gray-800">
                  {Number.isFinite(team0Stats.percentages.twoPoint) ? team0Stats.percentages.twoPoint.toFixed(1) : "0.0"}%
                </div>
                <div class="text-gray-500 font-medium px-4 text-xs">VS</div>
                <div class="font-bold text-white text-right flex-1 bg-[#0f1116] px-4 py-3 rounded-lg border border-gray-800">
                  {Number.isFinite(team1Stats.percentages.twoPoint) ? team1Stats.percentages.twoPoint.toFixed(1) : "0.0"}%
                </div>
              </div>
              <div class="text-center text-gray-500 text-xs font-semibold uppercase -mt-11 translate-y-2 pointer-events-none">
                2 Point %
              </div>

              <!-- 3 Point % -->
              <div class="flex items-center justify-between text-base pt-2">
                <div class="font-bold text-white text-left flex-1 bg-[#0f1116] px-4 py-3 rounded-lg border border-gray-800">
                  {Number.isFinite(team0Stats.percentages.threePoint) ? team0Stats.percentages.threePoint.toFixed(1) : "0.0"}%
                </div>
                <div class="text-gray-500 font-medium px-4 text-xs">VS</div>
                <div class="font-bold text-white text-right flex-1 bg-[#0f1116] px-4 py-3 rounded-lg border border-gray-800">
                  {Number.isFinite(team1Stats.percentages.threePoint) ? team1Stats.percentages.threePoint.toFixed(1) : "0.0"}%
                </div>
              </div>
              <div class="text-center text-gray-500 text-xs font-semibold uppercase -mt-11 translate-y-2 pointer-events-none">
                3 Point %
              </div>
            </div>

            <div class="grid grid-cols-2 gap-4 mt-4 h-[300px]">
              <div class="flex flex-col gap-2 relative bg-[#0f1116] rounded-xl border border-gray-800 p-2 overflow-hidden items-center">
                <div class="flex items-center gap-1.5 justify-center py-1 mt-1 z-10 w-full">
                  <div class="w-1.5 h-1.5 rounded-full" style="background-color: {team0Color}"></div>
                  <span class="text-xs text-gray-300 font-medium truncate max-w-[120px]">{team0Name}</span>
                </div>
                <div class="w-[85%] mx-auto h-[90%] scale-105 pointer-events-none mt-4 absolute">
                  <ShotChart shotPoints={shotPoints.filter((p) => p.team === team0Name)} {selectedShotId} hideHeader={true} />
                </div>
              </div>
              <div class="flex flex-col gap-2 relative bg-[#0f1116] rounded-xl border border-gray-800 p-2 overflow-hidden items-center">
                <div class="flex items-center gap-1.5 justify-center py-1 mt-1 z-10 w-full">
                  <div class="w-1.5 h-1.5 rounded-full" style="background-color: {team1Color}"></div>
                  <span class="text-xs text-gray-300 font-medium truncate max-w-[120px]">{team1Name}</span>
                </div>
                <div class="w-[85%] mx-auto h-[90%] scale-105 pointer-events-none mt-4 absolute">
                  <ShotChart shotPoints={shotPoints.filter((p) => p.team === team1Name)} {selectedShotId} hideHeader={true} />
                </div>
              </div>
            </div>
          </Card.Content>
        </Card.Root>
      {/if}
        {/if}
      </div>

      {#if session?.isConnected() && activeTab !== "comparison"}
        <div class="w-full mt-auto self-end">
          <ShotChart
            shotPoints={activeTab === "team1"
              ? shotPoints.filter((p) => p.team === team0Name)
              : activeTab === "team2"
                ? shotPoints.filter((p) => p.team === team1Name)
                : shotPoints}
            {selectedShotId}
          />
        </div>
      {/if}
    </div>
  </div>


  <div class="mt-6 w-full">
    <!-- Live Shot Log -->
      <Card.Root class="bg-[#1a1d24] border-gray-800 flex flex-col min-h-0 relative">
        <Card.Header class="pb-2 flex flex-row items-center justify-between">
          <Card.Title class="text-white">Live Shot Log</Card.Title>
          {#if teamsConfigured}
            <div class="flex bg-[#0f1116] rounded-full p-1 border border-gray-700">
              <button
                class="px-3 py-1 text-xs font-medium rounded-full transition-colors {eventLogFilter === 'all'
                  ? 'bg-[#9333ea] text-white'
                  : 'text-gray-400 hover:text-white'}"
                onclick={() => (eventLogFilter = "all")}>All</button
              >
              <button
                class="px-3 py-1 text-xs font-medium rounded-full transition-colors flex items-center gap-1 {eventLogFilter === 'team1'
                  ? 'bg-[#1a1d24] text-white'
                  : 'text-gray-400 hover:text-white'}"
                onclick={() => (eventLogFilter = "team1")}
                ><div class="w-1.5 h-1.5 rounded-full" style="background-color: {team0Color}"></div>
                {team0Name}</button
              >
              <button
                class="px-3 py-1 text-xs font-medium rounded-full transition-colors flex items-center gap-1 {eventLogFilter === 'team2'
                  ? 'bg-[#1a1d24] text-white'
                  : 'text-gray-400 hover:text-white'}"
                onclick={() => (eventLogFilter = "team2")}
                ><div class="w-1.5 h-1.5 rounded-full" style="background-color: {team1Color}"></div>
                {team1Name}</button
              >
            </div>
          {/if}
        </Card.Header>
        <Card.Content class="p-0 max-h-[400px] overflow-y-auto">
          
          {#if logs.length === 0 && !session?.isConnected()}
            <div class="flex flex-col items-center justify-center py-16 text-gray-500">
              <List class="w-10 h-10 mb-3 opacity-40" />
              <p class="text-sm font-medium">No shots recorded</p>
              <p class="text-xs text-gray-600 mt-1">Shot events will appear here during analysis</p>
            </div>
          {:else}
<div class="flex flex-col">
            {#each logs.filter((l) => eventLogFilter === "all" || (eventLogFilter === "team1" && l.team === team0Name) || (eventLogFilter === "team2" && l.team === team1Name)) as log}
              <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
              <div
                class="flex items-center gap-4 p-4 border-b border-gray-800 last:border-0 cursor-pointer transition-colors
                  {selectedShotId === log.id ? 'bg-white/10' : 'hover:bg-white/5'}"
                onclick={() => (selectedShotId = selectedShotId === log.id ? null : log.id)}
              >
                <div
                  class="w-6 h-6 rounded-full flex items-center justify-center shrink-0 {log.result === 'made'
                    ? 'bg-green-500/20 text-green-500'
                    : 'bg-red-500/20 text-red-500'}"
                >
                  {#if log.result === "made"}<Check class="w-4 h-4" />{:else}<X class="w-4 h-4" />{/if}
                </div>
                <div class="flex-1 flex items-center gap-2">
                  <span class="text-gray-400 font-mono text-xs shrink-0">[{formatMs(log.timestamp_ms)}]</span>
                  <span class="text-white font-medium text-sm">{log.type}</span>
                  <span class="text-gray-400 text-sm capitalize">{log.result}</span>
                  <span class="text-gray-600 text-sm hidden sm:inline-block truncate">- {log.location}</span>
                </div>
                {#if log.team}
                  <div
                    class="shrink-0 px-2 py-0.5 rounded text-xs font-medium bg-[#2a2d34] text-gray-300 border border-gray-700 max-w-[100px] truncate"
                  >
                    {log.team}
                  </div>
                {/if}
              </div>
            {/each}
          </div>
          {/if}
        </Card.Content>
      </Card.Root>
  </div>

  <!-- Global Start Overlay -->
  {#if !stream}
    <div class="fixed inset-0 z-[9999] bg-black/80 flex items-center justify-center backdrop-blur-sm">
      <div class="text-center space-y-6 p-8 bg-[#1a1d24] rounded-2xl border border-gray-700 max-w-md w-full shadow-2xl">
        <div class="w-20 h-20 bg-blue-600 rounded-full flex items-center justify-center mx-auto shadow-lg shadow-blue-600/20">
          <Video class="w-10 h-10 text-white" />
        </div>
        <div>
          <h2 class="text-2xl font-bold text-white mb-2">Ready to Analyze?</h2>
          <p class="text-gray-400">Connect your camera or simulate with a video file to start tracking shots.</p>
        </div>
        <div class="flex flex-col gap-3">
          <Button onclick={handleStartCamera} size="lg" class="w-full font-bold text-lg h-12">Start Camera</Button>
          <Button onclick={handleStartSimulation} variant="secondary" size="lg" class="w-full font-bold text-lg h-12">
            <Play class="w-5 h-5 mr-2" /> Simulate Video
          </Button>
        </div>
      </div>
    </div>
  {/if}
</div>
