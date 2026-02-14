<script lang="ts">
  import { onDestroy } from "svelte";
  import Button from "$lib/components/ui/button/button.svelte";
  import * as Card from "$lib/components/ui/card/index.js";
  import { Settings, Square, Check, X, Play, Video, Crosshair } from "lucide-svelte";
  import CalibrationOverlay from "$lib/components/CalibrationOverlay.svelte";
  import TeamSetupOverlay from "$lib/components/TeamSetupOverlay.svelte";
  import ShotChart from "$lib/components/ShotChart.svelte";
  import { createAnalysisSession } from "$lib/utils/webrtc-utils";
  import { videoClickToCalibrationPoint, submitCalibration, type CalibrationPoint } from "$lib/services/calibration";
  import { submitTeamColors } from "$lib/services/team-colors";
  import { BACKEND_URL } from "$lib";
  import { formatMs } from "$lib/utils/format";
  import type { Shot, GameStats } from "$lib/types";

  // --- DOM refs ---
  let videoElement: HTMLVideoElement | null = null;
  let stream = $state<MediaStream | null>(null);

  // --- Mode flags ---
  let simulationMode = $state<boolean>(false);
  let calibrationMode = $state(false);
  let calibrationPoints = $state<CalibrationPoint[]>([]);
  let isCalibrated = $state(false);
  let teamSetupMode = $state(false);
  let team0Color = $state("red");
  let team1Color = $state("blue");
  let teamsConfigured = $state(false);

  // --- Analysis state ---
  let session: ReturnType<typeof createAnalysisSession> | null = null;
  let logs = $state<Shot[]>([]);
  let shots = $state<Shot[]>([]);
  let selectedShotId = $state<number | null>(null);
  let stats = $state<GameStats>({
    totalShots: { made: 0, total: 0 },
    percentages: { fieldGoal: 0.0, twoPoint: 0.0, threePoint: 0.0 },
  });

  type ShotPoint = { id: number; x: number; y: number; result: Shot["result"] };
  const shotPoints = $derived<ShotPoint[]>(shots.map((s) => ({ id: s.id, x: s.coord.x, y: s.coord.y, result: s.result })));

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
    videoElement.src = `${BACKEND_URL}/video/video9.mp4`;
    await new Promise<void>((r) => videoElement!.addEventListener("canplay", () => r(), { once: true }));
    videoElement.playbackRate = 0.5;
    await videoElement.play();
    stream = (videoElement as any).captureStream();
    videoElement.pause();
    calibrationMode = true;
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
        teamSetupMode = true;
      } else {
        alert(`Calibration failed: ${data.error || "Unknown error"}`);
      }
    } catch {
      alert("Could not send calibration to server.");
    }
  }

  // --- Team colors handlers ---
  async function confirmTeamColors() {
    if (team0Color === team1Color) {
      alert("Select different colors.");
      return;
    }
    try {
      const data = await submitTeamColors(team0Color, team1Color);
      if (data.success) {
        teamsConfigured = true;
        teamSetupMode = false;
        videoElement?.play();
      } else {
        alert(`Team setup failed: ${data.error}`);
      }
    } catch {
      alert("Could not send team colors to server.");
    }
  }

  function skipTeamSetup() {
    teamSetupMode = false;
    videoElement?.play();
  }

  // --- Analysis ---
  async function startAnalysis() {
    if (!stream) {
      alert("Start camera or simulation first.");
      return;
    }
    try {
      session = createAnalysisSession(
        stream,
        {
          onStatsUpdate: (s) => (stats = s),
          onShotDetected: (shot) => {
            logs = [shot, ...logs].slice(0, 50);
            shots = [...shots, shot].slice(-200);
          },
        },
        { backendUrl: BACKEND_URL, simulationMode, isCalibrated },
      );
      await session.start();
      if (simulationMode && videoElement?.paused) videoElement.play();
    } catch (error) {
      console.error("Failed to start analysis:", error);
      alert("Could not establish a connection with the server.");
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
      calibrationMode = false;
      calibrationPoints = [];
      isCalibrated = false;
      teamSetupMode = false;
      teamsConfigured = false;
      stream = null;
    } else {
      stopCamera();
    }
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

        {#if teamSetupMode}
          <TeamSetupOverlay bind:team0Color bind:team1Color onconfirm={confirmTeamColors} onskip={skipTeamSetup} />
        {/if}
      </div>

      <!-- Recording Controls -->
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
                {:else if simulationMode && teamSetupMode}Configure team colors
                {:else if simulationMode && isCalibrated}Ready for analysis {teamsConfigured
                    ? `(Teams: ${team0Color} vs ${team1Color})`
                    : "(No team tracking)"}
                {:else}Duration: {recordingDuration}{/if}
              </p>
            </div>
          </div>
          <div class="flex gap-3">
            {#if simulationMode && stream && !calibrationMode && !isCalibrated}
              <Button onclick={enterCalibrationMode} variant="secondary" size="sm">
                <Crosshair class="w-4 h-4 mr-2" /> Calibrate
              </Button>
            {/if}
            {#if !stream}
              <Button disabled variant="secondary">Start Camera First</Button>
            {:else if !session?.isConnected()}
              <Button onclick={startAnalysis} variant="secondary" disabled={simulationMode && !isCalibrated}>
                <Play class="w-4 h-4 mr-2" /> Start Analysis
              </Button>
            {:else}
              <Button onclick={stopAnalysis} variant="destructive">
                <Square class="w-4 h-4 mr-2" /> Stop
              </Button>
            {/if}
          </div>
        </Card.Content>
      </Card.Root>

      <!-- Live Shot Log -->
      <Card.Root class="bg-[#1a1d24] border-gray-800">
        <Card.Header class="pb-2"><Card.Title class="text-white">Live Shot Log</Card.Title></Card.Header>
        <Card.Content class="p-0 max-h-[400px] overflow-y-auto">
          <div class="flex flex-col">
            {#each logs as log}
              <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
              <div
                class="flex items-center gap-4 p-4 border-b border-gray-800 last:border-0 cursor-pointer transition-colors
                  {selectedShotId === log.id ? 'bg-white/10' : 'hover:bg-white/5'}"
                onclick={() => (selectedShotId = selectedShotId === log.id ? null : log.id)}
              >
                <div
                  class="w-6 h-6 rounded-full flex items-center justify-center {log.result === 'made'
                    ? 'bg-green-500/20 text-green-500'
                    : 'bg-red-500/20 text-red-500'}"
                >
                  {#if log.result === "made"}<Check class="w-4 h-4" />{:else}<X class="w-4 h-4" />{/if}
                </div>
                <div class="flex-1">
                  <span class="text-gray-400 font-mono text-sm">[{formatMs(log.timestamp_ms)}]</span>
                  <span class="text-white font-medium ml-2">{log.type}</span>
                  <span class="text-gray-500 text-sm ml-1">- {log.location}</span>
                </div>
              </div>
            {/each}
          </div>
        </Card.Content>
      </Card.Root>
    </div>

    <!-- Right Column -->
    <div class="flex flex-col gap-6 self-start sticky top-6">
      <div class="grid grid-cols-2 gap-4">
        {#each [{ label: "Total Shots", value: `${stats.totalShots.made}/${stats.totalShots.total}` }, { label: "Field Goal %", value: `${Number.isFinite(stats.percentages.fieldGoal) ? stats.percentages.fieldGoal.toFixed(2) : "0.0"}%` }, { label: "2-Point %", value: `${Number.isFinite(stats.percentages.twoPoint) ? stats.percentages.twoPoint.toFixed(2) : "0.0"}%` }, { label: "3-Point %", value: `${Number.isFinite(stats.percentages.threePoint) ? stats.percentages.threePoint.toFixed(2) : "0.0"}%` }] as stat}
          <Card.Root class="bg-[#1a1d24] border-gray-800">
            <Card.Content class="p-4">
              <p class="text-gray-400 text-sm mb-1">{stat.label}</p>
              <p class="text-2xl font-bold text-white">{stat.value}</p>
            </Card.Content>
          </Card.Root>
        {/each}
      </div>

      <ShotChart {shotPoints} {selectedShotId} />
    </div>
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
