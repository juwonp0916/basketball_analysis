<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { createPeerConnection, sendOffer } from "$lib/utils/webrtc-utils";
  import Button from "$lib/components/ui/button/button.svelte";
  import * as Card from "$lib/components/ui/card/index.js";
  import { Settings, Circle, Square, Check, X, Play, Pause, Video, RotateCcw, Crosshair } from "lucide-svelte";
  import { Chart, Svg, Points, Group, Text } from "layerchart";
  import type { Shot, GameStats } from "@/lib/types";
  import { writable } from "svelte/store";

  // --- WebRTC & Camera Logic ---

  let videoElement: HTMLVideoElement | null = null;
  let videoContainer: HTMLDivElement | null = null;
  let stream = $state<MediaStream | null>(null);
  let pc: RTCPeerConnection | null = null;
  let analyticsChannel: RTCDataChannel | null = null;

  // --- Simulation & Calibration State ---
  const BACKEND_URL = "http://127.0.0.1:8000";
  let simulationMode = $state(false);
  let calibrationMode = $state(false);
  let calibrationModeSelection = $state(false); // Show mode selector
  let selectedCalibrationMode = $state<"4-point" | "6-point">("4-point");
  let calibrationPoints = $state<{ x: number; y: number }[]>([]);
  let isCalibrated = $state(false);

  // --- Team Color State ---
  let teamSetupMode = $state(false);
  let team0Color = $state("red");
  let team1Color = $state("blue");
  let teamsConfigured = $state(false);

  const AVAILABLE_COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "cyan",
    "white",
    "black",
    "gray",
    "brown",
    "navy",
    "maroon",
    "lime",
    "teal",
    "gold",
  ];

  const CALIBRATION_LABELS_6PT = [
    "Baseline Left Sideline",
    "Baseline Left Penalty Box",
    "Baseline Right Penalty Box",
    "Baseline Right Sideline",
    "Free Throw Line Left",
    "Free Throw Line Right",
  ];

  const CALIBRATION_LABELS_4PT = [
    "Baseline Left Penalty Box",
    "Baseline Right Penalty Box",
    "Free Throw Line Left",
    "Free Throw Line Right",
  ];

  const CALIBRATION_LABELS = $derived(selectedCalibrationMode === "4-point" ? CALIBRATION_LABELS_4PT : CALIBRATION_LABELS_6PT);

  const expectedCalibrationPoints = $derived(selectedCalibrationMode === "4-point" ? 4 : 6);

  const CALIBRATION_COLORS_6PT = [
    "#eab308", // yellow - sideline
    "#22c55e", // green - penalty box
    "#22c55e", // green - penalty box
    "#eab308", // yellow - sideline
    "#d946ef", // magenta - FT line
    "#d946ef", // magenta - FT line
  ];

  const CALIBRATION_COLORS_4PT = [
    "#22c55e", // green - penalty box
    "#22c55e", // green - penalty box
    "#d946ef", // magenta - FT line
    "#d946ef", // magenta - FT line
  ];

  const CALIBRATION_COLORS = $derived(selectedCalibrationMode === "4-point" ? CALIBRATION_COLORS_4PT : CALIBRATION_COLORS_6PT);

  let logs = $state<Shot[]>([]);
  let shots = $state<Shot[]>([]);
  let analysisStartMs = $state<number | null>(null);
  let selectedShotId = $state<number | null>(null);

  let stats = $state<GameStats>({
    totalShots: { made: 0, total: 0 },
    percentages: { fieldGoal: 0.0, twoPoint: 0.0, threePoint: 0.0 },
  });

  // --- Chart Dimensions ---
  const chartPadding = { top: 20, bottom: 20, left: 20, right: 20 };
  let containerWidth = $state(0);
  let containerHeight = $state(0);
  let innerWidth = $derived(Math.max(0, containerWidth - chartPadding.left - chartPadding.right));
  let innerHeight = $derived(Math.max(0, containerHeight - chartPadding.top - chartPadding.bottom));

  function formatMs(ms: number): string {
    const totalSec = Math.max(0, Math.floor(ms / 1000));
    const mm = String(Math.floor(totalSec / 60)).padStart(2, "0");
    const ss = String(totalSec % 60).padStart(2, "0");
    return `${mm}:${ss}`;
  }

  type ShotPoint = { id: number; x: number; y: number; result: Shot["result"] };
  const shotPoints = $derived<ShotPoint[]>(shots.map((s) => ({ id: s.id, x: s.coord.x, y: s.coord.y, result: s.result })));

  // --- Camera Mode ---

  async function handleStartCamera() {
    try {
      const constraints = {
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      };
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoElement && stream) {
        videoElement.srcObject = stream;
      } else {
        console.error("videoElement or stream is null", { videoElement, stream });
      }
    } catch (error) {
      console.error("Error accessing camera: ", error);
      alert("Could not access the camera.");
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      if (videoElement) videoElement.srcObject = null;
      stream = null;
    }
  }

  // --- Simulation Mode ---

  async function handleStartSimulation() {
    if (!videoElement) return;
    simulationMode = true;

    videoElement.crossOrigin = "anonymous";
    videoElement.src = `${BACKEND_URL}/video/video9.mp4`;

    await new Promise<void>((resolve) => {
      videoElement!.addEventListener("canplay", () => resolve(), { once: true });
    });

    // Slow playback so the backend YOLO pipeline can process more frames.
    // At 1x the queue floods (~80% skipped); 0.3x keeps most frames processable.
    videoElement.playbackRate = 0.5;

    // Play briefly to initialize the stream, then pause for calibration
    await videoElement.play();

    // Capture the stream from the video element
    stream = (videoElement as any).captureStream();

    // Pause for calibration mode selection
    videoElement.pause();
    calibrationModeSelection = true;
  }

  // --- Calibration ---

  function handleCalibrationClick(event: MouseEvent) {
    console.log(`[Calibration Click] Current: ${calibrationPoints.length}/${expectedCalibrationPoints}`);

    if (!videoElement) {
      console.error("[Calibration Click] No video element");
      return;
    }

    if (calibrationPoints.length >= expectedCalibrationPoints) {
      console.log("[Calibration Click] Already at max points, ignoring");
      return;
    }

    const rect = videoElement.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;

    // Scale to actual video resolution
    const scaleX = videoElement.videoWidth / videoElement.clientWidth;
    const scaleY = videoElement.videoHeight / videoElement.clientHeight;

    const newPoint = { x: clickX * scaleX, y: clickY * scaleY };
    console.log(`[Calibration Click] Adding point ${calibrationPoints.length + 1}:`, newPoint);

    calibrationPoints = [...calibrationPoints, { x: clickX * scaleX, y: clickY * scaleY }];
  }

  function resetCalibration() {
    calibrationPoints = [];
  }

  async function confirmCalibration() {
    const expectedPoints = selectedCalibrationMode === "4-point" ? 4 : 6;
    if (!videoElement || calibrationPoints.length !== expectedPoints) return;

    const payload = {
      points: calibrationPoints.map((p) => [p.x, p.y]),
      image_width: videoElement.videoWidth,
      image_height: videoElement.videoHeight,
      mode: selectedCalibrationMode,
    };

    try {
      const res = await fetch(`${BACKEND_URL}/calibration`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.success) {
        isCalibrated = true;
        calibrationMode = false;
        // Show team setup after calibration
        teamSetupMode = true;
      } else {
        alert(`Calibration failed: ${data.error || "Unknown error"}`);
      }
    } catch (err) {
      console.error("Calibration request failed:", err);
      alert("Could not send calibration to server.");
    }
  }

  async function confirmTeamColors() {
    if (team0Color === team1Color) {
      alert("Please select different colors for each team.");
      return;
    }

    try {
      const res = await fetch(`${BACKEND_URL}/team-colors`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ team0_color: team0Color, team1_color: team1Color }),
      });
      const data = await res.json();
      if (data.success) {
        teamsConfigured = true;
        teamSetupMode = false;
        // Resume video playback
        if (videoElement) videoElement.play();
      } else {
        alert(`Team setup failed: ${data.error || "Unknown error"}`);
      }
    } catch (err) {
      console.error("Team colors request failed:", err);
      alert("Could not send team colors to server.");
    }
  }

  function skipTeamSetup() {
    teamSetupMode = false;
    // Resume video playback without team detection
    if (videoElement) videoElement.play();
  }

  // Get calibration point position in element-relative pixels (for SVG overlay display)
  function toDisplayCoords(point: { x: number; y: number }) {
    if (!videoElement) return { x: 0, y: 0 };
    return {
      x: point.x / (videoElement.videoWidth / videoElement.clientWidth),
      y: point.y / (videoElement.videoHeight / videoElement.clientHeight),
    };
  }

  // --- Analysis ---

  async function startAnalysis() {
    if (!stream) {
      alert("Please start the camera or simulation first.");
      return;
    }
    try {
      analysisStartMs = Date.now();

      pc = createPeerConnection();

      analyticsChannel = pc.createDataChannel("analytics");

      analyticsChannel.onopen = () => {
        console.log("Analytics data channel opened.");
        // Auto-start detection if simulation mode and calibrated
        if (simulationMode && isCalibrated) {
          fetch(`${BACKEND_URL}/detection/start`, { method: "POST" })
            .then((res) => res.json())
            .then((data) => console.log("Detection started:", data))
            .catch((err) => console.error("Failed to start detection:", err));
        }
      };
      analyticsChannel.onclose = () => console.log("Analytics data channel closed.");
      analyticsChannel.onmessage = (event) => {
        const msg = JSON.parse(event.data) as WebRTCMessage;

        console.log(msg);

        if (msg.type === "frame_sync") {
          stats = msg.current_stats;
          return;
        }

        if (msg.type === "shot_event") {
          stats = msg.updated_stats;

          // msg.event_data is already a `Shot` matching your frontend types.ts
          const newShot = msg.event_data;

          logs = [newShot, ...logs].slice(0, 50);
          shots = [...shots, newShot].slice(-200);
          return;
        }
      };

      stream.getTracks().forEach((track) => pc?.addTrack(track, stream!));

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const answer = await sendOffer(pc, `${BACKEND_URL}/offer`);
      await pc.setRemoteDescription(new RTCSessionDescription(answer));
      console.log("WebRTC connection established!");

      // Resume video if simulation mode (in case it was paused)
      if (simulationMode && videoElement && videoElement.paused) {
        videoElement.play();
      }
    } catch (error) {
      console.error("Failed to start analysis session:", error);
      alert("Could not establish a connection with the server.");
    }
  }

  function stopAnalysis() {
    if (pc) {
      pc.close();
      pc = null;
      console.log("WebRTC connection closed.");
    }

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
    // Show mode selection instead of directly entering calibration
    calibrationModeSelection = true;
    calibrationPoints = [];
    isCalibrated = false;
  }

  function startCalibrationWithMode(mode: "4-point" | "6-point") {
    selectedCalibrationMode = mode;
    calibrationModeSelection = false;
    calibrationMode = true;
    calibrationPoints = [];
  }

  onMount(() => {});

  onDestroy(() => {
    stopCamera();
    stopAnalysis();
  });

  // --- UI State ---
  let isRecording = $state(false);
  let recordingDuration = $state("00:12:45");
</script>

<div class="min-h-screen bg-[#0f1116] text-white font-sans p-6">
  <!-- Header -->
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
    <!-- Left Column: Video & Logs -->
    <div class="lg:col-span-2 flex flex-col gap-6">
      <!-- Video Player -->
      <div
        bind:this={videoContainer}
        class="relative w-full aspect-video bg-black rounded-xl overflow-hidden border border-gray-800 shadow-2xl"
      >
        {#if !stream}
          <div class="absolute inset-0 flex flex-col items-center justify-center text-gray-500">
            <Video class="w-12 h-12 mb-4 opacity-50" />
            <p class="mb-4">Camera not started</p>
            <Button onclick={handleStartCamera} variant="secondary">Start Camera</Button>
          </div>
        {/if}
        <!-- svelte-ignore a11y_media_has_caption -->
        <video bind:this={videoElement} class="w-full h-full object-cover" autoplay playsinline muted crossorigin="anonymous"></video>

        <!-- Calibration Overlay -->
        {#if calibrationMode && videoElement}
          <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
          <div class="absolute inset-0 cursor-crosshair" onclick={handleCalibrationClick}>
            <svg class="absolute inset-0 w-full h-full" style="pointer-events: none;">
              <!-- Connecting lines -->
              {#if calibrationPoints.length >= 2}
                <!-- Baseline segments -->
                {#each [[0, 1], [1, 2], [2, 3]] as [a, b]}
                  {#if calibrationPoints.length > b}
                    {@const pa = toDisplayCoords(calibrationPoints[a])}
                    {@const pb = toDisplayCoords(calibrationPoints[b])}
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
              {#if calibrationPoints.length >= expectedCalibrationPoints}
                <!-- Draw complete box based on mode -->
                {#if selectedCalibrationMode === "4-point"}
                  <!-- 4-point mode: indices 0,1,2,3 -->
                  {@const p0 = toDisplayCoords(calibrationPoints[0])}
                  {@const p1 = toDisplayCoords(calibrationPoints[1])}
                  {@const p2 = toDisplayCoords(calibrationPoints[2])}
                  {@const p3 = toDisplayCoords(calibrationPoints[3])}
                  <!-- Baseline -->
                  <line x1={p0.x} y1={p0.y} x2={p1.x} y2={p1.y} stroke="#22c55e" stroke-width="2" opacity="0.7" />
                  <!-- FT line -->
                  <line x1={p2.x} y1={p2.y} x2={p3.x} y2={p3.y} stroke="#d946ef" stroke-width="2" opacity="0.7" />
                  <!-- Sides -->
                  <line x1={p0.x} y1={p0.y} x2={p2.x} y2={p2.y} stroke="#22c55e" stroke-width="2" opacity="0.7" />
                  <line x1={p1.x} y1={p1.y} x2={p3.x} y2={p3.y} stroke="#22c55e" stroke-width="2" opacity="0.7" />
                {:else}
                  <!-- 6-point mode: baseline points 0-3, FT line points 4-5 -->
                  {@const p1 = toDisplayCoords(calibrationPoints[1])}
                  {@const p2 = toDisplayCoords(calibrationPoints[2])}
                  {@const p4 = toDisplayCoords(calibrationPoints[4])}
                  {@const p5 = toDisplayCoords(calibrationPoints[5])}
                  <!-- FT line (points 4-5) -->
                  <line x1={p4.x} y1={p4.y} x2={p5.x} y2={p5.y} stroke="#d946ef" stroke-width="2" opacity="0.7" />
                  <!-- Penalty box sides -->
                  <line x1={p1.x} y1={p1.y} x2={p4.x} y2={p4.y} stroke="#22c55e" stroke-width="2" opacity="0.7" />
                  <line x1={p2.x} y1={p2.y} x2={p5.x} y2={p5.y} stroke="#22c55e" stroke-width="2" opacity="0.7" />
                {/if}
              {/if}

              <!-- Point markers -->
              {#each calibrationPoints as point, i}
                {@const dp = toDisplayCoords(point)}
                <circle cx={dp.x} cy={dp.y} r="8" fill={CALIBRATION_COLORS[i]} opacity="0.9" stroke="white" stroke-width="1.5" />
                <text x={dp.x} y={dp.y + 1} text-anchor="middle" dominant-baseline="middle" fill="black" font-size="10" font-weight="bold">
                  {i + 1}
                </text>
                <text
                  x={dp.x + 14}
                  y={dp.y + 4}
                  fill="white"
                  font-size="11"
                  font-weight="600"
                  style="text-shadow: 0 1px 3px rgba(0,0,0,0.8);"
                >
                  {CALIBRATION_LABELS[i]}
                </text>
              {/each}
            </svg>

            <!-- Calibration instructions + action buttons -->
            <div
              class="absolute top-4 left-1/2 -translate-x-1/2 bg-black/80 rounded-lg px-4 py-2 text-center pointer-events-auto flex flex-col items-center gap-2"
            >
              {#if calibrationPoints.length < 6}
                <p class="text-sm font-medium text-white pointer-events-none">
                  Click point {calibrationPoints.length + 1}/6:
                  <span class="text-blue-400">{CALIBRATION_LABELS[calibrationPoints.length]}</span>
                </p>
              {:else}
                <p class="text-sm font-medium text-green-400">All {expectedCalibrationPoints} points placed. Confirm or reset.</p>
              {/if}
              <div class="flex gap-3">
                <Button
                  onclick={(e: MouseEvent) => {
                    e.stopPropagation();
                    resetCalibration();
                  }}
                  variant="secondary"
                  size="sm"
                >
                  <RotateCcw class="w-4 h-4 mr-1" /> Reset
                </Button>
                {#if calibrationPoints.length === 6}
                  <Button
                    onclick={(e: MouseEvent) => {
                      e.stopPropagation();
                      confirmCalibration();
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
        {/if}

        <!-- Team Setup Overlay -->
        {#if teamSetupMode}
          <div class="absolute inset-0 bg-black/70 flex items-center justify-center">
            <div class="bg-[#1a1d24] rounded-xl p-6 max-w-md w-full mx-4 border border-gray-700">
              <h3 class="text-xl font-bold text-white mb-4 text-center">Team Colors Setup</h3>
              <p class="text-gray-400 text-sm mb-6 text-center">Select jersey colors for each team to enable team-based shot tracking.</p>

              <div class="space-y-4 mb-6">
                <div>
                  <label class="block text-sm font-medium text-gray-300 mb-2">Team 1 Jersey Color</label>
                  <select
                    bind:value={team0Color}
                    class="w-full bg-[#0f1116] border border-gray-600 rounded-lg px-4 py-2 text-white capitalize"
                  >
                    {#each AVAILABLE_COLORS as color}
                      <option value={color} class="capitalize">{color}</option>
                    {/each}
                  </select>
                </div>

                <div>
                  <label class="block text-sm font-medium text-gray-300 mb-2">Team 2 Jersey Color</label>
                  <select
                    bind:value={team1Color}
                    class="w-full bg-[#0f1116] border border-gray-600 rounded-lg px-4 py-2 text-white capitalize"
                  >
                    {#each AVAILABLE_COLORS as color}
                      <option value={color} class="capitalize">{color}</option>
                    {/each}
                  </select>
                </div>
              </div>

              {#if team0Color === team1Color}
                <p class="text-red-400 text-sm mb-4 text-center">Please select different colors for each team.</p>
              {/if}

              <div class="flex gap-3">
                <Button onclick={skipTeamSetup} variant="secondary" class="flex-1">Skip</Button>
                <Button onclick={confirmTeamColors} class="flex-1 bg-green-600 hover:bg-green-700" disabled={team0Color === team1Color}>
                  <Check class="w-4 h-4 mr-2" /> Confirm
                </Button>
              </div>
            </div>
          </div>
        {/if}

        <!-- Video Overlay Controls -->
        <!-- <div class="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent flex justify-between items-center"> -->
        <!--   <span class="text-sm font-mono">1:02:14</span> -->
        <!--   <div class="w-full mx-4 h-1 bg-gray-600 rounded-full overflow-hidden"> -->
        <!--     <div class="w-2/3 h-full bg-white"></div> -->
        <!--   </div> -->
        <!--   <span class="text-sm font-mono">1:45:30</span> -->
        <!-- </div> -->
      </div>

      <!-- Recording Controls -->
      <Card.Root class="bg-[#1a1d24] border-gray-800">
        <Card.Content class="flex items-center justify-between p-4">
          <div class="flex items-center gap-4">
            <div class="w-10 h-10 rounded-full bg-red-500/20 flex items-center justify-center">
              <div class="w-4 h-4 bg-red-500 rounded-full {isRecording ? 'animate-pulse' : ''}"></div>
            </div>
            <div>
              <h3 class="font-bold text-white">
                {simulationMode ? "Simulation Mode" : "Recording in Progress"}
              </h3>
              <p class="text-sm text-gray-400">
                {#if simulationMode && !isCalibrated}
                  Calibration required
                {:else if simulationMode && isCalibrated && teamSetupMode}
                  Configure team colors
                {:else if simulationMode && isCalibrated}
                  Ready for analysis {teamsConfigured ? `(Teams: ${team0Color} vs ${team1Color})` : "(No team tracking)"}
                {:else}
                  Duration: {recordingDuration}
                {/if}
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
            {:else if !pc}
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
        <Card.Header class="pb-2">
          <Card.Title class="text-white">Live Shot Log</Card.Title>
        </Card.Header>
        <Card.Content class="p-0 max-h-[400px] overflow-y-auto">
          <div class="flex flex-col">
            {#each logs as log}
              <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
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
                  {#if log.result === "made"}
                    <Check class="w-4 h-4" />
                  {:else}
                    <X class="w-4 h-4" />
                  {/if}
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

    <!-- Right Column: Stats & Chart -->
    <div class="flex flex-col gap-6 self-start sticky top-6">
      <!-- Stats Grid -->
      <div class="grid grid-cols-2 gap-4">
        <Card.Root class="bg-[#1a1d24] border-gray-800">
          <Card.Content class="p-4">
            <p class="text-gray-400 text-sm mb-1">Total Shots</p>
            <p class="text-2xl font-bold text-white">{stats.totalShots.made}/{stats.totalShots.total}</p>
          </Card.Content>
        </Card.Root>
        <Card.Root class="bg-[#1a1d24] border-gray-800">
          <Card.Content class="p-4">
            <p class="text-gray-400 text-sm mb-1">Field Goal %</p>
            <p class="text-2xl font-bold text-white">
              {Number.isFinite(stats.percentages.fieldGoal) ? stats.percentages.fieldGoal.toFixed(2) : "0.0"}%
            </p>
          </Card.Content>
        </Card.Root>
        <Card.Root class="bg-[#1a1d24] border-gray-800">
          <Card.Content class="p-4">
            <p class="text-gray-400 text-sm mb-1">2-Point %</p>
            <p class="text-2xl font-bold text-white">
              {Number.isFinite(stats.percentages.twoPoint) ? stats.percentages.twoPoint.toFixed(2) : "0.0"}%
            </p>
          </Card.Content>
        </Card.Root>
        <Card.Root class="bg-[#1a1d24] border-gray-800">
          <Card.Content class="p-4">
            <p class="text-gray-400 text-sm mb-1">3-Point %</p>
            <p class="text-2xl font-bold text-white">
              {Number.isFinite(stats.percentages.threePoint) ? stats.percentages.threePoint.toFixed(2) : "0.0"}%
            </p>
          </Card.Content>
        </Card.Root>
      </div>

      <!-- Shot Chart -->
      <Card.Root class="bg-[#1a1d24] border-gray-800 flex flex-col h-[800px]">
        <Card.Header class="pb-2">
          <div class="flex bg-[#0f1116] rounded-lg p-1 w-full">
            <button class="flex-1 py-1 text-sm font-medium rounded bg-[#1a1d24] text-white shadow">All</button>
            <button class="flex-1 py-1 text-sm font-medium text-gray-400 hover:text-white">Made</button>
            <button class="flex-1 py-1 text-sm font-medium text-gray-400 hover:text-white">Missed</button>
            <button class="flex-1 py-1 text-sm font-medium text-gray-400 hover:text-white">3PT</button>
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
                        <!-- Highlight ring for selected shot -->
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
    </div>
  </div>

  <!-- Global Start Overlay -->
  {#if !stream}
    <div class="fixed inset-0 z-[9999] bg-black/80 flex items-center justify-center backdrop-blur-sm pointer-events-auto">
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
