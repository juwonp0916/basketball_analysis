<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { createPeerConnection, sendOffer } from "$lib/webrtc_utils";
  import Button from "$lib/components/ui/button/button.svelte";
  import * as Card from "$lib/components/ui/card/index.js";
  import { Settings, Circle, Square, Check, X, Play, Pause, Video } from "lucide-svelte";
  import { Chart, Svg, Points, Group, Text } from "layerchart";
  import type { Shot, GameStats } from "@/lib/types";
  import { writable } from "svelte/store";

  // --- WebRTC & Camera Logic ---

  let videoElement: HTMLVideoElement | null = null;
  let stream = $state<MediaStream | null>(null);
  let pc: RTCPeerConnection | null = null;
  let analyticsChannel: RTCDataChannel | null = null;

  type FrameSyncPayload = {
    type: "frame_sync";
    sequence_id: number;
    video_timestamp_ms: number;
    current_stats: GameStats;
  };

  type ShotDetectedPayload = {
    type: "shot_event";
    video_timestamp_ms: number;
    event_data: Shot;
    updated_stats: GameStats;
  };

  type WebRTCMessage = FrameSyncPayload | ShotDetectedPayload;

  let logs = $state<Shot[]>([]);
  let shots = $state<Shot[]>([]);
  let analysisStartMs = $state<number | null>(null);

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

  async function startAnalysis() {
    if (!stream) {
      alert("Please start the camera first.");
      return;
    }
    try {
      analysisStartMs = Date.now();

      pc = createPeerConnection();

      analyticsChannel = pc.createDataChannel("analytics");

      analyticsChannel.onopen = () => console.log("Analytics data channel opened.");
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
      const answer = await sendOffer(pc, "http://127.0.0.1:8000/offer");
      await pc.setRemoteDescription(new RTCSessionDescription(answer));
      console.log("WebRTC connection established!");
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
    stopCamera();
  }

  onMount(() => {});

  onDestroy(() => {
    stopCamera();
    stopAnalysis();
  });

  // --- UI State & Mock Data ---
  let isRecording = $state(false);
  let recordingDuration = $state("00:12:45"); // Mock
</script>

<div class="min-h-screen bg-[#0f1116] text-white font-sans p-6">
  <!-- Header -->
  <header class="flex justify-between items-center mb-6">
    <div class="flex items-center gap-2">
      <div class="w-6 h-6 bg-blue-600 rounded-full"></div>
      <!-- Logo Placeholder -->
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
      <div class="relative w-full aspect-video bg-black rounded-xl overflow-hidden border border-gray-800 shadow-2xl">
        {#if !stream}
          <div class="absolute inset-0 flex flex-col items-center justify-center text-gray-500">
            <Video class="w-12 h-12 mb-4 opacity-50" />
            <p class="mb-4">Camera not started</p>
            <Button onclick={handleStartCamera} variant="secondary">Start Camera</Button>
          </div>
        {/if}
        <video bind:this={videoElement} class="w-full h-full object-cover" autoplay playsinline muted></video>

        <!-- Video Overlay Controls (Mock) -->
        <div class="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent flex justify-between items-center">
          <span class="text-sm font-mono">1:02:14</span>
          <div class="w-full mx-4 h-1 bg-gray-600 rounded-full overflow-hidden">
            <div class="w-2/3 h-full bg-white"></div>
          </div>
          <span class="text-sm font-mono">1:45:30</span>
        </div>
      </div>

      <!-- Recording Controls -->
      <Card.Root class="bg-[#1a1d24] border-gray-800">
        <Card.Content class="flex items-center justify-between p-4">
          <div class="flex items-center gap-4">
            <div class="w-10 h-10 rounded-full bg-red-500/20 flex items-center justify-center">
              <div class="w-4 h-4 bg-red-500 rounded-full {isRecording ? 'animate-pulse' : ''}"></div>
            </div>
            <div>
              <h3 class="font-bold text-white">Recording in Progress</h3>
              <p class="text-sm text-gray-400">Duration: {recordingDuration}</p>
            </div>
          </div>
          <div class="flex gap-3">
            {#if !stream}
              <Button disabled variant="secondary">Start Camera First</Button>
            {:else if !pc}
              <Button onclick={startAnalysis} variant="secondary">
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
      <Card.Root class="bg-[#1a1d24] border-gray-800 flex-1">
        <Card.Header class="pb-2">
          <Card.Title class="text-white">Live Shot Log</Card.Title>
        </Card.Header>
        <Card.Content class="p-0">
          <div class="flex flex-col">
            {#each logs as log}
              <div class="flex items-center gap-4 p-4 border-b border-gray-800 last:border-0 hover:bg-white/5 transition-colors">
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
    <div class="flex flex-col gap-6">
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
            <p class="text-2xl font-bold text-white">{stats.percentages.fieldGoal}%</p>
          </Card.Content>
        </Card.Root>
        <Card.Root class="bg-[#1a1d24] border-gray-800">
          <Card.Content class="p-4">
            <p class="text-gray-400 text-sm mb-1">2-Point %</p>
            <p class="text-2xl font-bold text-white">{stats.percentages.twoPoint}%</p>
          </Card.Content>
        </Card.Root>
        <Card.Root class="bg-[#1a1d24] border-gray-800">
          <Card.Content class="p-4">
            <p class="text-gray-400 text-sm mb-1">3-Point %</p>
            <p class="text-2xl font-bold text-white">{stats.percentages.threePoint}%</p>
          </Card.Content>
        </Card.Root>
      </div>

      <!-- Shot Chart -->
      <Card.Root class="bg-[#1a1d24] border-gray-800 flex-1 flex flex-col h-[500px]">
        <Card.Header class="pb-2">
          <!-- Tabs (Mock) -->
          <div class="flex bg-[#0f1116] rounded-lg p-1 w-full">
            <button class="flex-1 py-1 text-sm font-medium rounded bg-[#1a1d24] text-white shadow">All</button>
            <button class="flex-1 py-1 text-sm font-medium text-gray-400 hover:text-white">Made</button>
            <button class="flex-1 py-1 text-sm font-medium text-gray-400 hover:text-white">Missed</button>
            <button class="flex-1 py-1 text-sm font-medium text-gray-400 hover:text-white">3PT</button>
          </div>
        </Card.Header>
        <Card.Content class="flex-1 relative p-4 min-h-0">
          <!-- Basketball Court Visualization -->
          <div class="w-full h-full relative" bind:clientWidth={containerWidth} bind:clientHeight={containerHeight}>
            <Chart data={shotPoints} x="x" y="y" xDomain={[0, 50]} yDomain={[0, 47]} padding={chartPadding}>
              <Svg>
                <!-- Court Background -->
                <svg width={innerWidth} height={innerHeight} viewBox="0 0 50 47" preserveAspectRatio="none" style="overflow: visible;">
                  <g class="court-lines" stroke="#374151" stroke-width="0.5" fill="none">
                    <!-- Half Court Outline -->
                    <rect x="0" y="0" width="50" height="47" vector-effect="non-scaling-stroke" />
                    <!-- Key -->
                    <rect x="17" y="0" width="16" height="19" vector-effect="non-scaling-stroke" />
                    <!-- Free Throw Circle -->
                    <circle cx="25" cy="19" r="6" vector-effect="non-scaling-stroke" />
                    <!-- 3 Point Line (Simplified Arc) -->
                    <path d="M 3 0 L 3 14 Q 25 35 47 14 L 47 0" vector-effect="non-scaling-stroke" />
                    <!-- Hoop -->
                    <circle cx="25" cy="5.25" r="0.75" stroke="#ef4444" vector-effect="non-scaling-stroke" />
                    <!-- Backboard -->
                    <line x1="22" y1="4" x2="28" y2="4" vector-effect="non-scaling-stroke" />
                  </g>
                </svg>

                <!-- Shots -->
                <Points>
                  {#snippet children({ points })}
                    {#each points as point}
                      <circle
                        cx={point.x}
                        cy={point.y}
                        r="1.5"
                        fill={point.data.result === "made" ? "#10b981" : "#ef4444"}
                        stroke="none"
                        class="transition-all duration-300 hover:r-2"
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
          <p class="text-gray-400">Connect your camera to start tracking shots and viewing live analytics.</p>
        </div>
        <Button onclick={handleStartCamera} size="lg" class="w-full font-bold text-lg h-12">Start Camera</Button>
      </div>
    </div>
  {/if}
</div>
