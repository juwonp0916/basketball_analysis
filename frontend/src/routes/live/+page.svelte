<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { createPeerConnection, sendOffer } from "$lib/webrtc_utils";
  import RadialChart from "$lib/components/RadialChart.svelte";
  import Button from "$lib/components/ui/button/button.svelte";

  // TODO: Load data from backend
  const data1 = { player: "1e3db", shots_succs: 50, total_shots: 200, color: "var(--color-safari)" };
  const data2 = { player: "1e3dc", shots_succs: 120, total_shots: 200, color: "var(--color-safari)" };
  const data3 = { player: "1e3dd", shots_succs: 160, total_shots: 200, color: "var(--color-safari)" };

  let videoElement: HTMLVideoElement | null = null;
  let stream = $state<MediaStream | null>(null);

  let pc: RTCPeerConnection | null = null;

  async function startCamera() {
    try {
      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      };

      console.log(videoElement);

      stream = await navigator.mediaDevices.getUserMedia(constraints);

      if (videoElement && stream) {
        videoElement.srcObject = stream;
      }
    } catch (error) {
      console.error("Error accessing camera: ", error);
      alert("Could not access the camera. Please ensure you have a camera connected and have granted permission.");
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      if (videoElement) videoElement.srcObject = null;
      stream = null;
    }
  }

  // TOOD: Create analysis event handling
  async function startAnalysis() {
    if (!stream) {
      alert("Please start the camera first.");
      return;
    }

    try {
      pc = createPeerConnection();

      stream.getTracks().forEach((track) => pc?.addTrack(track, stream!));

      pc.ondatachannel = (event) => {
        const channel = event.channel;
        if (channel.label === "analytics") {
          console.log("Analytics data channel received!");

          channel.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log("Received analytics: ", data);
            // TODO: Update UI with received data
          };

          channel.onopen = () => console.log("Analytics channel opened.");
          channel.onclose = () => console.log("Analytics channel closed.");
        }
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // TODO: Update server url with env variables
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
</script>

<main class="flex flex-col items-center justify-center p-8">
  <h1 class="text-4xl font-bold mb-4">Live Analysis</h1>
  {#if !stream}
    <Button
      onclick={startCamera}
      class="w-full h-120 hover:bg-gray-50 bg-transparent border-dashed border-2 text-black text-lg font-light py-2 px-4 rounded-lg m-2 transition-colors duration-300"
      >Start Camera
    </Button>
  {/if}
  <div class:hidden={!stream} class="w-full mx-auto bg-black border-2 border-gray-600 rounded-lg shadow-lg overflow-hidden">
    <video bind:this={videoElement} class="w-full max-h-115 block" autoplay playsinline muted></video>
  </div>
  {#if stream}
    <!-- TODO: Create and handle analysis event -->
    <div>
      <Button onclick={startAnalysis} class="text-white font-bold py-2 px-4 rounded-lg m-2">Start Analysis</Button>
      <Button onclick={stopAnalysis} variant="destructive" class="text-white font-bold py-2 px-4 rounded-lg m-2">Stop Analysis</Button>
    </div>
    <div class="items-center text-center font-bold text-3xl m-5">Statistics</div>
    <div class="grid lg:grid-cols-3 sm:grid-cols-2 grid-cols-1 gap-5 p-3">
      <div><RadialChart shotData={data1} /></div>
      <div><RadialChart shotData={data2} /></div>
      <div><RadialChart shotData={data3} /></div>
      <div><RadialChart shotData={data2} /></div>
      <div><RadialChart shotData={data1} /></div>
      <div><RadialChart shotData={data3} /></div>
    </div>
  {/if}
</main>
