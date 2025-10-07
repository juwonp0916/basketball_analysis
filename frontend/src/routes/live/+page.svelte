<script lang="ts">
  import { onMount, onDestroy } from "svelte";

  let videoElement: HTMLVideoElement | null = null;
  let stream: MediaStream | null = null;

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

  async function startAnalysis() {
    if (!stream) {
      alert("Please start the camera first.");
      return;
    }

    pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });

    stream.getTracks().forEach((track) => pc?.addTrack(track, stream!));

    pc.ondatachannel = (event) => {
      const channel = event.channel;
      if (channel.label === "analytics") {
        console.log("Analytics data channel received!");

        channel.onmessage = (event) => {
          const data = JSON.parse(event.data);
          console.log("Received analytics:", data);
          // TODO: Update UI with received data
        };

        channel.onopen = () => console.log("Analytics channel opened.");
        channel.onclose = () => console.log("Analytics channel closed.");
      }
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // TODO: Update base url with env variables
    const response = await fetch("http://127.0.0.1:8000/offer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(pc.localDescription),
    });

    const answer = await response.json();
    await pc.setRemoteDescription(new RTCSessionDescription(answer));

    console.log("WebRTC connection established!");
  }

  function stopAnalysis() {
    if (pc) {
      pc.close();
      pc = null;
      console.log("WebRTC connection closed.");
    }
  }

  onDestroy(() => {
    stopCamera();
    stopAnalysis();
  });
</script>

<main class="flex flex-col items-center justify-center p-4">
  <h1 class="text-2xl font-bold mb-4">Live Camera Feed</h1>
  <div class="w-full max-w-4xl mx-auto bg-black border-2 border-gray-600 rounded-lg shadow-lg overflow-hidden">
    <video bind:this={videoElement} class="w-full h-auto block" autoplay playsinline muted></video>
  </div>

  <div class="mt-4">
    <button
      onclick={startCamera}
      class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg m-2 transition-colors duration-300"
    >
      Start Camera
    </button>
    <button onclick={startAnalysis} class="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-lg m-2">
      Start Analysis
    </button>
    <button onclick={stopAnalysis} class="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg m-2">
      Stop Analysis
    </button>
    <button
      onclick={stopCamera}
      class="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg m-2 transition-colors duration-300"
    >
      Stop Camera
    </button>
  </div>
</main>
