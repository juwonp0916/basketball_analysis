import type { Shot, GameStats, WebRTCMessage } from "$lib/types";

export interface WebRTCAnswer {
  sdp: string;
  type: RTCSdpType;
}

export type AnalysisCallbacks = {
  onStatsUpdate: (stats: GameStats) => void;
  onShotDetected: (shot: Shot) => void;
  onTeamColorsCalibrated?: (team0Color: string, team1Color: string) => void;
}

export type AnalysisSessionOptions = {
  backendUrl: string;
  simulationMode?: boolean;
  isCalibrated?: boolean;
}

export function createAnalysisSession(stream: MediaStream, callbacks: AnalysisCallbacks, options: AnalysisSessionOptions) {
  let pc: RTCPeerConnection | null = null;

  async function start() {
    pc = createPeerConnection();

    const channel = pc.createDataChannel("analytics");

    channel.onopen = () => {
      console.log("Analytics channel is open");

      if (options.simulationMode && options.isCalibrated) {
        fetch(`${options.backendUrl}/detection/start`, { method: "POST" })
          .then((res) => res.json())
          .then((data) => console.log("Detection started:", data))
          .catch((err) => console.error("Failed to start detection:", err));
      }
    };

    channel.onclose = () => console.log("Analytics channel is closed");

    channel.onmessage = (event) => {
      const msg = JSON.parse(event.data) as WebRTCMessage;
      // console.log(msg):

      if (msg.type === "frame_sync") {
        callbacks.onStatsUpdate(msg.current_stats);
      } else if (msg.type === "shot_event") {
        callbacks.onStatsUpdate(msg.updated_stats)
        callbacks.onShotDetected(msg.event_data);
      } else if (msg.type === "team_colors") {
        if (callbacks.onTeamColorsCalibrated) {
          callbacks.onTeamColorsCalibrated(msg.team0_color, msg.team1_color);
        }
      }
    };

    stream.getTracks().forEach((track) => pc!.addTrack(track, stream));

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const answer = await sendOffer(pc, `${options.backendUrl}/offer`);
    await pc.setRemoteDescription(new RTCSessionDescription(answer));
    console.log("WebRTC connection established");
  }

  function stop() {
    if (pc) {
      pc.close();
      pc = null;
      console.log("WebRTC connection closed");
    }
  }

  function isConnected() {
    return pc !== null;
  }

  return { start, stop, isConnected };
}


export function createPeerConnection(): RTCPeerConnection {
  const conf = {
    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
  };
  return new RTCPeerConnection(conf);
}

export async function sendOffer(pc: RTCPeerConnection, url: string): Promise<WebRTCAnswer> {
  if (!pc.localDescription) {
    throw new Error("Local description is not set on the PeerConnection.");
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(pc.localDescription),
  });

  if (!response.ok) {
    throw new Error(`Failed to send offer to server: ${response.statusText}`);
  }

  return await response.json();
}

