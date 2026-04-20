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
  isCalibrated?: boolean;
}

export function createAnalysisSession(stream: MediaStream, callbacks: AnalysisCallbacks, options: AnalysisSessionOptions) {
  let pc: RTCPeerConnection | null = null;

  async function start() {
    pc = createPeerConnection();

    const channel = pc.createDataChannel("analytics");

    channel.onopen = () => {
      console.log("Analytics channel is open");

      if (options.isCalibrated) {
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

    // Lock encoding quality: high bitrate + maintain resolution over framerate.
    // This prevents the browser from adaptively downscaling the video, which
    // degrades YOLO detection and jersey color accuracy.
    await lockSenderQuality(pc);
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


/**
 * Lock video sender encoding parameters to prevent adaptive quality degradation.
 *
 * By default, the browser's WebRTC congestion control is free to:
 *  - Reduce resolution (scale down frames)
 *  - Reduce framerate
 *  - Reduce bitrate / increase quantization
 *
 * For CV/ML workloads we need consistent resolution and faithful colors, so:
 *  1. Set degradationPreference to "maintain-resolution" — if the encoder must
 *     degrade, it should drop frames rather than downscale.
 *  2. Set a high maxBitrate (8 Mbps) so the encoder has room for quality.
 *     On a local network this is easily sustainable; on remote links the
 *     congestion controller will still cap actual throughput below this.
 *  3. Remove any scaleResolutionDownBy that the browser may have defaulted.
 */
async function lockSenderQuality(pc: RTCPeerConnection): Promise<void> {
  for (const sender of pc.getSenders()) {
    if (sender.track?.kind !== "video") continue;

    const params = sender.getParameters();

    // degradationPreference: keep resolution, sacrifice framerate if needed
    params.degradationPreference = "maintain-resolution";

    // Configure each encoding layer
    if (params.encodings && params.encodings.length > 0) {
      for (const enc of params.encodings) {
        enc.maxBitrate = 8_000_000;         // 8 Mbps ceiling
        enc.scaleResolutionDownBy = 1.0;    // no downscaling
        // maxFramerate is intentionally left unset so the source
        // framerate is preserved (typically 30fps from getUserMedia)
      }
    }

    try {
      await sender.setParameters(params);
      console.log("Video sender quality locked:", {
        degradationPreference: params.degradationPreference,
        encodings: params.encodings?.map(e => ({
          maxBitrate: e.maxBitrate,
          scaleResolutionDownBy: e.scaleResolutionDownBy,
        })),
      });
    } catch (err) {
      // setParameters can fail in some browsers / edge cases — not fatal
      console.warn("Could not lock sender quality:", err);
    }
  }
}

