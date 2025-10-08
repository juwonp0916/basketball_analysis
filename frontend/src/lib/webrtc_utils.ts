export interface WebRTCAnswer {
  sdp: string;
  type: RTCSdpType;
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

