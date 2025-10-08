import asyncio
import json
import logging
import av
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from aiortc import MediaStreamError, MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaRecorder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs = set()


class Offer(BaseModel):
    sdp: str
    type: str


async def save_frames_to_file(track: MediaStreamTrack, file_path: str):
    logging.info(f"Starting to save track {track.kind} to {file_path}")
    container = None
    try:
        first_frame = await track.recv()

        width = first_frame.width
        height = first_frame.height

        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1

        container = av.open(file_path, mode="w")
        stream = container.add_stream('libx264', rate=30)
        stream.width = first_frame.width
        stream.height = first_frame.height
        stream.pix_fmt = 'yuv420p'

        resized_frame = first_frame.to_ndarray(format="bgr24")
        resized_frame = resized_frame[:height, :width, :]  # Crop to the new dimensions
        frame_to_encode = av.VideoFrame.from_ndarray(resized_frame, format="bgr24")

        for packet in stream.encode(frame_to_encode):
            container.mux(packet)

        while True:
            try:
                frame = await track.recv()

                cropped_ndarray = frame.to_ndarray(format="bgr24")[:height, :width, :]
                new_frame = av.VideoFrame.from_ndarray(cropped_ndarray, format="bgr24")

                for packet in stream.encode(new_frame):
                    container.mux(packet)
            except MediaStreamError:
                logging.info("Track ended, stopping frame saving.")
                break

    except Exception as e:
        logging.error(f"Error saving frames to {file_path}: {e}")
    finally:
        if container:
            container.close()
            logging.info(f"Finished saving track {track.kind} to {file_path}")


@app.post("/offer")
async def offer(params: Offer):
    offer_desc = RTCSessionDescription(sdp=params.sdp, type=params.type)
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            logger.info(f"Closing peer connection due to state: {pc.connectionState}")
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track {track.kind} received")

        assert track.kind == "video", "Received track is not a video"

        # TODO 1: Replace with frame processing logic
        output_file = f"output/received_video_{track.id}.mp4"
        asyncio.create_task(save_frames_to_file(track, output_file))

        @track.on("ended")
        async def on_ended():
            logging.info(f"Track {track.kind} ended")

    # Channel for sending the analytics back to the client
    analytics_channel = pc.createDataChannel("analytics")

    @analytics_channel.on("open")
    async def on_open():
        logger.info("Data channel 'analytics' opened")

        # Send analytics data every 2 seconds
        while analytics_channel.readyState == "open":
            response_data = {"timestamp": asyncio.get_event_loop().time(), "message": "Processing analytics..."}
            analytics_channel.send(json.dumps(response_data))
            await asyncio.sleep(2)

    # Set the remote description from the client's offer
    await pc.setRemoteDescription(offer_desc)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
