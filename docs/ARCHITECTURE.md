# System architecture

[Project overview](../README.md) · [Setup and usage](../INSTRUCTION.md)

The application connects a browser dashboard to a local Python analysis service. REST requests handle signaling, calibration, and detection control; WebRTC carries the video stream and analytics messages.

## Frame-to-dashboard flow

| Stage | Responsibility | Source |
| --- | --- | --- |
| Capture | Obtain a camera stream or capture a simulation video's playback. | [Dashboard](../frontend/src/routes/live/v4/+page.svelte) |
| Connect | Negotiate WebRTC through `/offer`, create the `analytics` data channel, and request detection after calibration. | [WebRTC client](../frontend/src/lib/utils/webrtc-utils.ts) |
| Ingest | Decode incoming video frames and make them available through `SharedFrameBuffer`. | [Connection manager](../backend/src/utils/webrtc_manager.py) |
| Process | Consume frames asynchronously, run detector work in a thread, and serialize events and aggregate statistics. | [Shot processing pipeline](../backend/score_detection/shot_processing_pipeline.py) |
| Analyze | Identify shot attempts, outcomes, shooter positions, and team labels. | [Streaming detector](../backend/score_detection/streaming_shot_detector.py) |
| Present | Update shot charts, team views, statistics, and the event log from incoming JSON messages. | [Dashboard](../frontend/src/routes/live/v4/+page.svelte), [shot chart](../frontend/src/lib/components/ShotChart.svelte) |

## Shot detection and outcome classification

The configured basketball model exposes five classes: `ball`, `made`, `person`, `rim`, and `shoot`. The streaming detector maintains ball, hoop, and shooting-pose histories rather than treating each frame as an independent event.

Shooting-pose detections are checked using a temporal voting window. Shooter estimation uses shooting boxes and person detections, with fallbacks based on ball proximity and recent frames. Spatial and temporal checks suppress repeated detections of the same attempt.

The current streaming outcome logic uses ball/rim geometry and trajectory checks in [utils.py](../backend/score_detection/utils.py). The report describes an additional learned `made`-class cue; although that class is listed in the model configuration, the current streaming decision path does not explicitly consume it.

## Court calibration and localization

The dashboard collects four paint-corner correspondences in video coordinates. The backend maps them to FIBA reference points using an OpenCV homography. A projected court outline and per-point errors help the operator inspect and adjust alignment.

[ShotLocalizer](../backend/score_detection/localization.py) maps estimated shooter ground-contact positions to court coordinates. [TeamStatistics](../backend/score_detection/statistics.py) supplies zone and two-/three-point classification. The calibration modules also contain six-point variants; the current dashboard uses four-point calibration.

Calibration assumes a fixed camera and a planar court. The current streaming constructor mismatch is documented in the [setup guide](../INSTRUCTION.md#current-integration-caveats).

## Automatic team differentiation

[StreamingTeamDetector](../backend/score_detection/team_detector.py) extracts a torso region from each player bounding box and filters likely skin pixels. It estimates the dominant jersey color using per-crop K-Means in CIELAB space.

Features accumulate across frames before two-team clustering. Outlier filtering and a minimum color-separation check help reject poor initializations. Shooter classification compares color features with the learned team centroids and returns a team ID with a confidence score. Optional temporal voting is available when track IDs are supplied; the live inference path does not establish persistent player identities.

The pipeline retries team initialization and periodically rechecks the learned colors. This implementation has evolved from the report's earlier HSV-based, three-cluster description.

## Analytics messages and interface

The shared payload definitions live in [backend/schema.py](../backend/schema.py) and [frontend/src/lib/types.ts](../frontend/src/lib/types.ts).

| Message | Contents |
| --- | --- |
| `frame_sync` | Frame sequence, video timestamp, and aggregate statistics |
| `shot_event` | Shot ID, timestamp, 2PT/3PT type, made/missed outcome, chart coordinates, location label, optional team ID, team confidence, and updated statistics |
| `team_colors` | The two automatically estimated team colors |

The dashboard derives per-team shooting statistics from received shot events. It provides combined and team-specific charts, a comparison view, outcome/type filters, and event selection that highlights a corresponding chart marker.

## Implementation boundaries

- One stationary-camera analysis session with a primarily half-court view.
- Team-level shooting analytics; persistent player identity and individual player statistics are outside the implemented scope.
- In-memory session state and a bounded dashboard history.
- Local model inference, with the checked-in configuration selecting Apple MPS.

The report motivates CPU-accessible processing and frame skipping, but the current checkout uses different device and buffering settings. No throughput or latency guarantee is implied by the streaming architecture.
