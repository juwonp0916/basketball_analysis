# Basketball analytics dashboard

SvelteKit and TypeScript frontend for the basketball shot-analysis system. It handles camera capture, court calibration, WebRTC communication, and live shot/team visualization.

[Project overview](../README.md) · [Full setup guide](../INSTRUCTION.md) · [System architecture](../docs/ARCHITECTURE.md)

## Development

With the backend running on port 8000:

```bash
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

Open [the dashboard at `/live/v4`](http://127.0.0.1:5173/live/v4). The root route is a starter page.

## Entry points

- [Main dashboard](src/routes/live/v4/+page.svelte): camera/simulation controls, calibration, team views, statistics, and event log.
- [ShotChart](src/lib/components/ShotChart.svelte): half-court rendering, shot markers, and outcome/type filters.
- [CalibrationOverlay](src/lib/components/CalibrationOverlay.svelte) and [calibration service](src/lib/services/calibration.ts): reference-point selection and API submission.
- [WebRTC utilities](src/lib/utils/webrtc-utils.ts): signaling, video transport, analytics messages, and detection startup.
- [Backend URL](src/lib/index.ts): local API address.

Run `npm run check` for Svelte/TypeScript checks and `npm run build` for a production build.
