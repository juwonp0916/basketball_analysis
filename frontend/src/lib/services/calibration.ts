import { BACKEND_URL } from "$lib";

export const CALIBRATION_LABLES = [
  "Baseline Left Sideline",
  "Baseline Left Penalty Box",
  "Baseline Right Penalty Box",
  "Baseline Right Sideline",
  "Free Throw Line Left",
  "Free Throw Line Right",
] as const;

export const CALIBRATION_COLORS = [
  "#eab308",
  "#22c55e",
  "#22c55e",
  "#eab308",
  "#d946ef",
  "#d946ef",
] as const;

export const MAX_CALIBRATION_POINTS = 6;

export type CalibrationPoint = { x: number; y: number };

export function videoClickToCalibrationPoint(
  event: MouseEvent,
  videoEl: HTMLVideoElement
): CalibrationPoint {
  const rect = videoEl.getBoundingClientRect();
  const scaleX = videoEl.videoWidth / videoEl.clientWidth;
  const scaleY = videoEl.videoHeight / videoEl.clientHeight;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

export function toDisplayCoords(
  point: CalibrationPoint,
  videoEl: HTMLVideoElement
): { x: number; y: number } {
  return {
    x: point.x / (videoEl.videoWidth / videoEl.clientWidth),
    y: point.y / (videoEl.videoHeight / videoEl.clientHeight),
  };
}

export async function submitCalibration(
  points: CalibrationPoint[],
  imageWidth: number,
  imageHeight: number
): Promise<{ success: boolean; error?: string }> {
  const res = await fetch(`${BACKEND_URL}/calibration`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      points: points.map((p) => [p.x, p.y]),
      image_width: imageWidth,
      image_height: imageHeight,
    }),
  });
  return res.json();
}
