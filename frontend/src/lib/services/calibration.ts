import { BACKEND_URL } from "$lib";

export const CALIBRATION_LABLES = [
  "Baseline Left Paint Corner",
  "Baseline Right Paint Corner",
  "FT Line Left Corner",
  "FT Line Right Corner",
] as const;

export const CALIBRATION_COLORS = [
  "#eab308",
  "#22c55e",
  "#d946ef",
  "#3b82f6",
] as const;

export const MAX_CALIBRATION_POINTS = 4;

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

export type CalibrationResult = {
  success: boolean;
  error?: string;
  point_errors?: number[];
  avg_error?: number;
  court_outline_pixels?: number[][][];
};

export async function submitCalibration(
  points: CalibrationPoint[],
  imageWidth: number,
  imageHeight: number,
  mode: string = "4-point"
): Promise<CalibrationResult> {
  const res = await fetch(`${BACKEND_URL}/calibration`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      points: points.map((p) => [p.x, p.y]),
      image_width: imageWidth,
      image_height: imageHeight,
      mode,
    }),
  });
  return res.json();
}
