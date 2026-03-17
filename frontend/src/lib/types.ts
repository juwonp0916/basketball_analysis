export interface Point {
  x: number;
  y: number;
}

export interface TotalShots {
  made: number;
  total: number;
}

export interface Percentages {
  fieldGoal: number;
  twoPoint: number;
  threePoint: number;
}

export interface GameStats {
  totalShots: TotalShots;
  percentages: Percentages;
}

export interface Shot {
  id: number;
  timestamp_ms: number;
  type: "2pt" | "3pt";
  result: "made" | "missed";
  location: string;
  coord: Point;
  team?: string;
}


export type FrameSyncPayload = {
  type: "frame_sync";
  sequence_id: number;
  video_timestamp_ms: number;
  current_stats: GameStats;
};

export type ShotDetectedPayload = {
  type: "shot_event";
  video_timestamp_ms: number;
  event_data: Shot;
  updated_stats: GameStats;
};

export type WebRTCMessage = FrameSyncPayload | ShotDetectedPayload;
