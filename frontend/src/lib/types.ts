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
}
