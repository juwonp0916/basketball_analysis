import { BACKEND_URL } from "$lib";

export const AVAILABLE_COLORS = [
  "red", "blue", "green", "yellow", "orange", "purple", "pink",
  "cyan", "white", "black", "gray", "brown", "navy", "maroon",
  "lime", "teal", "gold",
] as const;

export type TeamColor = (typeof AVAILABLE_COLORS)[number];

export async function submitTeamColors(
  team0Color: string,
  team1Color: string
): Promise<{ success: boolean; error?: string }> {
  const res = await fetch(`${BACKEND_URL}/team-colors`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ team0_color: team0Color, team1_color: team1Color }),
  });
  return res.json();
}
