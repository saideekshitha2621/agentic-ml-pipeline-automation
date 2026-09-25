/** Shared chart tokens. Blue/orange/etc. follow the validated categorical order; the blue
 * ramp is the sequential scale. Text always wears text tokens, never a series color. */
export const viz = {
  surface: "#ffffff",
  border: "#e6e5e1",
  grid: "#ecebe7",
  textPrimary: "#1c1c1b",
  textSecondary: "#52514e",
  textMuted: "#6b6a66",
  highlight: "#256abf", // recommended / focus
  muted: "#86b6ef", // every non-highlighted bar (sequential blue step 250)
  neutralTrack: "#f1f0ed",
  noise: "#8a8985",
  categorical: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
  // sequential blue, light -> dark
  sequential: ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
} as const;

export const axisFont = { fontSize: "0.8125rem", color: viz.textSecondary } as const;

export function fmt(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Math.abs(value) >= 1000 ? value.toLocaleString(undefined, { maximumFractionDigits: 0 }) : value.toFixed(digits);
}

export function prettyAlgorithm(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
