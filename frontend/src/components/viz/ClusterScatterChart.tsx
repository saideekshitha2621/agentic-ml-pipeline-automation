import { useMemo, useRef, useState } from "react";
import { Box, ButtonBase, Typography } from "@mui/material";
import type { VisualizationData } from "../../types";
import ChartCard from "./ChartCard";
import { prettyAlgorithm, viz } from "./vizTheme";

const W = 720, H = 420, M = { top: 16, right: 20, bottom: 48, left: 52 };
const SHAPES = ["circle", "square", "triangle", "diamond"] as const;

function Mark({ shape, x, y, r, fill, opacity = 1, stroke = viz.surface }: {
  shape: string; x: number; y: number; r: number; fill: string; opacity?: number; stroke?: string;
}) {
  const common = { fill, fillOpacity: opacity, stroke, strokeWidth: 1 };
  if (shape === "square") return <rect x={x - r} y={y - r} width={2 * r} height={2 * r} {...common} />;
  if (shape === "triangle") return <polygon points={`${x},${y - r * 1.2} ${x + r * 1.1},${y + r * 0.9} ${x - r * 1.1},${y + r * 0.9}`} {...common} />;
  if (shape === "diamond") return <polygon points={`${x},${y - r * 1.25} ${x + r * 1.25},${y} ${x},${y + r * 1.25} ${x - r * 1.25},${y}`} {...common} />;
  return <circle cx={x} cy={y} r={r} {...common} />;
}

function niceTicks(min: number, max: number, count = 5): number[] {
  const step = (max - min) / count;
  return Array.from({ length: count + 1 }, (_, i) => min + step * i);
}

export default function ClusterScatterChart({ data }: { data: NonNullable<VisualizationData["cluster_plot"]> }) {
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  const [hover, setHover] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Encoding follows the cluster, not its size rank, so toggling one never repaints the rest.
  // Color is paired with a marker shape so identity never rests on color alone.
  const style = useMemo(() => {
    const map = new Map<number, { color: string; shape: string }>();
    data.clusters.filter((c) => !c.is_noise).forEach((c, i) =>
      map.set(c.label, { color: viz.categorical[i % viz.categorical.length], shape: SHAPES[Math.floor(i / 2) % SHAPES.length] }),
    );
    map.set(-1, { color: viz.noise, shape: "circle" });
    return map;
  }, [data.clusters]);
  const styleOf = (label: number) => style.get(label) ?? { color: viz.noise, shape: "circle" };

  const { xs, ys } = useMemo(() => {
    const xv = data.points.map((p) => p[0]), yv = data.points.map((p) => p[1]);
    const pad = (a: number, b: number) => { const d = (b - a) * 0.06 || 1; return [a - d, b + d] as const; };
    return { xs: pad(Math.min(...xv), Math.max(...xv)), ys: pad(Math.min(...yv), Math.max(...yv)) };
  }, [data.points]);

  const sx = (v: number) => M.left + ((v - xs[0]) / (xs[1] - xs[0])) * (W - M.left - M.right);
  const sy = (v: number) => H - M.bottom - ((v - ys[0]) / (ys[1] - ys[0])) * (H - M.top - M.bottom);

  const visible = data.points.map((p, i) => ({ p, i })).filter(({ p }) => !hidden.has(p[2]));

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = svgRef.current!.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * W, my = ((e.clientY - rect.top) / rect.height) * H;
    let best = -1, bestD = 14 * 14;
    for (const { p, i } of visible) {
      const d = (sx(p[0]) - mx) ** 2 + (sy(p[1]) - my) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
    setHover(best >= 0 ? best : null);
  };

  const hp = hover !== null ? data.points[hover] : null;
  const varText = (v: number) => `${(v * 100).toFixed(1)}%`;
  const label = (l: number) => (l === -1 ? "Noise" : `Cluster ${l + 1}`);
  const shown = data.points.length < data.n_points_total
    ? `${data.points.length.toLocaleString()} of ${data.n_points_total.toLocaleString()} points shown`
    : `${data.n_points_total.toLocaleString()} points`;

  return (
    <ChartCard
      title="Cluster visualization"
      subtitle={`${prettyAlgorithm(data.algorithm)} · 2-D PCA projection · ${shown} · explains ${varText(data.explained_variance[0] + data.explained_variance[1])} of variance`}
      span={2}
    >
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mb: 2 }}>
        {data.clusters.map((c) => {
          const s = styleOf(c.label), off = hidden.has(c.label);
          return (
            <ButtonBase
              key={c.label} aria-pressed={!off}
              onClick={() => setHidden((prev) => { const n = new Set(prev); if (n.has(c.label)) n.delete(c.label); else n.add(c.label); return n; })}
              sx={{ display: "flex", alignItems: "center", gap: 1, px: 1.25, py: 0.5, borderRadius: 5, border: `1px solid ${viz.border}`, opacity: off ? 0.45 : 1, "&:hover": { bgcolor: viz.neutralTrack } }}
            >
              <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden><Mark shape={s.shape} x={7} y={7} r={5} fill={s.color} stroke="none" /></svg>
              <Typography sx={{ fontSize: "0.8125rem", color: viz.textPrimary }}>
                {label(c.label)} <Box component="span" sx={{ color: viz.textMuted }}>· {c.size.toLocaleString()} ({c.pct}%)</Box>
              </Typography>
            </ButtonBase>
          );
        })}
      </Box>

      <Box sx={{ position: "relative", width: "100%" }}>
        <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`PCA scatter plot of ${data.clusters.length} clusters`}
          style={{ width: "100%", height: "auto", display: "block" }} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
          {niceTicks(ys[0], ys[1]).map((t) => (
            <g key={`y${t}`}>
              <line x1={M.left} x2={W - M.right} y1={sy(t)} y2={sy(t)} stroke={viz.grid} />
              <text x={M.left - 8} y={sy(t) + 4} textAnchor="end" fontSize="12" fill={viz.textSecondary}>{t.toFixed(1)}</text>
            </g>
          ))}
          {niceTicks(xs[0], xs[1]).map((t) => (
            <text key={`x${t}`} x={sx(t)} y={H - M.bottom + 18} textAnchor="middle" fontSize="12" fill={viz.textSecondary}>{t.toFixed(1)}</text>
          ))}
          <text x={(M.left + W - M.right) / 2} y={H - 8} textAnchor="middle" fontSize="12.5" fill={viz.textSecondary}>PC1 ({varText(data.explained_variance[0])} of variance)</text>
          <text transform={`translate(14 ${(M.top + H - M.bottom) / 2}) rotate(-90)`} textAnchor="middle" fontSize="12.5" fill={viz.textSecondary}>PC2 ({varText(data.explained_variance[1])})</text>
          {visible.map(({ p, i }) => {
            const s = styleOf(p[2]);
            return <Mark key={i} shape={s.shape} x={sx(p[0])} y={sy(p[1])} r={3.5} fill={s.color} opacity={hover !== null && hover !== i ? 0.55 : 0.85} />;
          })}
          {hp && <Mark shape={styleOf(hp[2]).shape} x={sx(hp[0])} y={sy(hp[1])} r={6} fill={styleOf(hp[2]).color} stroke={viz.textPrimary} />}
        </svg>
        {hp && (
          <Box sx={{
            position: "absolute", pointerEvents: "none", left: `${(sx(hp[0]) / W) * 100}%`, top: `${(sy(hp[1]) / H) * 100}%`,
            transform: "translate(-50%, calc(-100% - 12px))", bgcolor: viz.textPrimary, color: "#fff", px: 1.25, py: 0.75,
            borderRadius: 1.5, whiteSpace: "nowrap", fontSize: "0.8125rem", zIndex: 1,
          }}>
            <b>{label(hp[2])}</b> · PC1 {hp[0].toFixed(2)}, PC2 {hp[1].toFixed(2)}
          </Box>
        )}
      </Box>
    </ChartCard>
  );
}
