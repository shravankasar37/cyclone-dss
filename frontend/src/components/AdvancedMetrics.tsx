"use client";
/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useState } from "react";
import { get, type BasinStorm } from "@/lib/api";
import { Card, Stat } from "./ui";

type M = Record<string, any>;
const n1 = (v: number | null | undefined) => (v == null ? "—" : v.toFixed(1));
const pct = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`);

function Table({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <div className="neu-inset overflow-x-auto p-2">
      <table className="w-full text-xs tabular">
        <thead><tr className="text-left text-faint">{head.map((h) => <th key={h} className="p-2 font-semibold">{h}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i} className="border-t border-[color:var(--sd)]/40">{r.map((c, j) => <td key={j} className={`p-2 ${j === 0 ? "font-semibold text-ink" : "text-muted"}`}>{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

/** Sections for the models from the advanced Colab notebook; renders nothing until their metrics exist. */
export function AdvancedMetrics({ m }: { m: M }) {
  const mc = m.metrics_multichannel, seq = m.metrics_convlstm, det = m.metrics_detector;
  if (!mc && !seq && !det) return null;
  return (
    <div className="grid gap-6 xl:grid-cols-2">
      {det && (
        <Card title="5 · Full-basin detector (GridSat IR + WV)" subtitle={`Test seasons 2017–2024 · ${det.test_scenes} unseen basin scenes · match radius ${det.match_radius_km} km`}>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Precision" value={pct(det.test.precision)} sub={`${det.test.fp} false detections`} />
            <Stat label="Recall" value={pct(det.test.recall)} sub={`${det.test.fn} systems missed`} />
            <Stat label="F1" value={det.test.f1.toFixed(2)} sub={`threshold ${det.threshold} (set on 2016)`} />
            <Stat label="Centre error" value={det.test.median_loc_err_km != null ? Math.round(det.test.median_loc_err_km) : "—"} unit="km" sub="median" />
          </div>
          <p className="mt-3 text-[11px] text-faint">Counts every depression and cyclone in each scene (IMD ≥17 kt). Cyclone-vs-depression label correct for {pct(det.test.class_acc_matched)} of matched detections.</p>
        </Card>
      )}
      {mc && (
        <Card title="6 · Multi-channel satellite fusion" subtitle="One model, evaluated with different inputs on held-out 2016 NIO storms">
          <Table head={["Inputs", "MAE kt", "RMSE kt", "IMD class acc", "Macro-F1"]}
            rows={["ir", "ir+wv", "ir+wv+pmw"].filter((k) => mc[k]).map((k) => [k.toUpperCase().replaceAll("+", " + "), n1(mc[k].test.mae_kt), n1(mc[k].test.rmse_kt), pct(mc[k].test.class_acc), mc[k].test.macro_f1.toFixed(2)])} />
          <p className="mt-3 text-[11px] text-faint">
            Finding: the extra channels change the error by only {n1(Math.abs(mc["ir+wv+pmw"].test.mae_kt - mc.ir.test.mae_kt))} kt — within noise. Infrared carries almost all of the
            intensity signal, so the model runs in IR-only mode when other channels are missing without losing skill.
            {mc.ir?.test.pmw_available >= 0.99 && " (TCIR supplies a microwave field for every frame, likely gap-filled, so real-time microwave coverage would be lower.)"}
          </p>
        </Card>
      )}
      {seq && (
        <Card title="7 · ConvLSTM image-sequence outlook" subtitle="Last 8 IR+WV frames (21 h) → 24 h wind change and RI · same network with vs without images">
          <Table head={["Model", "Test set", "24 h change MAE", "Persistence", "RI PR-AUC", "RI base rate"]}
            rows={["convlstm_images", "scalars_only"].filter((k) => seq[k]).flatMap((k) => ["nio_test", "val_all"].map((s) => [
              k === "convlstm_images" ? "ConvLSTM + images" : "No images", s === "nio_test" ? "NIO 2016" : "All basins (val)",
              `${n1(seq[k][s].dv24_mae_kt)} kt`, `${n1(seq[k][s].persistence_dv24_mae_kt)} kt`, seq[k][s].ri_pr_auc?.toFixed(2) ?? "—", pct(seq[k][s].ri_base_rate)]))} />
          <p className="mt-3 text-[11px] text-faint">
            Finding: images slightly improve the 24 h wind-change forecast ({n1(seq.convlstm_images.val_all.dv24_mae_kt)} vs {n1(seq.scalars_only.val_all.dv24_mae_kt)} kt), but
            {seq.convlstm_images.val_all.ri_pr_auc < seq.scalars_only.val_all.ri_pr_auc
              ? ` do not improve rapid-intensification detection (PR-AUC ${seq.convlstm_images.val_all.ri_pr_auc.toFixed(2)} vs ${seq.scalars_only.val_all.ri_pr_auc.toFixed(2)}). The track-based RI model stays the primary warning; the image outlook is shown as experimental.`
              : " also improve rapid-intensification detection."}
            {" "}The 2016 NIO test storms contain no RI events, so RI cannot be scored there.
          </p>
        </Card>
      )}
    </div>
  );
}

/** Section 8: the full image-only pipeline on the six demo storms (fetched from the basin endpoints). */
export function BasinDemoTable() {
  const [rows, setRows] = useState<BasinStorm["summary"][] | null>(null);
  useEffect(() => {
    get<{ available: boolean; storms: { storm: string }[] }>("/api/basin/storms").then(async (s) => {
      if (!s.available) return setRows([]);
      const out = await Promise.all(s.storms.map((x) => get<BasinStorm>(`/api/basin/storms/${x.storm}`).then((d) => d.summary)));
      setRows(out);
    }).catch(() => setRows([]));
  }, []);
  if (!rows?.length) return null;
  const h = (v: number | null | undefined) => (v == null ? "—" : v > 0 ? `${Math.round(v)} h before` : v === 0 ? "same time" : `${Math.round(-v)} h after`);
  const title = (s: string) => s[0] + s.slice(1).toLowerCase();
  const leads = rows.map((r) => r.hours_before_cyclonic_storm).filter((v): v is number => v != null);
  return (
    <Card title="8 · Image-only pipeline on six demo storms" subtitle="Raw GridSat scenes → detect → track → intensity; none of these storms were used to train the detector (2016 = validation year)">
      <Table head={["Storm", "First detected vs depression", "vs Cyclonic Storm stage", "Fixes detected", "Centre error", "Intensity error", "Confirmed false tracks"]}
        rows={rows.map((r) => [`${title(r.storm)} ${r.start.slice(0, 4)}${r.wv_missing ? " *" : ""}`, h(r.hours_before_depression), h(r.hours_before_cyclonic_storm),
          pct(r.detection_rate), r.median_position_error_km != null ? `${Math.round(r.median_position_error_km)} km` : "—",
          r.intensity_mae_kt != null ? `${Math.round(r.intensity_mae_kt)} kt` : "—", r.confirmed_false ?? "—"])} />
      <p className="mt-3 text-[11px] text-faint">
        Every storm was identified from imagery alone {leads.length ? `${Math.round(Math.min(...leads))}–${Math.round(Math.max(...leads))} h` : ""} before it reached Cyclonic Storm strength.
        Confirmed = a track persisting 15 h. * GridSat has no water-vapour data for 2024; the detector used an IR-derived estimate.
      </p>
    </Card>
  );
}
