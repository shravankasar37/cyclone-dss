"use client";
import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { API, FEATURE_LABELS, get } from "@/lib/api";
import { assetUrl } from "@/lib/staticData";
import { usePalette } from "@/lib/theme";
import { Card, ErrorBox, Loading, Stat } from "@/components/ui";
import { AdvancedMetrics, BasinDemoTable } from "@/components/AdvancedMetrics";

/* eslint-disable @typescript-eslint/no-explicit-any */
type M = Record<string, any>;
const pct = (v: number, d = 0) => `${(v * 100).toFixed(d)}%`;
const n1 = (v: number) => v.toFixed(1);

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

export default function Validation() {
  const [m, setM] = useState<M | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const p = usePalette();
  useEffect(() => { get<M>("/api/metrics").then(setM).catch((e) => setErr(String(e.message ?? e))); }, []);
  if (err) return <ErrorBox msg={err} />;
  if (!m || !p) return <Loading h={600} />;

  const it = m.metrics, ri = m.ri_metrics, tr = m.track_metrics, gen = m.genesis_metrics;
  const A = it.experiments.A_nio_only.test, B = it.experiments.B_cross_basin.test;
  const leads = Object.keys(tr.model_test);
  const trackRows = leads.map((h) => ({ lead: h, model: Math.round(tr.model_test[h].track_err_km), persistence: Math.round(tr.persistence_test[h].track_err_km) }));
  const coverage = leads.map((h) => ({ lead: h, coverage: Math.round(tr.cone.test_coverage[h] * 100) }));
  const riSel = ri.variants[ri.selected];
  const shap = Object.entries(ri.shap_mean_abs_test as Record<string, number>).sort((a, b) => b[1] - a[1]).slice(0, 8)
    .map(([k, v]) => ({ f: FEATURE_LABELS[k] ?? k, v: Math.round(v * 1000) / 1000 }));
  const tip = { contentStyle: { background: p["--bg"], border: "none", borderRadius: 14, fontSize: 12, boxShadow: `6px 6px 14px ${p["--sd"]}` } };

  return (
    <div className="space-y-6">
      <section className="neu p-6">
        <h1 className="text-xl font-extrabold text-ink">How good is it? Every number below is on data the models never trained on.</h1>
        <p className="mt-2 max-w-3xl text-sm text-muted">
          Satellite model: held-out 2016 North Indian Ocean storms. Track, RI and genesis models: trained on seasons up to 2016, tuned on 2017–19,
          tested on 2020–25 (Amphan, Tauktae, Yaas, Mocha, Biparjoy, Michaung, Remal, Dana, Fengal…).
        </p>
        <div className="mt-5 grid grid-cols-2 gap-4 md:grid-cols-4">
          <Stat label="Satellite wind error" value={n1(B.mae_kt)} unit="kt MAE" sub={`RMSE ${n1(B.rmse_kt)} kt`} />
          <Stat label="Cross-basin transfer gain" value={`+${n1(it.cross_basin_test_mae_gain_pct)}%`} sub="vs NIO-only training" />
          <Stat label="RI storms caught (2020–25)" value={`${ri.test_ri_storms.filter((s: M) => s.flagged).length}/${ri.test_ri_storms.length}`} sub={`${ri.test_false_alarm_storms?.length ?? 0} false-alarm storms`} />
          <Stat label="Cone coverage (target 67%)" value={pct(tr.cone.test_coverage["24h"])} sub="at 24 h, test seasons" />
        </div>
      </section>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="1 · Satellite intensity (EfficientNet-B0 + Dvorak head)" subtitle={`Test: ${B.n} frames from ${B.storms} storms · train ${it.split_sizes.io_train} NIO + ${it.split_sizes.x_train} other-basin frames`}>
          <Table head={["Experiment", "MAE kt", "RMSE kt", "Bias", "IMD class acc", "Macro-F1", "T-number MAE"]}
            rows={[["A · NIO only", n1(A.mae_kt), n1(A.rmse_kt), n1(A.bias_kt), pct(A.class_acc_head), A.macro_f1_head.toFixed(2), A.t_number_mae.toFixed(2)],
              ["B · cross-basin → NIO", n1(B.mae_kt), n1(B.rmse_kt), n1(B.bias_kt), pct(B.class_acc_head), B.macro_f1_head.toFixed(2), B.t_number_mae.toFixed(2)]]} />
          <div className="neu-inset mt-4 overflow-hidden p-2">
            <img src={assetUrl("/api/metrics/intensity_eval.png", API)} alt="Predicted vs best-track wind and IMD class confusion matrix" className="w-full rounded-[14px] bg-white" />
          </div>
          <p className="mt-3 text-[11px] text-faint">
            External INSAT-3D test ({it.insat3d_external.n} images, different processing): {n1(it.insat3d_external.zero_shot_mae_kt)} kt raw,
            {" "}{n1(it.insat3d_external.linear_calibrated_5fold_mae_kt)} kt after linear calibration — a real domain shift we report rather than hide.
            No ESCS/SuCS frames exist in the 2016 test storms.
          </p>
        </Card>

        <Card title="2 · Rapid-intensification early warning (LightGBM)" subtitle={`Imbalance ${n1(ri.imbalance_neg_per_pos)}:1 → class-weighted + Platt-calibrated; threshold chosen on 2017–19`}>
          <Table head={["Features", "PR-AUC", "ROC-AUC", "Brier skill", "Precision", "Recall", "Hits / FA / Miss"]}
            rows={Object.entries(ri.variants as M).map(([k, v]) => [k === "track" ? "Track history" : "Track + ERA5 env", v.test.pr_auc.toFixed(2), v.test.roc_auc.toFixed(2),
              v.test.brier_skill_vs_climatology.toFixed(2), pct(v.test.precision), pct(v.test.recall), `${v.test.hits} / ${v.test.false_alarms} / ${v.test.misses}`])} />
          <p className="mt-2 text-[11px] text-faint">Base rate in test seasons: {pct(riSel.test.ri_cases / riSel.test.n, 1)} of 6-hourly fixes. Monthly-mean ERA5 did not improve ranking; daily environment fields are the next step.</p>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div>
              <div className="label mb-2">Test storms with RI</div>
              <Table head={["Storm", "Max P", "Flagged"]} rows={ri.test_ri_storms.map((s: M) => [`${s.name.split(":")[0]} ${s.season}`, pct(s.max_p), s.flagged ? "✓" : "✗"])} />
            </div>
            <div>
              <div className="label mb-2">Top drivers (mean |SHAP|)</div>
              <div className="neu-inset p-2">
                <ResponsiveContainer width="100%" height={230}>
                  <BarChart data={shap} layout="vertical" margin={{ left: 10, right: 10 }}>
                    <XAxis type="number" hide />
                    <YAxis type="category" dataKey="f" width={110} stroke={p["--faint"]} fontSize={10} tickLine={false} />
                    <Tooltip {...tip} />
                    <Bar dataKey="v" fill={p["--orange"]} radius={[0, 6, 6, 0]} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </Card>

        <Card title="3 · Track & intensity forecast (LSTM + MC dropout)" subtitle={`Test: ${tr.samples.test} forecasts · compared with persistence (extrapolating recent motion)`}>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="neu-inset p-2">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={trackRows} margin={{ top: 10, right: 10, left: -10 }}>
                  <CartesianGrid stroke={p["--sd"]} strokeOpacity={0.5} vertical={false} />
                  <XAxis dataKey="lead" stroke={p["--faint"]} fontSize={11} tickLine={false} />
                  <YAxis stroke={p["--faint"]} fontSize={11} tickLine={false} unit=" km" />
                  <Tooltip {...tip} /><Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="model" name="AI model" fill={p["--accent"]} radius={[6, 6, 0, 0]} isAnimationActive={false} />
                  <Bar dataKey="persistence" name="persistence" fill={p["--faint"]} radius={[6, 6, 0, 0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="neu-inset p-2">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={coverage} margin={{ top: 10, right: 10, left: -10 }}>
                  <CartesianGrid stroke={p["--sd"]} strokeOpacity={0.5} vertical={false} />
                  <XAxis dataKey="lead" stroke={p["--faint"]} fontSize={11} tickLine={false} />
                  <YAxis domain={[0, 100]} stroke={p["--faint"]} fontSize={11} tickLine={false} unit="%" />
                  <Tooltip {...tip} />
                  <ReferenceLine y={67} stroke={p["--red"]} strokeDasharray="5 4" label={{ value: "target 67%", fill: p["--red"], fontSize: 10, position: "insideTopRight" }} />
                  <Bar dataKey="coverage" name="truth inside cone" fill={p["--green"]} radius={[6, 6, 0, 0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <Table head={["Lead", "Track err", "Skill vs persistence", "Wind MAE", "Cone radius (median)"]}
            rows={leads.map((h) => [h, `${Math.round(tr.model_test[h].track_err_km)} km`, `${tr.skill_vs_persistence_pct[h].track.toFixed(0)}%`,
              `${n1(tr.model_test[h].wind_mae_kt)} kt`, `${Math.round(tr.cone.median_radius_km_test[h])} km`])} />
          <p className="mt-3 text-[11px] text-faint">
            Honest context: IMD&apos;s operational track errors in recent years are roughly 70–80 km at 24 h and ~150–180 km at 72 h — about half ours.
            Our model uses best-track history only (no NWP steering flow); it is decision support, not a replacement for official forecasts.
          </p>
        </Card>

        <Card title="4 · Cyclogenesis (depression → Cyclonic Storm within 72 h)" subtitle="LightGBM on track + ERA5 environment · test seasons 2020–25">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="ROC-AUC (all fixes)" value={gen.test.roc_auc.toFixed(2)} />
            <Stat label="Brier skill" value={gen.test.brier_skill_vs_climatology.toFixed(2)} sub="vs climatology" />
            <Stat label="Accuracy @ 50%" value={pct(gen.test.accuracy_at_0_5 ?? gen.test["accuracy_at_0.5"])} />
            <Stat label="From first fix" value={gen.test.first_fix_roc_auc?.toFixed(2) ?? "—"} sub={`AUC, ${gen.test.first_fix_systems} systems`} />
          </div>
          <p className="mt-3 text-[11px] text-faint">
            The first-fix score is the honest &quot;earliest warning&quot; number: from a system&apos;s very first 6-hourly position, before any trend is visible.
          </p>
                </Card>
      </div>
      <AdvancedMetrics m={m} />
      <BasinDemoTable />
    </div>
  );
}
