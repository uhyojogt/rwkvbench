import { StrictMode, useEffect, useState, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { Activity, BarChart3, Database, Gauge, HardDrive, RefreshCw } from "lucide-react";
import "./styles.css";

type BenchmarkRun = {
  run_id: string;
  created_at: string;
  backend: string;
  model: string;
  hardware: {
    os: string;
    python: string;
    cpu: string;
    gpu: string | null;
    driver: string | null;
    cuda: string | null;
  };
  config: {
    batch_size: number;
    prompt_len: number;
    gen_len: number;
    dtype: string;
  };
  metrics: {
    prefill_tps: number;
    decode_tps: number;
    total_tps: number;
    latency_p50_ms: number;
    latency_p95_ms: number;
    vram_peak_mb: number | null;
  };
  metadata: Record<string, string | number>;
};

const formatNumber = (value: number) =>
  new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);

const formatDecimal = (value: number) =>
  new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 }).format(value);

function App() {
  const [runs, setRuns] = useBenchmarkRuns();

  const bestDecode = maxBy(runs, (run) => run.metrics.decode_tps);
  const bestPrefill = maxBy(runs, (run) => run.metrics.prefill_tps);
  const peakVram = maxBy(runs, (run) => run.metrics.vram_peak_mb ?? 0);
  const hardware = runs[0]?.hardware;

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">RWKVBench</div>
          <h1>Benchmark Dashboard</h1>
        </div>
        <button className="iconButton" type="button" onClick={() => setRuns([])} title="Reload sample data">
          <RefreshCw size={18} />
        </button>
      </header>

      <section className="summaryGrid" aria-label="Benchmark summary">
        <MetricCard
          icon={<Database size={18} />}
          label="Runs"
          value={String(runs.length)}
          detail="torch-cuda sweep"
        />
        <MetricCard
          icon={<Gauge size={18} />}
          label="Best Decode"
          value={`${formatNumber(bestDecode?.metrics.decode_tps ?? 0)} tok/s`}
          detail={bestDecode ? runLabel(bestDecode) : "No data"}
        />
        <MetricCard
          icon={<Activity size={18} />}
          label="Best Prefill"
          value={`${formatNumber(bestPrefill?.metrics.prefill_tps ?? 0)} tok/s`}
          detail={bestPrefill ? runLabel(bestPrefill) : "No data"}
        />
        <MetricCard
          icon={<HardDrive size={18} />}
          label="Peak VRAM"
          value={`${formatNumber(peakVram?.metrics.vram_peak_mb ?? 0)} MB`}
          detail={peakVram ? runLabel(peakVram) : "No data"}
        />
      </section>

      <section className="hardwareBand" aria-label="Hardware">
        <div>
          <span className="label">GPU</span>
          <strong>{hardware?.gpu ?? "Unknown"}</strong>
        </div>
        <div>
          <span className="label">Driver</span>
          <strong>{hardware?.driver ?? "Unknown"}</strong>
        </div>
        <div>
          <span className="label">CUDA</span>
          <strong>{hardware?.cuda ?? "Unknown"}</strong>
        </div>
        <div>
          <span className="label">Python</span>
          <strong>{hardware?.python ?? "Unknown"}</strong>
        </div>
      </section>

      <section className="chartSection">
        <div className="sectionHeader">
          <div>
            <div className="eyebrow">Throughput</div>
            <h2>Decode TPS By Batch And Prompt</h2>
          </div>
          <BarChart3 size={20} />
        </div>
        <BarChart runs={runs} metric="decode_tps" />
      </section>

      <section className="tableSection">
        <div className="sectionHeader">
          <div>
            <div className="eyebrow">Runs</div>
            <h2>Result Matrix</h2>
          </div>
        </div>
        <ResultTable runs={runs} />
      </section>
    </main>
  );
}

function useBenchmarkRuns(): [BenchmarkRun[], (runs: BenchmarkRun[]) => void] {
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);

  useEffect(() => {
    if (runs.length === 0) {
      fetch("/data/summary.json")
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Failed to load summary.json: ${response.status}`);
          }
          return response.json() as Promise<BenchmarkRun[]>;
        })
        .then(setRuns)
        .catch((error) => {
          console.error(error);
          setRuns([]);
        });
    }
  }, [runs.length]);

  return [runs, setRuns];
}

function MetricCard({
  icon,
  label,
  value,
  detail,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <article className="metricCard">
      <div className="metricIcon">{icon}</div>
      <div>
        <span className="label">{label}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
    </article>
  );
}

function BarChart({ runs, metric }: { runs: BenchmarkRun[]; metric: keyof BenchmarkRun["metrics"] }) {
  const max = Math.max(...runs.map((run) => Number(run.metrics[metric]) || 0), 1);

  return (
    <div className="barChart">
      {runs.map((run) => {
        const value = Number(run.metrics[metric]) || 0;
        const height = Math.max((value / max) * 100, 3);

        return (
          <div className="barGroup" key={run.run_id}>
            <div className="barTrack" title={`${runLabel(run)}: ${formatNumber(value)} tok/s`}>
              <div className="barFill" style={{ height: `${height}%` }} />
            </div>
            <span>{run.config.batch_size}x</span>
            <small>{run.config.prompt_len}</small>
          </div>
        );
      })}
    </div>
  );
}

function ResultTable({ runs }: { runs: BenchmarkRun[] }) {
  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>Backend</th>
            <th>Batch</th>
            <th>Prompt</th>
            <th>Decode TPS</th>
            <th>Prefill TPS</th>
            <th>P50 ms</th>
            <th>VRAM</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.run_id}>
              <td>{run.backend}</td>
              <td>{run.config.batch_size}</td>
              <td>{run.config.prompt_len}</td>
              <td>{formatNumber(run.metrics.decode_tps)}</td>
              <td>{formatNumber(run.metrics.prefill_tps)}</td>
              <td>{formatDecimal(run.metrics.latency_p50_ms)}</td>
              <td>{formatNumber(run.metrics.vram_peak_mb ?? 0)} MB</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function runLabel(run: BenchmarkRun) {
  return `bs${run.config.batch_size} p${run.config.prompt_len}`;
}

function maxBy<T>(items: T[], selector: (item: T) => number): T | undefined {
  return items.reduce<T | undefined>((best, item) => {
    if (!best || selector(item) > selector(best)) {
      return item;
    }
    return best;
  }, undefined);
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
