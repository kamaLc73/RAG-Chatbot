import {
  Activity,
  AlertTriangle,
  BarChart3,
  CalendarDays,
  CheckCircle2,
  Database,
  Eye,
  Edit3,
  FileText,
  Layers,
  LineChart,
  Play,
  RefreshCw,
  Search,
  Target,
  Trash2,
  Video,
  X,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { useEffect, useMemo, useState, type FormEvent } from 'react';
import {
  deleteEvaluationRun,
  getEvaluationDataset,
  listEvaluationRuns,
  prepareEvaluationRun,
  renameEvaluationRun,
  syncEvaluationRuns,
  type EvaluationDatasetResponse,
  type EvaluationMetricOption,
  type EvaluationRun,
  type EvaluationRunsResponse,
  type EvaluationTask,
} from '../../api/admin';
import { cn } from '../../utils/helpers';
import Button from '../ui/Button';

type TaskFilter = 'all' | EvaluationTask;

const TASK_LABELS: Record<TaskFilter, string> = {
  all: 'Tous',
  docs: 'Documents',
  forms: 'Formulaires',
  videos: 'Vidéos',
  intentions: 'Intentions',
};

const TASK_ICONS: Record<EvaluationTask, LucideIcon> = {
  docs: FileText,
  forms: Database,
  videos: Video,
  intentions: Target,
};

const TASK_COLORS: Record<string, string> = {
  docs: '#4d7d13',
  forms: '#2563eb',
  videos: '#e11d48',
  intentions: '#7c3aed',
  vespa: '#4d7d13',
  chroma: '#2563eb',
  pgvector: '#7c3aed',
  unknown: '#64748b',
};

const DEFAULT_METRIC_OPTIONS: EvaluationMetricOption[] = [
  { key: 'primary_score', label: 'Score principal' },
  { key: 'latency_mean', label: 'Latence moyenne' },
  { key: 'errors', label: 'Erreurs' },
];

const DOC_RADAR_METRICS = [
  { key: 'answer_correctness', label: 'Exactitude' },
  { key: 'faithfulness', label: 'Fidélité' },
  { key: 'answer_relevancy', label: 'Pertinence' },
  { key: 'context_precision', label: 'Précision' },
  { key: 'context_recall', label: 'Rappel' },
];

function numberValue(value?: number | null) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString('fr-FR') : '-';
}

function percentValue(value?: number | null) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '-';
  return `${Math.round(value * 100)} %`;
}

function secondsValue(value?: number | null) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '-';
  return `${value.toFixed(2)} s`;
}

function metricValue(run: EvaluationRun, metricKey: string) {
  if (metricKey === 'primary_score') return run.primary_score ?? run.metrics.primary_score ?? null;
  return run.metrics?.[metricKey] ?? null;
}

function formatMetric(metricKey: string, value?: number | null) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '-';
  if (metricKey.includes('latency')) return secondsValue(value);
  if (metricKey === 'errors' || metricKey === 'correct' || metricKey === 'total') return numberValue(value);
  if (value >= 0 && value <= 1) return percentValue(value);
  return value.toFixed(2);
}

function formatRunDate(value?: string | null) {
  if (!value) return '-';
  const match = value.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
  if (!match) return value;
  const [, year, month, day, hour, minute, second] = match;
  const date = new Date(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute), Number(second));
  return date.toLocaleString('fr-FR', { dateStyle: 'medium', timeStyle: 'short' });
}

function shortRunName(name: string) {
  return name.replace(/^vespa_/, '').replace(/^chroma_/, '').replace(/^pgvector_/, '').replaceAll('_', ' ');
}

function compactRunName(name: string) {
  const cleaned = shortRunName(name);
  return cleaned.length > 28 ? `${cleaned.slice(0, 26)}…` : cleaned;
}

function backendLabel(value: string) {
  if (value === 'chroma') return 'Chroma';
  if (value === 'pgvector') return 'pgvector';
  if (value === 'vespa') return 'Vespa';
  return 'Inconnu';
}

function datasetValue(value: unknown): string {
  if (value == null || value === '') return '-';
  if (typeof value === 'boolean') return value ? 'Oui' : 'Non';
  if (Array.isArray(value)) return value.map((item) => datasetValue(item)).join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function availableMetrics(runs: EvaluationRun[], options: EvaluationMetricOption[]) {
  const available = new Set<string>();
  runs.forEach((run) => {
    Object.entries(run.metrics || {}).forEach(([key, value]) => {
      if (typeof value === 'number' && Number.isFinite(value)) available.add(key);
    });
    if (run.primary_score != null) available.add('primary_score');
  });

  const source = options.length ? options : DEFAULT_METRIC_OPTIONS;
  return source.filter((option) => available.has(option.key));
}

function StatCard({
  label,
  value,
  icon: Icon,
  tone = 'default',
}: {
  label: string;
  value: string | number;
  icon: LucideIcon;
  tone?: 'default' | 'primary' | 'blue' | 'purple' | 'danger';
}) {
  const toneClass = {
    default: 'text-muted-foreground',
    primary: 'text-primary',
    blue: 'text-blue-600',
    purple: 'text-purple-600',
    danger: 'text-destructive',
  }[tone];

  return (
    <div className="relative flex min-h-[120px] items-center justify-center rounded-lg border border-border bg-card px-6 py-4 shadow-sm">
      <div className={cn('absolute left-6 top-4 flex items-center gap-2 text-sm font-semibold', toneClass)}>
        <Icon className="h-4 w-4" />
        <span>{label}</span>
      </div>
      <p className="pt-5 text-center text-4xl font-bold leading-none text-foreground">{value}</p>
    </div>
  );
}

function MetricBarChart({
  runs,
  metricKey,
  metricLabel,
}: {
  runs: EvaluationRun[];
  metricKey: string;
  metricLabel: string;
}) {
  const values = runs
    .map((run) => ({ run, value: metricValue(run, metricKey) }))
    .filter((item): item is { run: EvaluationRun; value: number } => typeof item.value === 'number' && Number.isFinite(item.value));
  const max = Math.max(...values.map((item) => item.value), 0.0001);

  return (
    <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
      <div className="mb-5 flex items-center justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold text-foreground">Comparaison</h3>
          <p className="text-sm text-muted-foreground">{metricLabel}</p>
        </div>
        <BarChart3 className="h-5 w-5 text-muted-foreground" />
      </div>
      {values.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border bg-muted/30 p-8 text-center text-sm text-muted-foreground">
          Aucune donnée exploitable pour cette métrique.
        </p>
      ) : (
        <div className="space-y-4">
          {values.map(({ run, value }) => {
            const color = TASK_COLORS[run.backend] || TASK_COLORS[run.task] || TASK_COLORS.unknown;
            const width = Math.max(8, (value / max) * 100);
            return (
              <div key={run.id} className="grid gap-2">
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span className="min-w-0 truncate font-medium text-foreground">{shortRunName(run.run_name)}</span>
                  <span className="shrink-0 font-semibold text-foreground">{formatMetric(metricKey, value)}</span>
                </div>
                <div className="h-3 overflow-hidden rounded-full bg-muted">
                  <div className="h-full rounded-full" style={{ width: `${width}%`, backgroundColor: color }} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function TrendChart({
  runs,
  metricKey,
  metricLabel,
}: {
  runs: EvaluationRun[];
  metricKey: string;
  metricLabel: string;
}) {
  const points = runs
    .slice()
    .reverse()
    .map((run) => ({ run, value: metricValue(run, metricKey) }))
    .filter((item): item is { run: EvaluationRun; value: number } => typeof item.value === 'number' && Number.isFinite(item.value));
  const min = Math.min(...points.map((point) => point.value), 0);
  const max = Math.max(...points.map((point) => point.value), 1);
  const range = max - min || 1;
  const width = 520;
  const height = 210;
  const padding = 30;
  const bottomPadding = 38;
  const path = points
    .map((point, index) => {
      const x = padding + (index / Math.max(points.length - 1, 1)) * (width - padding * 2);
      const y = height - bottomPadding - ((point.value - min) / range) * (height - padding - bottomPadding);
      return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
    })
    .join(' ');

  return (
    <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold text-foreground">Évolution</h3>
          <p className="text-sm text-muted-foreground">{metricLabel} par date de run</p>
        </div>
        <LineChart className="h-5 w-5 text-muted-foreground" />
      </div>
      {points.length < 2 ? (
        <p className="rounded-lg border border-dashed border-border bg-muted/30 p-8 text-center text-sm text-muted-foreground">
          Sélectionne au moins deux runs avec cette métrique.
        </p>
      ) : (
        <div className="space-y-4 overflow-x-auto">
          <svg viewBox={`0 0 ${width} ${height}`} className="min-w-[520px]">
            <line x1={padding} x2={width - padding} y1={height - bottomPadding} y2={height - bottomPadding} stroke="#d9e4cd" />
            <line x1={padding} x2={padding} y1={padding} y2={height - bottomPadding} stroke="#d9e4cd" />
            <path d={path} fill="none" stroke="#4d7d13" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
            {points.map((point, index) => {
              const x = padding + (index / Math.max(points.length - 1, 1)) * (width - padding * 2);
              const y = height - bottomPadding - ((point.value - min) / range) * (height - padding - bottomPadding);
              return (
                <g key={point.run.id}>
                  <circle cx={x} cy={y} r="5" fill="#4d7d13" />
                  <text x={x} y={height - 10} textAnchor="middle" className="fill-muted-foreground text-[10px] font-semibold">
                    R{index + 1}
                  </text>
                  <text x={x} y={Math.max(14, y - 10)} textAnchor="middle" className="fill-foreground text-[10px] font-semibold">
                    {formatMetric(metricKey, point.value)}
                  </text>
                  <title>{`${point.run.run_name}: ${formatMetric(metricKey, point.value)}`}</title>
                </g>
              );
            })}
          </svg>
          <div className="grid min-w-[520px] gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {points.map((point, index) => (
              <div key={point.run.id} className="flex min-w-0 items-center gap-2 rounded-md bg-muted/40 px-3 py-2 text-xs">
                <span className="shrink-0 rounded-full bg-primary/10 px-2 py-0.5 font-semibold text-primary">R{index + 1}</span>
                <span className="min-w-0 truncate text-muted-foreground" title={shortRunName(point.run.run_name)}>
                  {compactRunName(point.run.run_name)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function DocsRadarChart({ runs }: { runs: EvaluationRun[] }) {
  const docsRuns = runs.filter((run) => run.task === 'docs').slice(0, 3);
  const size = 340;
  const center = size / 2;
  const radius = 118;
  const axisPoints = DOC_RADAR_METRICS.map((metric, index) => {
    const angle = -Math.PI / 2 + (index / DOC_RADAR_METRICS.length) * Math.PI * 2;
    return {
      ...metric,
      x: center + Math.cos(angle) * radius,
      y: center + Math.sin(angle) * radius,
      labelX: center + Math.cos(angle) * (radius + 34),
      labelY: center + Math.sin(angle) * (radius + 34),
      angle,
    };
  });
  const palette = ['#4d7d13', '#2563eb', '#7c3aed'];

  return (
    <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold text-foreground">Radar RAGAS</h3>
          <p className="text-sm text-muted-foreground">Jusqu’à 3 runs documents</p>
        </div>
        <Activity className="h-5 w-5 text-muted-foreground" />
      </div>
      {docsRuns.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border bg-muted/30 p-8 text-center text-sm text-muted-foreground">
          Aucun run document sélectionné.
        </p>
      ) : (
        <div className="grid items-center gap-5 xl:grid-cols-[380px_minmax(0,1fr)]">
          <svg viewBox={`0 0 ${size} ${size}`} className="mx-auto h-[340px] w-[340px] max-w-full">
            {[0.25, 0.5, 0.75, 1].map((level) => (
              <polygon
                key={level}
                points={axisPoints
                  .map((point) => `${center + Math.cos(point.angle) * radius * level},${center + Math.sin(point.angle) * radius * level}`)
                  .join(' ')}
                fill="none"
                stroke="#d9e4cd"
                strokeWidth="1"
              />
            ))}
            {axisPoints.map((point) => (
              <g key={point.key}>
                <line x1={center} y1={center} x2={point.x} y2={point.y} stroke="#d9e4cd" />
                <text x={point.labelX} y={point.labelY} textAnchor="middle" dominantBaseline="middle" className="fill-muted-foreground text-[10px]">
                  {point.label}
                </text>
              </g>
            ))}
            {docsRuns.map((run, runIndex) => {
              const color = palette[runIndex % palette.length];
              const polygon = axisPoints
                .map((point) => {
                  const value = Math.max(0, Math.min(1, Number(metricValue(run, point.key) || 0)));
                  return `${center + Math.cos(point.angle) * radius * value},${center + Math.sin(point.angle) * radius * value}`;
                })
                .join(' ');
              return <polygon key={run.id} points={polygon} fill={color} fillOpacity="0.14" stroke={color} strokeWidth="3" />;
            })}
          </svg>
          <div className="grid gap-2">
            {docsRuns.map((run, index) => (
              <div key={run.id} className="rounded-lg border border-border bg-background p-3">
                <div className="mb-2 flex items-center gap-2">
                  <span className="h-3 w-3 rounded-full" style={{ backgroundColor: palette[index % palette.length] }} />
                  <span className="truncate text-sm font-semibold text-foreground">{shortRunName(run.run_name)}</span>
                </div>
                <div className="grid grid-cols-5 gap-2 text-xs text-muted-foreground">
                  {DOC_RADAR_METRICS.map((metric) => (
                    <span key={metric.key} className="rounded-md bg-muted/40 px-2 py-1">
                      <span className="block truncate">{metric.label}</span>
                      <strong className="text-foreground">{formatMetric(metric.key, metricValue(run, metric.key))}</strong>
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function RunCard({
  run,
  selected,
  compareDisabled,
  onToggle,
  onDetails,
  onRename,
  onDelete,
}: {
  run: EvaluationRun;
  selected: boolean;
  compareDisabled: boolean;
  onToggle: () => void;
  onDetails: () => void;
  onRename: () => void;
  onDelete: () => void;
}) {
  const Icon = TASK_ICONS[run.task] || Layers;
  const color = TASK_COLORS[run.backend] || TASK_COLORS[run.task] || TASK_COLORS.unknown;

  return (
    <article className="rounded-lg border border-border bg-card p-4 shadow-sm transition hover:border-primary/50">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <span className="rounded-full border px-2.5 py-1 text-xs font-semibold" style={{ borderColor: color, color }}>
              {backendLabel(run.backend)}
            </span>
            <span className="rounded-full bg-muted px-2.5 py-1 text-xs font-semibold text-muted-foreground">{run.task_label}</span>
          </div>
          <h3 className="truncate text-base font-semibold text-foreground">{shortRunName(run.run_name)}</h3>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <span className="inline-flex items-center gap-1">
              <CalendarDays className="h-3.5 w-3.5" />
              {formatRunDate(run.created_at)}
            </span>
            <span className="inline-flex items-center gap-1">
              <Icon className="h-3.5 w-3.5" />
              {run.dataset_size ?? '-'} lignes
            </span>
          </div>
        </div>
        <label
          className={cn(
            'flex shrink-0 items-center gap-2 rounded-md border border-border bg-background px-3 py-2 text-xs font-semibold text-foreground',
            compareDisabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'
          )}
          title={compareDisabled ? 'Sélectionne uniquement des runs du même type.' : undefined}
        >
          <input
            type="checkbox"
            checked={selected}
            disabled={compareDisabled}
            onChange={onToggle}
            className="h-4 w-4 accent-primary disabled:cursor-not-allowed"
          />
          {compareDisabled ? 'Type différent' : 'Comparer'}
        </label>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        <div className="rounded-md bg-muted/40 p-3">
          <p className="text-xs text-muted-foreground">{run.primary_label}</p>
          <p className="mt-1 text-lg font-bold text-foreground">{formatMetric('primary_score', run.primary_score)}</p>
        </div>
        <div className="rounded-md bg-muted/40 p-3">
          <p className="text-xs text-muted-foreground">Latence</p>
          <p className="mt-1 text-lg font-bold text-foreground">{secondsValue(run.metrics.latency_mean)}</p>
        </div>
        <div className="rounded-md bg-muted/40 p-3">
          <p className="text-xs text-muted-foreground">Erreurs</p>
          <p className="mt-1 text-lg font-bold text-foreground">{numberValue(run.metrics.errors)}</p>
        </div>
      </div>
      <div className="mt-4 grid gap-2 sm:grid-cols-3">
        <Button type="button" variant="outline" size="sm" onClick={onDetails} className="gap-2">
          <Eye className="h-4 w-4" />
          Détails
        </Button>
        <Button type="button" variant="outline" size="sm" onClick={onRename} className="gap-2">
          <Edit3 className="h-4 w-4" />
          Renommer
        </Button>
        <Button type="button" variant="outline" size="sm" onClick={onDelete} className="user-delete-button gap-2">
          <Trash2 className="h-4 w-4" />
          Supprimer
        </Button>
      </div>
    </article>
  );
}

function RunDetails({ run }: { run: EvaluationRun }) {
  const metricEntries = Object.entries(run.metrics || {}).filter(([key, value]) => {
    if (key === 'latency_seconds' && run.metrics.latency_mean != null) return false;
    return typeof value === 'number' && Number.isFinite(value);
  });
  const summary = run.summary || {};
  const orgBreakdown =
    run.task === 'docs' && typeof summary.ragas_summary_by_org === 'object' && summary.ragas_summary_by_org
      ? (summary.ragas_summary_by_org as Record<string, { count?: number; metrics?: Record<string, number> }>)
      : null;

  return (
    <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Détail du run</p>
        </div>
        <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-semibold text-primary">{run.status}</span>
      </div>

      <div className="grid gap-2 text-sm sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-md bg-muted/40 px-3 py-2">
          <span className="block text-xs font-medium text-muted-foreground">Moteur</span>
          <strong className="mt-1 block truncate text-foreground">{backendLabel(run.backend)}</strong>
        </div>
        <div className="rounded-md bg-muted/40 px-3 py-2">
          <span className="block text-xs font-medium text-muted-foreground">Tâche</span>
          <strong className="mt-1 block truncate text-foreground">{run.task_label}</strong>
        </div>
        <div className="rounded-md bg-muted/40 px-3 py-2">
          <span className="block text-xs font-medium text-muted-foreground">Dataset</span>
          <strong className="mt-1 block truncate text-foreground">{run.dataset_size ?? '-'}</strong>
        </div>
        <div className="rounded-md bg-muted/40 px-3 py-2">
          <span className="block text-xs font-medium text-muted-foreground">Date</span>
          <strong className="mt-1 block truncate text-foreground" title={formatRunDate(run.created_at)}>
            {formatRunDate(run.created_at)}
          </strong>
        </div>
      </div>

      <div className="mt-5 border-t border-border pt-5">
        <p className="mb-3 text-sm font-semibold text-foreground">Métriques</p>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {metricEntries.map(([key, value]) => (
            <div key={key} className="rounded-md bg-muted/40 px-3 py-2 text-sm">
              <span className="block truncate text-xs text-muted-foreground">{key.replaceAll('_', ' ')}</span>
              <strong className="mt-1 block text-base text-foreground">{formatMetric(key, value)}</strong>
            </div>
          ))}
        </div>
      </div>

      {orgBreakdown && (
        <div className="mt-5 border-t border-border pt-5">
          <p className="mb-3 text-sm font-semibold text-foreground">RCAR / CNRA</p>
          <div className="grid gap-2 sm:grid-cols-2">
            {Object.entries(orgBreakdown).map(([org, payload]) => (
              <div key={org} className="rounded-md bg-muted/40 p-3 text-sm">
                <div className="flex items-center justify-between gap-3">
                  <strong className="uppercase text-foreground">{org}</strong>
                  <span className="text-xs text-muted-foreground">{payload.count ?? '-'} questions</span>
                  <span className="text-xs text-muted-foreground">
                    Exactitude : <strong className="text-foreground">{formatMetric('answer_correctness', payload.metrics?.answer_correctness)}</strong>
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function RunDetailsModal({ run, onClose }: { run: EvaluationRun; onClose: () => void }) {
  return (
    <div className="app-modal-overlay">
      <div className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-lg border border-border bg-background p-5 shadow-xl">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Run d’évaluation</p>
            <h3 className="mt-1 text-xl font-semibold text-foreground">{shortRunName(run.run_name)}</h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-border bg-background p-2 text-muted-foreground transition hover:bg-muted hover:text-foreground"
            aria-label="Fermer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <RunDetails run={run} />
      </div>
    </div>
  );
}

function DeleteRunModal({
  run,
  error,
  isDeleting,
  onClose,
  onConfirm,
}: {
  run: EvaluationRun;
  error: string;
  isDeleting: boolean;
  onClose: () => void;
  onConfirm: () => void;
}) {
  return (
    <div className="app-modal-overlay">
      <div className="w-full max-w-md rounded-lg border border-border bg-card p-6 shadow-xl">
        <div className="mb-4 flex items-start gap-3">
          <div className="rounded-full bg-destructive/10 p-2 text-destructive">
            <AlertTriangle className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-foreground">Supprimer le run</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              Cette action supprimera le run de la base et le dossier associé dans `data/evaluation/runs`.
            </p>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-muted/30 p-4 text-sm">
          <p className="font-semibold text-foreground">{shortRunName(run.run_name)}</p>
          <p className="mt-1 text-muted-foreground">{run.run_dir}</p>
        </div>

        {error && <p className="form-error mt-4">{error}</p>}

        <div className="mt-6 flex justify-end gap-2">
          <Button type="button" variant="outline" onClick={onClose} disabled={isDeleting}>
            Annuler
          </Button>
          <Button type="button" variant="outline" onClick={onConfirm} disabled={isDeleting} className="user-delete-button gap-2">
            <Trash2 className="h-4 w-4" />
            Supprimer
          </Button>
        </div>
      </div>
    </div>
  );
}

function RenameRunModal({
  run,
  value,
  error,
  isRenaming,
  onChange,
  onClose,
  onConfirm,
}: {
  run: EvaluationRun;
  value: string;
  error: string;
  isRenaming: boolean;
  onChange: (value: string) => void;
  onClose: () => void;
  onConfirm: () => void;
}) {
  return (
    <div className="app-modal-overlay">
      <div className="w-full max-w-md rounded-lg border border-border bg-card p-6 shadow-xl">
        <div className="mb-4 flex items-start gap-3">
          <div className="rounded-full bg-primary/10 p-2 text-primary">
            <Edit3 className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-foreground">Renommer le run</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              Le nom sera enregistré dans PostgreSQL et dans le fichier résumé du run.
            </p>
          </div>
        </div>

        <div className="grid gap-2">
          <label htmlFor="evaluation-run-name" className="text-sm font-semibold text-foreground">
            Nom du run
          </label>
          <input
            id="evaluation-run-name"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            className="h-11 rounded-lg border border-input bg-background px-3 text-sm"
            placeholder={run.run_name}
            autoFocus
          />
          <p className="text-xs text-muted-foreground">
            Les espaces et caractères spéciaux seront convertis en nom technique sûr.
          </p>
        </div>

        {error && <p className="form-error mt-4">{error}</p>}

        <div className="mt-6 flex justify-end gap-2">
          <Button type="button" variant="outline" onClick={onClose} disabled={isRenaming}>
            Annuler
          </Button>
          <Button type="button" onClick={onConfirm} disabled={isRenaming || value.trim().length === 0} className="gap-2">
            <Edit3 className="h-4 w-4" />
            Renommer
          </Button>
        </div>
      </div>
    </div>
  );
}

function DatasetModal({
  dataset,
  search,
  onSearchChange,
  onClose,
}: {
  dataset: EvaluationDatasetResponse;
  search: string;
  onSearchChange: (value: string) => void;
  onClose: () => void;
}) {
  const normalizedSearch = search.trim().toLowerCase();
  const rows = useMemo(() => {
    if (!normalizedSearch) return dataset.rows;
    return dataset.rows.filter((row) => {
      const haystack = [row.index, ...dataset.columns.map((column) => row[column])]
        .map((value) => datasetValue(value))
        .join(' ')
        .toLowerCase();
      return haystack.includes(normalizedSearch);
    });
  }, [dataset.columns, dataset.rows, normalizedSearch]);

  return (
    <div className="app-modal-overlay">
      <div className="flex max-h-[90vh] w-full max-w-6xl flex-col rounded-lg border border-border bg-card p-6 shadow-xl">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Dataset d'évaluation</p>
            <h3 className="mt-1 text-xl font-semibold text-foreground">{dataset.task_label}</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              {dataset.total} ligne{dataset.total > 1 ? 's' : ''} - {dataset.path}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-border bg-background p-2 text-muted-foreground transition hover:bg-muted hover:text-foreground"
            aria-label="Fermer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <label className="relative mb-4 block">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            value={search}
            onChange={(event) => onSearchChange(event.target.value)}
            className="h-11 w-full rounded-lg border border-input bg-background pl-10 pr-3 text-sm"
            placeholder="Rechercher dans le dataset..."
          />
        </label>

        <div className="min-h-0 overflow-auto rounded-lg border border-border">
          <table className="min-w-full border-collapse text-left text-sm">
            <thead className="sticky top-0 z-10 bg-muted text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="whitespace-nowrap border-b border-border px-3 py-3 font-semibold">#</th>
                {dataset.columns.map((column) => (
                  <th key={column} className="whitespace-nowrap border-b border-border px-3 py-3 font-semibold">
                    {column.replaceAll('_', ' ')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.index} className="border-b border-border/70 align-top last:border-b-0">
                  <td className="whitespace-nowrap px-3 py-3 font-semibold text-muted-foreground">{row.index}</td>
                  {dataset.columns.map((column) => (
                    <td key={`${row.index}-${column}`} className="max-w-[420px] px-3 py-3 text-foreground">
                      <div className="max-h-28 overflow-auto whitespace-pre-wrap leading-relaxed">
                        {datasetValue(row[column])}
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length === 0 && (
            <p className="p-8 text-center text-sm text-muted-foreground">Aucune ligne ne correspond à la recherche.</p>
          )}
        </div>

        <p className="mt-3 text-xs text-muted-foreground">
          {rows.length} ligne{rows.length > 1 ? 's' : ''} affichée{rows.length > 1 ? 's' : ''}.
        </p>
      </div>
    </div>
  );
}

export default function EvaluationPanel() {
  const [data, setData] = useState<EvaluationRunsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSyncing, setIsSyncing] = useState(false);
  const [error, setError] = useState('');
  const [taskFilter, setTaskFilter] = useState<TaskFilter>('all');
  const [metricKey, setMetricKey] = useState('primary_score');
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [detailsRun, setDetailsRun] = useState<EvaluationRun | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<EvaluationRun | null>(null);
  const [deleteError, setDeleteError] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);
  const [renameTarget, setRenameTarget] = useState<EvaluationRun | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [renameError, setRenameError] = useState('');
  const [isRenaming, setIsRenaming] = useState(false);
  const [datasetModal, setDatasetModal] = useState<EvaluationDatasetResponse | null>(null);
  const [datasetSearch, setDatasetSearch] = useState('');
  const [datasetError, setDatasetError] = useState('');
  const [isDatasetLoading, setIsDatasetLoading] = useState(false);
  const [showAllRuns, setShowAllRuns] = useState(false);
  const [runTask, setRunTask] = useState<EvaluationTask>('docs');
  const [runMode, setRunMode] = useState<'smoke' | 'full'>('smoke');
  const [runName, setRunName] = useState('');
  const [skipRagas, setSkipRagas] = useState(true);
  const [runMessage, setRunMessage] = useState('');
  const [isPreparing, setIsPreparing] = useState(false);

  const loadRuns = async () => {
    setIsLoading(true);
    setError('');
    try {
      setData(await listEvaluationRuns());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Impossible de charger les évaluations.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void loadRuns();
  }, []);

  const runs = data?.runs ?? [];
  const filteredRuns = useMemo(() => {
    return runs.filter((run) => (taskFilter === 'all' ? true : run.task === taskFilter));
  }, [runs, taskFilter]);

  useEffect(() => {
    setShowAllRuns(false);
  }, [taskFilter]);

  const selectedRuns = useMemo(() => {
    return filteredRuns.filter((run) => selectedRunIds.includes(run.id));
  }, [filteredRuns, selectedRunIds]);

  const selectedTask = selectedRuns[0]?.task ?? null;

  const metricOptions = useMemo(() => {
    return availableMetrics(selectedRuns.length > 0 ? selectedRuns : filteredRuns, data?.metric_options ?? []);
  }, [data?.metric_options, filteredRuns, selectedRuns]);

  useEffect(() => {
    if (metricOptions.length === 0) return;
    if (!metricOptions.some((option) => option.key === metricKey)) {
      setMetricKey(metricOptions[0].key);
    }
  }, [metricKey, metricOptions]);

  useEffect(() => {
    const filteredIds = new Set(filteredRuns.map((run) => run.id));
    setSelectedRunIds((current) => {
      return current.filter((id) => filteredIds.has(id));
    });
  }, [filteredRuns]);

  const visibleRuns = showAllRuns ? filteredRuns : filteredRuns.slice(0, 6);
  const hiddenRunsCount = Math.max(0, filteredRuns.length - visibleRuns.length);

  const metricLabel = metricOptions.find((option) => option.key === metricKey)?.label ?? 'Métrique';

  const toggleRun = (runId: string) => {
    const run = filteredRuns.find((item) => item.id === runId);
    if (!run) return;

    setSelectedRunIds((current) => {
      if (current.includes(runId)) return current.filter((id) => id !== runId);
      const currentRuns = filteredRuns.filter((item) => current.includes(item.id));
      const currentTask = currentRuns[0]?.task;
      if (currentTask && currentTask !== run.task) return current;
      return [...current, runId].slice(-6);
    });
  };

  const syncRuns = async () => {
    setIsSyncing(true);
    setError('');
    try {
      setData(await syncEvaluationRuns());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Synchronisation impossible.');
    } finally {
      setIsSyncing(false);
    }
  };

  const requestDeleteRun = (run: EvaluationRun) => {
    setDeleteTarget(run);
    setDeleteError('');
  };

  const closeDeleteRun = () => {
    setDeleteTarget(null);
    setDeleteError('');
  };

  const confirmDeleteRun = async () => {
    if (!deleteTarget) return;

    setIsDeleting(true);
    setDeleteError('');
    try {
      await deleteEvaluationRun(deleteTarget.id);
      setData(await syncEvaluationRuns());
      setSelectedRunIds((current) => current.filter((id) => id !== deleteTarget.id));
      if (detailsRun?.id === deleteTarget.id) setDetailsRun(null);
      closeDeleteRun();
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : 'Suppression impossible.');
    } finally {
      setIsDeleting(false);
    }
  };

  const requestRenameRun = (run: EvaluationRun) => {
    setRenameTarget(run);
    setRenameValue(run.run_name);
    setRenameError('');
  };

  const closeRenameRun = () => {
    setRenameTarget(null);
    setRenameValue('');
    setRenameError('');
  };

  const confirmRenameRun = async () => {
    if (!renameTarget) return;

    setIsRenaming(true);
    setRenameError('');
    try {
      const updated = await renameEvaluationRun(renameTarget.id, renameValue);
      setData((current) => {
        if (!current) return current;
        return {
          ...current,
          runs: current.runs.map((run) => (run.id === updated.id ? updated : run)),
        };
      });
      if (detailsRun?.id === updated.id) setDetailsRun(updated);
      closeRenameRun();
    } catch (err) {
      setRenameError(err instanceof Error ? err.message : 'Renommage impossible.');
    } finally {
      setIsRenaming(false);
    }
  };

  const openDataset = async () => {
    setIsDatasetLoading(true);
    setDatasetError('');
    try {
      const dataset = await getEvaluationDataset(runTask);
      setDatasetSearch('');
      setDatasetModal(dataset);
    } catch (err) {
      setDatasetError(err instanceof Error ? err.message : 'Chargement du dataset impossible.');
    } finally {
      setIsDatasetLoading(false);
    }
  };

  const launchRun = async (event: FormEvent) => {
    event.preventDefault();
    setRunMessage('');
    setIsPreparing(true);
    try {
      const result = await prepareEvaluationRun({
        task: runTask,
        mode: runMode,
        skip_ragas: skipRagas,
        limit: runMode === 'smoke' ? 3 : null,
        run_name: runName.trim() || null,
      });
      setRunName('');
      setRunMessage(
        result.run_name
          ? `${result.message} Run : ${result.run_name}${result.log_path ? ` - Log : ${result.log_path}` : ''}`
          : result.message
      );
    } catch (err) {
      setRunMessage(err instanceof Error ? err.message : 'Lancement impossible.');
    } finally {
      setIsPreparing(false);
    }
  };

  return (
    <section className="space-y-7">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold leading-tight text-foreground">Évaluation</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Visualiser les anciens runs, comparer les métriques et préparer les prochaines évaluations.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={() => void loadRuns()} disabled={isLoading} className="gap-2">
            <RefreshCw className="h-4 w-4" />
            Actualiser
          </Button>
          <Button onClick={() => void syncRuns()} disabled={isSyncing} className="gap-2">
            <Database className="h-4 w-4" />
            Synchroniser
          </Button>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/20 bg-destructive/10 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Éval. Documents" value={numberValue(data?.stats.by_task?.docs)} icon={FileText} tone="primary" />
        <StatCard label="Éval. Vidéos" value={numberValue(data?.stats.by_task?.videos)} icon={Video} tone="blue" />
        <StatCard label="Éval. Formulaires" value={numberValue(data?.stats.by_task?.forms)} icon={Database} />
        <StatCard label="Éval. Intentions" value={numberValue(data?.stats.by_task?.intentions)} icon={CheckCircle2} tone="purple" />
        <StatCard label="Erreurs" value={numberValue(data?.stats.total_errors)} icon={Activity} tone={data?.stats.total_errors ? 'danger' : 'default'} />
        <StatCard label="Exactitude docs" value={percentValue(data?.stats.best_docs_score)} icon={FileText} tone="primary" />
        <StatCard label="Exactitude ressources" value={percentValue(data?.stats.best_resource_hit_at_1)} icon={Target} tone="blue" />
        <StatCard label="Exactitude intentions" value={percentValue(data?.stats.best_intent_accuracy)} icon={CheckCircle2} tone="purple" />
      </div>

      <form onSubmit={launchRun} className="rounded-lg border border-border bg-card p-5 shadow-sm">
        <div className="mb-5">
          <h3 className="text-lg font-semibold text-foreground">Lancer un run</h3>
          <p className="mt-1 text-sm text-muted-foreground">
            Le run démarre en arrière-plan. Synchronise ensuite la liste pour récupérer les résultats.
          </p>
        </div>
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)_auto_auto_auto] lg:items-end">
          <label className="grid gap-2 text-sm font-semibold text-foreground">
            Type
            <select value={runTask} onChange={(event) => setRunTask(event.target.value as EvaluationTask)} className="h-11 rounded-lg border border-input bg-background px-3 text-sm font-normal">
              <option value="docs">Documents</option>
              <option value="forms">Formulaires</option>
              <option value="videos">Vidéos</option>
              <option value="intentions">Intentions</option>
            </select>
          </label>
          <label className="grid gap-2 text-sm font-semibold text-foreground">
            Mode
            <select value={runMode} onChange={(event) => setRunMode(event.target.value as 'smoke' | 'full')} className="h-11 rounded-lg border border-input bg-background px-3 text-sm font-normal">
              <option value="smoke">Test rapide</option>
              <option value="full">Benchmark complet</option>
            </select>
          </label>
          <label className="grid gap-2 text-sm font-semibold text-foreground">
            Nom du run
            <input
              value={runName}
              onChange={(event) => setRunName(event.target.value)}
              className="h-11 rounded-lg border border-input bg-background px-3 text-sm font-normal"
              placeholder="ex: docs_test_vespa"
            />
          </label>
          {runTask === 'docs' && (
            <label className="flex h-11 items-center gap-2 whitespace-nowrap text-sm font-semibold text-foreground">
              <input type="checkbox" checked={skipRagas} onChange={(event) => setSkipRagas(event.target.checked)} className="h-4 w-4 accent-primary" />
              Collecte sans RAGAS
            </label>
          )}
          <Button type="button" variant="outline" disabled={isDatasetLoading} onClick={() => void openDataset()} className="h-11 gap-2">
            <Eye className="h-4 w-4" />
            Voir dataset
          </Button>
          <Button type="submit" disabled={isPreparing} className="h-11 gap-2">
            <Play className="h-4 w-4" />
            Lancer
          </Button>
        </div>
        {datasetError && (
          <p className="mt-4 rounded-lg border border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">{datasetError}</p>
        )}
        {runMessage && (
          <p className="mt-4 rounded-lg border border-border bg-muted/30 p-3 text-sm text-muted-foreground">{runMessage}</p>
        )}
      </form>

      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-lg font-semibold text-foreground">Runs disponibles</h3>
            <p className="text-sm text-muted-foreground">
              {filteredRuns.length} run{filteredRuns.length > 1 ? 's' : ''} affiché{filteredRuns.length > 1 ? 's' : ''}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {(Object.keys(TASK_LABELS) as TaskFilter[]).map((task) => (
              <button
                key={task}
                type="button"
                onClick={() => setTaskFilter(task)}
                className={cn(
                  'rounded-md px-3 py-2 text-sm font-semibold transition',
                  taskFilter === task ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground hover:bg-accent'
                )}
              >
                {TASK_LABELS[task]}
              </button>
            ))}
            {selectedRunIds.length > 0 && (
              <Button variant="outline" size="sm" onClick={() => setSelectedRunIds([])}>
                Vider la sélection
              </Button>
            )}
          </div>
        </div>

        {isLoading && <p className="rounded-lg border border-border bg-card p-6 text-sm text-muted-foreground">Chargement des évaluations...</p>}
        {!isLoading && filteredRuns.length === 0 && (
          <p className="rounded-lg border border-dashed border-border bg-muted/30 p-10 text-center text-sm text-muted-foreground">
            Aucun run trouvé. Lance une synchronisation pour lire `data/evaluation/runs`.
          </p>
        )}
        <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
          {visibleRuns.map((run) => (
            <RunCard
              key={run.id}
              run={run}
              selected={selectedRunIds.includes(run.id)}
              compareDisabled={selectedTask != null && run.task !== selectedTask}
              onToggle={() => toggleRun(run.id)}
              onDetails={() => setDetailsRun(run)}
              onRename={() => requestRenameRun(run)}
              onDelete={() => requestDeleteRun(run)}
            />
          ))}
        </div>
        {filteredRuns.length > 6 && (
          <div className="flex justify-center pt-2">
            <Button variant="outline" onClick={() => setShowAllRuns((current) => !current)}>
              {showAllRuns ? 'Voir moins' : `Voir plus (${hiddenRunsCount})`}
            </Button>
          </div>
        )}
      </div>

      <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold text-foreground">Comparaison graphique</h3>
            <p className="text-sm text-muted-foreground">Compare les runs sélectionnés dans la grille.</p>
          </div>
          <label className="grid min-w-[240px] gap-2 text-sm font-semibold text-foreground">
            Métrique
            <select
              value={metricKey}
              onChange={(event) => setMetricKey(event.target.value)}
              className="h-11 rounded-lg border border-input bg-background px-3 text-sm font-normal"
            >
              {metricOptions.map((option) => (
                <option key={option.key} value={option.key}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="grid gap-5 2xl:grid-cols-2">
          <MetricBarChart runs={selectedRuns} metricKey={metricKey} metricLabel={metricLabel} />
          <TrendChart runs={selectedRuns} metricKey={metricKey} metricLabel={metricLabel} />
          <div className="2xl:col-span-2">
            <DocsRadarChart runs={selectedRuns} />
          </div>
        </div>
      </div>

      {detailsRun && <RunDetailsModal run={detailsRun} onClose={() => setDetailsRun(null)} />}
      {renameTarget && (
        <RenameRunModal
          run={renameTarget}
          value={renameValue}
          error={renameError}
          isRenaming={isRenaming}
          onChange={setRenameValue}
          onClose={closeRenameRun}
          onConfirm={() => void confirmRenameRun()}
        />
      )}
      {datasetModal && (
        <DatasetModal
          dataset={datasetModal}
          search={datasetSearch}
          onSearchChange={setDatasetSearch}
          onClose={() => setDatasetModal(null)}
        />
      )}
      {deleteTarget && (
        <DeleteRunModal
          run={deleteTarget}
          error={deleteError}
          isDeleting={isDeleting}
          onClose={closeDeleteRun}
          onConfirm={() => void confirmDeleteRun()}
        />
      )}
    </section>
  );
}
