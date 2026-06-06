import {
  AlertCircle,
  Brain,
  CheckCircle2,
  Clock,
  Code2,
  Database,
  FileText,
  Layers,
  PlayCircle,
  Search,
  Send,
  Settings2,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { FormEvent, useLayoutEffect, useMemo, useRef, useState } from 'react';
import {
  runRetrievalTest,
  type RetrievalTestChunk,
  type RetrievalTestMode,
  type RetrievalTestResource,
  type RetrievalTestResponse,
} from '../../api/admin';
import { cn } from '../../utils/helpers';
import FormCard from '../chat/FormCard';
import VideoCard from '../chat/VideoCard';
import Button from '../ui/Button';

type RetrievalOrg = 'all' | 'rcar' | 'cnra';

function formatLatency(value?: number | null) {
  if (value == null || Number.isNaN(value)) return '-';
  if (value < 1) return `${Math.round(value * 1000)} ms`;
  return `${value.toFixed(2)} s`;
}

function formatNumber(value?: number | null, decimals = 3) {
  if (value == null || Number.isNaN(value)) return '-';
  return value.toLocaleString('fr-FR', {
    maximumFractionDigits: decimals,
  });
}

function confidenceValue(value?: number) {
  return typeof value === 'number' ? `${Math.round(value * 100)} %` : '-';
}

function orgLabel(org: string) {
  if (org === 'rcar') return 'RCAR';
  if (org === 'cnra') return 'CNRA';
  return 'Les deux';
}

function modeLabel(mode: RetrievalTestMode) {
  return mode === 'retrieval' ? 'Récupération' : 'Complète';
}

function sourceLabel(value?: string) {
  if (!value) return 'Source';
  if (value === 'bibliotheque') return 'Bibliothèque';
  if (value === 'faq') return 'FAQ';
  if (value === 'web') return 'Web';
  return value;
}

function StatCard({
  label,
  value,
  subValue,
  icon: Icon,
  tone = 'default',
}: {
  label: string;
  value: string | number;
  subValue?: string;
  icon: LucideIcon;
  tone?: 'default' | 'primary' | 'blue' | 'warning' | 'success';
}) {
  const toneClass = {
    default: 'text-muted-foreground',
    primary: 'text-primary',
    blue: 'text-blue-600',
    warning: 'text-amber-600',
    success: 'text-emerald-600',
  }[tone];

  return (
    <div className="relative flex min-h-[112px] items-center justify-center rounded-lg border border-border bg-card px-5 py-4 shadow-sm">
      <div className={cn('absolute left-5 top-4 flex items-center gap-2 text-sm font-semibold', toneClass)}>
        <Icon className="h-4 w-4" />
        <span>{label}</span>
      </div>
      <div className="pt-6 text-center">
        <p className="break-words text-3xl font-bold leading-none text-foreground">{value}</p>
        {subValue && <p className="mt-2 text-sm font-semibold text-muted-foreground">{subValue}</p>}
      </div>
    </div>
  );
}

function ScorePill({ label, value }: { label: string; value?: number | null }) {
  return (
    <span className="inline-flex min-w-0 items-center gap-1 rounded-full bg-muted px-2.5 py-1 text-xs font-semibold text-muted-foreground">
      <span className="truncate">{label}</span>
      <strong className="text-foreground">{formatNumber(value)}</strong>
    </span>
  );
}

function ChunkCard({ chunk }: { chunk: RetrievalTestChunk }) {
  const [expanded, setExpanded] = useState(false);
  const [canExpand, setCanExpand] = useState(false);
  const textRef = useRef<HTMLParagraphElement | null>(null);
  const text = (chunk.text || '').trim();

  useLayoutEffect(() => {
    if (expanded) return;

    const element = textRef.current;
    if (!element) return;

    const measure = () => {
      setCanExpand(element.scrollHeight > element.clientHeight + 1);
    };

    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [expanded, text]);

  return (
    <article className="min-w-0 rounded-lg border border-border bg-card p-4 shadow-sm">
      <div className="grid gap-3">
        <div className="min-w-0">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <span className="rounded-full bg-primary/10 px-2.5 py-1 text-xs font-bold text-primary">
              #{chunk.rank}
            </span>
            {chunk.org && (
              <span className="rounded-full bg-muted px-2.5 py-1 text-xs font-semibold text-muted-foreground">
                {orgLabel(chunk.org)}
              </span>
            )}
            {chunk.source_type && (
              <span className="rounded-full bg-blue-50 px-2.5 py-1 text-xs font-semibold text-blue-700">
                {sourceLabel(chunk.source_type)}
              </span>
            )}
          </div>
          <h4 className="mt-3 break-words text-base font-semibold text-foreground">{chunk.title}</h4>
          {chunk.relative_source && (
            <p className="mt-1 line-clamp-2 break-all text-xs text-muted-foreground">{chunk.relative_source}</p>
          )}
        </div>
        <div className="flex min-w-0 max-w-full flex-wrap gap-2">
          <ScorePill label="Vespa" value={chunk.vespa_relevance} />
          <ScorePill label="Rerank" value={chunk.rerank_score} />
          {chunk.source_boost ? <ScorePill label="Boost" value={chunk.source_boost} /> : null}
        </div>
      </div>

      <p
        ref={textRef}
        className={cn(
          'mt-4 whitespace-pre-wrap break-words text-sm leading-7 text-muted-foreground',
          !expanded && 'line-clamp-5'
        )}
      >
        {text || 'Aucun texte dans ce chunk.'}
      </p>

      {canExpand && (
        <button
          type="button"
          onClick={() => setExpanded((current) => !current)}
          className="mt-3 text-sm font-semibold text-primary hover:underline"
        >
          {expanded ? 'Voir moins' : 'Voir plus'}
        </button>
      )}
    </article>
  );
}

function ResourceCandidates({
  title,
  icon: Icon,
  candidates,
}: {
  title: string;
  icon: LucideIcon;
  candidates: RetrievalTestResource[];
}) {
  if (candidates.length === 0) return null;

  return (
    <div className="min-w-0 rounded-lg border border-border bg-card p-4 shadow-sm">
      <div className="mb-3 flex min-w-0 items-center gap-2 text-base font-semibold text-muted-foreground">
        <Icon className="h-4 w-4 flex-shrink-0" />
        <span className="truncate">{title}</span>
      </div>
      <div className="grid gap-2">
        {candidates.slice(0, 8).map((candidate, index) => (
          <div
            key={`${candidate.id || candidate.title}-${index}`}
            className="grid min-w-0 gap-2 rounded-lg border border-border bg-background p-3 xl:grid-cols-[minmax(0,1fr)_minmax(220px,auto)]"
          >
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold text-foreground" title={candidate.title}>
                {candidate.rank ? `#${candidate.rank} ` : ''}
                {candidate.title}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">{candidate.org ? orgLabel(String(candidate.org)) : '-'}</p>
            </div>
            <div className="flex min-w-0 flex-wrap items-center gap-2 xl:justify-end">
              <ScorePill label="Score" value={typeof candidate.score === 'number' ? candidate.score : null} />
              <ScorePill label="Vespa" value={typeof candidate.vespa_score === 'number' ? candidate.vespa_score : null} />
              <ScorePill label="Rerank" value={typeof candidate.rerank_score === 'number' ? candidate.rerank_score : null} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PromptDebug({ result }: { result: RetrievalTestResponse }) {
  const timings = JSON.stringify(result.timings || {}, null, 2);

  return (
    <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
      <div className="mb-4 flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-muted-foreground">
        <Code2 className="h-4 w-4" />
        Debug technique
      </div>
      <div className="grid gap-3">
        <details className="rounded-lg border border-border bg-background p-4">
          <summary className="cursor-pointer text-sm font-semibold text-foreground">Prompt système rendu</summary>
          <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap rounded-md bg-muted/40 p-3 text-xs leading-6 text-muted-foreground">
            {result.prompt?.system || 'Prompt indisponible.'}
          </pre>
        </details>
        <details className="rounded-lg border border-border bg-background p-4">
          <summary className="cursor-pointer text-sm font-semibold text-foreground">Contexte documentaire envoyé</summary>
          <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap rounded-md bg-muted/40 p-3 text-xs leading-6 text-muted-foreground">
            {result.prompt?.context || 'Aucun contexte.'}
          </pre>
        </details>
        <details className="rounded-lg border border-border bg-background p-4">
          <summary className="cursor-pointer text-sm font-semibold text-foreground">Timings</summary>
          <pre className="mt-3 overflow-auto rounded-md bg-muted/40 p-3 text-xs text-muted-foreground">{timings}</pre>
        </details>
      </div>
    </div>
  );
}

export default function RetrievalTestPanel() {
  const [query, setQuery] = useState('');
  const [org, setOrg] = useState<RetrievalOrg>('all');
  const [mode, setMode] = useState<RetrievalTestMode>('full');
  const [includeResources, setIncludeResources] = useState(true);
  const [useIntentClassifier, setUseIntentClassifier] = useState(true);
  const [result, setResult] = useState<RetrievalTestResponse | null>(null);
  const [error, setError] = useState('');
  const [isRunning, setIsRunning] = useState(false);

  const chunks = result?.chunks ?? [];
  const videos = result?.videos ?? [];
  const forms = result?.forms ?? [];
  const videoCandidates = result?.video_candidates ?? [];
  const formCandidates = result?.form_candidates ?? [];
  const resourcesCount = videos.length + forms.length;
  const totalLatency = result?.timings?.total_seconds;
  const docsLatency = result?.timings?.docs_retrieval_seconds;
  const generationLatency = result?.timings?.generation_seconds;

  const candidateGroupsCount = useMemo(
    () => (videoCandidates.length > 0 ? 1 : 0) + (formCandidates.length > 0 ? 1 : 0),
    [videoCandidates.length, formCandidates.length]
  );

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!query.trim()) return;
    setIsRunning(true);
    setError('');
    setResult(null);
    try {
      setResult(
        await runRetrievalTest({
          query,
          org,
          mode,
          include_resources: includeResources,
          use_intent_classifier: useIntentClassifier,
        })
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Test de récupération impossible.');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <section className="space-y-7">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold leading-tight text-foreground">Test récupération</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Tester une requête et inspecter les chunks, ressources, timings et prompt.
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="rounded-lg border border-border bg-card p-5 shadow-sm">
        <div className="grid gap-4">
          <label className="grid gap-2">
            <span className="text-sm font-semibold text-foreground">Requête</span>
            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              rows={3}
              className="min-h-24 rounded-lg border border-input bg-background px-4 py-3 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
              placeholder="Exemple : quelles sont les conditions pour bénéficier d'une pension RCAR ?"
            />
          </label>

          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] lg:items-end">
            <div className="grid gap-2">
              <span className="text-sm font-semibold text-foreground">Organisme</span>
              <div className="grid grid-cols-3 rounded-lg bg-muted p-1">
                {[
                  ['all', 'Les deux'],
                  ['rcar', 'RCAR'],
                  ['cnra', 'CNRA'],
                ].map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setOrg(value as RetrievalOrg)}
                    className={cn(
                      'rounded-md px-3 py-2 text-sm font-semibold transition-colors',
                      org === value ? 'bg-primary text-primary-foreground shadow-sm' : 'text-muted-foreground hover:bg-background'
                    )}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid gap-2">
              <span className="text-sm font-semibold text-foreground">Mode</span>
              <div className="grid grid-cols-2 rounded-lg bg-muted p-1">
                {[
                  ['full', 'Réponse complète'],
                  ['retrieval', 'Récupération'],
                ].map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setMode(value as RetrievalTestMode)}
                    className={cn(
                      'rounded-md px-3 py-2 text-sm font-semibold transition-colors',
                      mode === value ? 'bg-primary text-primary-foreground shadow-sm' : 'text-muted-foreground hover:bg-background'
                    )}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <Button type="submit" isLoading={isRunning} className="gap-2">
              <Send className="h-4 w-4" />
              Lancer le test
            </Button>
          </div>

          <div className="flex flex-wrap gap-3 border-t border-border pt-4">
            <label className="inline-flex items-center gap-2 text-sm font-medium text-muted-foreground">
              <input
                type="checkbox"
                checked={includeResources}
                onChange={(event) => setIncludeResources(event.target.checked)}
                className="h-4 w-4 accent-primary"
              />
              Tester vidéos et formulaires
            </label>
            <label className="inline-flex items-center gap-2 text-sm font-medium text-muted-foreground">
              <input
                type="checkbox"
                checked={useIntentClassifier}
                onChange={(event) => setUseIntentClassifier(event.target.checked)}
                className="h-4 w-4 accent-primary"
              />
              Utiliser le classificateur d'intentions
            </label>
          </div>
        </div>
      </form>

      {error && (
        <div className="flex items-center gap-3 rounded-lg border border-destructive/20 bg-destructive/10 p-4 text-sm text-destructive">
          <AlertCircle className="h-5 w-5" />
          {error}
        </div>
      )}

      {result && (
        <>
          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-4">
            <StatCard label="Mode" value={modeLabel(result.mode)} icon={Settings2} tone="primary" />
            <StatCard label="Organisme" value={orgLabel(result.org)} icon={Search} />
            <StatCard
              label="Intention"
              value={result.intent || '-'}
              subValue={`Confiance ${confidenceValue(result.intent_confidence)}`}
              icon={Brain}
              tone="blue"
            />
            <StatCard label="Latence totale" value={formatLatency(totalLatency)} icon={Clock} tone="warning" />
            <StatCard label="Récupération docs" value={formatLatency(docsLatency)} icon={Database} tone="primary" />
            <StatCard label="Génération" value={formatLatency(generationLatency)} icon={CheckCircle2} tone="success" />
            <StatCard label="Chunks retenus" value={chunks.length} icon={Layers} tone="blue" />
            <StatCard label="Ressources" value={resourcesCount} icon={FileText} tone="success" />
          </div>

          {result.response && (
            <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
              <div className="mb-3 flex items-center gap-2 text-base font-semibold text-muted-foreground">
                <Search className="h-4 w-4" />
                Réponse
              </div>
              <p className="whitespace-pre-wrap text-sm leading-7 text-foreground">{result.response}</p>
            </div>
          )}

          <div className="min-w-0 rounded-lg border border-border bg-card p-5 shadow-sm">
            <div className="mb-4 flex items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-2 text-base font-semibold text-muted-foreground">
                <FileText className="h-4 w-4 flex-shrink-0" />
                <span className="truncate">Ressources suggérées</span>
              </div>
              <span className="flex-shrink-0 rounded-full bg-muted px-3 py-1 text-xs font-semibold text-muted-foreground">
                {resourcesCount}
              </span>
            </div>
            {resourcesCount === 0 ? (
              <p className="rounded-lg border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
                Aucune vidéo ou formulaire suggéré.
              </p>
            ) : (
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {videos.map((resource, index) => (
                  <VideoCard key={resource.id || `${resource.title}-${index}`} resource={resource} />
                ))}
                {forms.map((resource, index) => (
                  <FormCard key={resource.id || `${resource.title}-${index}`} resource={resource} />
                ))}
              </div>
            )}
          </div>

          {candidateGroupsCount > 0 && (
            <div className={cn('grid gap-5', candidateGroupsCount > 1 && 'xl:grid-cols-2')}>
              <ResourceCandidates title="Candidats vidéos" icon={PlayCircle} candidates={videoCandidates} />
              <ResourceCandidates title="Candidats formulaires" icon={FileText} candidates={formCandidates} />
            </div>
          )}

          <div className="rounded-lg border border-border bg-background p-5 shadow-sm">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-base font-semibold text-muted-foreground">Chunks documentaires</h3>
                <p className="mt-1 text-sm text-muted-foreground">Passages retenus après récupération Vespa et reranking.</p>
              </div>
              <span className="rounded-full bg-muted px-3 py-1 text-sm font-semibold text-muted-foreground">
                {chunks.length} chunk{chunks.length > 1 ? 's' : ''}
              </span>
            </div>
            {chunks.length === 0 ? (
              <p className="rounded-lg border border-dashed border-border bg-muted/30 p-8 text-center text-sm text-muted-foreground">
                Aucun chunk retourné.
              </p>
            ) : (
              <div className="grid gap-4 lg:grid-cols-2">
                {chunks.map((chunk) => (
                  <ChunkCard key={`${chunk.rank}-${chunk.title}`} chunk={chunk} />
                ))}
              </div>
            )}
          </div>

          <PromptDebug result={result} />
        </>
      )}
    </section>
  );
}
