import {
  Activity,
  AlertCircle,
  AlertTriangle,
  Brain,
  CheckCircle2,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  Zap,
} from 'lucide-react';
import { FormEvent, useEffect, useMemo, useState } from 'react';
import {
  classifyIntent,
  createIntentExample,
  deleteIntentExample,
  getIntentStatus,
  listIntentExamples,
  listIntents,
  reloadIntents,
  syncIntents,
  updateIntent,
  updateIntentExample,
  type IntentClassifyResult,
  type IntentExample,
  type IntentItem,
  type IntentStatus,
} from '../../api/admin';
import { cn } from '../../utils/helpers';
import Button from '../ui/Button';

function numberValue(value?: number) {
  return typeof value === 'number' ? value.toLocaleString('fr-FR') : '-';
}

function confidenceValue(value?: number) {
  return typeof value === 'number' ? `${Math.round(value * 100)} %` : '-';
}

function DeleteIntentExampleModal({
  example,
  error,
  isDeleting,
  onClose,
  onConfirm,
}: {
  example: IntentExample;
  error: string;
  isDeleting: boolean;
  onClose: () => void;
  onConfirm: () => void;
}) {
  return (
    <div className="app-modal-overlay">
      <div className="w-full max-w-lg rounded-lg border border-border bg-card p-6 shadow-xl">
        <div className="mb-4 flex items-start gap-3">
          <div className="rounded-full bg-destructive/10 p-2 text-destructive">
            <AlertTriangle className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-foreground">Supprimer l’exemple</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              Cette action est définitive. L’exemple sera retiré de la base de données et des fichiers d’intentions.
            </p>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-muted/30 p-4 text-sm text-foreground">
          <p className="whitespace-pre-wrap font-medium">{example.text}</p>
          <p className="mt-2 text-xs text-muted-foreground">
            Source : {example.source || '-'} · Statut : {example.enabled ? 'actif' : 'inactif'}
          </p>
        </div>

        {error && <p className="form-error mt-4">{error}</p>}

        <div className="mt-6 flex justify-end gap-2">
          <Button type="button" variant="outline" onClick={onClose} disabled={isDeleting}>
            Annuler
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={onConfirm}
            isLoading={isDeleting}
            className="user-delete-button gap-2"
          >
            <Trash2 className="h-4 w-4" />
            Supprimer définitivement
          </Button>
        </div>
      </div>
    </div>
  );
}

export default function IntentsManager() {
  const [intents, setIntents] = useState<IntentItem[]>([]);
  const [examples, setExamples] = useState<IntentExample[]>([]);
  const [status, setStatus] = useState<IntentStatus | null>(null);
  const [selectedIntent, setSelectedIntent] = useState<string>('');
  const [newExample, setNewExample] = useState('');
  const [testQuery, setTestQuery] = useState('');
  const [testResult, setTestResult] = useState<IntentClassifyResult | null>(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isReloading, setIsReloading] = useState(false);
  const [isSubmittingExample, setIsSubmittingExample] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<IntentExample | null>(null);
  const [deleteError, setDeleteError] = useState('');
  const [isDeletingExample, setIsDeletingExample] = useState(false);

  const selected = useMemo(
    () => intents.find((intent) => intent.name === selectedIntent) ?? intents[0],
    [intents, selectedIntent]
  );

  const stats = useMemo(() => {
    const activeIntents = intents.filter((intent) => intent.enabled).length;
    const totalExamples = intents.reduce((sum, intent) => sum + intent.examples, 0);
    const activeExamples = intents.reduce((sum, intent) => sum + (intent.enabled ? intent.active_examples : 0), 0);
    return { activeIntents, totalExamples, activeExamples };
  }, [intents]);

  const loadIntents = async (preferredIntent?: string) => {
    setIsLoading(true);
    setError('');
    try {
      const [nextIntents, nextStatus] = await Promise.all([listIntents(), getIntentStatus()]);
      setIntents(nextIntents);
      setStatus(nextStatus);
      const nextSelected = preferredIntent || selectedIntent || nextIntents[0]?.name || '';
      setSelectedIntent(nextSelected);
      if (nextSelected) {
        setExamples(await listIntentExamples(nextSelected));
      } else {
        setExamples([]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Impossible de charger les intentions.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void loadIntents();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectIntent = async (intentName: string) => {
    setSelectedIntent(intentName);
    setError('');
    try {
      setExamples(await listIntentExamples(intentName));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Impossible de charger les exemples.');
    }
  };

  const handleSync = async () => {
    setError('');
    try {
      await syncIntents();
      await loadIntents(selected?.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Synchronisation impossible.');
    }
  };

  const handleReload = async () => {
    setIsReloading(true);
    setError('');
    try {
      const result = await reloadIntents();
      setStatus((current) => ({ ...(current ?? { status: 'ok' }), last_reload: result }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Rechargement impossible.');
    } finally {
      setIsReloading(false);
    }
  };

  const handleToggleIntent = async (intent: IntentItem) => {
    setError('');
    try {
      await updateIntent(intent.name, { enabled: !intent.enabled });
      await loadIntents(intent.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Impossible de modifier cette intention.');
    }
  };

  const handleAddExample = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selected || !newExample.trim()) return;
    setIsSubmittingExample(true);
    setError('');
    try {
      await createIntentExample(selected.name, newExample);
      setNewExample('');
      await loadIntents(selected.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Impossible d’ajouter cet exemple.');
    } finally {
      setIsSubmittingExample(false);
    }
  };

  const toggleExample = async (example: IntentExample) => {
    setError('');
    try {
      await updateIntentExample(example.id, { enabled: !example.enabled });
      if (selected) {
        await loadIntents(selected.name);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Impossible de modifier cet exemple.');
    }
  };

  const requestDeleteExample = (example: IntentExample) => {
    setDeleteTarget(example);
    setDeleteError('');
  };

  const closeDeleteModal = () => {
    setDeleteTarget(null);
    setDeleteError('');
  };

  const confirmDeleteExample = async () => {
    if (!deleteTarget) return;

    setIsDeletingExample(true);
    setDeleteError('');
    try {
      await deleteIntentExample(deleteTarget.id);
      if (selected) {
        await loadIntents(selected.name);
      }
      closeDeleteModal();
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : 'Impossible de supprimer cet exemple.');
    } finally {
      setIsDeletingExample(false);
    }
  };

  const handleTest = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!testQuery.trim()) return;
    setIsTesting(true);
    setTestResult(null);
    setError('');
    try {
      setTestResult(await classifyIntent(testQuery));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Test impossible.');
    } finally {
      setIsTesting(false);
    }
  };

  return (
    <section className="space-y-7">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold leading-tight text-foreground">Intentions</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Gérer les exemples utilisés par le classificateur sans changer la logique métier du pipeline.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <Button variant="outline" onClick={() => void handleSync()} disabled={isLoading} className="gap-2">
            <RefreshCw className="h-4 w-4" />
            Synchroniser
          </Button>
          <Button onClick={() => void handleReload()} isLoading={isReloading} className="gap-2">
            <Zap className="h-4 w-4" />
            Recharger
          </Button>
        </div>
      </div>

      <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-4">
        <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
          <div className="flex items-center gap-2 text-sm font-semibold text-primary">
            <Brain className="h-4 w-4" />
            Intentions actives
          </div>
          <p className="mt-4 text-4xl font-bold text-foreground">{numberValue(stats.activeIntents)}</p>
        </div>
        <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
          <div className="flex items-center gap-2 text-sm font-semibold text-muted-foreground">
            <CheckCircle2 className="h-4 w-4" />
            Exemples actifs
          </div>
          <p className="mt-4 text-4xl font-bold text-foreground">{numberValue(stats.activeExamples)}</p>
        </div>
        <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
          <div className="flex items-center gap-2 text-sm font-semibold text-muted-foreground">
            <Activity className="h-4 w-4" />
            Total exemples
          </div>
          <p className="mt-4 text-4xl font-bold text-foreground">{numberValue(stats.totalExamples)}</p>
        </div>
        <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
          <div className="flex items-center gap-2 text-sm font-semibold text-muted-foreground">
            <Zap className="h-4 w-4" />
            Dernier reload
          </div>
          <p className="mt-4 text-2xl font-bold text-foreground">
            {status?.last_reload?.elapsed_ms ? `${status.last_reload.elapsed_ms} ms` : '-'}
          </p>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-3 rounded-lg border border-destructive/20 bg-destructive/10 p-4 text-sm text-destructive">
          <AlertCircle className="h-5 w-5" />
          {error}
        </div>
      )}

      <div className="grid gap-5 xl:grid-cols-[360px_minmax(0,1fr)]">
        <div className="rounded-lg border border-border bg-card p-4 shadow-sm">
          <h3 className="mb-3 text-sm font-bold uppercase tracking-wide text-muted-foreground">Liste des intentions</h3>
          <div className="grid gap-2">
            {intents.map((intent) => (
              <button
                key={intent.id}
                type="button"
                onClick={() => void selectIntent(intent.name)}
                className={cn(
                  'rounded-lg border p-3 text-left transition-colors',
                  selected?.name === intent.name
                    ? 'border-primary bg-primary/5'
                    : 'border-border bg-background hover:border-primary/40 hover:bg-muted/40'
                )}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="truncate font-semibold text-foreground">{intent.label_fr}</p>
                    <p className="mt-0.5 text-xs text-muted-foreground">{intent.name}</p>
                  </div>
                  <span className={cn('rounded-full px-2 py-1 text-xs font-semibold', intent.enabled ? 'bg-emerald-100 text-emerald-700' : 'bg-muted text-muted-foreground')}>
                    {intent.enabled ? 'Actif' : 'Inactif'}
                  </span>
                </div>
                <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
                  <span>{intent.supported ? 'Supportée' : 'Non supportée'}</span>
                  <span>{intent.active_examples}/{intent.examples} exemples</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-5">
          {selected && (
            <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <h3 className="text-xl font-semibold text-foreground">{selected.label_fr}</h3>
                  <p className="mt-1 text-sm text-muted-foreground">{selected.description || selected.name}</p>
                  {!selected.supported && (
                    <p className="mt-2 text-sm font-medium text-amber-700">Cette intention n’est pas encore supportée par le pipeline.</p>
                  )}
                </div>
                <Button variant="outline" onClick={() => void handleToggleIntent(selected)}>
                  {selected.enabled ? 'Désactiver' : 'Activer'}
                </Button>
              </div>

              <form onSubmit={handleAddExample} className="mt-5 grid gap-3">
                <label className="text-sm font-semibold text-foreground">Ajouter un exemple</label>
                <textarea
                  value={newExample}
                  onChange={(event) => setNewExample(event.target.value)}
                  rows={3}
                  className="rounded-lg border border-input bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                  placeholder="Exemple de phrase utilisateur..."
                />
                <div className="flex justify-end">
                  <Button type="submit" isLoading={isSubmittingExample} className="gap-2">
                    <Plus className="h-4 w-4" />
                    Ajouter
                  </Button>
                </div>
              </form>
            </div>
          )}

          <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
            <h3 className="mb-4 text-sm font-bold uppercase tracking-wide text-muted-foreground">Exemples</h3>
            {examples.length === 0 ? (
              <p className="rounded-lg border border-dashed border-border bg-muted/30 p-8 text-center text-sm text-muted-foreground">
                Aucun exemple pour cette intention.
              </p>
            ) : (
              <div className="grid max-h-[420px] gap-2 overflow-y-auto pr-1">
                {examples.map((example) => (
                  <div key={example.id} className="grid gap-3 rounded-lg border border-border bg-background p-3 md:grid-cols-[minmax(0,1fr)_auto_auto] md:items-center">
                    <p className={cn('text-sm text-foreground', !example.enabled && 'text-muted-foreground line-through')}>
                      {example.text}
                    </p>
                    <Button variant="outline" size="sm" onClick={() => void toggleExample(example)}>
                      {example.enabled ? 'Désactiver' : 'Activer'}
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => requestDeleteExample(example)} className="user-delete-button gap-2">
                      <Trash2 className="h-4 w-4" />
                      Supprimer
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
        <h3 className="mb-4 text-sm font-bold uppercase tracking-wide text-muted-foreground">Test rapide</h3>
        <form onSubmit={handleTest} className="grid gap-3 md:grid-cols-[minmax(0,1fr)_auto]">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              value={testQuery}
              onChange={(event) => setTestQuery(event.target.value)}
              className="h-11 w-full rounded-lg border border-input bg-background pl-10 pr-3 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
              placeholder="Tester une phrase..."
            />
          </div>
          <Button type="submit" isLoading={isTesting}>
            Tester
          </Button>
        </form>
        {testResult && (
          <div className="mt-4 rounded-lg border border-border bg-muted/30 p-4 text-sm">
            <div className="flex flex-wrap items-center gap-3">
              <span className="font-semibold text-foreground">Résultat : {testResult.intent || '-'}</span>
              <span className="text-muted-foreground">Confiance : {confidenceValue(testResult.confidence)}</span>
            </div>
            {testResult.reasoning && <p className="mt-2 text-muted-foreground">{testResult.reasoning}</p>}
          </div>
        )}
      </div>

      {deleteTarget && (
        <DeleteIntentExampleModal
          example={deleteTarget}
          error={deleteError}
          isDeleting={isDeletingExample}
          onClose={closeDeleteModal}
          onConfirm={() => void confirmDeleteExample()}
        />
      )}
    </section>
  );
}
