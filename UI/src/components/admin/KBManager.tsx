import {
  AlertCircle,
  AlertTriangle,
  Archive,
  Building2,
  Database,
  FileText,
  FolderTree,
  Layers,
  RefreshCw,
  Trash2,
  UploadCloud,
  Video,
  X,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { useEffect, useState, type FormEvent } from 'react';
import {
  deleteKbSource,
  getKbStats,
  importKbDocument,
  importKbForm,
  importKbVideo,
  listKbSources,
  type KbSource,
  type KbStatsResponse,
  type KbTreeNode,
} from '../../api/admin';
import { cn } from '../../utils/helpers';
import Button from '../ui/Button';

function numberValue(value?: number) {
  return typeof value === 'number' ? value.toLocaleString('fr-FR') : '-';
}

function StatCard({
  label,
  value,
  icon: Icon,
  tone = 'default',
}: {
  label: string;
  value: string | number | undefined;
  icon: LucideIcon;
  tone?: 'default' | 'primary' | 'blue' | 'purple' | 'shared';
}) {
  const toneClass = {
    default: 'text-muted-foreground',
    primary: 'text-primary',
    blue: 'text-blue-600',
    purple: 'text-purple-600',
    shared: 'text-amber-700',
  }[tone];

  return (
    <div className="relative flex min-h-[120px] items-center justify-center rounded-lg border border-border bg-card px-6 py-4 shadow-sm">
      <div className={cn('absolute left-6 top-4 flex items-center gap-2 text-sm font-semibold', toneClass)}>
        <Icon className="h-4 w-4" />
        <span>{label}</span>
      </div>
      <p className="pt-5 text-center text-4xl font-bold leading-none text-foreground">{value ?? '-'}</p>
    </div>
  );
}

function badgeClass(kind?: string) {
  if (kind === 'faq') return 'bg-orange-100 text-orange-700';
  if (kind === 'web') return 'bg-blue-100 text-blue-700';
  if (kind === 'bibliotheque') return 'bg-purple-100 text-purple-700';
  if (kind === 'form') return 'bg-emerald-100 text-emerald-700';
  if (kind === 'video') return 'bg-rose-100 text-rose-700';
  return 'bg-muted text-muted-foreground';
}

function nodeIcon(kind?: string) {
  if (kind === 'form') return FileText;
  if (kind === 'video') return Video;
  if (kind === 'org') return Building2;
  return Archive;
}

function nodeCount(node: KbTreeNode) {
  return node.chunks || node.items || node.documents || 0;
}

function compactCount(value?: number) {
  return typeof value === 'number' && value > 0 ? value.toLocaleString('fr-FR') : '0';
}

function TreeView({ nodes }: { nodes: KbTreeNode[] }) {
  if (nodes.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border bg-muted/30 p-10 text-center text-sm text-muted-foreground">
        Aucun contenu trouvé dans l’arborescence.
      </div>
    );
  }

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      {nodes.map((node) => {
        const children = node.children ?? [];
        return (
          <div key={node.id} className="rounded-lg border border-border bg-background p-3">
            <div className="flex w-full items-center justify-between gap-3 rounded-md px-2 py-2 text-left">
              <span className="flex min-w-0 items-center gap-3 text-base font-semibold text-foreground">
                <Building2 className="h-5 w-5 text-muted-foreground" />
                {node.label}
              </span>
              <span className="flex shrink-0 items-center gap-2 text-xs">
                <span className="rounded-full bg-muted px-2.5 py-1 font-semibold text-muted-foreground">{children.length} dossiers</span>
                {Boolean(node.shared_items) && (
                  <span className="rounded-full bg-amber-100 px-2.5 py-1 font-semibold text-amber-700">
                    {compactCount(node.shared_items)} partagés
                  </span>
                )}
              </span>
            </div>

            <div className="mt-2 grid gap-1">
              {children.map((child) => {
                const Icon = nodeIcon(child.kind);
                return (
                  <div key={child.id} className="grid min-h-10 grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-2 rounded-md px-3 py-1.5 text-sm hover:bg-muted/40">
                    <div className="flex min-w-0 items-center gap-3">
                      <Icon className="h-4 w-4 shrink-0 text-muted-foreground" />
                      <span className="truncate text-foreground">{child.label}</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className={cn('rounded-md px-2 py-1 text-xs font-semibold', badgeClass(child.kind))}>{child.badge || child.kind}</span>
                      {Boolean(child.shared_items) && (
                        <span className="rounded-md bg-amber-100 px-2 py-1 text-xs font-semibold text-amber-700">Partagé</span>
                      )}
                    </div>
                    <span className="rounded-full bg-muted px-2.5 py-1 text-xs font-semibold text-muted-foreground">{numberValue(nodeCount(child))}</span>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function orgLabel(value: KbSource['org']) {
  if (value === 'cnra') return 'CNRA';
  if (value === 'rcar') return 'RCAR';
  return 'Partagé';
}

function ImportDocumentModal({
  error,
  isImporting,
  onClose,
  onSubmit,
}: {
  error: string;
  isImporting: boolean;
  onClose: () => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}) {
  const [importKind, setImportKind] = useState<'document' | 'form' | 'video'>('document');
  const descriptions = {
    document: 'PDF, TXT, Markdown ou HTML. Le document sera découpé en chunks et indexé dans Vespa.',
    form: 'Formulaire indexé dans Vespa avec titre, catégorie, contenu et lien PDF.',
    video: 'Vidéo indexée dans Vespa avec URL YouTube et transcription.',
  };

  return (
    <div className="app-modal-overlay">
      <form onSubmit={onSubmit} className="max-h-[92vh] w-full max-w-2xl overflow-y-auto rounded-lg border border-border bg-card p-6 shadow-xl">
        <div className="mb-5 flex items-start justify-between gap-4">
          <div>
            <h3 className="text-xl font-semibold text-foreground">Importer une source</h3>
            <p className="mt-1 text-sm text-muted-foreground">{descriptions[importKind]}</p>
          </div>
          <button type="button" onClick={onClose} className="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground" aria-label="Fermer">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="grid gap-4">
          {error && <p className="form-error">{error}</p>}
          <label className="grid gap-2 text-sm font-semibold text-foreground">
            Type d’import
            <select
              name="import_kind"
              value={importKind}
              onChange={(event) => setImportKind(event.target.value as 'document' | 'form' | 'video')}
              className="h-11 rounded-lg border border-input bg-background px-3 text-sm font-normal"
            >
              <option value="document">Document</option>
              <option value="form">Formulaire</option>
              <option value="video">Vidéo</option>
            </select>
          </label>
          <label className="grid gap-2 text-sm font-semibold text-foreground">
            Titre
            <input
              name="title"
              required={importKind !== 'document'}
              className="h-11 rounded-lg border border-input bg-background px-3 text-sm font-normal outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
              placeholder={importKind === 'video' ? 'Titre de la vidéo' : importKind === 'form' ? 'Titre du formulaire' : 'Titre du document'}
            />
          </label>
          <label className="grid gap-2 text-sm font-semibold text-foreground">
            Organisme
            <select name="org" defaultValue="both" className="h-11 rounded-lg border border-input bg-background px-3 text-sm font-normal">
              <option value="both">Partagé CNRA / RCAR</option>
              <option value="cnra">CNRA</option>
              <option value="rcar">RCAR</option>
            </select>
          </label>
          {importKind === 'document' && (
            <>
              <label className="grid gap-2 text-sm font-semibold text-foreground">
                Dossier
                <input name="faq_scope" defaultValue="admin" className="h-11 rounded-lg border border-input bg-background px-3 text-sm font-normal outline-none focus:border-primary focus:ring-2 focus:ring-primary/20" placeholder="admin, galerie_documentaire..." />
              </label>
              <label className="grid gap-2 text-sm font-semibold text-foreground">
                Type source
                <select name="source_type" defaultValue="admin_upload" className="h-11 rounded-lg border border-input bg-background px-3 text-sm font-normal">
                  <option value="admin_upload">Import admin</option>
                  <option value="bibliotheque">Bibliothèque</option>
                  <option value="faq">FAQ</option>
                  <option value="web">Web</option>
                </select>
              </label>
              <label className="grid gap-2 text-sm font-semibold text-foreground">
                Fichier
                <input name="file" type="file" accept=".pdf,.txt,.md,.html,.htm" required className="rounded-lg border border-input bg-background px-3 py-2 text-sm font-normal" />
              </label>
            </>
          )}

          {importKind === 'form' && (
            <>
              <label className="grid gap-2 text-sm font-semibold text-foreground">
                Catégorie
                <input name="category" defaultValue="Import admin" className="h-11 rounded-lg border border-input bg-background px-3 text-sm font-normal outline-none focus:border-primary focus:ring-2 focus:ring-primary/20" />
              </label>
              <div className="grid gap-4 md:grid-cols-2">
                <label className="grid gap-2 text-sm font-semibold text-foreground">
                  URL PDF
                  <input name="pdf_url" className="h-11 rounded-lg border border-input bg-background px-3 text-sm font-normal outline-none focus:border-primary focus:ring-2 focus:ring-primary/20" placeholder="https://..." />
                </label>
                <label className="grid gap-2 text-sm font-semibold text-foreground">
                  URL page
                  <input name="page_url" className="h-11 rounded-lg border border-input bg-background px-3 text-sm font-normal outline-none focus:border-primary focus:ring-2 focus:ring-primary/20" placeholder="https://..." />
                </label>
              </div>
              <label className="grid gap-2 text-sm font-semibold text-foreground">
                Contenu du formulaire
                <textarea name="content" rows={5} className="rounded-lg border border-input bg-background px-3 py-2 text-sm font-normal outline-none focus:border-primary focus:ring-2 focus:ring-primary/20" placeholder="Texte extrait ou résumé du formulaire..." />
              </label>
              <label className="grid gap-2 text-sm font-semibold text-foreground">
                Fichier texte/PDF optionnel
                <input name="file" type="file" accept=".pdf,.txt,.md,.html,.htm" className="rounded-lg border border-input bg-background px-3 py-2 text-sm font-normal" />
              </label>
            </>
          )}

          {importKind === 'video' && (
            <>
              <div className="grid gap-4 md:grid-cols-2">
                <label className="grid gap-2 text-sm font-semibold text-foreground">
                  URL YouTube
                  <input name="url" className="h-11 rounded-lg border border-input bg-background px-3 text-sm font-normal outline-none focus:border-primary focus:ring-2 focus:ring-primary/20" placeholder="https://www.youtube.com/watch?v=..." />
                </label>
                <label className="grid gap-2 text-sm font-semibold text-foreground">
                  ID vidéo optionnel
                  <input name="video_id" className="h-11 rounded-lg border border-input bg-background px-3 text-sm font-normal outline-none focus:border-primary focus:ring-2 focus:ring-primary/20" placeholder="ID YouTube" />
                </label>
              </div>
              <label className="grid gap-2 text-sm font-semibold text-foreground">
                Date de publication
                <input name="upload_date" className="h-11 rounded-lg border border-input bg-background px-3 text-sm font-normal outline-none focus:border-primary focus:ring-2 focus:ring-primary/20" placeholder="2026-06-03" />
              </label>
              <label className="grid gap-2 text-sm font-semibold text-foreground">
                Transcription
                <textarea name="transcript" rows={7} required className="rounded-lg border border-input bg-background px-3 py-2 text-sm font-normal outline-none focus:border-primary focus:ring-2 focus:ring-primary/20" placeholder="Transcript complet de la vidéo..." />
              </label>
            </>
          )}
        </div>

        <div className="mt-6 flex justify-end gap-2">
          <Button type="button" variant="outline" onClick={onClose} disabled={isImporting}>
            Annuler
          </Button>
          <Button type="submit" isLoading={isImporting} className="gap-2">
            <UploadCloud className="h-4 w-4" />
            Importer
          </Button>
        </div>
      </form>
    </div>
  );
}

function DeleteSourceModal({
  source,
  error,
  isDeleting,
  onClose,
  onConfirm,
}: {
  source: KbSource;
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
            <h3 className="text-xl font-semibold text-foreground">Supprimer la source</h3>
            <p className="mt-1 text-sm text-muted-foreground">Cette action supprimera les items Vespa créés par cet import.</p>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-muted/30 p-4 text-sm text-foreground">
          <p className="font-semibold">{source.title}</p>
          <p className="mt-1 text-muted-foreground">
            {orgLabel(source.org)} - {source.items_count} item{source.items_count > 1 ? 's' : ''} Vespa
          </p>
        </div>

        {error && <p className="form-error mt-4">{error}</p>}

        <div className="mt-6 flex justify-end gap-2">
          <Button type="button" variant="outline" onClick={onClose} disabled={isDeleting}>
            Annuler
          </Button>
          <Button type="button" variant="outline" onClick={onConfirm} isLoading={isDeleting} className="user-delete-button">
            <Trash2 className="h-4 w-4" />
            Supprimer définitivement
          </Button>
        </div>
      </div>
    </div>
  );
}

export default function KBManager() {
  const [data, setData] = useState<KbStatsResponse | null>(null);
  const [sources, setSources] = useState<KbSource[]>([]);
  const [isImportModalOpen, setIsImportModalOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<KbSource | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState('');
  const [importError, setImportError] = useState('');
  const [deleteError, setDeleteError] = useState('');

  const loadData = async () => {
    setIsLoading(true);
    setError('');
    try {
      const [payload, nextSources] = await Promise.all([getKbStats(), listKbSources()]);
      setData(payload);
      setSources(nextSources);
      if (payload.status !== 'ok') {
        setError(payload.error || 'Base de connaissances indisponible.');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Impossible de charger la base de connaissances.');
      setData(null);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void loadData();
  }, []);

  const stats = data?.stats ?? {};
  const tree = data?.tree ?? [];

  const handleImportDocument = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setImportError('');
    setIsImporting(true);
    try {
      const form = new FormData(event.currentTarget);
      const importKind = String(form.get('import_kind') || 'document');
      if (importKind === 'form') {
        await importKbForm(form);
      } else if (importKind === 'video') {
        await importKbVideo(form);
      } else {
        await importKbDocument(form);
      }
      setIsImportModalOpen(false);
      await loadData();
    } catch (err) {
      setImportError(err instanceof Error ? err.message : 'Impossible d’importer ce document.');
    } finally {
      setIsImporting(false);
    }
  };

  const requestDelete = (source: KbSource) => {
    setDeleteTarget(source);
    setDeleteError('');
  };

  const closeDeleteModal = () => {
    setDeleteTarget(null);
    setDeleteError('');
  };

  const confirmDelete = async () => {
    if (!deleteTarget) return;
    setIsDeleting(true);
    setDeleteError('');
    try {
      await deleteKbSource(deleteTarget.id);
      closeDeleteModal();
      await loadData();
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : 'Impossible de supprimer cette source.');
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <section className="space-y-7">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold leading-tight text-foreground">Base de connaissances</h2>
          <p className="mt-1 text-sm text-muted-foreground">Consulter les statistiques et l’organisation des données indexées dans Vespa</p>
        </div>
        <div className="flex flex-wrap gap-3">
          <Button variant="outline" onClick={() => void loadData()} disabled={isLoading} className="gap-2">
            <RefreshCw className="h-4 w-4" />
            Actualiser
          </Button>
          <Button className="gap-2" onClick={() => setIsImportModalOpen(true)}>
            <UploadCloud className="h-4 w-4" />
            Importer
          </Button>
        </div>
      </div>

      <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
        <StatCard label="Organismes" value={numberValue(stats.organisms)} icon={Building2} />
        <StatCard label="Documents ingérés" value={numberValue(stats.source_documents)} icon={FileText} tone="primary" />
        <StatCard label="Éléments partagés" value={numberValue(stats.shared_items)} icon={Layers} tone="shared" />
        <StatCard label="Chunks docs" value={numberValue(stats.doc_chunks)} icon={Database} tone="blue" />
        <StatCard label="Formulaires" value={numberValue(stats.forms)} icon={FileText} tone="primary" />
        <StatCard label="Vidéos" value={numberValue(stats.videos)} icon={Video} tone="blue" />
        <StatCard label="Unités indexées" value={numberValue(stats.indexed_units)} icon={Layers} />
        <StatCard label="Dossiers" value={numberValue(stats.buckets)} icon={FolderTree} tone="purple" />
        <StatCard label="Type dominant" value={stats.top_source_type || '-'} icon={Archive} />
      </div>

      {error && (
        <div className="flex items-center gap-3 rounded-lg border border-destructive/20 bg-destructive/10 p-4 text-sm text-destructive">
          <AlertCircle className="h-5 w-5" />
          {error}
        </div>
      )}

      {data?.sampled && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
          L’arborescence affiche un échantillon limité. Pour une base plus volumineuse, il faudra ajouter une pagination Vespa dédiée.
        </div>
      )}

      <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="text-sm font-bold uppercase tracking-wide text-muted-foreground">Arborescence</h3>
          <span className="text-xs font-medium text-muted-foreground">Les contenus partagés sont comptés dans CNRA et RCAR</span>
        </div>

        {isLoading ? (
          <p className="rounded-lg border border-border bg-muted/30 p-6 text-sm text-muted-foreground">Chargement de la base de connaissances...</p>
        ) : (
          <TreeView nodes={tree} />
        )}
      </div>

      <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
        <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold uppercase tracking-wide text-muted-foreground">Sources importées</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              Documents ajoutés depuis ce panneau. La suppression retire aussi les chunks Vespa créés par l’import.
            </p>
          </div>
          <span className="rounded-full bg-muted px-3 py-1 text-sm font-semibold text-muted-foreground">
            {sources.length} source{sources.length > 1 ? 's' : ''}
          </span>
        </div>

        {sources.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border bg-muted/30 p-8 text-center text-sm text-muted-foreground">
            Aucun import admin pour l’instant.
          </div>
        ) : (
          <div className="grid gap-2">
            {sources.map((source) => (
              <div
                key={source.id}
                className="grid gap-3 rounded-lg border border-border bg-background p-4 shadow-sm md:grid-cols-[minmax(0,1fr)_auto_auto_auto] md:items-center"
              >
                <div className="min-w-0">
                  <p className="truncate text-base font-semibold text-foreground" title={source.title}>
                    {source.title}
                  </p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {orgLabel(source.org)} - {source.type} - {source.source_type} - {source.items_count} item
                    {source.items_count > 1 ? 's' : ''} Vespa
                  </p>
                  {source.error && <p className="mt-2 text-sm font-medium text-destructive">{source.error}</p>}
                </div>
                <span
                  className={cn(
                    'w-fit rounded-full px-3 py-1 text-xs font-semibold',
                    source.status === 'indexed' && 'bg-emerald-100 text-emerald-700',
                    source.status === 'processing' && 'bg-blue-100 text-blue-700',
                    source.status === 'error' && 'bg-destructive/10 text-destructive',
                    source.status !== 'indexed' &&
                      source.status !== 'processing' &&
                      source.status !== 'error' &&
                      'bg-muted text-muted-foreground'
                  )}
                >
                  {source.status}
                </span>
                <span className="text-sm text-muted-foreground">
                  {new Date(source.created_at).toLocaleDateString('fr-FR')}
                </span>
                <Button variant="outline" size="sm" onClick={() => requestDelete(source)} className="user-delete-button gap-2">
                  <Trash2 className="h-4 w-4" />
                  Supprimer
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>

      {isImportModalOpen && (
        <ImportDocumentModal
          error={importError}
          isImporting={isImporting}
          onClose={() => setIsImportModalOpen(false)}
          onSubmit={(event) => void handleImportDocument(event)}
        />
      )}

      {deleteTarget && (
        <DeleteSourceModal
          source={deleteTarget}
          error={deleteError}
          isDeleting={isDeleting}
          onClose={closeDeleteModal}
          onConfirm={() => void confirmDelete()}
        />
      )}
    </section>
  );
}
