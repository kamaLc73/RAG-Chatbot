import {
  AlertCircle,
  AlertTriangle,
  CalendarDays,
  ChevronDown,
  Clock,
  Database,
  Filter,
  MessageSquare,
  RefreshCw,
  Search,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  TrendingUp,
  UserRound,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { deleteAdminConversation, listAdminConversations, listUsers } from '../../api/admin';
import type { ChatMessage, Conversation, Organization, User } from '../../api/types';
import { cn, formatConversationTitle } from '../../utils/helpers';
import FormCard from '../chat/FormCard';
import VideoCard from '../chat/VideoCard';
import Button from '../ui/Button';

interface ConversationsManagerProps {
  selectedUser?: User | null;
  onClearSelectedUser?: () => void;
}

type FeedbackFilter = 'all' | 'positive' | 'negative' | 'none';
type SortMode = 'recent' | 'oldest' | 'most-messages' | 'highest-latency';
type TypeFilter = 'all' | Organization;

function formatDate(value?: string) {
  if (!value) return '-';

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '-';

  return date.toLocaleString('fr-FR', {
    dateStyle: 'medium',
    timeStyle: 'short',
  });
}

function formatLatency(value?: number | null) {
  if (value == null) return '-';
  if (value < 1) return `${Math.round(value * 1000)} ms`;
  return `${value.toFixed(2)} s`;
}

function conversationUserLabel(conversation: Conversation) {
  return conversation.user_name || conversation.username || conversation.user_email || 'Utilisateur supprimé';
}

function orgLabel(org: Organization | string) {
  return org === 'CNRA & RCAR' ? 'Les deux' : org;
}

function messageFeedback(message: ChatMessage) {
  if (message.feedback === 1) return 'positive';
  if (message.feedback === -1) return 'negative';
  return null;
}

function chunkText(chunk: NonNullable<ChatMessage['chunks']>[number]) {
  return String(chunk.text || '').trim();
}

function chunkTitle(chunk: NonNullable<ChatMessage['chunks']>[number]) {
  const metadata = chunk.metadata || {};
  return String(metadata.title || metadata.source || metadata.url || 'Chunk documentaire');
}

function orgBadgeClass(org: Organization | string) {
  if (org === 'RCAR') return 'border-blue-200 bg-blue-50 text-blue-700';
  if (org === 'CNRA') return 'border-primary/20 bg-primary/10 text-primary';
  return 'border-slate-200 bg-slate-100 text-slate-700';
}

function StatCard({
  label,
  value,
  icon: Icon,
  tone = 'default',
}: {
  label: string;
  value: number | string;
  icon: LucideIcon;
  tone?: 'default' | 'rcar' | 'cnra' | 'positive' | 'negative';
}) {
  const toneClass = {
    default: 'text-muted-foreground',
    rcar: 'text-blue-600',
    cnra: 'text-primary',
    positive: 'text-primary',
    negative: 'text-destructive',
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

function MetricPill({ label, value, icon: Icon }: { label: string; value: string | number; icon: LucideIcon }) {
  return (
    <div className="relative flex min-h-[76px] items-center justify-center rounded-lg border border-border bg-background p-3">
      <div className="absolute left-3 top-3 flex items-center gap-2 text-xs font-semibold text-muted-foreground">
        <Icon className="h-3.5 w-3.5" />
        {label}
      </div>
      <p className="pt-4 text-center text-xl font-bold leading-none text-foreground">{value}</p>
    </div>
  );
}

function ConversationSnippet({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';

  return (
    <div className={cn('rounded-lg p-4', isUser ? 'bg-primary/5' : 'bg-muted/35')}>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span
          className={cn(
            'flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-bold',
            isUser ? 'bg-primary text-primary-foreground' : 'bg-accent text-accent-foreground'
          )}
        >
          {isUser ? 'Q' : 'R'}
        </span>
        <span className="text-sm font-semibold text-foreground">{isUser ? 'Question' : 'Réponse'}</span>
        <span className="text-xs text-muted-foreground">{formatDate(message.created_at)}</span>
      </div>
      <p className="line-clamp-3 whitespace-pre-wrap text-sm leading-6 text-foreground">{message.content}</p>
    </div>
  );
}

function ChunkPreview({
  chunk,
  index,
}: {
  chunk: NonNullable<ChatMessage['chunks']>[number];
  index: number;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [canExpand, setCanExpand] = useState(false);
  const textRef = useRef<HTMLParagraphElement | null>(null);
  const text = chunkText(chunk);

  useLayoutEffect(() => {
    if (isExpanded) return;

    const element = textRef.current;
    if (!element) return;

    const measure = () => {
      setCanExpand(element.scrollHeight > element.clientHeight + 1);
    };

    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [isExpanded, text]);

  return (
    <div className="rounded-md border border-border bg-card p-3">
      <p className="mb-1 truncate text-xs font-semibold text-foreground">
        Chunk {index + 1} - {chunkTitle(chunk)}
      </p>
      <p ref={textRef} className={cn('text-xs leading-5 text-muted-foreground', !isExpanded && 'line-clamp-3')}>
        {text || 'Contenu du chunk non disponible.'}
      </p>
      {canExpand && (
        <button
          type="button"
          onClick={() => setIsExpanded((current) => !current)}
          className="mt-2 text-xs font-semibold text-primary hover:underline"
        >
          {isExpanded ? 'Voir moins' : 'Voir plus'}
        </button>
      )}
    </div>
  );
}

function MessageResources({ message }: { message: ChatMessage }) {
  const resources = message.resources ?? [];
  if (resources.length === 0) return null;

  const videos = resources.filter((resource) => resource.type === 'video');
  const forms = resources.filter((resource) => resource.type === 'form');
  const others = resources.filter((resource) => resource.type !== 'video' && resource.type !== 'form');

  return (
    <div className="mt-4 rounded-lg border border-border bg-muted/20 p-3">
      <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Ressources suggérées</p>
      <div className="grid gap-3 md:grid-cols-2">
        {videos.map((resource, index) => (
          <VideoCard key={`video-${resource.id || resource.url || index}`} resource={resource} />
        ))}
        {forms.map((resource, index) => (
          <FormCard key={`form-${resource.id || resource.url || resource.pdf_url || index}`} resource={resource} />
        ))}
        {others.map((resource, index) => (
          <a
            key={`resource-${resource.id || resource.url || index}`}
            className="resource-card"
            href={resource.url || resource.page_url || resource.pdf_url || '#'}
            target="_blank"
            rel="noreferrer"
          >
            <Database className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
            <div>
              <strong>{resource.title}</strong>
              <p>{resource.description || resource.source || 'Ressource suggérée'}</p>
            </div>
          </a>
        ))}
      </div>
    </div>
  );
}

function MessageDetail({ message, showSources }: { message: ChatMessage; showSources: boolean }) {
  const isUser = message.role === 'user';
  const chunks = message.chunks ?? [];
  const feedback = messageFeedback(message);

  return (
    <div className="rounded-lg border border-border bg-background p-4">
      <div className="flex gap-3">
        <div
          className={cn(
            'flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-sm font-bold',
            isUser ? 'bg-primary text-primary-foreground' : 'bg-accent text-accent-foreground'
          )}
        >
          {isUser ? 'Q' : 'R'}
        </div>
        <div className="min-w-0 flex-1">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-foreground">{isUser ? 'Question' : 'Réponse'}</span>
              <span className="text-xs text-muted-foreground">{formatDate(message.created_at)}</span>
            </div>
            {!isUser && (
              <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                <span className="inline-flex items-center gap-1">
                  <Clock className="h-3.5 w-3.5" />
                  {formatLatency(message.latency_seconds)}
                </span>
                <span>{message.context_docs ?? chunks.length ?? 0} chunks</span>
                {feedback === 'positive' && (
                  <span className="inline-flex items-center gap-1 text-primary">
                    <ThumbsUp className="h-3.5 w-3.5" />
                    Positif
                  </span>
                )}
                {feedback === 'negative' && (
                  <span className="inline-flex items-center gap-1 text-destructive">
                    <ThumbsDown className="h-3.5 w-3.5" />
                    Négatif
                  </span>
                )}
              </div>
            )}
          </div>

          <p className="whitespace-pre-wrap text-sm leading-6 text-foreground">{message.content}</p>

          {!isUser && <MessageResources message={message} />}

          {!isUser && showSources && chunks.length > 0 && (
            <div className="mt-4 rounded-lg border border-dashed border-border bg-muted/20 p-3">
              <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Chunks récupérés</p>
              <div className="grid gap-2">
                {chunks.map((chunk, index) => (
                  <ChunkPreview key={`${message.id}-chunk-${index}`} chunk={chunk} index={index} />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function DeleteConversationModal({
  conversation,
  error,
  isDeleting,
  onClose,
  onConfirm,
}: {
  conversation: Conversation;
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
            <h3 className="text-xl font-semibold text-foreground">Supprimer la conversation</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              Cette action est définitive. La conversation et ses messages seront supprimés de la base de données.
            </p>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-muted/30 p-4 text-sm text-foreground">
          <p className="font-semibold">{formatConversationTitle(conversation.title, conversation.organization)}</p>
          <p className="mt-1 text-muted-foreground">{conversationUserLabel(conversation)}</p>
          <p className="mt-1 text-xs text-muted-foreground">Dernière activité : {formatDate(conversation.updated_at)}</p>
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
            className="user-delete-button"
          >
            <Trash2 className="h-4 w-4" />
            Supprimer définitivement
          </Button>
        </div>
      </div>
    </div>
  );
}

function ConversationCard({
  conversation,
  showSources,
  onDelete,
}: {
  conversation: Conversation;
  showSources: boolean;
  onDelete: (conversation: Conversation) => void;
}) {
  const messages = conversation.messages ?? [];
  const firstQuestion = messages.find((message) => message.role === 'user');
  const firstAnswer = messages.find((message) => message.role === 'assistant');
  const previewMessages = [firstQuestion, firstAnswer].filter((message): message is ChatMessage => Boolean(message));
  const [isExpanded, setIsExpanded] = useState(false);
  const positiveFeedback =
    conversation.positive_feedback ?? messages.filter((message) => message.feedback === 1).length;
  const negativeFeedback =
    conversation.negative_feedback ?? messages.filter((message) => message.feedback === -1).length;
  const feedbackTotal = conversation.feedback_count ?? positiveFeedback + negativeFeedback;
  const contextDocs =
    conversation.total_context_docs ??
    messages.reduce((total, message) => total + (message.context_docs ?? message.chunks?.length ?? 0), 0);

  return (
    <article className="overflow-hidden rounded-lg border border-border bg-card shadow-sm">
      <div className="border-b border-border px-5 py-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0 space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="truncate text-lg font-semibold text-foreground">
                {formatConversationTitle(conversation.title, conversation.organization)}
              </h3>
              <span className={cn('rounded-full border px-2.5 py-1 text-xs font-semibold', orgBadgeClass(conversation.organization))}>
                {orgLabel(conversation.organization)}
              </span>
            </div>
            <div className="flex flex-wrap gap-x-5 gap-y-2 text-sm text-muted-foreground">
              <span className="inline-flex items-center gap-1.5">
                <UserRound className="h-4 w-4" />
                {conversationUserLabel(conversation)}
              </span>
              <span className="inline-flex items-center gap-1.5">
                <CalendarDays className="h-4 w-4" />
                {formatDate(conversation.updated_at)}
              </span>
              <span>ID : {conversation.id}</span>
            </div>
            {conversation.user_email && <p className="text-xs text-muted-foreground">{conversation.user_email}</p>}
          </div>
          <Button variant="outline" size="sm" onClick={() => onDelete(conversation)} className="user-delete-button shrink-0 gap-2">
            <Trash2 className="h-4 w-4" />
            Supprimer
          </Button>
        </div>
      </div>

      <div className="grid gap-4 p-5 lg:grid-cols-[minmax(0,1fr)_340px] lg:items-start">
        <div className="grid gap-3">
          {isExpanded ? (
            messages.length > 0 ? (
              messages.map((message) => <MessageDetail key={message.id} message={message} showSources={showSources} />)
            ) : (
              <p className="rounded-lg border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
                Aucun message enregistré pour cette conversation.
              </p>
            )
          ) : previewMessages.length > 0 ? (
            previewMessages.map((message) => <ConversationSnippet key={message.id} message={message} />)
          ) : (
            <p className="rounded-lg border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
              Aucun message enregistré pour cette conversation.
            </p>
          )}
        </div>

        <div className="grid gap-3 rounded-lg border border-border bg-muted/20 p-3 sm:grid-cols-2">
          <MetricPill label="Questions" value={conversation.question_count ?? messages.filter((message) => message.role === 'user').length} icon={MessageSquare} />
          <MetricPill label="Réponses" value={conversation.answer_count ?? messages.filter((message) => message.role === 'assistant').length} icon={MessageSquare} />
          <MetricPill label="Feedbacks" value={feedbackTotal} icon={TrendingUp} />
          <MetricPill label="Chunks" value={contextDocs} icon={Database} />
          <div className="sm:col-span-2">
            <MetricPill label="Latence moy." value={formatLatency(conversation.avg_latency_seconds)} icon={Clock} />
          </div>
        </div>
      </div>

      <button
        type="button"
        onClick={() => setIsExpanded((current) => !current)}
        className="flex w-full items-center justify-between gap-3 border-t border-border px-5 py-3 text-left text-sm font-semibold text-primary transition-colors hover:bg-accent/40"
      >
        <span>{isExpanded ? 'Voir moins' : 'Voir toute la conversation'}</span>
        <ChevronDown className={cn('h-4 w-4 shrink-0 transition-transform', isExpanded && 'rotate-180')} />
      </button>
    </article>
  );
}

export default function ConversationsManager({ selectedUser, onClearSelectedUser }: ConversationsManagerProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [query, setQuery] = useState('');
  const [userFilter, setUserFilter] = useState('');
  const [typeFilter, setTypeFilter] = useState<TypeFilter>('all');
  const [feedbackFilter, setFeedbackFilter] = useState<FeedbackFilter>('all');
  const [sortMode, setSortMode] = useState<SortMode>('recent');
  const [showSources, setShowSources] = useState(true);
  const [deleteTarget, setDeleteTarget] = useState<Conversation | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState('');
  const [deleteError, setDeleteError] = useState('');
  const conversationsListRef = useRef<HTMLDivElement | null>(null);

  const loadData = async () => {
    setIsLoading(true);
    setError('');
    try {
      const [nextConversations, nextUsers] = await Promise.all([listAdminConversations(), listUsers()]);
      setConversations(nextConversations);
      setUsers(nextUsers);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Impossible de charger les conversations.');
      setConversations([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void loadData();
  }, []);

  useEffect(() => {
    if (selectedUser) setUserFilter(String(selectedUser.id));
  }, [selectedUser]);

  const stats = useMemo(() => {
    const allMessages = conversations.flatMap((conversation) => conversation.messages ?? []);
    const assistantMessages = allMessages.filter((message) => message.role === 'assistant');
    const feedbackMessages = assistantMessages.filter((message) => message.feedback != null);
    const latencies = assistantMessages
      .map((message) => message.latency_seconds)
      .filter((value): value is number => typeof value === 'number');
    const totalChunks = conversations.reduce((total, conversation) => {
      if (typeof conversation.total_context_docs === 'number') {
        return total + conversation.total_context_docs;
      }
      return (
        total +
        (conversation.messages ?? []).reduce(
          (messageTotal, message) => messageTotal + (message.context_docs ?? message.chunks?.length ?? 0),
          0
        )
      );
    }, 0);

    return {
      both: conversations.filter((conversation) => conversation.organization === 'CNRA & RCAR').length,
      rcar: conversations.filter((conversation) => conversation.organization === 'RCAR').length,
      cnra: conversations.filter((conversation) => conversation.organization === 'CNRA').length,
      feedbacks: feedbackMessages.length,
      positive: feedbackMessages.filter((message) => message.feedback === 1).length,
      negative: feedbackMessages.filter((message) => message.feedback === -1).length,
      avgLatency: latencies.length ? formatLatency(latencies.reduce((sum, value) => sum + value, 0) / latencies.length) : '-',
      totalChunks,
      messages: conversations.reduce((total, conversation) => total + (conversation.message_count ?? conversation.messages?.length ?? 0), 0),
    };
  }, [conversations]);

  const visibleConversations = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = conversations.filter((conversation) => {
      const messages = conversation.messages ?? [];
      const matchesSelectedUser = userFilter ? String(conversation.user_id ?? '') === userFilter : true;
      const matchesType = typeFilter === 'all' ? true : conversation.organization === typeFilter;
      const matchesSearch = needle
        ? [
            conversation.id,
            conversation.title,
            conversationUserLabel(conversation),
            conversation.username,
            conversation.user_email,
            ...messages.map((message) => message.content),
          ]
            .filter(Boolean)
            .some((value) => String(value).toLowerCase().includes(needle))
        : true;
      const feedbacks = messages.map(messageFeedback).filter((value): value is 'positive' | 'negative' => value != null);
      const matchesFeedback =
        feedbackFilter === 'all' ||
        (feedbackFilter === 'none' && feedbacks.length === 0) ||
        (feedbackFilter !== 'none' && feedbacks.includes(feedbackFilter));

      return matchesSelectedUser && matchesType && matchesSearch && matchesFeedback;
    });

    return filtered.sort((a, b) => {
      if (sortMode === 'oldest') return new Date(a.updated_at).getTime() - new Date(b.updated_at).getTime();
      if (sortMode === 'most-messages') return (b.message_count ?? 0) - (a.message_count ?? 0);
      if (sortMode === 'highest-latency') return (b.avg_latency_seconds ?? 0) - (a.avg_latency_seconds ?? 0);
      return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
    });
  }, [conversations, feedbackFilter, query, sortMode, typeFilter, userFilter]);

  useEffect(() => {
    if (!selectedUser) return;

    const timeoutId = window.setTimeout(() => {
      conversationsListRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 120);

    return () => window.clearTimeout(timeoutId);
  }, [selectedUser, visibleConversations.length]);

  const clearUserFilter = () => {
    setUserFilter('');
    onClearSelectedUser?.();
  };

  const requestDelete = (conversation: Conversation) => {
    setDeleteTarget(conversation);
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
      await deleteAdminConversation(deleteTarget.id);
      setConversations((current) => current.filter((item) => item.id !== deleteTarget.id));
      closeDeleteModal();
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : 'Impossible de supprimer cette conversation.');
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <section className="space-y-7">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold leading-tight text-foreground">Conversations</h2>
          <p className="mt-1 text-sm text-muted-foreground">Voir et gérer les conversations des utilisateurs avec les retours</p>
        </div>
        <Button variant="outline" onClick={() => void loadData()} disabled={isLoading} className="gap-2">
          <RefreshCw className="h-4 w-4" />
          Actualiser
        </Button>
      </div>

      <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
        <StatCard label="Les deux" value={stats.both} icon={MessageSquare} />
        <StatCard label="Feedback" value={stats.feedbacks} icon={TrendingUp} />
        <StatCard label="Latence moy." value={stats.avgLatency} icon={Clock} />
        <StatCard label="RCAR" value={stats.rcar} icon={MessageSquare} tone="rcar" />
        <StatCard label="Positive" value={stats.positive} icon={ThumbsUp} tone="positive" />
        <StatCard label="Total chunks" value={stats.totalChunks} icon={Database} />
        <StatCard label="CNRA" value={stats.cnra} icon={MessageSquare} tone="cnra" />
        <StatCard label="Négative" value={stats.negative} icon={ThumbsDown} tone="negative" />
        <StatCard label="Messages" value={stats.messages} icon={MessageSquare} />
      </div>

      <div className="relative">
        <Search className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground" />
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Rechercher par ID de conversation ou contenu de message..."
          className="h-12 w-full rounded-lg border border-input bg-background pl-12 pr-4 text-sm outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/20"
        />
      </div>

      <div className="rounded-lg border border-border bg-card p-5 shadow-sm">
        <div className="mb-4 flex items-center gap-2 font-semibold text-foreground">
          <Filter className="h-5 w-5 text-muted-foreground" />
          Filtres
        </div>
        <label className="mb-4 flex items-center gap-2 border-b border-border pb-4 text-sm text-foreground">
          <input type="checkbox" checked={showSources} onChange={(event) => setShowSources(event.target.checked)} className="h-4 w-4 accent-primary" />
          Afficher les sources documentaires
        </label>
        <div className="grid gap-5 md:grid-cols-2">
          <label className="grid gap-2 text-sm font-semibold text-foreground">
            Utilisateur
            <select value={userFilter} onChange={(event) => setUserFilter(event.target.value)} className="h-12 rounded-lg border border-input bg-background px-3 text-sm font-normal">
              <option value="">Tous les utilisateurs</option>
              {users.map((user) => (
                <option key={user.id} value={user.id}>
                  {user.name || user.username || user.email}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-2 text-sm font-semibold text-foreground">
            Type
            <select value={typeFilter} onChange={(event) => setTypeFilter(event.target.value as TypeFilter)} className="h-12 rounded-lg border border-input bg-background px-3 text-sm font-normal">
              <option value="all">Tous les types</option>
              <option value="CNRA & RCAR">Les deux</option>
              <option value="RCAR">RCAR</option>
              <option value="CNRA">CNRA</option>
            </select>
          </label>
          <label className="grid gap-2 text-sm font-semibold text-foreground">
            Feedback
            <select value={feedbackFilter} onChange={(event) => setFeedbackFilter(event.target.value as FeedbackFilter)} className="h-12 rounded-lg border border-input bg-background px-3 text-sm font-normal">
              <option value="all">Tous</option>
              <option value="positive">Positif</option>
              <option value="negative">Négatif</option>
              <option value="none">Sans feedback</option>
            </select>
          </label>
          <label className="grid gap-2 text-sm font-semibold text-foreground">
            Trier par
            <select value={sortMode} onChange={(event) => setSortMode(event.target.value as SortMode)} className="h-12 rounded-lg border border-input bg-background px-3 text-sm font-normal">
              <option value="recent">Plus récentes</option>
              <option value="oldest">Plus anciennes</option>
              <option value="most-messages">Plus de messages</option>
              <option value="highest-latency">Latence élevée</option>
            </select>
          </label>
        </div>
        {selectedUser && (
          <Button variant="outline" size="sm" onClick={clearUserFilter} className="mt-4">
            Voir toutes les conversations
          </Button>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-3 rounded-lg border border-destructive/20 bg-destructive/10 p-4 text-sm text-destructive">
          <AlertCircle className="h-5 w-5" />
          {error}
        </div>
      )}

      <div ref={conversationsListRef} className="scroll-mt-4 space-y-4">
        {isLoading && <p className="rounded-lg border border-border bg-card p-6 text-sm text-muted-foreground">Chargement des conversations...</p>}
        {!isLoading && visibleConversations.length === 0 && (
          <p className="rounded-lg border border-dashed border-border bg-muted/30 p-10 text-center text-sm text-muted-foreground">
            Aucune conversation trouvée.
          </p>
        )}
        {visibleConversations.map((conversation) => (
          <ConversationCard
            key={conversation.id}
            conversation={conversation}
            showSources={showSources}
            onDelete={requestDelete}
          />
        ))}
      </div>

      <p className="pt-2 text-center text-sm text-muted-foreground">
        Affichage de {visibleConversations.length} sur {conversations.length} conversation{conversations.length > 1 ? 's' : ''}
      </p>

      {deleteTarget && (
        <DeleteConversationModal
          conversation={deleteTarget}
          error={deleteError}
          isDeleting={isDeleting}
          onClose={closeDeleteModal}
          onConfirm={() => void confirmDelete()}
        />
      )}
    </section>
  );
}

