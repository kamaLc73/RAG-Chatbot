import { apiRequest } from './client';
import type { AdminStats, Conversation, User } from './types';

export async function getAdminStats() {
  return apiRequest<AdminStats>('/admin/stats');
}

export async function listUsers() {
  return apiRequest<User[]>('/admin/users');
}

export async function createUser(payload: {
  username: string;
  email: string;
  password: string;
  full_name?: string | null;
  is_superuser: boolean;
}) {
  return apiRequest<User>('/admin/users', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function updateUser(id: string, payload: {
  username?: string;
  email?: string;
  password?: string;
  full_name?: string | null;
  is_superuser?: boolean;
}) {
  return apiRequest<User>(`/admin/users/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(payload),
  });
}

export async function updateUserStatus(id: string, is_active: boolean) {
  return apiRequest<User>(`/admin/users/${id}`, {
    method: 'PATCH',
    body: JSON.stringify({ is_active }),
  });
}

export async function deleteUser(id: string) {
  return apiRequest<void>(`/admin/users/${id}`, {
    method: 'DELETE',
  });
}

export async function listAdminConversations() {
  return apiRequest<Conversation[]>('/admin/conversations');
}

export async function deleteAdminConversation(id: string) {
  return apiRequest<void>(`/admin/conversations/${id}`, {
    method: 'DELETE',
  });
}

export async function listKbItems() {
  return apiRequest<Array<{ id: string; title: string; source: string; status: string }>>('/admin/kb');
}

export interface KbTreeNode {
  id: string;
  label: string;
  kind: string;
  badge?: string;
  chunks?: number;
  documents?: number;
  items?: number;
  shared_items?: number;
  children?: KbTreeNode[];
}

export interface KbStatsResponse {
  status: string;
  error?: string;
  schema_counts?: {
    doc_chunks?: number;
    forms?: number;
    videos?: number;
  };
  stats?: {
    organisms?: number;
    source_documents?: number;
    doc_chunks?: number;
    forms?: number;
    videos?: number;
    indexed_units?: number;
    shared_items?: number;
    buckets?: number;
    top_source_type?: string;
  };
  source_types?: Record<string, number>;
  tree?: KbTreeNode[];
  sampled?: boolean;
}

export async function getKbStats() {
  return apiRequest<KbStatsResponse>('/admin/kb/stats');
}

export interface KbSource {
  id: string;
  type: 'document' | 'form' | 'video';
  title: string;
  org: 'cnra' | 'rcar' | 'both';
  source_type: string;
  source_path?: string | null;
  source_url?: string | null;
  status: string;
  error?: string | null;
  items_count: number;
  created_at: string;
  updated_at: string;
}

export async function listKbSources() {
  return apiRequest<KbSource[]>('/admin/kb/sources');
}

export async function importKbDocument(formData: FormData) {
  return apiRequest<KbSource>('/admin/kb/import/document', {
    method: 'POST',
    body: formData,
  });
}

export async function importKbForm(formData: FormData) {
  return apiRequest<KbSource>('/admin/kb/import/form', {
    method: 'POST',
    body: formData,
  });
}

export async function importKbVideo(formData: FormData) {
  return apiRequest<KbSource>('/admin/kb/import/video', {
    method: 'POST',
    body: formData,
  });
}

export async function deleteKbSource(id: string) {
  return apiRequest<void>(`/admin/kb/sources/${id}`, {
    method: 'DELETE',
  });
}

export interface IntentItem {
  id: string;
  name: string;
  label_fr: string;
  description?: string | null;
  enabled: boolean;
  builtin: boolean;
  supported: boolean;
  examples: number;
  active_examples: number;
  created_at: string;
  updated_at: string;
}

export interface IntentExample {
  id: string;
  intent_id: string;
  text: string;
  enabled: boolean;
  source: string;
  created_at: string;
  updated_at: string;
}

export interface IntentStatus {
  status: string;
  runtime?: Record<string, unknown>;
  database?: { total: number; active: number };
  last_reload?: {
    status: string;
    elapsed_ms: number;
    examples: number;
    intents: number;
    loaded: boolean;
  } | null;
}

export interface IntentClassifyResult {
  status: string;
  intent?: string;
  confidence?: number;
  reasoning?: string;
  loaded?: boolean;
  error?: string;
}

export async function listIntents() {
  return apiRequest<IntentItem[]>('/admin/intents');
}

export async function getIntentStatus() {
  return apiRequest<IntentStatus>('/admin/intents/status');
}

export async function syncIntents() {
  return apiRequest<{ status: string; message: string }>('/admin/intents/sync', {
    method: 'POST',
  });
}

export async function reloadIntents() {
  return apiRequest<NonNullable<IntentStatus['last_reload']>>('/admin/intents/reload', {
    method: 'POST',
  });
}

export async function classifyIntent(query: string) {
  return apiRequest<IntentClassifyResult>('/admin/intents/classify', {
    method: 'POST',
    body: JSON.stringify({ query }),
  });
}

export async function updateIntent(name: string, payload: { enabled?: boolean; label_fr?: string; description?: string | null }) {
  return apiRequest<IntentItem>(`/admin/intents/${name}`, {
    method: 'PATCH',
    body: JSON.stringify(payload),
  });
}

export async function listIntentExamples(name: string) {
  return apiRequest<IntentExample[]>(`/admin/intents/${name}/examples`);
}

export async function createIntentExample(name: string, text: string) {
  return apiRequest<IntentExample>(`/admin/intents/${name}/examples`, {
    method: 'POST',
    body: JSON.stringify({ text }),
  });
}

export async function updateIntentExample(id: string, payload: { text?: string; enabled?: boolean }) {
  return apiRequest<IntentExample>(`/admin/intents/examples/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(payload),
  });
}

export async function deleteIntentExample(id: string) {
  return apiRequest<void>(`/admin/intents/examples/${id}`, {
    method: 'DELETE',
  });
}

export type RetrievalTestMode = 'retrieval' | 'full';

export interface RetrievalTestResource {
  id?: string;
  type: 'video' | 'form' | 'document';
  title: string;
  description?: string;
  url?: string;
  thumbnail_url?: string;
  pdf_url?: string;
  page_url?: string;
  org?: string;
  score?: number | null;
  vespa_score?: number | null;
  rerank_score?: number | null;
  rank?: number | null;
  ranking_mode?: string | null;
  [key: string]: unknown;
}

export interface RetrievalTestChunk {
  rank: number;
  title: string;
  text: string;
  org?: string;
  source_type?: string;
  relative_source?: string;
  chunk_index?: number | string | null;
  vespa_rank?: number | string | null;
  vespa_relevance?: number | null;
  rerank_score?: number | null;
  rerank_raw_score?: number | null;
  source_boost?: number | null;
  rerank_mode?: string | null;
  metadata?: Record<string, unknown>;
}

export interface RetrievalTestPrompt {
  system?: string;
  context?: string;
  org_identity?: string;
  supplementary_hint?: string;
}

export interface RetrievalTestResponse {
  status: string;
  mode: RetrievalTestMode;
  query: string;
  org: 'cnra' | 'rcar' | 'all' | string;
  response?: string;
  intent?: string;
  intent_confidence?: number;
  intent_reasoning?: string;
  loaded?: boolean;
  cached?: boolean;
  timings?: Record<string, number>;
  chunks?: RetrievalTestChunk[];
  videos?: RetrievalTestResource[];
  forms?: RetrievalTestResource[];
  video_candidates?: RetrievalTestResource[];
  form_candidates?: RetrievalTestResource[];
  gates?: Record<string, { enabled?: boolean; signal?: boolean; [key: string]: unknown }>;
  prompt?: RetrievalTestPrompt;
}

export async function runRetrievalTest(payload: {
  query: string;
  org?: 'cnra' | 'rcar' | 'all';
  mode?: RetrievalTestMode;
  include_resources?: boolean;
  use_intent_classifier?: boolean;
}) {
  return apiRequest<RetrievalTestResponse>('/admin/retrieval-test/query', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export type EvaluationTask = 'docs' | 'forms' | 'videos' | 'intentions';

export interface EvaluationMetricOption {
  key: string;
  label: string;
}

export interface EvaluationRun {
  id: string;
  run_dir: string;
  run_name: string;
  task: EvaluationTask;
  task_label: string;
  backend: string;
  status: string;
  dataset?: string | null;
  dataset_size?: number | null;
  branch?: string | null;
  commit?: string | null;
  created_at?: string | null;
  file_updated_at?: string | null;
  synced_at?: string | null;
  metrics: Record<string, number | null | undefined>;
  primary_score?: number | null;
  primary_label: string;
  breakdowns?: Record<string, unknown>;
  summary?: Record<string, unknown>;
}

export interface EvaluationStats {
  total_runs: number;
  by_task: Record<EvaluationTask, number>;
  latest_run?: string | null;
  best_docs_score?: number | null;
  best_resource_hit_at_1?: number | null;
  best_intent_accuracy?: number | null;
  avg_latency?: number | null;
  total_errors: number;
}

export interface EvaluationRunsResponse {
  status: string;
  runs: EvaluationRun[];
  stats: EvaluationStats;
  metric_options: EvaluationMetricOption[];
}

export interface EvaluationDatasetResponse {
  status: string;
  task: EvaluationTask;
  task_label: string;
  path: string;
  total: number;
  columns: string[];
  rows: Array<Record<string, unknown> & { index: number }>;
}

export async function listEvaluationRuns() {
  return apiRequest<EvaluationRunsResponse>('/admin/evaluation/runs');
}

export async function syncEvaluationRuns() {
  return apiRequest<EvaluationRunsResponse>('/admin/evaluation/runs/sync', {
    method: 'POST',
  });
}

export async function deleteEvaluationRun(id: string) {
  return apiRequest<void>(`/admin/evaluation/runs/${id}`, {
    method: 'DELETE',
  });
}

export async function renameEvaluationRun(id: string, run_name: string) {
  return apiRequest<EvaluationRun>(`/admin/evaluation/runs/${id}`, {
    method: 'PATCH',
    body: JSON.stringify({ run_name }),
  });
}

export async function getEvaluationDataset(task: EvaluationTask) {
  return apiRequest<EvaluationDatasetResponse>(`/admin/evaluation/datasets/${task}`);
}

export async function prepareEvaluationRun(payload: {
  task: EvaluationTask;
  mode: 'smoke' | 'full';
  skip_ragas?: boolean;
  limit?: number | null;
  run_name?: string | null;
}) {
  return apiRequest<{
    status: string;
    message: string;
    task: EvaluationTask;
    mode: string;
    pid?: number;
    run_name?: string;
    log_path?: string;
  }>('/admin/evaluation/runs', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function runEvaluation() {
  return apiRequest<{ score: number; passed: number; failed: number; status?: string; message?: string }>('/admin/evaluation/run', {
    method: 'POST',
  });
}
