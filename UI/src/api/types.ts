export type Organization = 'CNRA' | 'RCAR' | 'CNRA & RCAR';

export interface User {
  id: string;
  email: string;
  username?: string;
  name: string;
  full_name?: string | null;
  role: 'user' | 'admin';
  organization?: Organization;
  is_active?: boolean;
  is_superuser?: boolean;
  created_at?: string;
}

export interface AuthResponse {
  access_token: string;
  refresh_token?: string;
  user: User;
}

export interface Conversation {
  id: string;
  title: string;
  title_source?: 'auto_pending' | 'auto_generating' | 'auto' | 'user' | string;
  title_generated_at?: string | null;
  organization: Organization;
  org?: 'cnra' | 'rcar' | 'all';
  created_at?: string;
  updated_at: string;
  user_id?: string | number | null;
  user_name?: string | null;
  username?: string | null;
  user_email?: string | null;
  message_count?: number;
  question_count?: number;
  answer_count?: number;
  feedback_count?: number;
  positive_feedback?: number;
  negative_feedback?: number;
  avg_latency_seconds?: number | null;
  total_context_docs?: number;
  max_context_docs?: number;
  messages?: ChatMessage[];
}

export interface Resource {
  id?: string;
  title: string;
  description?: string;
  url?: string;
  thumbnail_url?: string;
  pdf_url?: string;
  page_url?: string;
  type: 'video' | 'form' | 'document';
  source?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  resources?: Resource[];
  chunks?: Array<{ text?: string; metadata?: Record<string, unknown> }>;
  context_docs?: number | null;
  latency_seconds?: number | null;
  feedback?: 1 | -1 | null;
  feedback_comment?: string | null;
  feedback_at?: string | null;
  intent?: string | null;
  intent_confidence?: number | null;
}

export interface ChatResponse {
  conversation_id: string;
  message_id?: string;
  answer: string;
  resources?: Resource[];
  context_docs?: number | null;
  latency_seconds?: number | null;
}

export interface AdminStats {
  users: number;
  conversations: number;
  documents: number;
  intents: number;
}
