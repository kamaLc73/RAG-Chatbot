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
  organization: Organization;
  updated_at: string;
  user_id?: string | number | null;
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
}

export interface ChatResponse {
  conversation_id: string;
  answer: string;
  resources?: Resource[];
}

export interface AdminStats {
  users: number;
  conversations: number;
  documents: number;
  intents: number;
}
