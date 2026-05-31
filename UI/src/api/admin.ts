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

export async function listKbItems() {
  return apiRequest<Array<{ id: string; title: string; source: string; status: string }>>('/admin/kb');
}

export async function listIntents() {
  return apiRequest<Array<{ id: string; name: string; examples: number; enabled: boolean }>>('/admin/intents');
}

export async function runEvaluation() {
  return apiRequest<{ score: number; passed: number; failed: number }>('/admin/evaluation/run', {
    method: 'POST',
  });
}
