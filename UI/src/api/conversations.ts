import { apiRequest } from './client';
import type { Conversation, Organization } from './types';

export async function listConversations() {
  return apiRequest<Conversation[]>('/conversations');
}

export async function createConversation(organization: Organization) {
  return apiRequest<Conversation>('/conversations', {
    method: 'POST',
    body: JSON.stringify({ organization }),
  });
}

export async function deleteConversation(id: string) {
  return apiRequest<void>(`/conversations/${id}`, { method: 'DELETE' });
}
