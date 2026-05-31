import { apiRequest } from './client';
import type { ChatMessage, ChatResponse, Organization } from './types';

export async function sendChatMessage(payload: {
  message: string;
  organization: Organization;
  conversation_id?: string;
}) {
  return apiRequest<ChatResponse>('/chat/query', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function transcribeAudio(file: File) {
  const form = new FormData();
  form.append('file', file);
  return apiRequest<{ text: string }>('/chat/transcribe', {
    method: 'POST',
    body: form,
  });
}

export async function getConversationMessages(conversationId: string) {
  return apiRequest<ChatMessage[]>(`/conversations/${conversationId}/messages`);
}
