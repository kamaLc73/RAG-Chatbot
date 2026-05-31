import type { Conversation } from '../api/types';

export function cn(...classes: (string | boolean | null | undefined)[]) {
  return classes.filter(Boolean).join(' ');
}

export function formatDate(dateString: string) {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (Number.isNaN(date.getTime())) return '';
  if (diffMins < 1) return 'maintenant';
  if (diffMins < 60) return `${diffMins} min`;
  if (diffHours < 24) return `${diffHours} h`;
  if (diffDays < 7) return `${diffDays} j`;

  return date.toLocaleDateString('fr-FR', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });
}

export interface GroupedConversations {
  today: Conversation[];
  yesterday: Conversation[];
  lastWeek: Conversation[];
  older: Conversation[];
}

export function groupConversationsByDate(conversations: Conversation[]): GroupedConversations {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const lastWeek = new Date(today);
  lastWeek.setDate(lastWeek.getDate() - 7);

  return conversations.reduce<GroupedConversations>(
    (groups, conversation) => {
      const date = new Date(conversation.updated_at);
      if (date >= today) groups.today.push(conversation);
      else if (date >= yesterday) groups.yesterday.push(conversation);
      else if (date >= lastWeek) groups.lastWeek.push(conversation);
      else groups.older.push(conversation);
      return groups;
    },
    { today: [], yesterday: [], lastWeek: [], older: [] }
  );
}

export function organizationLabel(value: string) {
  if (value === 'CNRA') return 'CNRA';
  if (value === 'RCAR') return 'RCAR';
  return 'CNRA & RCAR';
}

export function formatConversationTitle(title: string | undefined, organization: string) {
  const label = organizationLabel(organization);
  const normalized = (title || '').trim();

  if (!normalized || normalized === 'Conversation' || normalized === 'Nouvelle conversation') {
    return `Nouvelle conversation ${label}`;
  }

  const legacyMatch = normalized.match(/^New\s+(.+?)\s+Conversation$/i);
  if (legacyMatch) {
    return `Nouvelle conversation ${organizationLabel(legacyMatch[1].replace('CNRA-RECORE', 'CNRA'))}`;
  }

  return normalized.replaceAll('CNRA-RECORE', 'CNRA');
}
