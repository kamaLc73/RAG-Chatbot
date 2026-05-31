import { useEffect, useState } from 'react';
import { listAdminConversations } from '../../api/admin';
import type { Conversation, User } from '../../api/types';
import { formatConversationTitle } from '../../utils/helpers';
import Button from '../ui/Button';

interface ConversationsManagerProps {
  selectedUser?: User | null;
  onClearSelectedUser?: () => void;
}

export default function ConversationsManager({ selectedUser, onClearSelectedUser }: ConversationsManagerProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);

  useEffect(() => {
    void listAdminConversations().then(setConversations).catch(() => setConversations([]));
  }, []);

  const visibleConversations = selectedUser
    ? conversations.filter((conversation) => String(conversation.user_id ?? '') === String(selectedUser.id))
    : conversations;

  return (
    <section className="admin-panel">
      <div className="panel-heading">
        <div>
          <h2>Conversations</h2>
          {selectedUser && (
            <p className="mt-1 text-sm text-muted-foreground">
              Conversations de {selectedUser.name || selectedUser.username || selectedUser.email}
            </p>
          )}
        </div>
        {selectedUser && (
          <Button variant="outline" onClick={onClearSelectedUser}>
            Voir toutes les conversations
          </Button>
        )}
      </div>

      <div className="table-list">
        {visibleConversations.map((conversation) => (
          <div className="table-row" key={conversation.id}>
            <span>
              <strong>{formatConversationTitle(conversation.title, conversation.organization)}</strong>
              <small>{new Date(conversation.updated_at).toLocaleString('fr-FR')}</small>
            </span>
            <span>{conversation.organization}</span>
          </div>
        ))}
        {visibleConversations.length === 0 && (
          <p className="empty">
            {selectedUser ? 'Aucune conversation pour cet utilisateur.' : 'Aucune conversation chargée.'}
          </p>
        )}
      </div>
    </section>
  );
}
