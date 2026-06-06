import { Check, Edit3, MessageSquare, Trash2, X } from 'lucide-react';
import { useState } from 'react';
import type { Conversation } from '../../api/types';
import { cn, formatConversationTitle, formatDate } from '../../utils/helpers';
import Button from '../ui/Button';

interface ConversationItemProps {
  conversation: Conversation;
  isActive: boolean;
  onClick: () => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
}

export default function ConversationItem({ conversation, isActive, onClick, onDelete, onRename }: ConversationItemProps) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [draftTitle, setDraftTitle] = useState(formatConversationTitle(conversation.title, conversation.organization));

  const startEditing = () => {
    setConfirmDelete(false);
    setDraftTitle(formatConversationTitle(conversation.title, conversation.organization));
    setIsEditing(true);
  };

  const cancelEditing = () => {
    setIsEditing(false);
    setDraftTitle(formatConversationTitle(conversation.title, conversation.organization));
  };

  const saveTitle = () => {
    const nextTitle = draftTitle.trim();
    if (!nextTitle) return;
    onRename(conversation.id, nextTitle);
    setIsEditing(false);
  };

  return (
    <div
      className={cn(
        'group relative rounded-lg px-3 py-3 transition-all duration-200',
        isEditing ? 'cursor-default' : 'cursor-pointer',
        isActive
          ? 'border-2 border-primary/70 bg-primary/10 text-primary'
          : 'border-2 border-transparent text-foreground hover:border-primary/50'
      )}
      onClick={isEditing ? undefined : onClick}
    >
      <div className="flex items-start gap-3">
        <MessageSquare className={cn('mt-0.5 h-4 w-4 flex-shrink-0', isActive ? 'text-primary' : 'text-muted-foreground')} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2">
            {isEditing ? (
              <input
                value={draftTitle}
                onChange={(event) => setDraftTitle(event.target.value)}
                onClick={(event) => event.stopPropagation()}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    saveTitle();
                  }
                  if (event.key === 'Escape') {
                    event.preventDefault();
                    cancelEditing();
                  }
                }}
                className="min-w-0 flex-1 rounded-md border border-input bg-background px-2 py-1 text-sm font-semibold text-foreground outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                autoFocus
              />
            ) : (
              <h3 className="truncate text-sm font-semibold" title={formatConversationTitle(conversation.title, conversation.organization)}>
                {formatConversationTitle(conversation.title, conversation.organization)}
              </h3>
            )}
            <div
              className="flex items-center gap-1 opacity-100 transition-opacity sm:opacity-0 sm:group-hover:opacity-100"
              onClick={(event) => event.stopPropagation()}
            >
              {isEditing ? (
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={saveTitle}
                    className="h-9 w-9 flex-shrink-0 p-0 text-primary hover:bg-primary/10"
                    aria-label="Enregistrer le titre"
                  >
                    <Check className="h-5 w-5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={cancelEditing}
                    className="h-9 w-9 flex-shrink-0 p-0"
                    aria-label="Annuler la modification"
                  >
                    <X className="h-5 w-5" />
                  </Button>
                </>
              ) : confirmDelete ? (
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onDelete(conversation.id)}
                    className="h-9 w-9 flex-shrink-0 p-0 text-destructive hover:bg-destructive/10"
                    aria-label="Confirmer la suppression"
                  >
                    <Check className="h-5 w-5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setConfirmDelete(false)}
                    className="h-9 w-9 flex-shrink-0 p-0"
                    aria-label="Annuler la suppression"
                  >
                    <X className="h-5 w-5" />
                  </Button>
                </>
              ) : (
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={startEditing}
                    className="h-9 w-9 flex-shrink-0 p-0 text-primary hover:bg-primary/10"
                    aria-label="Modifier le titre"
                  >
                    <Edit3 className="h-5 w-5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setConfirmDelete(true)}
                    className="h-9 w-9 flex-shrink-0 p-0 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                    aria-label="Supprimer la conversation"
                  >
                    <Trash2 className="h-5 w-5" />
                  </Button>
                </>
              )}
            </div>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">{formatDate(conversation.updated_at)}</p>
        </div>
      </div>
    </div>
  );
}
