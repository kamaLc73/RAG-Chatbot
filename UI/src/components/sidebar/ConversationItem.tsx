import { Check, MessageSquare, Trash2, X } from 'lucide-react';
import { useState } from 'react';
import type { Conversation } from '../../api/types';
import { cn, formatConversationTitle, formatDate } from '../../utils/helpers';
import Button from '../ui/Button';

interface ConversationItemProps {
  conversation: Conversation;
  isActive: boolean;
  onClick: () => void;
  onDelete: (id: string) => void;
}

export default function ConversationItem({ conversation, isActive, onClick, onDelete }: ConversationItemProps) {
  const [confirmDelete, setConfirmDelete] = useState(false);

  return (
    <div
      className={cn(
        'group relative cursor-pointer rounded-lg px-3 py-3 transition-all duration-200',
        isActive
          ? 'border-2 border-primary/70 bg-primary/10 text-primary'
          : 'border-2 border-transparent text-foreground hover:border-primary/50'
      )}
      onClick={onClick}
    >
      <div className="flex items-start gap-3">
        <MessageSquare className={cn('mt-0.5 h-4 w-4 flex-shrink-0', isActive ? 'text-primary' : 'text-muted-foreground')} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2">
            <h3 className="truncate text-sm font-semibold" title={formatConversationTitle(conversation.title, conversation.organization)}>
              {formatConversationTitle(conversation.title, conversation.organization)}
            </h3>
            <div
              className="flex items-center gap-1 opacity-100 transition-opacity sm:opacity-0 sm:group-hover:opacity-100"
              onClick={(event) => event.stopPropagation()}
            >
              {confirmDelete ? (
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
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setConfirmDelete(true)}
                  className="h-9 w-9 flex-shrink-0 p-0 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                  aria-label="Supprimer la conversation"
                >
                  <Trash2 className="h-5 w-5" />
                </Button>
              )}
            </div>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">{formatDate(conversation.updated_at)}</p>
        </div>
      </div>
    </div>
  );
}
