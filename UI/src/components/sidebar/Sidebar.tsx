import { Plus, X } from 'lucide-react';
import { useEffect, useMemo } from 'react';
import { deleteConversation, updateConversationTitle } from '../../api/conversations';
import { useChat } from '../../contexts/ChatContext';
import TypeSelector from '../chat/TypeSelector';
import Button from '../ui/Button';
import ConversationItem from './ConversationItem';
import { cn, groupConversationsByDate, organizationLabel } from '../../utils/helpers';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function Sidebar({ isOpen, onClose }: SidebarProps) {
  const {
    organization,
    setOrganization,
    conversations,
    activeConversationId,
    refreshConversations,
    selectConversation,
    startConversation,
  } = useChat();

  useEffect(() => {
    void refreshConversations();
  }, [refreshConversations]);

  const filteredConversations = useMemo(
    () => conversations.filter((conversation) => conversation.organization === organization),
    [conversations, organization]
  );

  const grouped = useMemo(() => groupConversationsByDate(filteredConversations), [filteredConversations]);

  const handleNewConversation = async () => {
    await startConversation();
    if (window.innerWidth < 768) onClose();
  };

  const handleSelectConversation = async (conversationId: string) => {
    await selectConversation(conversationId);
    if (window.innerWidth < 768) onClose();
  };

  const handleDeleteConversation = async (conversationId: string) => {
    await deleteConversation(conversationId);
    await refreshConversations();
  };

  const handleRenameConversation = async (conversationId: string, title: string) => {
    await updateConversationTitle(conversationId, title);
    await refreshConversations();
  };

  const renderGroup = (title: string, items: typeof filteredConversations) => {
    if (items.length === 0) return null;

    return (
      <div className="mb-6">
        <h3 className="mb-2 px-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">{title}</h3>
        <div className="space-y-1">
          {items.map((conversation) => (
            <ConversationItem
              key={conversation.id}
              conversation={conversation}
              isActive={activeConversationId === conversation.id}
              onClick={() => void handleSelectConversation(conversation.id)}
              onDelete={(id) => void handleDeleteConversation(id)}
              onRename={(id, title) => void handleRenameConversation(id, title)}
            />
          ))}
        </div>
      </div>
    );
  };

  return (
    <>
      {isOpen && <div className="fixed inset-0 z-40 bg-black/50 md:hidden" onClick={onClose} aria-hidden="true" />}

      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-50 flex w-[min(20rem,calc(100vw-2rem))] flex-col border-r border-primary/20 bg-background shadow-md transition-transform duration-300 ease-in-out md:static md:w-[18rem] md:translate-x-0 lg:w-[20rem] xl:w-[22rem]',
          isOpen ? 'translate-x-0' : '-translate-x-full'
        )}
        aria-label="Conversations"
      >
        <div className="space-y-3 border-b border-border p-3 sm:p-4">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold">Conversations</h2>
            <Button variant="ghost" size="sm" onClick={onClose} className="md:hidden" aria-label="Fermer la barre latérale">
              <X className="h-4 w-4" />
            </Button>
          </div>

          <TypeSelector value={organization} onChange={setOrganization} />

          <button
            type="button"
            onClick={() => void handleNewConversation()}
            className="flex w-full items-center gap-3 rounded-lg bg-[#323E48] px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-[#323E48]/90"
          >
            <span className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-white/20">
              <Plus className="h-4 w-4" />
            </span>
            Nouvelle conversation {organizationLabel(organization)}
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-3 sm:p-4">
          {filteredConversations.length === 0 ? (
            <div className="py-8 text-center">
              <p className="text-sm text-muted-foreground">Aucune conversation pour le moment</p>
              <p className="mt-1 text-xs text-muted-foreground">Cliquez sur Nouvelle conversation pour commencer</p>
            </div>
          ) : (
            <>
              {renderGroup("Aujourd'hui", grouped.today)}
              {renderGroup('Hier', grouped.yesterday)}
              {renderGroup('Cette semaine', grouped.lastWeek)}
              {renderGroup('Plus ancien', grouped.older)}
            </>
          )}
        </div>
      </aside>
    </>
  );
}
