import { Loader2, MessageSquare } from 'lucide-react';
import { useEffect, useMemo, useRef } from 'react';
import { useChat } from '../../contexts/ChatContext';
import { formatConversationTitle, organizationLabel } from '../../utils/helpers';
import ChatMessage from './ChatMessage';
import MessageInput from './MessageInput';

export default function ChatInterface() {
  const { organization, conversations, activeConversationId, messages, isSending, error, sendMessage } = useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const currentConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === activeConversationId),
    [activeConversationId, conversations]
  );

  const label = organizationLabel(organization);
  const title = formatConversationTitle(currentConversation?.title, currentConversation?.organization || organization);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [messages.length, isSending]);

  return (
    <div className="flex h-full min-w-0 flex-col bg-background">
      <div className="border-b border-primary/20 bg-background p-3 sm:p-4">
        <div className="mx-auto max-w-4xl">
          <h2 className="truncate text-base font-semibold text-foreground sm:text-lg">{title}</h2>
          <p className="mt-1 text-xs text-muted-foreground">Conversation {label}</p>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto overscroll-y-contain border-l border-r border-primary/20 bg-background md:border-l-0" role="log" aria-live="polite">
        <div className="mx-auto flex min-h-full max-w-4xl flex-col gap-3 p-3 sm:gap-4 sm:p-4">
          {error && (
            <div className="rounded-lg border border-destructive/20 bg-destructive/10 p-4 text-sm text-destructive" role="alert">
              {error}
            </div>
          )}

          {messages.length === 0 && !isSending && (
            <div className="flex flex-1 flex-col items-center justify-center p-4 text-center sm:p-8">
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 sm:h-20 sm:w-20">
                <MessageSquare className="h-7 w-7 text-primary sm:h-9 sm:w-9" />
              </div>
              <h2 className="mb-2 text-xl font-semibold">Démarrer une conversation</h2>
              <p className="max-w-sm text-muted-foreground">
                Posez-moi vos questions sur les informations de retraite et de pension {label}.
              </p>
            </div>
          )}

          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}

          {isSending && (
            <div className="flex items-center gap-3 rounded-lg bg-secondary/50 p-4">
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-muted">
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              </div>
              <div>
                <span className="text-sm font-medium">Assistant</span>
                <p className="text-sm text-muted-foreground">Recherche en cours...</p>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="border-l border-r border-t border-primary/20 bg-background md:border-l-0">
        <div className="mx-auto max-w-4xl">
          <MessageInput
            onSend={(content) => void sendMessage(content)}
            disabled={isSending}
            placeholder={`Posez une question sur ${label}...`}
          />
        </div>
      </div>
    </div>
  );
}
