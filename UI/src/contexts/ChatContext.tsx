import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react';
import * as chatApi from '../api/chat';
import * as conversationsApi from '../api/conversations';
import type { ChatMessage, Conversation, Organization } from '../api/types';

interface ChatContextValue {
  organization: Organization;
  setOrganization: (organization: Organization) => void;
  conversations: Conversation[];
  activeConversationId?: string;
  messages: ChatMessage[];
  isSending: boolean;
  error?: string;
  refreshConversations: () => Promise<void>;
  startConversation: () => Promise<void>;
  selectConversation: (id: string) => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
}

const ChatContext = createContext<ChatContextValue | undefined>(undefined);

function localMessage(role: ChatMessage['role'], content: string, resources: ChatMessage['resources'] = []): ChatMessage {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    resources,
    created_at: new Date().toISOString(),
  };
}

export function ChatProvider({ children }: { children: ReactNode }) {
  const [organization, setOrganization] = useState<Organization>('CNRA & RCAR');
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string>();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string>();

  const changeOrganization = useCallback(
    (nextOrganization: Organization) => {
      if (nextOrganization === organization) return;
      setOrganization(nextOrganization);
      setActiveConversationId(undefined);
      setMessages([]);
      setError(undefined);
    },
    [organization]
  );

  const refreshConversations = useCallback(async () => {
    try {
      setConversations(await conversationsApi.listConversations());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load conversations');
    }
  }, []);

  const startConversation = useCallback(async () => {
    const conversation = await conversationsApi.createConversation(organization);
    setConversations((current) => [conversation, ...current]);
    setActiveConversationId(conversation.id);
    setMessages([]);
  }, [organization]);

  const selectConversation = useCallback(async (id: string) => {
    setActiveConversationId(id);
    setMessages(await chatApi.getConversationMessages(id));
  }, []);

  const sendMessage = useCallback(
    async (content: string) => {
      setError(undefined);
      setIsSending(true);
      setMessages((current) => [...current, localMessage('user', content)]);
      try {
        const response = await chatApi.sendChatMessage({
          message: content,
          organization,
          conversation_id: activeConversationId,
        });
        setActiveConversationId(response.conversation_id);
        setMessages((current) => [
          ...current,
          localMessage('assistant', response.answer, response.resources ?? []),
        ]);
        void refreshConversations();
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unable to send message';
        setError(message);
        setMessages((current) => [...current, localMessage('assistant', message)]);
      } finally {
        setIsSending(false);
      }
    },
    [activeConversationId, organization, refreshConversations]
  );

  const value = useMemo(
    () => ({
      organization,
      setOrganization: changeOrganization,
      conversations,
      activeConversationId,
      messages,
      isSending,
      error,
      refreshConversations,
      startConversation,
      selectConversation,
      sendMessage,
    }),
    [
      organization,
      changeOrganization,
      conversations,
      activeConversationId,
      messages,
      isSending,
      error,
      refreshConversations,
      startConversation,
      selectConversation,
      sendMessage,
    ]
  );

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat() {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used inside ChatProvider');
  }
  return context;
}
