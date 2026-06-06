import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import * as chatApi from '../api/chat';
import * as conversationsApi from '../api/conversations';
import { useAuth } from './AuthContext';
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
  updateMessageFeedback: (messageId: string, feedback: 1 | -1 | null) => void;
}

const ChatContext = createContext<ChatContextValue | undefined>(undefined);

function localMessage(
  role: ChatMessage['role'],
  content: string,
  resources: ChatMessage['resources'] = [],
  patch: Partial<ChatMessage> = {}
): ChatMessage {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    resources,
    created_at: new Date().toISOString(),
    ...patch,
  };
}

export function ChatProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const currentUserId = user?.id ?? null;
  const activeUserIdRef = useRef<string | null>(currentUserId);
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
    if (!currentUserId) {
      setConversations([]);
      return;
    }

    const requestUserId = currentUserId;
    try {
      const nextConversations = await conversationsApi.listConversations();
      if (activeUserIdRef.current === requestUserId) {
        setConversations(nextConversations);
      }
    } catch (err) {
      if (activeUserIdRef.current === requestUserId) {
        setError(err instanceof Error ? err.message : 'Impossible de charger les conversations');
      }
    }
  }, [currentUserId]);

  useEffect(() => {
    activeUserIdRef.current = currentUserId;
    setOrganization('CNRA & RCAR');
    setConversations([]);
    setActiveConversationId(undefined);
    setMessages([]);
    setError(undefined);
    setIsSending(false);

    if (currentUserId) {
      void refreshConversations();
    }
  }, [currentUserId, refreshConversations]);

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
          localMessage('assistant', response.answer, response.resources ?? [], {
            id: response.message_id ?? crypto.randomUUID(),
            context_docs: response.context_docs,
            latency_seconds: response.latency_seconds,
          }),
        ]);
        void refreshConversations();
        window.setTimeout(() => void refreshConversations(), 3000);
        window.setTimeout(() => void refreshConversations(), 9000);
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

  const updateMessageFeedback = useCallback((messageId: string, feedback: 1 | -1 | null) => {
    setMessages((current) =>
      current.map((message) => (message.id === messageId ? { ...message, feedback } : message))
    );
  }, []);

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
      updateMessageFeedback,
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
      updateMessageFeedback,
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
