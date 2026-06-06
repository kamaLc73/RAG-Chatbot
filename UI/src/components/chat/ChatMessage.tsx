import { useState } from 'react';
import { ThumbsDown, ThumbsUp } from 'lucide-react';
import { submitMessageFeedback } from '../../api/chat';
import type { ChatMessage as ChatMessageType } from '../../api/types';
import { useChat } from '../../contexts/ChatContext';
import { cn } from '../../utils/helpers';
import FormCard from './FormCard';
import MarkdownRenderer from './MarkdownRenderer';
import VideoCard from './VideoCard';

export default function ChatMessage({ message }: { message: ChatMessageType }) {
  const { updateMessageFeedback } = useChat();
  const isUser = message.role === 'user';
  const [feedback, setFeedback] = useState<1 | -1 | null>(message.feedback ?? null);
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);
  const videos = message.resources?.filter((resource) => resource.type === 'video') ?? [];
  const forms = message.resources?.filter((resource) => resource.type === 'form') ?? [];
  const documents = message.resources?.filter((resource) => resource.type === 'document') ?? [];

  const handleFeedback = async (nextFeedback: 1 | -1) => {
    if (isSubmittingFeedback) return;
    const resolvedFeedback = feedback === nextFeedback ? null : nextFeedback;
    const previousFeedback = feedback;
    setFeedback(resolvedFeedback);
    updateMessageFeedback(message.id, resolvedFeedback);
    setIsSubmittingFeedback(true);
    try {
      await submitMessageFeedback(message.id, resolvedFeedback);
    } catch {
      setFeedback(previousFeedback);
      updateMessageFeedback(message.id, previousFeedback);
    } finally {
      setIsSubmittingFeedback(false);
    }
  };

  return (
    <article className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div
        className={cn(
          'max-w-[min(760px,92%)] rounded-lg border px-3 py-3 text-sm leading-7 shadow-soft sm:max-w-[min(760px,86%)] sm:px-4',
          isUser
            ? 'border-primary bg-primary text-primary-foreground'
            : 'border-border bg-card text-card-foreground'
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <MarkdownRenderer content={message.content} />
        )}
        {(videos.length > 0 || forms.length > 0 || documents.length > 0) && (
          <div className="mt-3 grid min-w-0 gap-2">
            {videos.map((resource, index) => (
              <VideoCard key={resource.id ?? `${resource.title}-${index}`} resource={resource} />
            ))}
            {[...forms, ...documents].map((resource, index) => (
              <FormCard key={resource.id ?? `${resource.title}-${index}`} resource={resource} />
            ))}
          </div>
        )}
        {!isUser && (
          <div className="mt-3 flex flex-wrap items-center gap-1 border-t border-border/70 pt-2">
            <button
              type="button"
              onClick={() => void handleFeedback(1)}
              disabled={isSubmittingFeedback}
              className={cn(
                'inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-primary/10 hover:text-primary disabled:opacity-50',
                feedback === 1 && 'bg-primary/10 text-primary'
              )}
              aria-label="Réponse utile"
              title="Réponse utile"
            >
              <ThumbsUp className={cn('h-4 w-4', feedback === 1 && 'fill-current')} />
            </button>
            <button
              type="button"
              onClick={() => void handleFeedback(-1)}
              disabled={isSubmittingFeedback}
              className={cn(
                'inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive disabled:opacity-50',
                feedback === -1 && 'bg-destructive/10 text-destructive'
              )}
              aria-label="Réponse peu utile"
              title="Réponse peu utile"
            >
              <ThumbsDown className={cn('h-4 w-4', feedback === -1 && 'fill-current')} />
            </button>
          </div>
        )}
      </div>
    </article>
  );
}
