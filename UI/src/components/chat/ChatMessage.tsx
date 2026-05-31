import type { ChatMessage as ChatMessageType } from '../../api/types';
import { cn } from '../../utils/helpers';
import FormCard from './FormCard';
import MarkdownRenderer from './MarkdownRenderer';
import VideoCard from './VideoCard';

export default function ChatMessage({ message }: { message: ChatMessageType }) {
  const isUser = message.role === 'user';
  const videos = message.resources?.filter((resource) => resource.type === 'video') ?? [];
  const forms = message.resources?.filter((resource) => resource.type === 'form') ?? [];
  const documents = message.resources?.filter((resource) => resource.type === 'document') ?? [];

  return (
    <article className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div
        className={cn(
          'max-w-[min(760px,86%)] rounded-lg border px-4 py-3 text-sm leading-7 shadow-soft',
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
          <div className="mt-3 grid gap-2">
            {videos.map((resource, index) => (
              <VideoCard key={resource.id ?? `${resource.title}-${index}`} resource={resource} />
            ))}
            {[...forms, ...documents].map((resource, index) => (
              <FormCard key={resource.id ?? `${resource.title}-${index}`} resource={resource} />
            ))}
          </div>
        )}
      </div>
    </article>
  );
}
