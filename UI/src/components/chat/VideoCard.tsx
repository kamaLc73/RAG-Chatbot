import { PlayCircle } from 'lucide-react';
import type { Resource } from '../../api/types';

export default function VideoCard({ resource }: { resource: Resource }) {
  const href = resource.url || '#';
  const thumbnail = resource.thumbnail_url;
  return (
    <a className="resource-card items-start" href={href} target="_blank" rel="noreferrer">
      {thumbnail ? (
        <div className="relative h-14 w-24 flex-shrink-0 overflow-hidden rounded-md border border-border bg-muted">
          <img
            src={thumbnail}
            alt={resource.title}
            className="h-full w-full object-cover"
            loading="lazy"
          />
          <div className="absolute inset-0 flex items-center justify-center bg-black/10">
            <PlayCircle className="h-6 w-6 text-white drop-shadow" />
          </div>
        </div>
      ) : (
        <PlayCircle className="mt-0.5 h-5 w-5 flex-shrink-0 text-primary" />
      )}
      <div>
        <strong>{resource.title}</strong>
        <p>{resource.description || resource.source || 'Vidéo de support'}</p>
      </div>
    </a>
  );
}
