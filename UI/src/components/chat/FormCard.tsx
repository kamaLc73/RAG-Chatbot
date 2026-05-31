import { FileText } from 'lucide-react';
import type { Resource } from '../../api/types';

export default function FormCard({ resource }: { resource: Resource }) {
  const href = resource.url || resource.pdf_url || resource.page_url || '#';
  const thumbnail = resource.thumbnail_url;
  return (
    <a className="resource-card items-start" href={href} target="_blank" rel="noreferrer">
      <div className="relative h-14 w-24 flex-shrink-0 overflow-hidden rounded-md border border-border bg-muted">
        {thumbnail ? (
          <img
            src={thumbnail}
            alt={resource.title}
            className="h-full w-full object-cover"
            loading="lazy"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center text-primary">
            <FileText className="h-6 w-6" />
          </div>
        )}
      </div>
      <div>
        <strong>{resource.title}</strong>
        <p>{resource.description || resource.source || 'Formulaire utile'}</p>
      </div>
    </a>
  );
}
