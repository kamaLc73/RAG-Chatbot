import type { Organization } from '../../api/types';
import { cn } from '../../utils/helpers';

interface TypeSelectorProps {
  value: Organization;
  onChange: (organization: Organization) => void;
  disabled?: boolean;
}

const options: Array<{ value: Organization; label: string; title: string }> = [
  { value: 'CNRA & RCAR', label: 'Les deux', title: 'Recherche dans CNRA et RCAR' },
  { value: 'RCAR', label: 'RCAR', title: "Régime Collectif d'Allocation de Retraite" },
  { value: 'CNRA', label: 'CNRA', title: "Caisse Nationale de Retraites et d'Assurances" },
];

export default function TypeSelector({ value, onChange, disabled = false }: TypeSelectorProps) {
  return (
    <div className="flex w-full gap-2 rounded-lg bg-secondary p-1" role="tablist">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          disabled={disabled}
          title={option.title}
          onClick={() => onChange(option.value)}
          className={cn(
            'min-w-0 flex-1 rounded-md px-2 py-2 text-xs font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50 sm:px-3 sm:text-sm',
            value === option.value
              ? 'bg-primary text-primary-foreground shadow-sm'
              : 'text-muted-foreground hover:bg-background/60 hover:text-foreground'
          )}
          role="tab"
          aria-selected={value === option.value}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
