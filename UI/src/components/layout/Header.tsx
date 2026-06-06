import { Menu } from 'lucide-react';
import Button from '../ui/Button';
import PageNavigation from './PageNavigation';
import UserMenu from './UserMenu';

interface HeaderProps {
  onMenuClick: () => void;
}

export default function Header({ onMenuClick }: HeaderProps) {
  return (
    <header className="relative z-50 border-b border-border bg-background">
      <div className="flex min-w-0 items-center gap-2 px-3 py-2 sm:gap-4 sm:px-4 sm:py-3">
        <Button
          variant="ghost"
          size="sm"
          onClick={onMenuClick}
          className="flex-shrink-0 md:hidden"
          aria-label="Ouvrir les conversations"
        >
          <Menu className="h-5 w-5" />
        </Button>

        <div className="flex min-w-0 flex-shrink-0 items-center gap-2 sm:gap-3">
          <img src="/prevoyance_logo.png" alt="CDG Prévoyance" className="h-12 w-auto object-contain sm:h-16" />
          <div className="hidden sm:block">
            <h1 className="text-sm font-semibold leading-tight">Assistant CDG Prévoyance</h1>
            <p className="text-xs text-muted-foreground">RCAR & CNRA</p>
          </div>
        </div>

        <div className="flex-1" />

        <div className="flex min-w-0 items-center gap-2 sm:gap-3">
          <PageNavigation className="hidden md:flex" />
          <UserMenu />
        </div>
      </div>
    </header>
  );
}

