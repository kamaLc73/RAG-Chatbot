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
      <div className="flex items-center gap-4 px-4 py-3">
        <Button
          variant="ghost"
          size="sm"
          onClick={onMenuClick}
          className="flex-shrink-0 md:hidden"
          aria-label="Ouvrir les conversations"
        >
          <Menu className="h-5 w-5" />
        </Button>

        <div className="flex flex-shrink-0 items-center gap-3">
          <img src="/prevoyance_logo.png" alt="CDG Prévoyance" className="h-16 w-auto object-contain" />
          <div className="hidden sm:block">
            <h1 className="text-sm font-semibold leading-tight">Assistant Prévoyance</h1>
            <p className="text-xs text-muted-foreground">RCAR & CNRA</p>
          </div>
        </div>

        <div className="flex-1" />

        <div className="flex items-center gap-3">
          <PageNavigation className="hidden md:flex" />
          <UserMenu />
        </div>
      </div>
    </header>
  );
}
