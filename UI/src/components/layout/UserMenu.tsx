import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ChevronDown, LogOut, Shield, User } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import Button from '../ui/Button';
import { cn } from '../../utils/helpers';

export default function UserMenu() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const [menuPosition, setMenuPosition] = useState<{ top: number; left: number } | null>(null);

  const updateMenuPosition = () => {
    const button = buttonRef.current;
    if (!button) return;

    const rect = button.getBoundingClientRect();
    const menuWidth = 240;
    const margin = 12;
    let left = rect.right - menuWidth;
    left = Math.min(left, window.innerWidth - menuWidth - margin);
    left = Math.max(left, margin);

    setMenuPosition({ top: rect.bottom + 8, left });
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      if (menuRef.current?.contains(target)) return;
      if (dropdownRef.current?.contains(target)) return;
      setIsOpen(false);
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return undefined;

    updateMenuPosition();

    const handleReposition = () => updateMenuPosition();
    window.addEventListener('resize', handleReposition);
    window.addEventListener('scroll', handleReposition, true);

    return () => {
      window.removeEventListener('resize', handleReposition);
      window.removeEventListener('scroll', handleReposition, true);
    };
  }, [isOpen]);

  if (!user) {
    return null;
  }

  const displayName = user.name || user.email || 'Utilisateur';
  const secondaryLine = user.email && user.email !== displayName ? user.email : undefined;

  const handleLogout = () => {
    logout();
    navigate('/login', { replace: true });
  };

  return (
    <div className="relative" ref={menuRef}>
      <Button
        ref={buttonRef}
        variant="ghost"
        size="sm"
        onClick={() => setIsOpen((current) => !current)}
        aria-expanded={isOpen}
        aria-haspopup="true"
        aria-label="Menu utilisateur"
        className="flex items-center gap-2"
      >
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary">
          <User className="h-4 w-4 text-primary-foreground" />
        </div>
        <span className="hidden max-w-32 truncate text-sm font-medium sm:inline">{displayName}</span>
        <ChevronDown className={cn('h-4 w-4 transition-transform', isOpen && 'rotate-180')} />
      </Button>

      {isOpen &&
        menuPosition &&
        createPortal(
          <div
            ref={dropdownRef}
            className="fixed z-[9999] w-60 rounded-lg border border-border bg-background py-1 shadow-lg"
            style={{ top: menuPosition.top, left: menuPosition.left }}
            role="menu"
          >
            <div className="border-b border-border px-4 py-3">
              <p className="text-sm font-medium">{displayName}</p>
              {secondaryLine && <p className="mt-0.5 text-xs text-muted-foreground">{secondaryLine}</p>}
            </div>

            <div className="py-1">
              {user.role === 'admin' && (
                <button
                  type="button"
                  onClick={() => {
                    navigate('/admin');
                    setIsOpen(false);
                  }}
                  className="flex w-full items-center gap-3 px-4 py-2 text-sm text-foreground transition-colors hover:bg-accent"
                  role="menuitem"
                >
                  <Shield className="h-4 w-4" />
                  Panneau d'administration
                </button>
              )}

              {user.role === 'admin' && <div className="my-1 h-px bg-border" />}

              <button
                type="button"
                onClick={handleLogout}
                className="logout-menu-item"
                role="menuitem"
              >
                <LogOut className="h-4 w-4" />
                Se déconnecter
              </button>
            </div>
          </div>,
          document.body
        )}
    </div>
  );
}
