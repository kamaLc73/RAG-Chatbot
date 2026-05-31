import { LogOut, MessageSquare, Shield } from 'lucide-react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import Button from '../ui/Button';

interface PageNavigationProps {
  className?: string;
}

export default function PageNavigation({ className = '' }: PageNavigationProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuth();

  const handleLogout = () => {
    logout();
    navigate('/login', { replace: true });
  };

  return (
    <div className={`items-center gap-2 ${className}`}>
      <Button
        variant={location.pathname.startsWith('/chat') ? 'primary' : 'outline'}
        size="sm"
        onClick={() => navigate('/chat')}
        className="gap-2"
      >
        <MessageSquare className="h-4 w-4" />
        Chat
      </Button>
      {user?.role === 'admin' && (
        <Button
          variant={location.pathname.startsWith('/admin') ? 'primary' : 'outline'}
          size="sm"
          onClick={() => navigate('/admin')}
          className="gap-2"
        >
          <Shield className="h-4 w-4" />
          Admin
        </Button>
      )}
      <Button variant="outline" size="sm" onClick={handleLogout} className="gap-2 text-destructive hover:text-destructive">
        <LogOut className="h-4 w-4" />
        Se déconnecter
      </Button>
    </div>
  );
}
