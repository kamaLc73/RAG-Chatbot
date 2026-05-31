import { FormEvent, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import Button from '../components/ui/Button';
import Input from '../components/ui/Input';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../components/ui/Card';

const localDevAdmin = {
  email: 'admin@cdg.dev',
  password: '0123456789aze',
};

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const shouldPrefillLocalAdmin = import.meta.env.DEV;
  const [mode, setMode] = useState<'user' | 'admin'>(shouldPrefillLocalAdmin ? 'admin' : 'user');
  const [identifier, setIdentifier] = useState(shouldPrefillLocalAdmin ? localDevAdmin.email : '');
  const [password, setPassword] = useState(shouldPrefillLocalAdmin ? localDevAdmin.password : '');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError('');
    setIsLoading(true);
    try {
      await login({ identifier, password, mode });
      navigate('/chat', { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connexion impossible');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md border-2 border-primary/20">
        <CardHeader className="space-y-4">
          <div className="flex justify-center">
            <img src="/prevoyance_logo.png" alt="CDG Prévoyance" className="h-24 w-auto object-contain" />
          </div>
          <div className="space-y-1 text-center">
            <CardTitle>{mode === 'admin' ? 'Connexion Admin' : 'Bienvenue'}</CardTitle>
            <CardDescription>
              {mode === 'admin' ? 'Accès administrateur' : 'Connectez-vous à votre chatbot'}
            </CardDescription>
          </div>
        </CardHeader>

        <form onSubmit={(event) => void submit(event)}>
          <CardContent className="space-y-4">
            {error && <p className="form-error">{error}</p>}

            <Input
              label={mode === 'admin' ? 'Email' : 'Identifiant'}
              type={mode === 'admin' ? 'email' : 'text'}
              placeholder={mode === 'admin' ? 'admin@exemple.com' : 'votre.identifiant'}
              value={identifier}
              onChange={(event) => setIdentifier(event.target.value)}
              required
              autoComplete={mode === 'admin' ? 'email' : 'username'}
              autoFocus
            />

            <div>
              <Input
                label="Mot de passe"
                type={showPassword ? 'text' : 'password'}
                placeholder="Entrez votre mot de passe"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
                autoComplete="current-password"
              />
              <button
                type="button"
                onClick={() => setShowPassword((current) => !current)}
                className="mt-1 text-sm text-primary hover:underline"
                tabIndex={-1}
              >
                {showPassword ? 'Masquer' : 'Afficher'} le mot de passe
              </button>
            </div>
          </CardContent>

          <CardFooter className="flex flex-col space-y-4">
            <Button type="submit" className="w-full" isLoading={isLoading} disabled={isLoading}>
              {mode === 'admin' ? 'Connexion Admin' : 'Connexion'}
            </Button>
            <button
              type="button"
              onClick={() => {
                const nextMode = mode === 'admin' ? 'user' : 'admin';
                setMode(nextMode);
                setIdentifier(shouldPrefillLocalAdmin && nextMode === 'admin' ? localDevAdmin.email : '');
                setPassword(shouldPrefillLocalAdmin && nextMode === 'admin' ? localDevAdmin.password : '');
                setError('');
              }}
              className="text-sm text-muted-foreground transition-colors hover:text-primary"
            >
              {mode === 'admin' ? 'Utilisateur' : 'Admin'}
            </button>
            <p className="text-center text-sm text-muted-foreground">
              Pas encore de compte ?{' '}
              <Link to="/signup" className="font-medium text-primary hover:underline">
                Créer un compte
              </Link>
            </p>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}
