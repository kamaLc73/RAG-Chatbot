import { FormEvent, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import type { Organization } from '../api/types';
import { useAuth } from '../contexts/AuthContext';
import Button from '../components/ui/Button';
import Input from '../components/ui/Input';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../components/ui/Card';

export default function Signup() {
  const { signup } = useAuth();
  const navigate = useNavigate();
  const [fullName, setFullName] = useState('');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [organization, setOrganization] = useState<Organization>('CNRA & RCAR');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError('');

    if (password !== confirmPassword) {
      setError('Les mots de passe ne correspondent pas');
      return;
    }

    setIsLoading(true);
    try {
      await signup({ name: fullName || username, username, email, password, organization });
      navigate('/chat', { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Création impossible');
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
            <CardTitle>Création de compte</CardTitle>
            <CardDescription>Démarrer avec l'assistant RCAR/CNRA</CardDescription>
          </div>
        </CardHeader>

        <form onSubmit={(event) => void submit(event)}>
          <CardContent className="space-y-4">
            {error && <p className="form-error">{error}</p>}

            <Input
              label="Nom complet"
              value={fullName}
              onChange={(event) => setFullName(event.target.value)}
              placeholder="Nom et prénom"
              autoComplete="name"
            />
            <Input
              label="Identifiant"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              placeholder="votre.identifiant"
              autoComplete="username"
              required
            />
            <Input
              label="Email"
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="vous@exemple.com"
              autoComplete="email"
              required
            />
            <div>
              <label htmlFor="organization" className="mb-2 block text-sm font-medium text-foreground">
                Organisation
              </label>
              <select
                id="organization"
                value={organization}
                onChange={(event) => setOrganization(event.target.value as Organization)}
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <option>CNRA</option>
                <option>RCAR</option>
                <option>CNRA & RCAR</option>
              </select>
            </div>
            <div>
              <Input
                label="Mot de passe"
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="Créer un mot de passe"
                autoComplete="new-password"
                required
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
            <Input
              label="Confirmer le mot de passe"
              type={showPassword ? 'text' : 'password'}
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              placeholder="Confirmer votre mot de passe"
              autoComplete="new-password"
              required
            />
          </CardContent>

          <CardFooter className="flex flex-col space-y-4">
            <Button type="submit" className="w-full" isLoading={isLoading} disabled={isLoading}>
              Créer le compte
            </Button>
            <p className="text-center text-sm text-muted-foreground">
              Déjà inscrit ?{' '}
              <Link to="/login" className="font-medium text-primary hover:underline">
                Connexion
              </Link>
            </p>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}
