import { FormEvent, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import Button from '../components/ui/Button';
import Input from '../components/ui/Input';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '../components/ui/Card';

const localDevAdmin = {
  email: 'admin@cdg.dev',
  password: '0123456789aze',
};

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const shouldPrefillLocalAdmin = import.meta.env.DEV;
  const [mode, setMode] = useState<'user' | 'admin'>(
    shouldPrefillLocalAdmin ? 'admin' : 'user',
  );
  const [identifier, setIdentifier] = useState(
    shouldPrefillLocalAdmin ? localDevAdmin.email : '',
  );
  const [password, setPassword] = useState(
    shouldPrefillLocalAdmin ? localDevAdmin.password : '',
  );
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
    <div className="auth-shell">
      <div className="auth-stage">
        <div className="auth-image-panel" aria-hidden="true">
          <img src="/img_cdg.png" alt="" />
          <div className="auth-image-overlay">
            <div className="auth-image-brand">
              <img src="/logoCDG.png" alt="" />
              <p>Assistant CDG Prévoyance</p>
            </div>
          </div>
        </div>

        <div className="auth-form-panel">
          <Card className="auth-card">

            <CardHeader className="auth-card-header">
              <div className="auth-logo-wrap">
                <img
                  src="/prevoyance_logo.png"
                  alt="CDG Prévoyance"
                  className="auth-logo"
                />
              </div>
              <div className="auth-titles">
                <CardTitle className="auth-title">
                  {mode === 'admin' ? 'Connexion Admin' : 'Bienvenue'}
                </CardTitle>
                <CardDescription className="auth-subtitle">
                  {mode === 'admin'
                    ? 'Accès espace administrateur'
                    : 'Connectez-vous à votre espace'}
                </CardDescription>
              </div>
            </CardHeader>

            <form onSubmit={(event) => void submit(event)}>
              <CardContent className="auth-card-content">
                {error && <p className="form-error">{error}</p>}

                <Input
                  label={mode === 'admin' ? 'Email' : 'Identifiant'}
                  type={mode === 'admin' ? 'email' : 'text'}
                  placeholder={
                    mode === 'admin' ? 'admin@cdg.dev' : 'votre.identifiant'
                  }
                  value={identifier}
                  onChange={(e) => setIdentifier(e.target.value)}
                  required
                  autoComplete={mode === 'admin' ? 'email' : 'username'}
                  autoFocus
                />

                <div className="auth-password-field">
                  <Input
                    label="Mot de passe"
                    type={showPassword ? 'text' : 'password'}
                    placeholder="••••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    autoComplete="current-password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((v) => !v)}
                    className="auth-toggle-pw"
                    tabIndex={-1}
                  >
                    {showPassword ? 'Masquer' : 'Afficher'} le mot de passe
                  </button>
                </div>
              </CardContent>

              <CardFooter className="auth-card-footer">
                <Button
                  type="submit"
                  className="auth-submit-btn w-full"
                  isLoading={isLoading}
                  disabled={isLoading}
                >
                  {mode === 'admin' ? 'Connexion Admin' : 'Se connecter'}
                </Button>

                <div className="auth-footer-links">
                  <button
                    type="button"
                    onClick={() => {
                      const next = mode === 'admin' ? 'user' : 'admin';
                      setMode(next);
                      setIdentifier(
                        shouldPrefillLocalAdmin && next === 'admin'
                          ? localDevAdmin.email
                          : '',
                      );
                      setPassword(
                        shouldPrefillLocalAdmin && next === 'admin'
                          ? localDevAdmin.password
                          : '',
                      );
                      setError('');
                    }}
                    className="auth-mode-toggle"
                  >
                    {mode === 'admin'
                      ? '← Espace utilisateur'
                      : 'Accès administrateur →'}
                  </button>

                  <p className="auth-signup-prompt">
                    Pas encore de compte ?{' '}
                    <Link to="/signup" className="auth-link">
                      Créer un compte
                    </Link>
                  </p>
                </div>
              </CardFooter>
            </form>

          </Card>
        </div>

      </div>
    </div>
  );
}
