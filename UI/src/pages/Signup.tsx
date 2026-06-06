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

export default function Signup() {
  const { signup } = useAuth();
  const navigate = useNavigate();
  const [fullName, setFullName] = useState('');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
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
      await signup({ name: fullName || username, username, email, password });
      navigate('/chat', { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Création impossible');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="auth-shell auth-shell--signup">
      <div className="auth-stage auth-stage--signup">
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
                <CardTitle className="auth-title">Créer un compte</CardTitle>
                <CardDescription className="auth-subtitle">
                  Démarrez avec l'assistant RCAR / CNRA
                </CardDescription>
              </div>
            </CardHeader>

            <form onSubmit={(event) => void submit(event)}>
              <CardContent className="auth-card-content">
                {error && <p className="form-error">{error}</p>}

                <Input
                  label="Nom complet"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="Nom et prénom"
                  autoComplete="name"
                />
                <Input
                  label="Identifiant"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="votre.identifiant"
                  autoComplete="username"
                  required
                />
                <Input
                  label="Email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="vous@exemple.com"
                  autoComplete="email"
                  required
                />

                <div className="auth-password-field">
                  <Input
                    label="Mot de passe"
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Créer un mot de passe"
                    autoComplete="new-password"
                    required
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

                <Input
                  label="Confirmer le mot de passe"
                  type={showPassword ? 'text' : 'password'}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Confirmer votre mot de passe"
                  autoComplete="new-password"
                  required
                />
              </CardContent>

              <CardFooter className="auth-card-footer">
                <Button
                  type="submit"
                  className="auth-submit-btn w-full"
                  isLoading={isLoading}
                  disabled={isLoading}
                >
                  Créer mon compte
                </Button>

                <div className="auth-footer-links">
                  <p className="auth-signup-prompt">
                    Déjà inscrit ?{' '}
                    <Link to="/login" className="auth-link">
                      Se connecter
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
