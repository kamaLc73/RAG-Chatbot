import {
  AlertTriangle,
  AtSign,
  CalendarDays,
  Edit3,
  Mail,
  MessageSquare,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  Trash2,
  X,
} from 'lucide-react';
import { FormEvent, useEffect, useMemo, useState } from 'react';
import { createUser, deleteUser, listUsers, updateUser } from '../../api/admin';
import type { User } from '../../api/types';
import { useAuth } from '../../contexts/AuthContext';
import Button from '../ui/Button';
import Input from '../ui/Input';

interface UserFormState {
  full_name: string;
  username: string;
  email: string;
  password: string;
  is_superuser: boolean;
}

const emptyForm: UserFormState = {
  full_name: '',
  username: '',
  email: '',
  password: '',
  is_superuser: false,
};

function initials(user: User) {
  const source = user.name || user.username || user.email;
  return source.trim().charAt(0).toUpperCase() || 'U';
}

function formatJoinedDate(value?: string) {
  if (!value) return 'Non renseigné';
  return new Date(value).toLocaleString('fr-FR', {
    dateStyle: 'medium',
    timeStyle: 'short',
  });
}

function userToForm(user: User): UserFormState {
  return {
    full_name: user.full_name || user.name || '',
    username: user.username || '',
    email: user.email,
    password: '',
    is_superuser: user.role === 'admin' || Boolean(user.is_superuser),
  };
}

function UserModal({
  mode,
  form,
  error,
  isSaving,
  onChange,
  onClose,
  onSubmit,
}: {
  mode: 'create' | 'edit';
  form: UserFormState;
  error: string;
  isSaving: boolean;
  onChange: (patch: Partial<UserFormState>) => void;
  onClose: () => void;
  onSubmit: (event: FormEvent) => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <form onSubmit={onSubmit} className="w-full max-w-lg rounded-lg border border-border bg-card p-6 shadow-xl">
        <div className="mb-5 flex items-start justify-between gap-4">
          <div>
            <h3 className="text-xl font-semibold text-foreground">
              {mode === 'create' ? 'Créer un utilisateur' : 'Modifier l’utilisateur'}
            </h3>
            <p className="mt-1 text-sm text-muted-foreground">
              {mode === 'create'
                ? 'Renseignez les informations du nouveau compte.'
                : 'Modifiez le nom, les identifiants, le mot de passe ou le rôle.'}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground"
            aria-label="Fermer"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="grid gap-4">
          {error && <p className="form-error">{error}</p>}
          <Input
            label="Nom complet"
            value={form.full_name}
            onChange={(event) => onChange({ full_name: event.target.value })}
            placeholder="Nom complet"
          />
          <Input
            label="Identifiant"
            value={form.username}
            onChange={(event) => onChange({ username: event.target.value })}
            placeholder="identifiant"
            required
          />
          <Input
            label="Email"
            type="email"
            value={form.email}
            onChange={(event) => onChange({ email: event.target.value })}
            placeholder="utilisateur@cdg.dev"
            required
          />
          <Input
            label={mode === 'create' ? 'Mot de passe' : 'Nouveau mot de passe'}
            type="password"
            value={form.password}
            onChange={(event) => onChange({ password: event.target.value })}
            placeholder={mode === 'create' ? 'Mot de passe' : 'Laisser vide pour ne pas changer'}
            required={mode === 'create'}
          />
          <label className="flex items-center gap-3 rounded-lg border border-border bg-background p-3 text-sm font-medium text-foreground">
            <input
              type="checkbox"
              checked={form.is_superuser}
              onChange={(event) => onChange({ is_superuser: event.target.checked })}
              className="h-4 w-4 accent-primary"
            />
            Administrateur
          </label>
        </div>

        <div className="mt-6 flex justify-end gap-2">
          <Button type="button" variant="outline" onClick={onClose}>
            Annuler
          </Button>
          <Button type="submit" isLoading={isSaving}>
            {mode === 'create' ? 'Créer' : 'Enregistrer'}
          </Button>
        </div>
      </form>
    </div>
  );
}

function DeleteUserModal({
  user,
  error,
  isDeleting,
  onClose,
  onConfirm,
}: {
  user: User;
  error: string;
  isDeleting: boolean;
  onClose: () => void;
  onConfirm: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <div className="w-full max-w-md rounded-lg border border-border bg-card p-6 shadow-xl">
        <div className="mb-4 flex items-start gap-3">
          <div className="rounded-full bg-destructive/10 p-2 text-destructive">
            <AlertTriangle className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-foreground">Supprimer l’utilisateur</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              Cette action est définitive. Le compte sera supprimé de la base de données.
            </p>
          </div>
        </div>

        <div className="rounded-lg border border-border bg-muted/30 p-4 text-sm text-foreground">
          <p className="font-semibold">{user.name || user.username || user.email}</p>
          <p className="mt-1 text-muted-foreground">{user.email}</p>
        </div>

        {error && <p className="form-error mt-4">{error}</p>}

        <div className="mt-6 flex justify-end gap-2">
          <Button type="button" variant="outline" onClick={onClose} disabled={isDeleting}>
            Annuler
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={onConfirm}
            isLoading={isDeleting}
            className="user-delete-button"
          >
            <Trash2 className="h-4 w-4" />
            Supprimer définitivement
          </Button>
        </div>
      </div>
    </div>
  );
}

interface UsersManagerProps {
  onViewConversations?: (user: User) => void;
}

export default function UsersManager({ onViewConversations }: UsersManagerProps) {
  const { user: currentUser, syncUser } = useAuth();
  const [users, setUsers] = useState<User[]>([]);
  const [query, setQuery] = useState('');
  const [modalMode, setModalMode] = useState<'create' | 'edit' | null>(null);
  const [editingUser, setEditingUser] = useState<User | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<User | null>(null);
  const [form, setForm] = useState<UserFormState>(emptyForm);
  const [error, setError] = useState('');
  const [deleteError, setDeleteError] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const loadUsers = async () => {
    setIsLoading(true);
    try {
      setUsers(await listUsers());
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void loadUsers().catch(() => setUsers([]));
  }, []);

  const filteredUsers = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return users;
    return users.filter((user) =>
      [user.name, user.full_name, user.username, user.email, user.id]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(needle))
    );
  }, [query, users]);

  const openCreate = () => {
    setModalMode('create');
    setEditingUser(null);
    setForm(emptyForm);
    setError('');
  };

  const openEdit = (user: User) => {
    setModalMode('edit');
    setEditingUser(user);
    setForm(userToForm(user));
    setError('');
  };

  const closeModal = () => {
    setModalMode(null);
    setEditingUser(null);
    setForm(emptyForm);
    setError('');
  };

  const submitForm = async (event: FormEvent) => {
    event.preventDefault();
    setError('');
    setIsSaving(true);

    const username = form.username.trim();
    const email = form.email.trim();
    const fullName = form.full_name.trim();

    if (!username || !email) {
      setError('Veuillez renseigner un identifiant et un email.');
      setIsSaving(false);
      return;
    }

    if (modalMode === 'create' && !form.password) {
      setError('Veuillez renseigner un mot de passe.');
      setIsSaving(false);
      return;
    }

    try {
      if (modalMode === 'create') {
        await createUser({
          username,
          email,
          password: form.password,
          full_name: fullName || null,
          is_superuser: form.is_superuser,
        });
      } else if (modalMode === 'edit' && editingUser) {
        const updatedUser = await updateUser(editingUser.id, {
          username,
          email,
          password: form.password || undefined,
          full_name: fullName || null,
          is_superuser: form.is_superuser,
        });
        if (currentUser && String(currentUser.id) === String(updatedUser.id)) {
          syncUser(updatedUser);
        }
      }
      await loadUsers();
      closeModal();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Impossible d’enregistrer l’utilisateur.');
    } finally {
      setIsSaving(false);
    }
  };

  const requestDelete = (user: User) => {
    setDeleteTarget(user);
    setDeleteError('');
  };

  const closeDeleteModal = () => {
    setDeleteTarget(null);
    setDeleteError('');
  };

  const confirmDelete = async () => {
    if (!deleteTarget) return;
    if (currentUser && String(currentUser.id) === String(deleteTarget.id)) {
      setDeleteError('Impossible de supprimer le compte actuellement connecté.');
      return;
    }

    setIsDeleting(true);
    setDeleteError('');
    try {
      await deleteUser(deleteTarget.id);
      await loadUsers();
      closeDeleteModal();
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : 'Impossible de supprimer cet utilisateur.');
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <section className="users-manager">
      <div className="users-toolbar">
        <div className="users-heading">
          <h2>Utilisateurs</h2>
          <p>Gérer et consulter les comptes utilisateur</p>
        </div>
        <div className="users-actions">
          <span>
            Total : {users.length} utilisateur{users.length > 1 ? 's' : ''}
          </span>
          <Button variant="outline" onClick={() => void loadUsers()} disabled={isLoading} className="gap-2">
            <RefreshCw className="h-4 w-4" />
            Actualiser
          </Button>
          <Button onClick={openCreate} className="gap-2">
            <Plus className="h-4 w-4" />
            Créer un utilisateur
          </Button>
        </div>
      </div>

      <div className="users-search">
        <Search className="users-search-icon" />
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Rechercher par identifiant, email ou nom..."
          className="h-12 w-full rounded-lg border border-input bg-background pl-12 pr-4 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
        />
      </div>

      <div className="users-list">
        {filteredUsers.map((user) => {
          const isAdmin = user.role === 'admin' || Boolean(user.is_superuser);
          const isCurrentUser = currentUser ? String(currentUser.id) === String(user.id) : false;
          return (
            <article key={user.id} className="user-card">
              <div className="user-avatar">{initials(user)}</div>

              <div className="user-card-body">
                <div className="user-title-row">
                  <h3>{user.name || user.username}</h3>
                  {isAdmin && (
                    <span className="user-admin-badge">
                      <ShieldCheck className="h-4 w-4" />
                      Admin
                    </span>
                  )}
                </div>

                <div className="user-meta">
                  <span>
                    <Mail />
                    Email : {user.email}
                  </span>
                  <span>
                    <AtSign />
                    Identifiant : {user.username || '-'}
                  </span>
                  <span>
                    <CalendarDays />
                    Créé le : {formatJoinedDate(user.created_at)}
                  </span>
                </div>
              </div>

              <div className="user-card-actions">
                <Button variant="outline" onClick={() => onViewConversations?.(user)} className="gap-2">
                  <MessageSquare className="h-4 w-4" />
                  Voir ses conversations
                </Button>
                <Button variant="outline" onClick={() => openEdit(user)} className="gap-2">
                  <Edit3 className="h-4 w-4" />
                  Modifier
                </Button>
                <Button
                  variant="outline"
                  onClick={() => requestDelete(user)}
                  className="user-delete-button"
                  disabled={isCurrentUser}
                  title={isCurrentUser ? 'Impossible de supprimer le compte connecté' : 'Supprimer définitivement'}
                >
                  <Trash2 className="h-4 w-4" />
                  Supprimer
                </Button>
              </div>
            </article>
          );
        })}

        {filteredUsers.length === 0 && (
          <div className="rounded-lg border border-dashed border-border bg-muted/30 p-10 text-center text-sm text-muted-foreground">
            Aucun utilisateur trouvé.
          </div>
        )}
      </div>

      <p className="users-count-footer">
        Affichage de {filteredUsers.length} sur {users.length} utilisateur{users.length > 1 ? 's' : ''}
      </p>

      {modalMode && (
        <UserModal
          mode={modalMode}
          form={form}
          error={error}
          isSaving={isSaving}
          onChange={(patch) => setForm((current) => ({ ...current, ...patch }))}
          onClose={closeModal}
          onSubmit={(event) => void submitForm(event)}
        />
      )}

      {deleteTarget && (
        <DeleteUserModal
          user={deleteTarget}
          error={deleteError}
          isDeleting={isDeleting}
          onClose={closeDeleteModal}
          onConfirm={() => void confirmDelete()}
        />
      )}
    </section>
  );
}
