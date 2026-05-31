import { createContext, useContext, useMemo, useState, type ReactNode } from 'react';
import { login as loginApi, logout as logoutApi, signup as signupApi } from '../api/auth';
import { tokenStore } from '../api/client';
import type { Organization, User } from '../api/types';

interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  login: (payload: { identifier: string; password: string; mode: 'user' | 'admin' }) => Promise<void>;
  signup: (payload: { name: string; email: string; password: string; organization?: Organization; username?: string }) => Promise<void>;
  syncUser: (user: User) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(() => tokenStore.getUser());

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      isAuthenticated: Boolean(user && tokenStore.getToken()),
      login: async ({ identifier, password, mode }) => {
        const loginPayload =
          mode === 'admin'
            ? { email: identifier, password, mode }
            : { username: identifier, email: identifier.includes('@') ? identifier : undefined, password, mode };
        const nextUser = await loginApi(loginPayload);
        setUser(nextUser);
      },
      signup: async (payload) => {
        const nextUser = await signupApi(payload);
        setUser(nextUser);
      },
      syncUser: (nextUser) => {
        tokenStore.setUser(nextUser);
        setUser(nextUser);
      },
      logout: () => {
        logoutApi();
        setUser(null);
      },
    }),
    [user]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used inside AuthProvider');
  }
  return context;
}
