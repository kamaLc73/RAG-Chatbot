import { apiRequest, tokenStore } from './client';
import type { AuthResponse, User } from './types';

interface LoginPayload {
  email?: string;
  username?: string;
  password: string;
  mode?: 'user' | 'admin';
}

interface SignupPayload extends LoginPayload {
  name: string;
}

export async function login(payload: LoginPayload) {
  const data = await apiRequest<AuthResponse>('/auth/login', {
    method: 'POST',
    body: JSON.stringify(payload),
    skipAuth: true,
  });
  tokenStore.setToken(data.access_token);
  tokenStore.setUser(data.user);
  return data.user;
}

export async function signup(payload: SignupPayload) {
  const data = await apiRequest<AuthResponse>('/auth/signup', {
    method: 'POST',
    body: JSON.stringify(payload),
    skipAuth: true,
  });
  tokenStore.setToken(data.access_token);
  tokenStore.setUser(data.user);
  return data.user;
}

export async function me() {
  return apiRequest<User>('/auth/me');
}

export function logout() {
  tokenStore.clear();
}
